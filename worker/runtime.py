"""Mtumishi runtime execution for leased jobs."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

from datetime import date
from typing import Any, Callable, Dict, Optional

from geocongoai import GeoSpatialContext, Mtumishi
from geocongoai.mtumishi.tools import GeoTool
from geocongoai.mtumishi.events import GeoTraceEvent
from geocongoai.mtumishi.planner import MtumishiResult
try:
    from geocongoai.mtumishi.runtimes.native import _detect_conversational_intent
except ImportError:
    # This SDK helper is private. Its absence must not stop the worker.
    _detect_conversational_intent = lambda _prompt: False

from worker.forest_change import analyze_forest_change
from worker.study_report import build_study_report
from worker.models import LeasedRun, WorkerConfig


QGIS_AREA_TOOL = "qgis_headless_measure_area"
logger = logging.getLogger(__name__)


def measure_web_area(geometry: Any) -> Dict[str, Any]:
    script = Path(__file__).with_name("qgis_headless.py")
    try:
        with TemporaryDirectory(prefix="mtumishi-qgis-") as workspace:
            process = subprocess.run(
                ["/usr/bin/python3", "-I", str(script.resolve())],
                input=json.dumps(geometry), text=True, capture_output=True,
                cwd=workspace,
                env={"PATH": "/usr/bin:/bin", "QT_QPA_PLATFORM": "offscreen",
                     "HOME": workspace, "XDG_RUNTIME_DIR": workspace,
                     "LANG": os.environ.get("LANG", "C.UTF-8")},
                timeout=30, check=True,
            )
        return json.loads(process.stdout)
    except subprocess.TimeoutExpired:
        return {"error": "La mesure QGIS a dépassé 30 secondes."}
    except (OSError, subprocess.CalledProcessError, ValueError):
        logger.exception("Headless QGIS area measurement failed")
        return {"error": "Le moteur QGIS headless est indisponible."}


def tool_failure(payload: Any) -> Optional[str]:
    if isinstance(payload, dict) and (payload.get("status") in ("error", "failed") or payload.get("error")):
        return str(payload.get("error") or payload.get("message") or "Échec de l’outil.")[:500]
    return None


def build_context(raw_context: Dict[str, Any]) -> GeoSpatialContext:
    """Build a `GeoSpatialContext` from the job payload."""
    payload = dict(raw_context or {})
    metadata = dict(payload.get("metadata") or {})
    for key in ("environment", "client", "geometry", "area_km2", "points_count"):
        if key in payload:
            metadata[key] = payload[key]
    payload["metadata"] = metadata
    return GeoSpatialContext.from_dict(payload)


def estimate_final_units(run: LeasedRun, result: Dict[str, Any]) -> int:
    """Basic billing fallback until token-based pricing is exposed by the runtime."""
    del result
    return max(1, run.units_reserved)


def collect_execution_steps(result_steps, memory_steps, event_steps, is_web):
    """Recover executed tool outputs even when Antigravity omits ToolResult chunks.

    GeoTool writes actual results to MtumishiMemory before returning to the model.
    Events and MtumishiResult may only contain starts on SDK 0.4.5. Merge by
    occurrence per tool, preserving repeated invocations and actual failures.
    """
    sources = []
    recorded = []
    if isinstance(memory_steps, (list, tuple)):
        for step in memory_steps:
            if isinstance(step, dict):
                recorded.append(dict(step))
            else:
                recorded.append({
                    "tool": step.tool_name, "args": step.arguments,
                    "result": step.result, "status": step.status,
                })
    sources.append(recorded)
    sources.append(result_steps if isinstance(result_steps, (list, tuple)) else [])
    sources.append(event_steps if isinstance(event_steps, (list, tuple)) else [])
    merged = []
    by_tool = {}
    for source in sources:
        seen = {}
        for step in source:
            if not isinstance(step, dict):
                continue
            name = str(step.get("tool") or "")
            if not name or name.startswith("dev_") or (is_web and name.startswith("qgis_") and name != QGIS_AREA_TOOL):
                continue
            occurrence = seen.get(name, 0)
            seen[name] = occurrence + 1
            previous = by_tool.setdefault(name, [])
            if occurrence < len(previous):
                existing = previous[occurrence]
                if existing.get("status") in (None, "running", "queued") and step.get("status") not in (None, "running", "queued"):
                    existing.update(step)
                continue
            copy = dict(step)
            merged.append(copy)
            previous.append(copy)
    return merged


def execute_run(
    run: LeasedRun,
    config: WorkerConfig,
    on_event: Optional[Callable[[Any], None]] = None,
    steps_collector: Optional[list] = None,
) -> Dict[str, Any]:
    """Execute a single Mtumishi job."""
    # Detect execution environment: Web vs QGIS
    client_env = (run.context_json or {}).get("environment") or "web"
    is_web = (client_env == "web")
    context = run.context_json or {}
    study = context.get("gundua_study")
    report_requested = is_web and context.get("report_pdf") is True and isinstance(study, str) and bool(study.strip())

    mode = context.get("gundua_mode")
    direct_requested = is_web and mode == "direct"
    direct_forest = report_requested and study == "deforestation" and mode in (None, "direct")
    unsupported_direct = direct_requested and not direct_forest
    geometry = context.get("geometry")
    generated_deliverables = []

    def forest_change(before_year: int, after_year: int) -> Dict[str, Any]:
        if not isinstance(geometry, dict):
            return {"status": "error", "error": "Dessinez un polygone sur la carte avant de lancer l’étude."}
        try:
            stats, layers = analyze_forest_change(geometry, before_year, after_year)
        except ValueError as exc:
            return {"status": "error", "error": str(exc)[:500]}
        except Exception:
            logger.exception("Sentinel-2 forest change analysis failed")
            return {"status": "error", "error": "Impossible de lire ou traiter les acquisitions Sentinel-2."}
        generated_deliverables[:] = layers
        return stats

    # Priority: if conversational intent (greetings, help, identity, etc.), route to native for instant response (< 50ms)
    prompt_str = run.prompt or ""
    is_conversational = False
    try:
        is_conversational = bool(_detect_conversational_intent(prompt_str))
    except Exception:
        pass

    runtime_type = "deterministic"
    model_name = None
    mtumishi = None
    if not (direct_forest or unsupported_direct):
        runtime_type = "native" if is_conversational else run.requested_runtime or config.default_runtime
        model_name = run.model_name or config.default_model_name
        if is_web and runtime_type == "antigravity" and model_name == "gemini-2.5-flash":
            model_name = config.default_model_name
        runtime_kwargs = {"model_name": model_name}
        if runtime_type != "native":
            runtime_kwargs.update(api_key=config.gemini_api_key, workspaces=[])
        mtumishi = Mtumishi(
            runtime_type=runtime_type,
            qgis_rpc_port=config.qgis_rpc_port,
            runtime_kwargs=runtime_kwargs,
        )

    # When running on the Web: remove all desktop QGIS and file-dev tools
    # so the agent never attempts QGIS actions or dev tools
    if is_web and hasattr(mtumishi, "registry") and hasattr(mtumishi.registry, "_tools"):
        web_forbidden_prefixes = ("qgis_", "dev_")
        mtumishi.registry._tools = {
            name: tool for name, tool in mtumishi.registry._tools.items()
            if not any(name.startswith(p) for p in web_forbidden_prefixes)
        }

    context_override = build_context(run.context_json)
    if mtumishi is not None and is_web and not is_conversational and geometry:

        def measure_selected_area() -> Dict[str, Any]:
            return measure_web_area(geometry)

        mtumishi.register_tool(GeoTool(
            name=QGIS_AREA_TOOL,
            description="Mesure la superficie ellipsoïdale de la zone dessinée et produit sa couche GeoJSON.",
            parameters={"type": "object", "properties": {}},
            handler=measure_selected_area,
        ))

        mtumishi.register_tool(GeoTool(
            name="sentinel2_forest_change",
            description="Compare les composites Sentinel-2 d’août à septembre de deux années sur le polygone dessiné. Livre le raster de variation NDVI, son GeoTIFF et les contours candidats. Ne confirme pas une déforestation.",
            parameters={"type": "object", "properties": {
                "before_year": {"type": "integer"}, "after_year": {"type": "integer"},
            }, "required": ["before_year", "after_year"]},
            handler=forest_change,
        ))

    # Add environment contextual hint for the agent prompt if in Web mode
    execution_prompt = run.prompt
    if is_web and not is_conversational:
        execution_prompt = (
            "[Contexte d'exécution : Application Web GeoCongo AI. Tu es exécuté sur l'interface cartographique web (pas dans QGIS Desktop). Pour mesurer la superficie de la zone dessinée, utilise qgis_headless_measure_area si disponible. Pour une comparaison de déforestation ou de végétation entre deux années, utilise sentinel2_forest_change si disponible. Concentre-toi sur l'imagerie satellitaire (STAC), la géologie (Pekua) et la détection d'indices minéraux/végétaux (Gundua) pour affichage sur la carte interactive.]\n\n"
            + run.prompt
        )

    try:
        if direct_forest:
            today = date.today()
            last = today.year if today.month >= 10 else today.year - 1
            args = {"before_year": context.get("forest_before_year", last - 1),
                    "after_year": context.get("forest_after_year", last)}
            name = "sentinel2_forest_change"
            if on_event:
                on_event(GeoTraceEvent.tool_call(name, args))
            before, after = args["before_year"], args["after_year"]
            if type(before) is not int or type(after) is not int or not 2018 <= before < after <= last:
                stats = {"status": "error", "error": "Choisissez deux années croissantes entre 2018 et la dernière saison août–septembre terminée."}
            else:
                stats = forest_change(**args)
            failure = tool_failure(stats)
            if on_event:
                on_event(GeoTraceEvent.tool_result(name, stats, status="error" if failure else "success"))
            if failure:
                summary = failure
            else:
                summary = (
                    f"Août–septembre {args['before_year']} et {args['after_year']} · Sentinel-2 L2A à 20 m. "
                    f"Couverture comparable : {stats['coverage_percent']} %. "
                    f"Zones candidates : {stats['candidate_count']}, soit {stats['candidate_ha']} ha. "
                    "Les contours indiquent des baisses spectrales à vérifier sur des données complémentaires."
                )
            result = MtumishiResult(summary=summary, status="failed" if failure else "success",
                                    steps=[{"tool": name, "args": args, "status": "error" if failure else "success", "result": stats}],
                                    context=context_override)
        elif unsupported_direct:
            result = MtumishiResult(summary="Cette étude est disponible avec l’assistant Mtumishi.", status="failed",
                                    context=context_override)
        elif on_event and hasattr(mtumishi, "run_with_stream"):
            # Never rerun the analysis after a streaming error: tools may already have executed.
            result = mtumishi.run_with_stream(
                prompt=execution_prompt,
                on_event=on_event,
                context_override=context_override,
            )
        else:
            result = mtumishi.run(prompt=execution_prompt, context_override=context_override)
    except Exception:
        if not report_requested:
            raise
        logger.exception("Gundua study execution interrupted")
        result = MtumishiResult(summary="L’exécution de l’étude a été interrompue. Les résultats partiels disponibles figurent dans le rapport.", status="failed")

    cleaned_steps = collect_execution_steps(
        result.steps,
        getattr(getattr(mtumishi, "memory", None), "steps", None),
        steps_collector,
        is_web,
    )

    # Keep every actual tool result in steps. The API adapts them to the web contract.
    # A STAC scene preview must not be stretched over the AOI or turned into a synthetic result.
    geojson_payload = getattr(result, "geojson", None)
    raster_overlay_payload = getattr(result, "raster_overlay", None)
    deliverables = getattr(result, "deliverables", None)
    if not isinstance(deliverables, (dict, list)):
        deliverables = {}

    for step in cleaned_steps:
        failure = tool_failure(step.get("result"))
        if failure:
            step["status"] = "error"
            step["error"] = failure

    sdk_status = getattr(result.status, "value", result.status)
    failed_steps = [step for step in cleaned_steps if step.get("status") in ("error", "failed", "running", "queued")
                    or isinstance(step.get("result"), dict) and step["result"].get("error")]
    status = "completed" if sdk_status in ("success", "completed") and not failed_steps else "failed"
    summary = result.summary or ""
    if any(step.get("status") in ("running", "queued") for step in cleaned_steps):
        summary += "\n\nCertains résultats d’outils n’ont pas pu être récupérés."
    if report_requested and not cleaned_steps and not unsupported_direct:
        status = "failed"
        summary += "\n\nAucun outil d’analyse n’a renvoyé de résultat pour cette étude."
    if report_requested:
        try:
            generated_deliverables.append(build_study_report(
                study[:100], run.id, status, context.get("bbox"), summary, cleaned_steps, generated_deliverables,
            ))
        except Exception:
            logger.exception("PDF generation failed for Mtumishi run %s", run.id)
            status = "failed"
            summary += "\n\nLe rapport PDF n’a pas pu être généré. Les données disponibles restent accessibles."
    if generated_deliverables:
        deliverables = [deliverables, *generated_deliverables]

    return {
        "status": status,
        "analysisType": study if report_requested else None,
        "sdk_status": sdk_status,
        "deliverables": deliverables,
        "summary": summary,
        "steps": cleaned_steps,
        "qgis_scripts": [] if is_web else result.qgis_scripts,
        "geojson": geojson_payload,
        "raster_overlay": raster_overlay_payload,
        "context": result.context.to_dict() if result.context else None,
        "metadata": {
            "user_id": run.user_id,
            "run_id": run.id,
            "environment": client_env,
            "requested_runtime": runtime_type,
            "model_name": model_name,
            "worker_id": config.worker_id,
        },
    }
