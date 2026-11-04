"""Mtumishi runtime execution for leased jobs."""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from geocongoai import GeoSpatialContext, Mtumishi
try:
    from geocongoai.mtumishi.runtimes.native import _detect_conversational_intent
except ImportError:
    # This SDK helper is private. Its absence must not stop the worker.
    _detect_conversational_intent = lambda _prompt: False

from worker.models import LeasedRun, WorkerConfig


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
            if not name or name.startswith("dev_") or (is_web and name.startswith("qgis_")):
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

    # Priority: if conversational intent (greetings, help, identity, etc.), route to native for instant response (< 50ms)
    prompt_str = run.prompt or ""
    is_conversational = False
    try:
        is_conversational = bool(_detect_conversational_intent(prompt_str))
    except Exception:
        pass

    if is_conversational:
        runtime_type = "native"
    else:
        runtime_type = run.requested_runtime or config.default_runtime

    if runtime_type == "native":
        runtime_kwargs = {
            "model_name": run.model_name or config.default_model_name,
        }
    else:
        # antigravity runtime — never provide directory workspaces to prevent dev_list_directory!
        runtime_kwargs = {
            "api_key": config.gemini_api_key,
            "model_name": run.model_name or config.default_model_name,
            "workspaces": [],
        }

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

    # Add environment contextual hint for the agent prompt if in Web mode
    execution_prompt = run.prompt
    if is_web and not is_conversational:
        execution_prompt = (
            "[Contexte d'exécution : Application Web GeoCongo AI. Tu es exécuté sur l'interface cartographique web (pas dans QGIS Desktop). N'utilise aucun script ni outil QGIS. Concentre-toi sur l'imagerie satellitaire (STAC), la géologie (Pekua) et la détection d'indices minéraux/végétaux (Gundua) pour affichage sur la carte interactive.]\n\n"
            + run.prompt
        )

    if on_event and hasattr(mtumishi, "run_with_stream"):
        # Never rerun the analysis after a streaming error: tools may already have executed.
        result = mtumishi.run_with_stream(
            prompt=execution_prompt,
            on_event=on_event,
            context_override=context_override,
        )
    else:
        result = mtumishi.run(prompt=execution_prompt, context_override=context_override)

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

    sdk_status = getattr(result.status, "value", result.status)
    failed_steps = [step for step in cleaned_steps if step.get("status") in ("error", "failed", "running", "queued")
                    or isinstance(step.get("result"), dict) and step["result"].get("error")]
    status = "completed" if sdk_status in ("success", "completed") and not failed_steps else "failed"
    return {
        "status": status,
        "sdk_status": sdk_status,
        "deliverables": deliverables,
        "summary": result.summary if not any(step.get("status") in ("running", "queued") for step in cleaned_steps)
            else result.summary + "\n\nCertains résultats d’outils n’ont pas pu être récupérés ; l’exécution ne peut pas être déclarée terminée.",
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
            "model_name": run.model_name or config.default_model_name,
            "worker_id": config.worker_id,
        },
    }
