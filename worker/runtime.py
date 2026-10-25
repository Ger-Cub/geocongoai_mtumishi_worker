"""Mtumishi runtime execution for leased jobs."""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from geocongoai import GeoSpatialContext, Mtumishi
from geocongoai.mtumishi.runtimes.native import _detect_conversational_intent

from worker.models import LeasedRun, WorkerConfig


def build_context(raw_context: Dict[str, Any]) -> GeoSpatialContext:
    """Build a `GeoSpatialContext` from the job payload."""
    return GeoSpatialContext.from_dict(raw_context or {})


def estimate_final_units(run: LeasedRun, result: Dict[str, Any]) -> int:
    """Basic billing fallback until token-based pricing is exposed by the runtime."""
    del result
    return max(1, run.units_reserved)


def execute_run(
    run: LeasedRun,
    config: WorkerConfig,
    on_event: Optional[Callable[[Any], None]] = None,
) -> Dict[str, Any]:
    """Execute a single Mtumishi job."""
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

    context_override = build_context(run.context_json)

    if on_event and hasattr(mtumishi, "run_with_stream"):
        try:
            result = mtumishi.run_with_stream(
                prompt=run.prompt,
                on_event=on_event,
                context_override=context_override,
            )
        except Exception:
            result = mtumishi.run(
                prompt=run.prompt,
                context_override=context_override,
            )
    else:
        result = mtumishi.run(
            prompt=run.prompt,
            context_override=context_override,
        )

    # Filter out any internal dev/workspace tools
    cleaned_steps = [
        st for st in (result.steps or [])
        if not str(st.get("tool", "")).startswith("dev_")
    ]

    return {
        "status": result.status,
        "summary": result.summary,
        "steps": cleaned_steps,
        "qgis_scripts": result.qgis_scripts,
        "context": result.context.to_dict() if result.context else None,
        "metadata": {
            "user_id": run.user_id,
            "run_id": run.id,
            "requested_runtime": runtime_type,
            "model_name": run.model_name or config.default_model_name,
            "worker_id": config.worker_id,
        },
    }
