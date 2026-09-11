"""Mtumishi runtime execution for leased jobs."""
from __future__ import annotations

from typing import Any, Dict

from geocongoai import GeoSpatialContext, Mtumishi

from worker.models import LeasedRun, WorkerConfig


def build_context(raw_context: Dict[str, Any]) -> GeoSpatialContext:
    """Build a `GeoSpatialContext` from the job payload."""
    return GeoSpatialContext.from_dict(raw_context or {})


def estimate_final_units(run: LeasedRun, result: Dict[str, Any]) -> int:
    """Basic billing fallback until token-based pricing is exposed by the runtime."""
    del result
    return max(1, run.units_reserved)


def execute_run(run: LeasedRun, config: WorkerConfig) -> Dict[str, Any]:
    """Execute a single Mtumishi job."""
    mtumishi = Mtumishi(
        runtime_type=run.requested_runtime or "antigravity",
        qgis_rpc_port=config.qgis_rpc_port,
        runtime_kwargs={
            "api_key": config.gemini_api_key,
            "model_name": run.model_name or config.default_model_name,
            "workspaces": config.workspaces,
        },
    )

    result = mtumishi.run(
        prompt=run.prompt,
        context_override=build_context(run.context_json),
    )

    return {
        "status": result.status,
        "summary": result.summary,
        "steps": result.steps,
        "qgis_scripts": result.qgis_scripts,
        "context": result.context.to_dict() if result.context else None,
        "metadata": {
            "user_id": run.user_id,
            "run_id": run.id,
            "requested_runtime": run.requested_runtime,
            "model_name": run.model_name or config.default_model_name,
            "worker_id": config.worker_id,
        },
    }
