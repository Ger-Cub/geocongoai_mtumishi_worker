"""Lightweight models for the worker."""
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class WorkerConfig:
    """Runtime configuration loaded from environment variables."""

    supabase_url: str
    supabase_service_role_key: str
    gemini_api_key: str
    worker_id: str
    poll_seconds: float
    lease_seconds: int
    qgis_rpc_port: int
    default_reserved_units: int
    default_model_name: str
    workspaces: list[str]


@dataclass
class LeasedRun:
    """Row fetched from `agent_runs`."""

    id: str
    user_id: str
    status: str
    prompt: str
    context_json: Dict[str, Any]
    requested_runtime: str
    model_name: Optional[str]
    units_reserved: int
    api_key_id: Optional[str] = None
