"""Supabase data access for the worker."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from supabase import Client, create_client

from worker.models import LeasedRun, WorkerConfig


def utc_now_iso() -> str:
    """Return an ISO-8601 timestamp in UTC."""
    return datetime.now(timezone.utc).isoformat()


class SupabaseRunStore:
    """Encapsulate all Supabase reads and writes for agent runs."""

    def __init__(self, config: WorkerConfig):
        self.config = config
        self.client: Client = create_client(
            config.supabase_url,
            config.supabase_service_role_key,
        )

    def lease_next_run(self) -> Optional[LeasedRun]:
        """Lease the next queued run using an RPC with `FOR UPDATE SKIP LOCKED`."""
        response = self.client.rpc(
            "lease_next_mtumishi_run",
            {
                "p_worker_id": self.config.worker_id,
                "p_lease_seconds": self.config.lease_seconds,
            },
        ).execute()

        rows = response.data or []
        if not rows:
            return None

        row = rows[0]
        return LeasedRun(
            id=row["id"],
            user_id=row["user_id"],
            status=row["status"],
            prompt=row["prompt"],
            context_json=row.get("context_json") or {},
            requested_runtime=row.get("requested_runtime") or "antigravity",
            model_name=row.get("model_name"),
            units_reserved=int(row.get("units_reserved") or self.config.default_reserved_units),
            api_key_id=row.get("api_key_id"),
        )

    def mark_completed(self, run: LeasedRun, result: Dict[str, Any], units_final: int) -> None:
        """Persist successful execution results."""
        payload = {
            "status": "completed",
            "summary": result.get("summary"),
            "steps_json": result.get("steps", []),
            "qgis_scripts_json": result.get("qgis_scripts", []),
            "result_json": result,
            "units_final": units_final,
            "lease_until": None,
            "finished_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
        }
        self.client.table("agent_runs").update(payload).eq("id", run.id).execute()

        self.client.rpc(
            "finalize_mtumishi_units",
            {
                "p_user_id": run.user_id,
                "p_run_id": run.id,
                "p_reserved": run.units_reserved,
                "p_final": units_final,
            },
        ).execute()

    def mark_failed(self, run: LeasedRun, error_message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Persist execution failures."""
        payload = {
            "status": "failed",
            "error_json": {
                "message": error_message,
                "details": details or {},
            },
            "lease_until": None,
            "finished_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
        }
        self.client.table("agent_runs").update(payload).eq("id", run.id).execute()
