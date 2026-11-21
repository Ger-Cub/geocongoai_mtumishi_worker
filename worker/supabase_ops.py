"""Supabase data access for the worker."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from supabase import Client, create_client

from worker.models import LeasedRun, WorkerConfig

logger = logging.getLogger(__name__)


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
        """Persist results and final billing atomically; never bypass lease validation."""
        self.client.rpc(
            "complete_mtumishi_run",
            {
                "p_run_id": run.id,
                "p_worker_id": self.config.worker_id,
                "p_status": "completed",
                "p_final": units_final,
                "p_summary": result.get("summary"),
                "p_steps": result.get("steps", []),
                "p_qgis_scripts": result.get("qgis_scripts", []),
                "p_result": result,
            },
        ).execute()

    def heartbeat(self, run_id: str, lease_seconds: int = 180) -> None:
        """Extend the run lease so active long-running jobs never expire prematurely."""
        self._update_active_run(run_id, {}, lease_seconds)

    def update_progress(self, run_id: str, steps: list, summary_fragment: Optional[str] = None, extend_lease_seconds: int = 180) -> None:
        """Update live progress during execution and refresh the worker lease."""
        payload: Dict[str, Any] = {"steps_json": steps}
        if summary_fragment is not None:
            payload["summary"] = summary_fragment
        self._update_active_run(run_id, payload, extend_lease_seconds)

    def _update_active_run(self, run_id: str, payload: Dict[str, Any], lease_seconds: int) -> None:
        now = datetime.now(timezone.utc)
        payload["lease_until"] = (now + timedelta(seconds=lease_seconds)).isoformat()
        payload["updated_at"] = now.isoformat()
        try:
            (
                self.client.table("agent_runs")
                .update(payload)
                .eq("id", run_id)
                .eq("status", "running")
                .eq("worker_id", self.config.worker_id)
                .gt("lease_until", now.isoformat())
                .execute()
            )
        except Exception:
            logger.exception("Failed to update active Mtumishi run %s", run_id)

    def mark_failed(self, run: LeasedRun, error_message: str, details: Optional[Dict[str, Any]] = None, result: Optional[Dict[str, Any]] = None) -> None:
        """Persist execution failures."""
        self.client.rpc(
            "complete_mtumishi_run",
            {
                "p_run_id": run.id, "p_worker_id": self.config.worker_id,
                "p_status": "failed", "p_final": 0, "p_summary": error_message,
                "p_steps": (result or {}).get("steps", []),
                "p_result": {**(result or {}), "error": {"message": error_message, "details": details or {}}},
            },
        ).execute()
