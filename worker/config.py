"""Configuration loading for the worker."""
import os
import socket
from pathlib import Path

from worker.models import WorkerConfig


def _load_dotenv() -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.is_file():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def load_config() -> WorkerConfig:
    """Load worker config from `.env` or process environment."""
    _load_dotenv()

    worker_id = os.getenv("MTUMISHI_WORKER_ID") or f"{socket.gethostname()}-{os.getpid()}"
    workspaces_raw = os.getenv("MTUMISHI_WORKSPACES", ".")
    workspaces = [item.strip() for item in workspaces_raw.split(",") if item.strip()]

    return WorkerConfig(
        supabase_url=_require_env("SUPABASE_URL"),
        supabase_service_role_key=_require_env("SUPABASE_SERVICE_ROLE_KEY"),
        gemini_api_key=_require_env("GEMINI_API_KEY"),
        worker_id=worker_id,
        poll_seconds=float(os.getenv("MTUMISHI_POLL_SECONDS", "2")),
        lease_seconds=int(os.getenv("MTUMISHI_LEASE_SECONDS", "120")),
        qgis_rpc_port=int(os.getenv("MTUMISHI_QGIS_RPC_PORT", "65534")),
        default_reserved_units=int(os.getenv("MTUMISHI_DEFAULT_RESERVED_UNITS", "12")),
        default_model_name=os.getenv("MTUMISHI_MODEL_NAME", "gemini-2.5-flash"),
        workspaces=workspaces or ["."],
    )
