"""Main entrypoint for the GeoCongoAI Mtumishi worker."""
from __future__ import annotations

import logging
import time
import traceback
from typing import Any

from worker.config import load_config
from worker.runtime import estimate_final_units, execute_run
from worker.supabase_ops import SupabaseRunStore


def configure_logging() -> None:
    """Configure process-wide logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def main() -> None:
    """Run the worker loop forever."""
    configure_logging()
    logger = logging.getLogger("geocongoai.mtumishi.worker")

    config = load_config()
    store = SupabaseRunStore(config)

    logger.info("Worker started with id=%s", config.worker_id)

    while True:
        run = store.lease_next_run()
        if run is None:
            time.sleep(config.poll_seconds)
            continue

        logger.info("Leased run id=%s user_id=%s runtime=%s", run.id, run.user_id, run.requested_runtime)

        steps_collector: list = []
        text_chunks: list = []
        last_update_time = [0.0]

        def _on_event(ev: Any) -> None:
            now = time.time()
            ev_type = getattr(ev, "type", None)
            meta = getattr(ev, "metadata", {}) or {}
            type_val = ev_type.value if hasattr(ev_type, "value") else str(ev_type)

            if type_val == "tool_call":
                tool_name = meta.get("tool") or "tool"
                # Exclude internal dev workspace tools
                if str(tool_name).startswith("dev_"):
                    return
                args = meta.get("args") or {}
                steps_collector.append({
                    "step": len(steps_collector) + 1,
                    "tool": tool_name,
                    "args": args,
                    "status": "running",
                })
                store.update_progress(run.id, steps_collector)
                last_update_time[0] = now
            elif type_val == "tool_result":
                tool_name = meta.get("tool")
                if str(tool_name).startswith("dev_"):
                    return
                res = meta.get("result")
                status = meta.get("status") or "success"
                for st in reversed(steps_collector):
                    if st.get("tool") == tool_name and st.get("status") == "running":
                        st["status"] = status
                        st["result"] = res
                        break
                store.update_progress(run.id, steps_collector)
                last_update_time[0] = now
            elif type_val == "text":
                content = getattr(ev, "content", "")
                if content:
                    text_chunks.append(content)
                    if now - last_update_time[0] > 1.2:
                        store.update_progress(run.id, steps_collector, "".join(text_chunks))
                        last_update_time[0] = now

        try:
            result = execute_run(run, config, on_event=_on_event)
            units_final = estimate_final_units(run, result)
            store.mark_completed(run, result, units_final)
            logger.info("Completed run id=%s units_final=%s", run.id, units_final)
        except Exception as exc:
            logger.exception("Run failed id=%s", run.id)
            store.mark_failed(
                run,
                error_message=str(exc),
                details={"traceback": traceback.format_exc()},
            )


if __name__ == "__main__":
    main()
