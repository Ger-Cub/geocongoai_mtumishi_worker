"""Main entrypoint for the GeoCongoAI Mtumishi worker."""
from __future__ import annotations

import logging
import threading
import time
import traceback
from typing import Any

from worker.config import load_config
from worker.runtime import estimate_final_units, execute_run
from worker.supabase_ops import SupabaseRunStore
from worker.realtime import MtumishiRealtimePublisher


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
    realtime = MtumishiRealtimePublisher(store.client)

    logger.info("Worker started with id=%s", config.worker_id)

    while True:
        try:
            run = store.lease_next_run()
        except Exception:
            logger.exception("Unable to lease a run; retrying")
            time.sleep(config.poll_seconds)
            continue
        if run is None:
            time.sleep(config.poll_seconds)
            continue

        logger.info("Leased run id=%s user_id=%s runtime=%s", run.id, run.user_id, run.requested_runtime)
        realtime.emit(run.id, "run_started", {"worker_id": config.worker_id, "runtime": run.requested_runtime, "model_name": run.model_name})

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
                realtime.emit(run.id, "tool_started", {"tool": tool_name, "args": args}, event_name=tool_name)
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
                realtime.emit(run.id, "tool_completed", {"tool": tool_name, "status": status, "result": res}, event_name=tool_name)
                last_update_time[0] = now
            elif type_val == "text":
                content = getattr(ev, "content", "")
                if content:
                    text_chunks.append(content)
                    if now - last_update_time[0] > 1.2:
                        store.update_progress(run.id, steps_collector, "".join(text_chunks))
                        last_update_time[0] = now

        heartbeat_stop = threading.Event()

        def _heartbeat_worker() -> None:
            while not heartbeat_stop.wait(timeout=15):
                store.heartbeat(run.id, lease_seconds=180)

        hb_thread = threading.Thread(target=_heartbeat_worker, daemon=True)
        hb_thread.start()

        try:
            result = execute_run(run, config, on_event=_on_event, steps_collector=steps_collector)
            if result.get("status") != "completed":
                message = result.get("summary") or f"Agent returned status {result.get('status')!r}"
                store.mark_failed(run, error_message=message, result=result)
                realtime.emit(run.id, "run_failed", {"error": message}, event_name="run_failed")
                continue
            units_final = estimate_final_units(run, result)
            # Commit the durable result before optional live notifications.
            store.mark_completed(run, result, units_final)

            if result.get("geojson"):
                realtime.emit_geojson_layer(
                    run.id,
                    f"geojson-{run.id}",
                    result["geojson"],
                    "Calque Mtumishi",
                    opacity=0.85,
                )

            if result.get("raster_overlay"):
                realtime.emit_raster_layer(
                    run.id,
                    f"raster-{run.id}",
                    {
                        "image_url": result["raster_overlay"].get("url"),
                        "bounds": result["raster_overlay"].get("bounds"),
                        "name": result["raster_overlay"].get("name"),
                        "opacity": result["raster_overlay"].get("opacity", 0.85),
                    },
                    result["raster_overlay"].get("name", "Couche raster Mtumishi"),
                    opacity=result["raster_overlay"].get("opacity", 0.85),
                )

            realtime.emit(run.id, "run_completed", {"status": "completed", "units_final": units_final, "summary": result.get("summary")}, event_name="run_completed")
            logger.info("Completed run id=%s units_final=%s", run.id, units_final)
        except Exception as exc:
            logger.exception("Run failed id=%s", run.id)
            realtime.emit(run.id, "run_failed", {"error": str(exc), "traceback": traceback.format_exc()}, event_name="run_failed")
            try:
                store.mark_failed(run, error_message=str(exc), details={"traceback": traceback.format_exc()})
            except Exception:
                logger.exception("Unable to finalize failed run id=%s", run.id)
        finally:
            heartbeat_stop.set()


if __name__ == "__main__":
    main()
