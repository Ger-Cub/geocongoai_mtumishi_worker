"""Main entrypoint for the GeoCongoAI Mtumishi worker."""
from __future__ import annotations

import logging
import time
import traceback

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

        try:
            result = execute_run(run, config)
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
