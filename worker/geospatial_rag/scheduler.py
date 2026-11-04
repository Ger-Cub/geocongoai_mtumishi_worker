"""Automated Background Ingestion Scheduler for Geospatial RAG (Sentinel-2 Congo Basin)."""
from __future__ import annotations

import datetime
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

from supabase import create_client

from worker.config import load_config
from worker.geospatial_rag.ingest import ingest_congo_basin_tiles
from worker.geospatial_rag.stac_client import GEOLOGICAL_REGIONS, get_region_details

logger = logging.getLogger("geocongoai.geospatial_rag.scheduler")

DEFAULT_CYCLE_REGIONS = [
    "katanga_copperbelt",
    "kivu_maniema",
    "virunga_volcanoes",
    "kasai_craton",
    "ituri_kilo_moto",
    "cuvette_centrale",
    "bas_congo",
]


class GeospatialAutoIngestScheduler:
    """
    Automated daemon scheduler that periodically cycles across geological regions
    of the Congo Basin, fetching and indexing Sentinel-2 tiles into satellite_embeddings.
    """

    def __init__(
        self,
        enabled: Optional[bool] = None,
        interval_minutes: Optional[int] = None,
        batch_size: Optional[int] = None,
        max_clouds: Optional[float] = None,
        target_total_tiles: Optional[int] = None,
        regions_cycle: Optional[List[str]] = None,
    ):
        if enabled is None:
            env_val = os.getenv("GEOSPATIAL_AUTO_INGEST_ENABLED", "1").lower()
            self.enabled = env_val in ("1", "true", "yes", "on")
        else:
            self.enabled = enabled

        self.interval_minutes = interval_minutes or int(os.getenv("GEOSPATIAL_AUTO_INGEST_INTERVAL_MINUTES", "360"))
        self.batch_size = batch_size or int(os.getenv("GEOSPATIAL_AUTO_INGEST_BATCH_SIZE", "5"))
        self.max_clouds = max_clouds or float(os.getenv("GEOSPATIAL_AUTO_INGEST_MAX_CLOUDS", "15.0"))
        self.target_total_tiles = target_total_tiles or int(os.getenv("GEOSPATIAL_AUTO_INGEST_TARGET_TOTAL", "500"))

        raw_regions = os.getenv("GEOSPATIAL_AUTO_INGEST_REGIONS", "")
        if raw_regions and raw_regions.lower() != "all":
            self.regions_cycle = [r.strip() for r in raw_regions.split(",") if r.strip()]
        else:
            self.regions_cycle = regions_cycle or list(DEFAULT_CYCLE_REGIONS)

        self._current_region_idx = 0
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._is_ingesting = False
        self._lock = threading.Lock()

        # Telemetry & Status
        self._last_run: Optional[Dict[str, Any]] = None
        self._next_run_timestamp: Optional[str] = None
        self._run_history: List[Dict[str, Any]] = []
        self._last_cached_db_count: Optional[int] = None
        self._last_cached_db_time: float = 0.0

    def get_total_tiles_in_db(self, force_refresh: bool = False) -> int:
        """Query total count of indexed tiles in satellite_embeddings (cached for 60s)."""
        now = time.time()
        if not force_refresh and self._last_cached_db_count is not None and (now - self._last_cached_db_time < 60.0):
            return self._last_cached_db_count

        try:
            config = load_config()
            supabase = create_client(config.supabase_url, config.supabase_service_role_key)
            res = supabase.table("satellite_embeddings").select("id", count="exact").limit(1).execute()
            count = res.count if res.count is not None else (len(res.data) if res.data else 0)
            self._last_cached_db_count = count
            self._last_cached_db_time = now
            return count
        except Exception as exc:
            logger.warning("Could not fetch satellite_embeddings count: %s", exc)
            return self._last_cached_db_count or 0

    def start(self) -> bool:
        """Start the background scheduler thread."""
        with self._lock:
            if self._thread and self._thread.is_alive():
                logger.info("Auto-ingest scheduler is already running.")
                return True

            if not self.enabled:
                logger.info("Auto-ingest scheduler is disabled by configuration (GEOSPATIAL_AUTO_INGEST_ENABLED=0).")
                return False

            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._worker_loop,
                name="geospatial-auto-ingest-worker",
                daemon=True,
            )
            self._thread.start()
            logger.info(
                "Started Geospatial Auto-Ingest Scheduler: interval=%d min, batch_size=%d, max_clouds=%.1f%%, regions=%s",
                self.interval_minutes,
                self.batch_size,
                self.max_clouds,
                self.regions_cycle,
            )
            return True

    def stop(self) -> None:
        """Stop the background scheduler thread gracefully."""
        with self._lock:
            self._stop_event.set()
            if self._thread and self._thread.is_alive():
                logger.info("Stopping Geospatial Auto-Ingest Scheduler...")
                self._thread.join(timeout=5.0)
                self._thread = None
                logger.info("Geospatial Auto-Ingest Scheduler stopped.")

    def _worker_loop(self) -> None:
        """Main loop executed by the background thread."""
        logger.info("Auto-ingest background worker loop entered.")

        # Première exécution après un court délai de démarrage (15 secondes)
        initial_wait_seconds = int(os.getenv("GEOSPATIAL_AUTO_INGEST_INITIAL_DELAY_SECONDS", "15"))
        if self._stop_event.wait(timeout=initial_wait_seconds):
            return

        while not self._stop_event.is_set():
            current_region = self.regions_cycle[self._current_region_idx % len(self.regions_cycle)]
            self._current_region_idx += 1

            # Vérifier le plafond cible
            if self.target_total_tiles > 0:
                current_total = self.get_total_tiles_in_db(force_refresh=True)
                if current_total >= self.target_total_tiles:
                    logger.info(
                        "Target total of %d tiles reached in satellite_embeddings (%d existing). Idling...",
                        self.target_total_tiles,
                        current_total,
                    )
                    # Dormir pendant l'intervalle complet
                    sleep_sec = self.interval_minutes * 60
                    self._update_next_run(sleep_sec)
                    if self._stop_event.wait(timeout=sleep_sec):
                        break
                    continue

            # Exécuter l'ingestion du batch
            self._execute_cycle(current_region)

            # Calculer la prochaine exécution
            sleep_sec = max(60, self.interval_minutes * 60)
            self._update_next_run(sleep_sec)
            logger.info("Next auto-ingestion scheduled in %d minutes (%s)", self.interval_minutes, self._next_run_timestamp)

            if self._stop_event.wait(timeout=sleep_sec):
                break

    def _update_next_run(self, sleep_sec: int) -> None:
        next_dt = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=sleep_sec)
        self._next_run_timestamp = next_dt.isoformat()

    def _execute_cycle(self, region_key: str, is_manual: bool = False) -> Dict[str, Any]:
        """Execute one ingestion run for a single region."""
        start_time = time.time()
        start_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        canonical_key, _, region_info = get_region_details(region_key)

        logger.info(
            "Executing %s ingestion cycle for region '%s' (%s) [batch_size=%d, max_clouds=%.1f%%]",
            "manual" if is_manual else "scheduled",
            canonical_key,
            region_info["name"],
            self.batch_size,
            self.max_clouds,
        )

        with self._lock:
            self._is_ingesting = True

        result_summary: Dict[str, Any] = {
            "timestamp": start_iso,
            "region_key": canonical_key,
            "region_name": region_info["name"],
            "batch_size": self.batch_size,
            "is_manual": is_manual,
            "status": "running",
        }

        try:
            res = ingest_congo_basin_tiles(
                limit=self.batch_size,
                max_cloud_cover=self.max_clouds,
                region_key=canonical_key,
                dry_run=False,
            )
            duration = round(time.time() - start_time, 2)
            result_summary.update({
                "status": res.get("status", "success"),
                "duration_seconds": duration,
                "indexed_count": res.get("indexed_count", 0),
                "skipped_count": res.get("skipped_count", 0),
                "scanned_count": res.get("scanned_count", 0),
                "indexed_items": res.get("indexed_items", []),
                "error": res.get("error"),
            })
            self.get_total_tiles_in_db(force_refresh=True)

        except Exception as exc:
            duration = round(time.time() - start_time, 2)
            logger.exception("Ingestion cycle failed for '%s': %s", canonical_key, exc)
            result_summary.update({
                "status": "error",
                "duration_seconds": duration,
                "indexed_count": 0,
                "error": str(exc),
            })
        finally:
            with self._lock:
                self._is_ingesting = False
                self._last_run = result_summary
                self._run_history.append(result_summary)
                if len(self._run_history) > 20:
                    self._run_history.pop(0)

        return result_summary

    def trigger_now(
        self,
        limit: Optional[int] = None,
        max_clouds: Optional[float] = None,
        region: Optional[str] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Trigger an immediate, on-demand ingestion run in a background thread."""
        batch = limit or self.batch_size
        clouds = max_clouds if max_clouds is not None else self.max_clouds
        target_region = region or self.regions_cycle[self._current_region_idx % len(self.regions_cycle)]

        if self._is_ingesting:
            return {
                "status": "rejected",
                "message": "An ingestion cycle is already currently running. Please wait for it to complete.",
            }

        def _run_task():
            start_time = time.time()
            start_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
            canonical_key, _, region_info = get_region_details(target_region)
            with self._lock:
                self._is_ingesting = True
            try:
                res = ingest_congo_basin_tiles(
                    limit=batch,
                    max_cloud_cover=clouds,
                    region_key=target_region,
                    dry_run=dry_run,
                )
                duration = round(time.time() - start_time, 2)
                summary = {
                    "timestamp": start_iso,
                    "region_key": canonical_key,
                    "region_name": region_info["name"],
                    "batch_size": batch,
                    "is_manual": True,
                    "dry_run": dry_run,
                    "status": res.get("status", "success"),
                    "duration_seconds": duration,
                    "indexed_count": res.get("indexed_count", 0),
                    "skipped_count": res.get("skipped_count", 0),
                    "scanned_count": res.get("scanned_count", 0),
                    "indexed_items": res.get("indexed_items", []),
                }
            except Exception as exc:
                duration = round(time.time() - start_time, 2)
                summary = {
                    "timestamp": start_iso,
                    "region_key": canonical_key,
                    "region_name": region_info["name"],
                    "is_manual": True,
                    "status": "error",
                    "duration_seconds": duration,
                    "error": str(exc),
                }
            finally:
                with self._lock:
                    self._is_ingesting = False
                    self._last_run = summary
                    self._run_history.append(summary)
                    if len(self._run_history) > 20:
                        self._run_history.pop(0)

        t = threading.Thread(target=_run_task, name="manual-ingest-trigger", daemon=True)
        t.start()

        return {
            "status": "started",
            "message": f"Immediate ingestion launched for region '{target_region}' (limit={batch}, max_clouds={clouds}%, dry_run={dry_run})",
            "target_region": target_region,
            "batch_size": batch,
        }

    def get_status(self) -> Dict[str, Any]:
        """Return comprehensive JSON status of the scheduler and indexed tiles."""
        is_alive = bool(self._thread and self._thread.is_alive())
        next_region_key = self.regions_cycle[self._current_region_idx % len(self.regions_cycle)]
        _, _, next_region_info = get_region_details(next_region_key)

        return {
            "enabled": self.enabled,
            "is_scheduler_running": is_alive,
            "is_ingestion_in_progress": self._is_ingesting,
            "configuration": {
                "interval_minutes": self.interval_minutes,
                "batch_size": self.batch_size,
                "max_clouds": self.max_clouds,
                "target_total_tiles": self.target_total_tiles,
                "regions_cycle": self.regions_cycle,
            },
            "next_scheduled_region": {
                "key": next_region_key,
                "name": next_region_info["name"],
                "targets": next_region_info.get("targets", []),
            },
            "next_run_estimated": self._next_run_timestamp,
            "last_run": self._last_run,
            "total_tiles_in_db": self.get_total_tiles_in_db(),
            "recent_runs": self._run_history[-5:],
        }


# Global singleton instance
_SCHEDULER_INSTANCE: Optional[GeospatialAutoIngestScheduler] = None
_INIT_LOCK = threading.Lock()


def get_scheduler() -> GeospatialAutoIngestScheduler:
    """Get or initialize the global scheduler singleton."""
    global _SCHEDULER_INSTANCE
    with _INIT_LOCK:
        if _SCHEDULER_INSTANCE is None:
            _SCHEDULER_INSTANCE = GeospatialAutoIngestScheduler()
        return _SCHEDULER_INSTANCE


def start_auto_ingest_scheduler() -> bool:
    """Convenience function to start the global scheduler."""
    scheduler = get_scheduler()
    return scheduler.start()


def stop_auto_ingest_scheduler() -> None:
    """Convenience function to stop the global scheduler."""
    global _SCHEDULER_INSTANCE
    with _INIT_LOCK:
        if _SCHEDULER_INSTANCE:
            _SCHEDULER_INSTANCE.stop()
