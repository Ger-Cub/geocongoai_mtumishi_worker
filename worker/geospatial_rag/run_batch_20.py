"""Batch ingest 20 tiles across 4 key geological regions of the Congo Basin."""
import logging
import sys
from worker.geospatial_rag.ingest import ingest_congo_basin_tiles

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("geocongoai.batch_ingest")

REGIONS = [
    ("virunga_volcanoes", 5, 25.0, "Chaîne volcanique des Virunga & Rift (Analogue Tharsis Mars)"),
    ("katanga_copperbelt", 5, 20.0, "Arc cuprifère du Katanga (Altérations hydrothermales Cu-Co)"),
    ("kasai_craton", 5, 25.0, "Craton archéen du Kasaï (Socle cristallin & kimberlites - Analogue Lune)"),
    ("ituri_kilo_moto", 5, 25.0, "Ceintures de roches vertes Kilo-Moto (BIF archéens & or orogénique)"),
]

def run():
    total_new = 0
    logger.info("=== Starting batch ingestion of 20 geological tiles ===")
    for region_key, limit, max_clouds, desc in REGIONS:
        logger.info(">>> Processing region: %s (%s) - Target: %d tiles, max clouds: %.1f%%", region_key, desc, limit, max_clouds)
        try:
            res = ingest_congo_basin_tiles(
                limit=limit,
                max_cloud_cover=max_clouds,
                region_key=region_key,
                scan_max_items=50,
            )
            indexed = res.get("indexed_count", 0)
            total_new += indexed
            logger.info(">>> Completed region '%s': %d new tiles inserted.", region_key, indexed)
        except Exception as exc:
            logger.exception("Error processing region '%s': %s", region_key, exc)

    logger.info("=== Batch ingestion completed. Total new tiles inserted: %d ===", total_new)

if __name__ == "__main__":
    run()

