"""Ingestion pipeline for Sentinel-2 tiles in the Congo Basin into satellite_embeddings."""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from typing import Any, Callable, Dict, List, Optional

from supabase import create_client

from worker.config import load_config
from worker.geospatial_rag.embedding_client import QwenVLEmbeddingClient
from worker.geospatial_rag.image_processor import (
    build_swir_nir_red_composite,
    image_to_jpeg_bytes,
)
from worker.geospatial_rag.stac_client import (
    CONGO_BASIN_BBOX,
    GEOLOGICAL_REGIONS,
    extract_tile_assets,
    get_region_details,
    search_congo_basin_tiles_iter,
)

logger = logging.getLogger("geocongoai.geospatial_rag.ingest")


def format_geojson_polygon(bbox: List[float]) -> Dict[str, Any]:
    """Convert bbox [min_lon, min_lat, max_lon, max_lat] to GeoJSON Polygon."""
    min_x, min_y, max_x, max_y = bbox
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [min_x, min_y],
                [max_x, min_y],
                [max_x, max_y],
                [min_x, max_y],
                [min_x, min_y],
            ]
        ],
    }


def ingest_congo_basin_tiles(
    limit: int = 5,
    max_cloud_cover: float = 15.0,
    datetime_range: str = "2024-01-01/2026-11-01",
    region_key: Optional[str] = None,
    dry_run: bool = False,
    scan_max_items: int = 150,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """
    Execute the ingestion pipeline for Sentinel-2 tiles targeting a geological region in the Congo Basin.
    Continuously streams STAC items until `limit` *new* unindexed tiles are inserted into Supabase.
    """
    config = load_config()
    supabase = create_client(config.supabase_url, config.supabase_service_role_key)
    embedder = QwenVLEmbeddingClient()

    canonical_key, target_bbox, region_info = get_region_details(region_key)
    logger.info(
        "Starting ingestion for region '%s' (%s): target=%d new tiles, max_clouds=%.1f%%, bbox=%s",
        canonical_key,
        region_info["name"],
        limit,
        max_cloud_cover,
        target_bbox,
    )

    # 1. Obtenir le flux de dalles STAC Planetary Computer
    try:
        items_stream = search_congo_basin_tiles_iter(
            bbox=target_bbox,
            max_cloud_cover=max_cloud_cover,
            datetime_range=datetime_range,
            max_items=scan_max_items,
        )
    except Exception as exc:
        logger.exception("Failed to connect or query Planetary Computer STAC: %s", exc)
        return {
            "status": "error",
            "error": str(exc),
            "region_key": canonical_key,
            "region_name": region_info["name"],
            "indexed_count": 0,
            "skipped_count": 0,
            "scanned_count": 0,
            "indexed_items": [],
        }

    ingested_count = 0
    skipped_count = 0
    scanned_count = 0
    indexed_items: List[str] = []

    for item in items_stream:
        scanned_count += 1
        item_id = item.id

        # 2. Vérifier si la dalle est déjà indexée (idempotence)
        try:
            existing = (
                supabase.table("satellite_embeddings")
                .select("id")
                .eq("stac_item_id", item_id)
                .limit(1)
                .execute()
            )
            if existing.data:
                skipped_count += 1
                logger.debug("[%d] Tile %s already indexed; continuing search...", scanned_count, item_id)
                continue
        except Exception as exc:
            logger.warning("Supabase lookup failed for %s: %s; skipping item", item_id, exc)
            continue

        assets_info = extract_tile_assets(item)
        swir_url = assets_info.get("swir_url")
        nir_url = assets_info.get("nir_url")
        red_url = assets_info.get("red_url")
        scl_url = assets_info.get("scl_url")

        if not (swir_url and nir_url and red_url):
            logger.warning("Tile %s missing required SWIR/NIR/Red bands; skipping.", item_id)
            continue

        logger.info(
            "Found NEW tile [%d/%d target] %s (clouds=%.1f%%, date=%s)",
            ingested_count + 1,
            limit,
            item_id,
            assets_info.get("cloud_cover", 0.0),
            assets_info.get("datetime"),
        )

        try:
            # 3. Prétraitement 3 canaux (SWIR, NIR, Rouge) + filtrage des nuages SCL (Zero Local Heavy Storage)
            composite_img, cloud_fraction = build_swir_nir_red_composite(
                swir_url=swir_url,
                nir_url=nir_url,
                red_url=red_url,
                scl_url=scl_url,
            )

            img_bytes = image_to_jpeg_bytes(composite_img)

            # 4. Génération de l'embedding multimodal 4096d via Qwen3-VL-Embedding-8B
            embedding_4096 = embedder.embed_image(img_bytes)

            bbox_coords = assets_info["bbox"]
            min_x, min_y, max_x, max_y = bbox_coords
            polygon_geojson = format_geojson_polygon(bbox_coords)
            polygon_wkt = f"SRID=4326;POLYGON(({min_x} {min_y}, {max_x} {min_y}, {max_x} {max_y}, {min_x} {max_y}, {min_x} {min_y}))"

            metadata = {
                "stac_item_id": item_id,
                "acquisition_date": assets_info["datetime"],
                "cloud_cover": assets_info["cloud_cover"],
                "masked_cloud_fraction": round(cloud_fraction * 100.0, 2),
                "composition": "SWIR(B12/B11)-NIR(B08)-Red(B04)",
                "target_region": region_info["name"],
                "region_key": canonical_key,
                "geological_targets": region_info.get("targets", []),
                "region_description": region_info.get("description", ""),
                "thumbnail_url": assets_info.get("thumbnail_url"),
                "stac_assets": {
                    "swir": swir_url,
                    "nir": nir_url,
                    "red": red_url,
                },
                "properties": assets_info["properties"],
            }

            if dry_run:
                logger.info("[DRY RUN] Would insert tile %s into satellite_embeddings", item_id)
                ingested_count += 1
                indexed_items.append(item_id)
                if progress_callback:
                    progress_callback({
                        "event": "tile_indexed",
                        "item_id": item_id,
                        "ingested_count": ingested_count,
                        "target_limit": limit,
                        "dry_run": True,
                    })
                if ingested_count >= limit:
                    break
                continue

            # 5. Insertion dans la table satellite_embeddings
            record = {
                "stac_item_id": item_id,
                "embedding": embedding_4096,
                "bbox": polygon_wkt,
                "geojson": polygon_geojson,
                "metadata": metadata,
            }

            supabase.table("satellite_embeddings").insert(record).execute()
            ingested_count += 1
            indexed_items.append(item_id)
            logger.info("Successfully indexed tile %s in satellite_embeddings (%d/%d)", item_id, ingested_count, limit)

            if progress_callback:
                progress_callback({
                    "event": "tile_indexed",
                    "item_id": item_id,
                    "ingested_count": ingested_count,
                    "target_limit": limit,
                    "dry_run": False,
                })

            if ingested_count >= limit:
                logger.info("Reached target of %d new tiles for region '%s'", limit, canonical_key)
                break

        except Exception as exc:
            logger.exception("Failed to ingest tile %s: %s", item_id, exc)

    status_str = "success" if ingested_count >= limit else ("partial" if ingested_count > 0 else "no_new_tiles")
    result = {
        "status": status_str,
        "region_key": canonical_key,
        "region_name": region_info["name"],
        "indexed_count": ingested_count,
        "skipped_count": skipped_count,
        "scanned_count": scanned_count,
        "indexed_items": indexed_items,
    }

    logger.info(
        "Ingestion summary for '%s': %d newly indexed, %d skipped, %d scanned total.",
        canonical_key,
        ingested_count,
        skipped_count,
        scanned_count,
    )
    return result


def main() -> None:
    """CLI entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    parser = argparse.ArgumentParser(description="Ingest Sentinel-2 tiles into satellite_embeddings (Congo Basin)")
    parser.add_argument("--limit", type=int, default=5, help="Number of new tiles to index")
    parser.add_argument("--max-clouds", type=float, default=15.0, help="Max cloud cover percentage")
    parser.add_argument("--date-range", type=str, default="2024-01-01/2026-11-01", help="STAC date range")
    parser.add_argument("--region", type=str, default="all", help="Geological region (katanga_copperbelt, kivu_maniema, virunga_volcanoes, kasai_craton, ituri_kilo_moto, cuvette_centrale, bas_congo, all)")
    parser.add_argument("--dry-run", action="store_true", help="Simulate ingestion without writing to DB")
    parser.add_argument("--auto", action="store_true", help="Run automated scheduler in continuous loop")
    parser.add_argument("--interval-minutes", type=int, default=360, help="Interval in minutes for --auto mode (default: 360m = 6h)")

    args = parser.parse_args()

    if args.auto:
        logger.info("Starting automated ingestion scheduler loop (interval=%d minutes)...", args.interval_minutes)
        from worker.geospatial_rag.scheduler import GeospatialAutoIngestScheduler
        scheduler = GeospatialAutoIngestScheduler(
            interval_minutes=args.interval_minutes,
            batch_size=args.limit,
            max_clouds=args.max_clouds,
        )
        scheduler.start()
        try:
            while True:
                time.sleep(1)
        except (KeyboardInterrupt, SystemExit):
            logger.info("Stopping scheduler...")
            scheduler.stop()
            sys.exit(0)

    ingest_congo_basin_tiles(
        limit=args.limit,
        max_cloud_cover=args.max_clouds,
        datetime_range=args.date_range,
        region_key=args.region,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
