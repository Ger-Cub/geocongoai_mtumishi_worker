"""Ingestion pipeline for Sentinel-2 tiles in the Congo Basin into satellite_embeddings."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any, Dict, List, Optional

from supabase import create_client

from worker.config import load_config
from worker.geospatial_rag.embedding_client import QwenVLEmbeddingClient
from worker.geospatial_rag.image_processor import (
    build_swir_nir_red_composite,
    image_to_jpeg_bytes,
)
from worker.geospatial_rag.stac_client import (
    CONGO_BASIN_BBOX,
    extract_tile_assets,
    search_congo_basin_tiles,
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
    limit: int = 10,
    max_cloud_cover: float = 15.0,
    datetime_range: str = "2024-01-01/2026-11-01",
    dry_run: bool = False,
) -> int:
    """Execute the full one-off ingestion pipeline for the Congo Basin."""
    config = load_config()
    supabase = create_client(config.supabase_url, config.supabase_service_role_key)
    embedder = QwenVLEmbeddingClient()

    logger.info("Starting Congo Basin satellite tiles ingestion (limit=%d, max_clouds=%.1f%%)", limit, max_cloud_cover)

    # 1. Interroger STAC Planetary Computer pour le Bassin du Congo
    items = search_congo_basin_tiles(
        bbox=CONGO_BASIN_BBOX,
        max_cloud_cover=max_cloud_cover,
        datetime_range=datetime_range,
        limit=limit,
    )

    if not items:
        logger.warning("No Sentinel-2 tiles found matching criteria.")
        return 0

    ingested_count = 0

    for idx, item in enumerate(items, start=1):
        item_id = item.id
        logger.info("[%d/%d] Processing tile %s...", idx, len(items), item_id)

        # 2. Vérifier si la dalle est déjà indexée (idempotence)
        existing = (
            supabase.table("satellite_embeddings")
            .select("id")
            .eq("stac_item_id", item_id)
            .limit(1)
            .execute()
        )
        if existing.data:
            logger.info("Tile %s already indexed; skipping.", item_id)
            continue

        assets_info = extract_tile_assets(item)
        swir_url = assets_info.get("swir_url")
        nir_url = assets_info.get("nir_url")
        red_url = assets_info.get("red_url")
        scl_url = assets_info.get("scl_url")

        if not (swir_url and nir_url and red_url):
            logger.warning("Tile %s missing required SWIR/NIR/Red bands; skipping.", item_id)
            continue

        try:
            # 3. Prétraitement 3 canaux (SWIR, NIR, Rouge) + filtrage des nuages SCL (Zero Local Heavy Storage)
            logger.info("Building 3-channel SWIR/NIR/Red composite with cloud masking for %s", item_id)
            composite_img, cloud_fraction = build_swir_nir_red_composite(
                swir_url=swir_url,
                nir_url=nir_url,
                red_url=red_url,
                scl_url=scl_url,
            )

            img_bytes = image_to_jpeg_bytes(composite_img)

            # 4. Génération de l'embedding multimodal 4096d via Qwen3-VL-Embedding-8B
            logger.info("Generating Qwen3-VL-Embedding-8B (4096d) for tile %s", item_id)
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
                "target_region": "Bassin du Congo / RDC",
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
                continue

            # 5. Insertion dans la table isolée satellite_embeddings
            record = {
                "stac_item_id": item_id,
                "embedding": embedding_4096,
                "bbox": polygon_wkt,
                "geojson": polygon_geojson,
                "metadata": metadata,
            }

            supabase.table("satellite_embeddings").insert(record).execute()
            logger.info("Successfully indexed tile %s in satellite_embeddings (4096d vector)", item_id)
            ingested_count += 1

        except Exception as exc:
            logger.exception("Failed to ingest tile %s: %s", item_id, exc)

    logger.info("Ingestion complete. %d/%d tiles indexed.", ingested_count, len(items))
    return ingested_count


def main() -> None:
    """CLI entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    parser = argparse.ArgumentParser(description="Ingest Sentinel-2 tiles into satellite_embeddings (Congo Basin)")
    parser.add_argument("--limit", type=int, default=5, help="Maximum tiles to index")
    parser.add_argument("--max-clouds", type=float, default=15.0, help="Max cloud cover percentage")
    parser.add_argument("--date-range", type=str, default="2024-01-01/2026-11-01", help="STAC date range")
    parser.add_argument("--dry-run", action="store_true", help="Simulate ingestion without writing to DB")

    args = parser.parse_args()
    ingest_congo_basin_tiles(
        limit=args.limit,
        max_cloud_cover=args.max_clouds,
        datetime_range=args.date_range,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

