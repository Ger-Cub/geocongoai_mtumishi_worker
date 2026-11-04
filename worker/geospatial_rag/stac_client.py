"""STAC client for Microsoft Planetary Computer targeting the Congo Basin."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import planetary_computer
from pystac import Item
from pystac_client import Client

logger = logging.getLogger("geocongoai.geospatial_rag.stac")

# Emprise géographique délimitant le Bassin du Congo (RDC et régions géologiques clés :
# Chaîne des Virunga, Arc cuprifère du Katanga, Craton du Kasaï, Cuvette centrale)
CONGO_BASIN_BBOX: List[float] = [11.5, -13.5, 31.5, 5.5]
PLANETARY_COMPUTER_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION_SENTINEL2_L2A = "sentinel-2-l2a"


def get_stac_client() -> Client:
    """Return a pystac_client Client configured with Planetary Computer authentication."""
    return Client.open(
        PLANETARY_COMPUTER_STAC_URL,
        modifier=planetary_computer.sign_inplace,
    )


def search_congo_basin_tiles(
    bbox: Optional[List[float]] = None,
    max_cloud_cover: float = 15.0,
    datetime_range: str = "2024-01-01/2026-11-01",
    limit: int = 20,
) -> List[Item]:
    """Search Sentinel-2 L2A tiles in the Congo Basin with low cloud cover."""
    target_bbox = bbox or CONGO_BASIN_BBOX
    client = get_stac_client()

    logger.info(
        "Searching STAC Sentinel-2 tiles in bbox=%s, max_cloud_cover=%.1f%%, range=%s",
        target_bbox,
        max_cloud_cover,
        datetime_range,
    )

    search = client.search(
        collections=[COLLECTION_SENTINEL2_L2A],
        bbox=target_bbox,
        datetime=datetime_range,
        query={"eo:cloud_cover": {"lt": max_cloud_cover}},
        max_items=limit,
    )

    items = list(search.items())
    logger.info("Found %d Sentinel-2 tiles matching criteria", len(items))
    return items


def extract_tile_assets(item: Item) -> Dict[str, Any]:
    """Extract signed URLs and metadata for SWIR, NIR, Red, and SCL cloud mask."""
    assets = item.assets

    # SWIR band : B12 (2190 nm) preferred for hydrothermal/mineral detection, fallback to B11 (1610 nm)
    swir_asset = assets.get("B12") or assets.get("B11")
    nir_asset = assets.get("B08")
    red_asset = assets.get("B04")
    scl_asset = assets.get("SCL")  # Scene Classification Layer (cloud/shadow mask)
    visual_asset = assets.get("rendered_preview") or assets.get("visual")

    return {
        "item_id": item.id,
        "datetime": item.datetime.isoformat() if item.datetime else None,
        "cloud_cover": item.properties.get("eo:cloud_cover", 0.0),
        "bbox": item.bbox,
        "geometry": item.geometry,
        "swir_url": swir_asset.href if swir_asset else None,
        "nir_url": nir_asset.href if nir_asset else None,
        "red_url": red_asset.href if red_asset else None,
        "scl_url": scl_asset.href if scl_asset else None,
        "thumbnail_url": visual_asset.href if visual_asset else None,
        "properties": {
            "platform": item.properties.get("platform", "Sentinel-2"),
            "constellation": item.properties.get("constellation", "Sentinel-2"),
            "instruments": item.properties.get("instruments", ["MSI"]),
            "mgrs_tile": item.properties.get("s2:mgrs_tile"),
        },
    }

