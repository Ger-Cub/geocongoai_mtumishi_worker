"""STAC client for Microsoft Planetary Computer targeting the Congo Basin."""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterator, List, Optional, Tuple

import planetary_computer
from pystac import Item
from pystac_client import Client

logger = logging.getLogger("geocongoai.geospatial_rag.stac")

# Emprise géographique délimitant le Bassin du Congo (RDC et régions géologiques clés :
# Chaîne des Virunga, Arc cuprifère du Katanga, Craton du Kasaï, Cuvette centrale)
CONGO_BASIN_BBOX: List[float] = [11.5, -13.5, 31.5, 5.5]
PLANETARY_COMPUTER_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION_SENTINEL2_L2A = "sentinel-2-l2a"

# Régions géologiques et métallogéniques prioritaires de la RDC et du Bassin du Congo
GEOLOGICAL_REGIONS: Dict[str, Dict[str, Any]] = {
    "congo_basin": {
        "name": "Bassin du Congo (Global)",
        "bbox": [11.5, -13.5, 31.5, 5.5],
        "description": "Ensemble du Bassin du Congo et de la République Démocratique du Congo",
        "targets": ["Vue d'ensemble", "Bassin intracratonique", "Couverture végétale & hydrographie"],
    },
    "katanga_copperbelt": {
        "name": "Arc cuprifère du Katanga (Central African Copperbelt)",
        "bbox": [25.0, -12.0, 28.5, -9.5],
        "description": "Kolwezi, Tenke Fungurume, Likasi, Lubumbashi",
        "targets": ["Cuivre (Cu)", "Cobalt (Co)", "Altérations hydrothermales", "Malachite", "Dépôts sédimentaires katangiens"],
    },
    "kivu_maniema": {
        "name": "Ceinture orogénique du Kivu - Maniema (Kibaran Belt)",
        "bbox": [26.0, -4.5, 29.5, -1.0],
        "description": "Goma, Bukavu, Walikale, Kindu, Kalima, Mwenga",
        "targets": ["Coltan (Ta-Nb)", "Cassitérite (Sn)", "Wolframite (W)", "Or (Au)", "Pegmatites granitiques"],
    },
    "virunga_volcanoes": {
        "name": "Chaîne volcanique des Virunga & Rift Albertin",
        "bbox": [29.0, -1.8, 29.8, -1.1],
        "description": "Nyiragongo, Nyamuragira, Champs de lave du Nord-Kivu",
        "targets": ["Structures circulaires", "Basaltes & laves récentes", "Anomalies thermiques", "Caldeiras", "Rift est-africain"],
    },
    "kasai_craton": {
        "name": "Craton archéen du Kasaï",
        "bbox": [20.5, -7.5, 24.5, -5.5],
        "description": "Tshikapa, Mbuji-Mayi, Bakwanga",
        "targets": ["Kimberlites", "Diamants alluvionnaires", "Socle cristallin granito-gneissique"],
    },
    "ituri_kilo_moto": {
        "name": "Ceintures de roches vertes Kilo-Moto (Ituri / Haut-Uélé)",
        "bbox": [28.0, 1.0, 31.0, 3.8],
        "description": "Watsa, Kibali, Mongbwalu, Bunia, Doko",
        "targets": ["Or filonien orogénique", "Banded Iron Formations (BIF)", "Ceintures de roches vertes archéennes"],
    },
    "cuvette_centrale": {
        "name": "Cuvette Centrale Congolaise",
        "bbox": [18.0, -2.5, 24.0, 1.5],
        "description": "Mbandaka, Inongo, Parc de la Salonga, Tourbières",
        "targets": ["Tourbières géantes", "Hydrocarbures sédimentaires", "Structures annulaires d'impact (ex. Bossémtélé)"],
    },
    "bas_congo": {
        "name": "Orogène Ouest-Congolien & Littoral Atlantique (Bas-Congo)",
        "bbox": [12.5, -6.0, 15.0, -4.5],
        "description": "Matadi, Boma, Muanda, Tshela, Monts de Cristal",
        "targets": ["Bauxite", "Phosphates", "Hydrocarbures côtiers", "Schistes & métasédiments"],
    },
}


def get_region_details(region_key: Optional[str] = None) -> Tuple[str, List[float], Dict[str, Any]]:
    """Resolve region key into (canonical_key, bbox, region_info)."""
    key = (region_key or "congo_basin").lower().strip()
    if key in ("all", "congo", "drc", "global"):
        key = "congo_basin"

    if key in GEOLOGICAL_REGIONS:
        info = GEOLOGICAL_REGIONS[key]
        return key, info["bbox"], info

    # Clé non reconnue : repli sur le Bassin du Congo global
    logger.warning("Unknown geological region '%s'; falling back to 'congo_basin'", region_key)
    return "congo_basin", CONGO_BASIN_BBOX, GEOLOGICAL_REGIONS["congo_basin"]


def get_stac_client() -> Client:
    """Return a pystac_client Client configured with Planetary Computer authentication."""
    return Client.open(
        PLANETARY_COMPUTER_STAC_URL,
        modifier=planetary_computer.sign_inplace,
    )


def search_congo_basin_tiles_iter(
    bbox: Optional[List[float]] = None,
    max_cloud_cover: float = 15.0,
    datetime_range: str = "2024-01-01/2026-11-01",
    max_items: Optional[int] = None,
) -> Iterator[Item]:
    """Stream Sentinel-2 L2A tiles in target bbox with low cloud cover page-by-page."""
    target_bbox = bbox or CONGO_BASIN_BBOX
    client = get_stac_client()

    logger.info(
        "Streaming STAC Sentinel-2 tiles: bbox=%s, max_cloud=%.1f%%, range=%s, max_items=%s",
        target_bbox,
        max_cloud_cover,
        datetime_range,
        max_items,
    )

    search = client.search(
        collections=[COLLECTION_SENTINEL2_L2A],
        bbox=target_bbox,
        datetime=datetime_range,
        query={"eo:cloud_cover": {"lt": max_cloud_cover}},
        max_items=max_items,
    )

    yield from search.items()


def search_congo_basin_tiles(
    bbox: Optional[List[float]] = None,
    max_cloud_cover: float = 15.0,
    datetime_range: str = "2024-01-01/2026-11-01",
    limit: int = 20,
) -> List[Item]:
    """Search Sentinel-2 L2A tiles in the Congo Basin with low cloud cover (returns a list)."""
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
