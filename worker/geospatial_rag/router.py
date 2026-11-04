"""FastAPI router for Geospatial & Multimodal RAG search endpoints."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from supabase import create_client

from worker.config import load_config
from worker.geospatial_rag.embedding_client import QwenVLEmbeddingClient
from worker.geospatial_rag.image_processor import (
    build_swir_nir_red_composite,
    image_to_jpeg_bytes,
    process_user_uploaded_image,
)
from worker.geospatial_rag.stac_client import (
    extract_tile_assets,
    search_congo_basin_tiles,
)

logger = logging.getLogger("geocongoai.geospatial_rag.router")

router = APIRouter(prefix="/api/v1/geospatial-search", tags=["geospatial-search"])


class TextSearchRequest(BaseModel):
    query: str = Field(..., description="Geological textual description or prompt (e.g. ring structure, hydrothermal alteration)")
    match_count: int = Field(default=10, ge=1, le=50)
    match_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    filter_metadata: Optional[Dict[str, Any]] = Field(default=None)


class BboxSearchRequest(BaseModel):
    bbox: List[float] = Field(..., min_length=4, max_length=4, description="[min_lon, min_lat, max_lon, max_lat]")
    match_count: int = Field(default=10, ge=1, le=50)
    match_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    filter_metadata: Optional[Dict[str, Any]] = Field(default=None)


def _get_supabase_client():
    config = load_config()
    return create_client(config.supabase_url, config.supabase_service_role_key)


@router.get("/health")
def health_check() -> Dict[str, Any]:
    """Health status and configuration for geospatial search."""
    return {
        "status": "healthy",
        "service": "GeoCongoAI Geospatial RAG Analog Finder",
        "model": "Qwen/Qwen3-VL-Embedding-8B",
        "dimension": 4096,
        "spectral_bands": ["SWIR", "NIR", "Red"],
        "target_region": "Congo Basin (DRC)",
    }


@router.post("/text")
def search_by_text(request: TextSearchRequest) -> Dict[str, Any]:
    """Search for satellite analogues using a textual geological prompt."""
    query_text = request.query.strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Query prompt cannot be empty.")

    embedder = QwenVLEmbeddingClient()
    supabase = _get_supabase_client()

    try:
        embedding_4096 = embedder.embed_text(query_text)
        rpc_params = {
            "query_embedding": embedding_4096,
            "match_threshold": request.match_threshold,
            "match_count": request.match_count,
            "filter_metadata": request.filter_metadata or {},
        }
        res = supabase.rpc("match_satellite_tiles", rpc_params).execute()
        results = res.data or []

        return {
            "query_type": "text",
            "query": query_text,
            "match_count": len(results),
            "results": results,
        }
    except Exception as exc:
        logger.exception("Text search failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}")


@router.post("/image")
async def search_by_image(
    file: UploadFile = File(...),
    match_count: int = Form(default=10),
    match_threshold: float = Form(default=0.0),
) -> Dict[str, Any]:
    """Search for satellite analogues using a user-uploaded geological/satellite image."""
    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    embedder = QwenVLEmbeddingClient()
    supabase = _get_supabase_client()

    try:
        processed_bytes = process_user_uploaded_image(raw_bytes)
        embedding_4096 = embedder.embed_image(processed_bytes)

        rpc_params = {
            "query_embedding": embedding_4096,
            "match_threshold": match_threshold,
            "match_count": match_count,
            "filter_metadata": {},
        }
        res = supabase.rpc("match_satellite_tiles", rpc_params).execute()
        results = res.data or []

        return {
            "query_type": "image",
            "filename": file.filename,
            "match_count": len(results),
            "results": results,
        }
    except Exception as exc:
        logger.exception("Image search failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Image search failed: {exc}")


@router.post("/bbox")
def search_by_bbox(request: BboxSearchRequest) -> Dict[str, Any]:
    """
    Search for satellite analogues from a geographic bounding box / polygon.
    Fetches corresponding Sentinel-2 tiles on the fly, computes SWIR/NIR/Red composite,
    generates 4096d embedding, and searches database.
    """
    bbox = request.bbox
    embedder = QwenVLEmbeddingClient()
    supabase = _get_supabase_client()

    try:
        # 1. Récupérer dalle Sentinel-2 correspondante via STAC à la volée
        items = search_congo_basin_tiles(
            bbox=bbox,
            max_cloud_cover=20.0,
            limit=1,
        )

        if not items:
            raise HTTPException(
                status_code=404,
                detail="No clear Sentinel-2 imagery found for the requested bounding box.",
            )

        item = items[0]
        assets_info = extract_tile_assets(item)
        swir_url = assets_info.get("swir_url")
        nir_url = assets_info.get("nir_url")
        red_url = assets_info.get("red_url")
        scl_url = assets_info.get("scl_url")

        if not (swir_url and nir_url and red_url):
            raise HTTPException(
                status_code=502,
                detail="STAC tile found but missing required SWIR/NIR/Red spectral bands.",
            )

        # 2. Prétraitement 3 canaux (SWIR/NIR/Rouge) à la volée
        composite_img, cloud_fraction = build_swir_nir_red_composite(
            swir_url=swir_url,
            nir_url=nir_url,
            red_url=red_url,
            scl_url=scl_url,
        )
        img_bytes = image_to_jpeg_bytes(composite_img)

        # 3. Génération d'embedding 4096d
        query_embedding_4096 = embedder.embed_image(img_bytes)

        # 4. Requête de similarité vectorielle
        rpc_params = {
            "query_embedding": query_embedding_4096,
            "match_threshold": request.match_threshold,
            "match_count": request.match_count,
            "filter_metadata": request.filter_metadata or {},
        }
        res = supabase.rpc("match_satellite_tiles", rpc_params).execute()
        results = res.data or []

        return {
            "query_type": "bbox",
            "bbox": bbox,
            "query_stac_item_id": item.id,
            "query_preview_url": assets_info.get("thumbnail_url"),
            "cloud_cover": assets_info.get("cloud_cover"),
            "match_count": len(results),
            "results": results,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Bbox search failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Bbox search failed: {exc}")

