"""FastAPI application for the GeoCongoAI Mtumishi worker."""
from __future__ import annotations

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from worker.geospatial_rag.router import router as geospatial_router

logger = logging.getLogger("geocongoai.worker.app")

app = FastAPI(
    title="GeoCongoAI Mtumishi Worker & Geospatial RAG API",
    description="Backend service for GeoCongoAI agent runs and multimodal satellite search (NASA Space Apps Challenge 2026)",
    version="1.1.0",
)

# Autoriser les requêtes CORS depuis le frontend React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Monter le sous-routeur de recherche géospatiale
app.include_router(geospatial_router)


@app.get("/")
def root():
    return {
        "service": "GeoCongoAI Mtumishi Worker",
        "status": "online",
        "version": "1.1.0",
        "endpoints": {
            "health": "/api/v1/geospatial-search/health",
            "text_search": "/api/v1/geospatial-search/text",
            "image_search": "/api/v1/geospatial-search/image",
            "bbox_search": "/api/v1/geospatial-search/bbox",
            "ingest_status": "/api/v1/geospatial-search/ingest/status",
            "ingest_regions": "/api/v1/geospatial-search/ingest/regions",
            "ingest_trigger": "/api/v1/geospatial-search/ingest/trigger",
            "ingest_toggle": "/api/v1/geospatial-search/ingest/toggle",
        },
    }
