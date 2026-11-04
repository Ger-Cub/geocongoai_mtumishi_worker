"""Utilities to publish structured Mtumishi GIS events to Supabase Realtime."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from supabase import Client


class MtumishiRealtimePublisher:
    """Publish run events and layers for live GIS rendering."""

    def __init__(self, client: Client):
        self.client = client

    def emit(self, run_id: str, event_type: str, payload: Dict[str, Any], event_name: Optional[str] = None) -> None:
        row = {
            "run_id": run_id,
            "event_type": event_type,
            "event_name": event_name,
            "payload": payload or {},
        }
        try:
            self.client.table("agent_run_events").insert(row).execute()
        except Exception:
            logging.getLogger(__name__).exception("Realtime event publication failed for run %s", run_id)

    def emit_layer(self, run_id: str, layer_id: str, name: str, layer_type: str, payload: Dict[str, Any], *, visible: bool = True, opacity: float = 0.85) -> None:
        row = {
            "run_id": run_id,
            "layer_id": layer_id,
            "name": name,
            "layer_type": layer_type,
            "payload": payload or {},
            "visible": visible,
            "opacity": opacity,
        }
        try:
            self.client.table("agent_run_layers").upsert(row, on_conflict="run_id,layer_id").execute()
        except Exception:
            logging.getLogger(__name__).exception("GIS layer publication failed for run %s", run_id)

    def emit_geojson_layer(self, run_id: str, layer_id: str, geojson: Dict[str, Any], name: str, *, opacity: float = 0.85) -> None:
        self.emit_layer(
            run_id,
            layer_id,
            name,
            "vector_geojson",
            {"geojson": geojson},
            visible=True,
            opacity=opacity,
        )
        self.emit(run_id, "layer_ready", {"layer_id": layer_id, "type": "vector_geojson", "name": name, "opacity": opacity}, event_name="geojson_layer_ready")

    def emit_raster_layer(self, run_id: str, layer_id: str, raster: Dict[str, Any], name: str, *, opacity: float = 0.85) -> None:
        self.emit_layer(
            run_id,
            layer_id,
            name,
            "raster_image",
            raster,
            visible=True,
            opacity=opacity,
        )
        self.emit(run_id, "layer_ready", {"layer_id": layer_id, "type": "raster_image", "name": name, "opacity": opacity, **raster}, event_name="raster_layer_ready")
