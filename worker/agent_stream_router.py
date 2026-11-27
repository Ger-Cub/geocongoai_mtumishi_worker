"""FastAPI SSE streaming router for real-time Mtumishi AI web agent sessions."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import urllib.parse
import urllib.request
import uuid
from collections import OrderedDict
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from supabase import create_client

from geocongoai import Mtumishi
from geocongoai.mtumishi.tools import GeoTool
from worker.config import load_config
from worker.forest_change import analyze_forest_change
from worker.runtime import (
    QGIS_AREA_TOOL,
    build_context,
    measure_web_area,
    tool_failure,
)

try:
    from geocongoai.mtumishi.runtimes.native import _detect_conversational_intent
except ImportError:
    _detect_conversational_intent = lambda _prompt: False

logger = logging.getLogger("geocongoai.worker.agent_stream")

router = APIRouter(prefix="/api/v1/agent", tags=["mtumishi-agent-stream"])

# Bounded in-memory session cache for multi-turn conversation memory
_MAX_SESSIONS = 100
_SESSIONS: "OrderedDict[str, Mtumishi]" = OrderedDict()


class AgentStreamRequest(BaseModel):
    """Payload for starting or continuing a real-time Mtumishi SSE session."""

    prompt: str = Field(..., min_length=1, max_length=32000)
    context: Optional[Dict[str, Any]] = Field(default=None)
    session_id: Optional[str] = Field(default=None)
    model_name: Optional[str] = Field(default=None)
    runtime: Optional[str] = Field(default="auto")


def _sse_event(event: str, data: Dict[str, Any]) -> str:
    """Format a single Server-Sent Event frame."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


def _get_or_create_session(session_id: str, runtime_type: str, model_name: str, api_key: str, qgis_rpc_port: int) -> Mtumishi:
    """Retrieve an existing Mtumishi instance with its memory or create a fresh one."""
    if session_id in _SESSIONS:
        inst = _SESSIONS.pop(session_id)
        if inst.runtime_type == runtime_type:
            _SESSIONS[session_id] = inst
            return inst

    runtime_kwargs: Dict[str, Any] = {"model_name": model_name}
    if runtime_type != "native":
        runtime_kwargs.update(api_key=api_key, workspaces=[])

    inst = Mtumishi(
        runtime_type=runtime_type,
        qgis_rpc_port=qgis_rpc_port,
        runtime_kwargs=runtime_kwargs,
    )
    # Strip desktop-only QGIS / dev tools for Web sessions
    if hasattr(inst, "registry") and hasattr(inst.registry, "_tools"):
        inst.registry._tools = {
            name: tool
            for name, tool in inst.registry._tools.items()
            if not name.startswith(("qgis_", "dev_"))
        }

    _SESSIONS[session_id] = inst
    while len(_SESSIONS) > _MAX_SESSIONS:
        _SESSIONS.popitem(last=False)
    return inst


def _register_web_tools(
    mtumishi: Mtumishi,
    geometry: Optional[Dict[str, Any]],
    pending_questions: List[Dict[str, Any]],
    pending_deliverables: List[Dict[str, Any]],
) -> None:
    """Register web-compatible GeoTools (area, Sentinel-2, human-in-the-loop) on the agent."""
    if isinstance(geometry, dict):
        mtumishi.register_tool(
            GeoTool(
                name=QGIS_AREA_TOOL,
                description="Mesure la superficie ellipsoïdale de la zone dessinée et produit sa couche GeoJSON.",
                parameters={"type": "object", "properties": {}},
                handler=lambda: measure_web_area(geometry),
            )
        )

        def _forest_change(before_year: int, after_year: int) -> Dict[str, Any]:
            try:
                stats, layers = analyze_forest_change(geometry, before_year, after_year)
                pending_deliverables.extend(layers)
                return stats
            except ValueError as exc:
                return {"status": "error", "error": str(exc)[:500]}
            except Exception:
                logger.exception("Sentinel-2 forest change analysis failed")
                return {"status": "error", "error": "Impossible de traiter les acquisitions Sentinel-2."}

        mtumishi.register_tool(
            GeoTool(
                name="sentinel2_forest_change",
                description="Compare les composites Sentinel-2 d'août à septembre de deux années sur le polygone dessiné.",
                parameters={
                    "type": "object",
                    "properties": {
                        "before_year": {"type": "integer"},
                        "after_year": {"type": "integer"},
                    },
                    "required": ["before_year", "after_year"],
                },
                handler=_forest_change,
            )
        )

    def _ask_user_confirmation(question: str, options: Optional[List[str]] = None) -> Dict[str, Any]:
        payload = {"question": question, "options": options or ["Oui, continuer", "Non, annuler"]}
        pending_questions.append(payload)
        return {"status": "awaiting_user_response", "question": question, "options": payload["options"]}

    mtumishi.register_tool(
        GeoTool(
            name="ask_user_confirmation",
            description="Pose une question de clarification ou demande confirmation à l'utilisateur avant d'exécuter une analyse lourde.",
            parameters={
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "Question claire à poser à l'utilisateur"},
                    "options": {"type": "array", "items": {"type": "string"}, "description": "Choix rapides proposés"},
                },
                "required": ["question"],
            },
            handler=_ask_user_confirmation,
        )
    )

    def _geocode_place(query: str, limit: int = 3) -> Dict[str, Any]:
        clean_query = (query or "").strip()
        if not clean_query:
            return {"status": "error", "error": "Le nom du lieu ne peut pas être vide."}

        geocoding_base = (os.getenv("GEOCODING_API_URL") or "https://nominatim.openstreetmap.org").rstrip("/")
        api_key = (os.getenv("GEOCODING_API_KEY") or "").strip()
        params: Dict[str, Any] = {
            "q": clean_query,
            "format": "jsonv2",
            "addressdetails": 1,
            "limit": max(1, min(int(limit or 3), 5)),
        }
        if api_key:
            params["key"] = api_key

        url = f"{geocoding_base}/search?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "GeoCongoAI/1.0 (gerardcubakabisimwa@gmail.com)",
                "Referer": "https://www.geocongoai.com",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=12) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            logger.warning("Geocoding lookup failed for %r: %s", clean_query, exc)
            return {"status": "error", "error": f"Échec du géocodage pour '{clean_query}': {str(exc)[:200]}"}

        if not isinstance(raw, list) or not raw:
            return {"status": "not_found", "query": clean_query, "results": [], "message": f"Aucun lieu trouvé pour '{clean_query}'."}

        parsed_results: List[Dict[str, Any]] = []
        for item in raw:
            try:
                lat = float(item["lat"])
                lon = float(item["lon"])
            except (KeyError, TypeError, ValueError):
                continue
            raw_bbox = item.get("boundingbox")
            bbox = None
            if isinstance(raw_bbox, list) and len(raw_bbox) == 4:
                try:
                    south, north, west, east = [float(x) for x in raw_bbox]
                    bbox = [west, south, east, north]
                except (TypeError, ValueError):
                    bbox = None
            parsed_results.append({
                "display_name": item.get("display_name"),
                "lat": lat,
                "lon": lon,
                "bbox": bbox,
                "type": item.get("type"),
                "address": item.get("address") or {},
            })

        if not parsed_results:
            return {"status": "not_found", "query": clean_query, "results": []}

        top = parsed_results[0]
        features: List[Dict[str, Any]] = [
            {
                "type": "Feature",
                "properties": {"name": top.get("display_name") or clean_query, "query": clean_query, "lat": top["lat"], "lon": top["lon"]},
                "geometry": {"type": "Point", "coordinates": [top["lon"], top["lat"]]},
            }
        ]
        if top.get("bbox"):
            w, s, e, n = top["bbox"]
            features.append({
                "type": "Feature",
                "properties": {"name": f"Emprise - {clean_query}", "bbox": top["bbox"]},
                "geometry": {"type": "Polygon", "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]},
            })

        pending_deliverables.append({
            "id": f"geocode-{uuid.uuid4().hex[:8]}",
            "kind": "vector_layer",
            "title": f"Localisation : {clean_query}",
            "format": "geojson",
            "displayable": True,
            "geojson": {"type": "FeatureCollection", "features": features},
            "bbox": top.get("bbox"),
        })

        return {
            "status": "success",
            "query": clean_query,
            "lat": top["lat"],
            "lon": top["lon"],
            "bbox": top.get("bbox"),
            "display_name": top.get("display_name"),
            "address": top.get("address"),
            "results": parsed_results,
        }

    mtumishi.register_tool(
        GeoTool(
            name="geocode_place",
            description="Convertit un nom de lieu, ville, mine, province ou adresse (ex: 'Kolwezi', 'Goma', 'Parc des Virunga') en coordonnées géographiques (latitude, longitude, bbox) et l'affiche sur la carte.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Nom du lieu ou adresse à géocoder"},
                    "limit": {"type": "integer", "description": "Nombre max de résultats (1 à 5)", "default": 3},
                },
                "required": ["query"],
            },
            handler=_geocode_place,
        )
    )


def _extract_deliverables(result_payload: Any, extra_layers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collect any GeoJSON/raster/file deliverables produced by a tool call."""
    items: List[Dict[str, Any]] = list(extra_layers)
    if isinstance(result_payload, dict):
        if isinstance(result_payload.get("geojson"), dict):
            items.append({
                "kind": "vector_geojson",
                "name": result_payload.get("name") or "Couche vectorielle Mtumishi",
                "geojson": result_payload["geojson"],
            })
        if isinstance(result_payload.get("raster_overlay"), dict):
            items.append({
                "kind": "raster_image",
                **result_payload["raster_overlay"],
            })
        if isinstance(result_payload.get("deliverables"), list):
            items.extend(result_payload["deliverables"])
    return items


def _verify_supabase_jwt(authorization: Optional[str], supabase_url: str, service_key: str) -> Optional[str]:
    """Validate Supabase Bearer JWT and return user_id when auth is enforced or token is present."""
    require_auth = os.getenv("MTUMISHI_REQUIRE_AUTH", "0").lower() in ("1", "true", "yes")
    if not authorization or not authorization.startswith("Bearer "):
        if require_auth:
            raise HTTPException(status_code=401, detail="Authentification Supabase requise (Bearer token manquant)")
        return None

    token = authorization.split(" ", 1)[1].strip()
    if not token:
        if require_auth:
            raise HTTPException(status_code=401, detail="Jeton Bearer invalide")
        return None

    try:
        client = create_client(supabase_url, service_key)
        user_resp = client.auth.get_user(token)
        user = getattr(user_resp, "user", None)
        if user and getattr(user, "id", None):
            return str(user.id)
        if require_auth:
            raise HTTPException(status_code=401, detail="Session Supabase expirée ou invalide")
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("Supabase JWT validation error: %s", exc)
        if require_auth:
            raise HTTPException(status_code=401, detail="Impossible de valider la session Supabase")
    return None


def _split_layers_and_artifacts(items: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Separate map-renderable layers from downloadable artifact files."""
    layers: List[Dict[str, Any]] = []
    artifacts: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind") or "")
        if kind in ("vector_geojson", "raster_image"):
            layers.append(item)
        if item.get("url") or item.get("contentBase64") or item.get("geojson") or kind == "file":
            artifacts.append({
                "id": item.get("id") or str(uuid.uuid4()),
                "name": item.get("name") or item.get("fileName") or "Livrable Mtumishi",
                "fileName": item.get("fileName") or item.get("name") or "livrable.json",
                "kind": kind or "file",
                "url": item.get("url"),
                "contentBase64": item.get("contentBase64"),
                "geojson": item.get("geojson"),
            })
    return layers, artifacts


@router.get("/health")
def agent_health() -> Dict[str, Any]:
    """Verify that the streaming agent endpoint is online."""
    return {"status": "online", "service": "mtumishi-agent-stream", "active_sessions": len(_SESSIONS)}


@router.post("/stream")
async def stream_agent(
    req: AgentStreamRequest,
    authorization: Optional[str] = Header(default=None),
) -> StreamingResponse:
    """Stream Mtumishi thoughts, tool calls, map layers, artifacts, questions, and text via SSE."""
    prompt = req.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt vide")

    config = load_config()
    user_id = _verify_supabase_jwt(authorization, config.supabase_url, config.supabase_service_role_key)

    session_id = req.session_id or str(uuid.uuid4())
    context_dict = req.context or {}
    geometry = context_dict.get("geometry") if isinstance(context_dict, dict) else None
    is_conversational = bool(_detect_conversational_intent(prompt))

    runtime_type = "native" if (req.runtime == "native" or (req.runtime == "auto" and is_conversational)) else "antigravity"
    model_name = req.model_name or config.default_model_name

    async def event_generator() -> AsyncIterator[str]:
        t0 = time.time()
        yield _sse_event("status", {
            "session_id": session_id,
            "user_id": user_id,
            "runtime": runtime_type,
            "message": "Connexion à Mtumishi AI...",
        })

        # Fast path for greetings / help / identity (< 50ms)
        if runtime_type == "native":
            try:
                mtumishi = _get_or_create_session(session_id, "native", model_name, config.gemini_api_key, config.qgis_rpc_port)
                ctx = build_context(context_dict)
                result = mtumishi.run(prompt, context_override=ctx)
                text = result.summary or "Bonjour ! 😊 Comment puis-je vous aider aujourd'hui ?"
                yield _sse_event("text", {"content": text})
                yield _sse_event("done", {
                    "session_id": session_id,
                    "runtime_used": "native",
                    "execution_ms": round((time.time() - t0) * 1000, 2),
                })
                return
            except Exception as exc:
                logger.warning("Native stream fast-path failed: %s", exc)

        # Full agentic streaming path
        pending_questions: List[Dict[str, Any]] = []
        pending_deliverables: List[Dict[str, Any]] = []
        mtumishi = _get_or_create_session(session_id, "antigravity", model_name, config.gemini_api_key, config.qgis_rpc_port)
        _register_web_tools(mtumishi, geometry, pending_questions, pending_deliverables)
        ctx = build_context(context_dict)

        execution_prompt = (
            "[Contexte d'exécution : Application Web GeoCongo AI. Tu es exécuté sur la carte interactive Web. "
            "Tu peux utiliser les outils disponibles (Pekua, Gundua, STAC, qgis_headless_measure_area, sentinel2_forest_change, ask_user_confirmation). "
            "Si une emprise est ambiguë ou une opération est coûteuse, pose une question via ask_user_confirmation.]\n\n"
            + prompt
        )

        emitted_text = False
        try:
            async for ev in mtumishi.stream(execution_prompt, context_override=ctx):
                ev_type = ev.type.value if hasattr(ev.type, "value") else str(ev.type)
                meta = ev.metadata or {}

                if ev_type == "thought" and ev.content:
                    yield _sse_event("thought", {"content": ev.content})
                elif ev_type == "tool_call":
                    tool_name = str(meta.get("tool") or "tool")
                    if not tool_name.startswith("dev_"):
                        yield _sse_event("tool_call", {"tool": tool_name, "args": meta.get("args") or {}})
                elif ev_type == "tool_result":
                    tool_name = str(meta.get("tool") or "tool")
                    if tool_name.startswith("dev_"):
                        continue
                    res = meta.get("result")
                    err = tool_failure(res)
                    status = "error" if err else (meta.get("status") or "success")
                    yield _sse_event("tool_result", {"tool": tool_name, "status": status, "result": res, "error": err})

                    if pending_questions:
                        q_payload = pending_questions.pop(0)
                        yield _sse_event("question", q_payload)

                    deliverables = _extract_deliverables(res, pending_deliverables)
                    pending_deliverables.clear()
                    if deliverables:
                        layers, artifacts = _split_layers_and_artifacts(deliverables)
                        for layer in layers:
                            yield _sse_event("layer", layer)
                        for artifact in artifacts:
                            yield _sse_event("artifact", artifact)
                elif ev_type == "text" and ev.content:
                    emitted_text = True
                    yield _sse_event("text", {"content": ev.content})
                elif ev_type == "status" and ev.content:
                    yield _sse_event("status", {"message": ev.content})
                elif ev_type == "error" and ev.content:
                    yield _sse_event("error", {"message": ev.content})

            # Recover any tool outputs recorded in MtumishiMemory if the SDK skipped tool_result frames
            mem_steps = getattr(getattr(mtumishi, "memory", None), "steps", None) or []
            for mstep in mem_steps:
                t_name = mstep.get("tool") if isinstance(mstep, dict) else getattr(mstep, "tool_name", None)
                t_res = mstep.get("result") if isinstance(mstep, dict) else getattr(mstep, "result", None)
                if t_name and not str(t_name).startswith("dev_"):
                    deliverables = _extract_deliverables(t_res, pending_deliverables)
                    pending_deliverables.clear()
                    if deliverables:
                        layers, artifacts = _split_layers_and_artifacts(deliverables)
                        for layer in layers:
                            yield _sse_event("layer", layer)
                        for artifact in artifacts:
                            yield _sse_event("artifact", artifact)
        except asyncio.CancelledError:
            logger.info("Client disconnected from SSE stream session=%s", session_id)
            return
        except Exception as exc:
            logger.exception("Error during Mtumishi SSE stream session=%s", session_id)
            if not emitted_text:
                yield _sse_event("error", {"message": f"Erreur lors du streaming Mtumishi : {exc}"})

        yield _sse_event("done", {
            "session_id": session_id,
            "runtime_used": "antigravity",
            "execution_ms": round((time.time() - t0) * 1000, 2),
        })

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
