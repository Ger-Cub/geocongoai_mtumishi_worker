"""FastAPI router for Mtumishi real-time synchronous chat endpoints."""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from worker.config import load_config

logger = logging.getLogger("geocongoai.worker.chat")

router = APIRouter(prefix="/api/v1/chat", tags=["mtumishi-chat"])


class ChatMessageRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=32000, description="User prompt or greeting")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Optional GIS context (bbox, province, etc.)")
    model_name: Optional[str] = Field(default=None, description="Model to use (defaults to gemini-2.5-flash)")
    runtime: Optional[str] = Field(default="auto", description="'auto', 'native', or 'antigravity'")


class ChatMessageResponse(BaseModel):
    status: str
    reply: str
    summary: Optional[str] = None
    is_conversational: bool = False
    runtime_used: str = "native"
    steps: List[Dict[str, Any]] = Field(default_factory=list)
    qgis_scripts: List[str] = Field(default_factory=list)
    execution_ms: float = 0.0


@router.get("/health")
def chat_health() -> Dict[str, Any]:
    """Instant health check to verify that Mtumishi chat worker is alive."""
    return {
        "status": "online",
        "service": "mtumishi-chat",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.post("", response_model=ChatMessageResponse)
def fast_chat(req: ChatMessageRequest) -> ChatMessageResponse:
    """Execute a synchronous conversational or analytical query without blocking on the DB queue."""
    t0 = time.time()
    prompt = req.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt vide")

    # 1. Vérification intention conversationnelle (salutations, politesse, aide, identité)
    is_conversational = False
    try:
        from geocongoai.mtumishi.runtimes.native import _detect_conversational_intent
        is_conversational = bool(_detect_conversational_intent(prompt))
    except Exception:
        pass

    # 2. Si conversationnel ou mode forcé 'native', réponse ultra-rapide (< 50ms)
    should_use_native = (req.runtime == "native") or (req.runtime == "auto" and is_conversational)

    if should_use_native:
        try:
            from geocongoai import Mtumishi
            mtumishi = Mtumishi(runtime_type="native")
            result = mtumishi.run(prompt)
            reply = result.summary or "Bonjour ! Comment puis-je vous aider aujourd'hui ?"
            exec_ms = round((time.time() - t0) * 1000, 2)
            return ChatMessageResponse(
                status="completed",
                reply=reply,
                summary=result.summary,
                is_conversational=True,
                runtime_used="native",
                steps=[],
                qgis_scripts=[],
                execution_ms=exec_ms,
            )
        except Exception as exc:
            logger.warning("Native conversational run failed: %s, falling back to LLM", exc)

    # 3. Requête nécessitant le modèle IA (Gemini / Antigravity)
    try:
        from geocongoai import Mtumishi
        config = load_config()
        model = req.model_name or config.default_model_name
        runtime_kwargs = {
            "model_name": model,
            "api_key": config.gemini_api_key,
            "workspaces": [],
        }
        mtumishi = Mtumishi(
            runtime_type="antigravity",
            qgis_rpc_port=config.qgis_rpc_port,
            runtime_kwargs=runtime_kwargs,
        )
        result = mtumishi.run(prompt)
        reply = result.summary or "Analyse terminée."
        exec_ms = round((time.time() - t0) * 1000, 2)
        return ChatMessageResponse(
            status="completed",
            reply=reply,
            summary=result.summary,
            is_conversational=is_conversational,
            runtime_used="antigravity",
            steps=getattr(result, "steps", []) or [],
            qgis_scripts=getattr(result, "qgis_scripts", []) or [],
            execution_ms=exec_ms,
        )
    except Exception as exc:
        logger.exception("Fast chat execution failed: %s", exc)
        # Fallback de secours gracieux en natif
        try:
            from geocongoai import Mtumishi
            mtumishi = Mtumishi(runtime_type="native")
            res = mtumishi.run(prompt)
            exec_ms = round((time.time() - t0) * 1000, 2)
            return ChatMessageResponse(
                status="completed",
                reply=res.summary or "Service temporairement en mode dégradé.",
                summary=res.summary,
                is_conversational=True,
                runtime_used="native-fallback",
                steps=[],
                qgis_scripts=[],
                execution_ms=exec_ms,
            )
        except Exception:
            raise HTTPException(status_code=500, detail=f"Erreur Mtumishi: {exc}")
