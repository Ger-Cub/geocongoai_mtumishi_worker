"""Embedding client for Qwen3-VL-Embedding-8B (multimodal, dimension 4096)."""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
from typing import List, Optional, Union

import httpx
import numpy as np

logger = logging.getLogger("geocongoai.geospatial_rag.embedding")

EMBEDDING_DIM = 4096
DEFAULT_QWEN_VL_MODEL = "Qwen/Qwen3-VL-Embedding-8B"
HF_API_URL_TEMPLATE = "https://api-inference.huggingface.co/models/{model}"


class QwenVLEmbeddingClient:
    """Client for generating 4096-dimensional embeddings with Qwen3-VL-Embedding-8B."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = DEFAULT_QWEN_VL_MODEL,
        endpoint_url: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_API_KEY")
        self.model_name = model_name
        self.endpoint_url = endpoint_url or os.getenv("QWEN_VL_ENDPOINT") or HF_API_URL_TEMPLATE.format(model=model_name)
        self.timeout = float(os.getenv("QWEN_VL_TIMEOUT", "60.0"))

    def _normalize_vector(self, vec: List[float]) -> List[float]:
        """L2-normalize a vector so that cosine similarity equals dot product."""
        arr = np.array(vec, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr.tolist()

    def _generate_deterministic_fallback(self, seed_data: Union[str, bytes]) -> List[float]:
        """Generate a deterministic 4096-dim L2-normalized vector for testing or offline environments."""
        if isinstance(seed_data, str):
            seed_bytes = seed_data.encode("utf-8")
        else:
            seed_bytes = seed_data

        seed_int = int(hashlib.sha256(seed_bytes).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed_int)
        raw_vec = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
        norm = np.linalg.norm(raw_vec)
        if norm > 0:
            raw_vec = raw_vec / norm
        return raw_vec.tolist()

    def embed_text(self, text: str) -> List[float]:
        """Generate a 4096-dim embedding for a text query."""
        if not text or not text.strip():
            raise ValueError("Query text cannot be empty.")

        cleaned_text = text.strip()

        if not self.api_key:
            logger.info("No HUGGINGFACE_API_KEY configured; using deterministic fallback embedding for text.")
            return self._generate_deterministic_fallback(cleaned_text)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "inputs": cleaned_text,
            "parameters": {"dimension": EMBEDDING_DIM},
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(self.endpoint_url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()

                if isinstance(data, list):
                    # Handle flat list or list of lists
                    vec = data[0] if isinstance(data[0], list) else data
                    if len(vec) == EMBEDDING_DIM:
                        return self._normalize_vector(vec)

                if isinstance(data, dict) and "embeddings" in data:
                    vec = data["embeddings"][0]
                    if len(vec) == EMBEDDING_DIM:
                        return self._normalize_vector(vec)

                logger.warning("Unexpected response format from embedding API; falling back to deterministic vector.")
                return self._generate_deterministic_fallback(cleaned_text)

        except Exception as exc:
            logger.warning("Failed to invoke Qwen3-VL text embedding API (%s); using fallback vector.", exc)
            return self._generate_deterministic_fallback(cleaned_text)

    def embed_image(self, image_bytes: bytes) -> List[float]:
        """Generate a 4096-dim embedding for a preprocessed image."""
        if not image_bytes:
            raise ValueError("Image bytes cannot be empty.")

        if not self.api_key:
            logger.info("No HUGGINGFACE_API_KEY configured; using deterministic fallback embedding for image.")
            return self._generate_deterministic_fallback(image_bytes)

        b64_img = base64.b64encode(image_bytes).decode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "inputs": {
                "image": b64_img,
            },
            "parameters": {"dimension": EMBEDDING_DIM},
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(self.endpoint_url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()

                if isinstance(data, list):
                    vec = data[0] if isinstance(data[0], list) else data
                    if len(vec) == EMBEDDING_DIM:
                        return self._normalize_vector(vec)

                if isinstance(data, dict) and "embeddings" in data:
                    vec = data["embeddings"][0]
                    if len(vec) == EMBEDDING_DIM:
                        return self._normalize_vector(vec)

                logger.warning("Unexpected response format from embedding API; falling back to deterministic vector.")
                return self._generate_deterministic_fallback(image_bytes)

        except Exception as exc:
            logger.warning("Failed to invoke Qwen3-VL image embedding API (%s); using fallback vector.", exc)
            return self._generate_deterministic_fallback(image_bytes)

