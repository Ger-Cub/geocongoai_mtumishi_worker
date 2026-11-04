"""Unit and integration tests for Geospatial & Multimodal RAG."""
from __future__ import annotations

import io
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image
from starlette.testclient import TestClient

from worker.app import app
from worker.geospatial_rag.embedding_client import (
    EMBEDDING_DIM,
    QwenVLEmbeddingClient,
)
from worker.geospatial_rag.image_processor import (
    image_to_jpeg_bytes,
    mask_and_interpolate_clouds,
    normalize_band,
    process_user_uploaded_image,
)
from worker.geospatial_rag.stac_client import CONGO_BASIN_BBOX


class TestGeospatialRAG(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_endpoint(self):
        """Test health check returns 200 with model information."""
        resp = self.client.get("/api/v1/geospatial-search/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data.get("status"), "healthy")
        self.assertEqual(data.get("dimension"), 4096)
        self.assertEqual(data.get("model"), "Qwen/Qwen3-VL-Embedding-8B")

    def test_qwen_embedding_dimension_and_norm(self):
        """Test Qwen3-VL client generates strictly 4096-dim L2-normalized vectors."""
        embedder = QwenVLEmbeddingClient()
        vec = embedder.embed_text("Structures annulaires volcaniques des Virunga")
        self.assertEqual(len(vec), EMBEDDING_DIM)
        norm = np.linalg.norm(np.array(vec, dtype=np.float32))
        self.assertAlmostEqual(norm, 1.0, places=4)

    def test_qwen_image_embedding_dimension(self):
        """Test image embedding generation generates strictly 4096 dimensions."""
        embedder = QwenVLEmbeddingClient()
        img = Image.new("RGB", (64, 64), color=(200, 100, 50))
        img_bytes = image_to_jpeg_bytes(img)
        vec = embedder.embed_image(img_bytes)
        self.assertEqual(len(vec), EMBEDDING_DIM)
        norm = np.linalg.norm(np.array(vec, dtype=np.float32))
        self.assertAlmostEqual(norm, 1.0, places=4)

    def test_band_normalization(self):
        """Test 2-98% percentile normalization to uint8 [0, 255]."""
        raw = np.array([[100.0, 500.0], [1000.0, 4000.0]], dtype=np.float32)
        normed = normalize_band(raw)
        self.assertEqual(normed.dtype, np.uint8)
        self.assertEqual(normed.shape, raw.shape)

    def test_cloud_masking_interpolation(self):
        """Test that cloud masking replaces cloudy pixels with median values."""
        swir = np.ones((10, 10), dtype=np.float32) * 50.0
        nir = np.ones((10, 10), dtype=np.float32) * 80.0
        red = np.ones((10, 10), dtype=np.float32) * 40.0
        # Simuler une zone nuageuse
        swir[0, 0] = 999.0
        swir_clean, _, _, _ = mask_and_interpolate_clouds(swir, nir, red, scl_url=None)
        self.assertEqual(swir_clean.shape, (10, 10))

    def test_congo_basin_bbox_bounds(self):
        """Verify Congo Basin bounding box coordinates are well-formed."""
        min_x, min_y, max_x, max_y = CONGO_BASIN_BBOX
        self.assertLess(min_x, max_x)
        self.assertLess(min_y, max_y)
        # DRC covers equator from ~ -13.5 to +5.5 lat
        self.assertLess(min_y, 0.0)
        self.assertGreater(max_y, 0.0)

    @patch("worker.geospatial_rag.router._get_supabase_client")
    def test_text_search_endpoint(self, mock_get_supabase):
        """Test text search endpoint calls match_satellite_tiles RPC."""
        mock_supabase = MagicMock()
        mock_rpc = MagicMock()
        mock_rpc.execute.return_value.data = [
            {
                "id": "11111111-2222-3333-4444-555555555555",
                "stac_item_id": "S2A_MSIL2A_20240815T080611_R078_T35LPH_20240815T120000",
                "similarity": 0.892,
                "metadata": {
                    "target_region": "Virunga",
                    "cloud_cover": 2.1,
                },
            }
        ]
        mock_supabase.rpc.return_value = mock_rpc
        mock_get_supabase.return_value = mock_supabase

        resp = self.client.post(
            "/api/v1/geospatial-search/text",
            json={"query": "Structures annulaires des Virunga", "match_count": 5},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data.get("query_type"), "text")
        self.assertEqual(len(data.get("results")), 1)
        self.assertEqual(data["results"][0]["similarity"], 0.892)

    @patch("worker.geospatial_rag.router._get_supabase_client")
    def test_image_search_endpoint(self, mock_get_supabase):
        """Test image upload search endpoint."""
        mock_supabase = MagicMock()
        mock_rpc = MagicMock()
        mock_rpc.execute.return_value.data = [
            {
                "id": "22222222-3333-4444-5555-666666666666",
                "stac_item_id": "S2B_MSIL2A_20240901T081500_R021_T34KDA",
                "similarity": 0.841,
                "metadata": {"target_region": "Katanga"},
            }
        ]
        mock_supabase.rpc.return_value = mock_rpc
        mock_get_supabase.return_value = mock_supabase

        img = Image.new("RGB", (32, 32), color=(120, 150, 80))
        img_bytes = image_to_jpeg_bytes(img)

        resp = self.client.post(
            "/api/v1/geospatial-search/image",
            files={"file": ("test_rock.jpg", img_bytes, "image/jpeg")},
            data={"match_count": 3},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data.get("query_type"), "image")
        self.assertEqual(len(data.get("results")), 1)


if __name__ == "__main__":
    unittest.main()

