"""Preprocess satellite bands into a 3-channel (SWIR/NIR/Red) false-color composite with cloud masking."""
from __future__ import annotations

import io
import logging
from typing import Optional, Tuple

import numpy as np
from PIL import Image
import rasterio
from rasterio.enums import Resampling
from rasterio.io import MemoryFile

logger = logging.getLogger("geocongoai.geospatial_rag.processor")

# SCL (Scene Classification Layer) values for cloud masking
# 3: Cloud shadow, 8: Cloud medium probability, 9: Cloud high probability, 10: Thin cirrus
CLOUD_SCL_CLASSES = {3, 8, 9, 10}


def fetch_band_window(
    band_url: str,
    target_shape: Tuple[int, int] = (512, 512),
) -> np.ndarray:
    """Read a remote COG band into memory and resample to standard target shape."""
    with rasterio.open(band_url) as src:
        data = src.read(
            1,
            out_shape=target_shape,
            resampling=Resampling.bilinear,
        ).astype(np.float32)
        return data


def mask_and_interpolate_clouds(
    swir: np.ndarray,
    nir: np.ndarray,
    red: np.ndarray,
    scl_url: Optional[str] = None,
    target_shape: Tuple[int, int] = (512, 512),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Identify cloudy pixels using SCL mask and replace them with median local reflectance."""
    cloud_fraction = 0.0

    if not scl_url:
        return swir, nir, red, cloud_fraction

    try:
        with rasterio.open(scl_url) as src:
            scl = src.read(
                1,
                out_shape=target_shape,
                resampling=Resampling.nearest,
            )
        cloud_mask = np.isin(scl, list(CLOUD_SCL_CLASSES))
        cloud_pixels = np.sum(cloud_mask)
        total_pixels = cloud_mask.size
        cloud_fraction = float(cloud_pixels) / float(total_pixels) if total_pixels > 0 else 0.0

        if cloud_fraction > 0:
            logger.info("Cloud masking applied: %.2f%% pixels masked and interpolated", cloud_fraction * 100)
            for channel in (swir, nir, red):
                valid_pixels = channel[~cloud_mask]
                fill_val = np.median(valid_pixels) if valid_pixels.size > 0 else 0.0
                channel[cloud_mask] = fill_val

    except Exception as exc:
        logger.warning("Could not apply SCL cloud masking: %s", exc)

    return swir, nir, red, cloud_fraction


def normalize_band(band: np.ndarray, p_low: float = 2.0, p_high: float = 98.0) -> np.ndarray:
    """Apply a 2% - 98% percentile contrast stretch and normalize to 0-255 uint8."""
    finite_vals = band[np.isfinite(band)]
    if finite_vals.size == 0:
        return np.zeros_like(band, dtype=np.uint8)

    v_min, v_max = np.percentile(finite_vals, (p_low, p_high))
    if v_max <= v_min:
        v_max = v_min + 1.0

    clipped = np.clip(band, v_min, v_max)
    normalized = ((clipped - v_min) / (v_max - v_min) * 255.0).astype(np.uint8)
    return normalized


def build_swir_nir_red_composite(
    swir_url: str,
    nir_url: str,
    red_url: str,
    scl_url: Optional[str] = None,
    target_shape: Tuple[int, int] = (512, 512),
) -> Tuple[Image.Image, float]:
    """
    Construct a 3-channel false-color composite (SWIR=R, NIR=G, Red=B) with cloud filtering.
    Zero Local Heavy Storage: operates in-memory only.
    """
    swir_data = fetch_band_window(swir_url, target_shape=target_shape)
    nir_data = fetch_band_window(nir_url, target_shape=target_shape)
    red_data = fetch_band_window(red_url, target_shape=target_shape)

    swir_clean, nir_clean, red_clean, cloud_frac = mask_and_interpolate_clouds(
        swir_data, nir_data, red_data, scl_url=scl_url, target_shape=target_shape
    )

    r_band = normalize_band(swir_clean)
    g_band = normalize_band(nir_clean)
    b_band = normalize_band(red_clean)

    rgb_array = np.dstack((r_band, g_band, b_band))
    image = Image.fromarray(rgb_array, mode="RGB")
    return image, cloud_frac


def image_to_jpeg_bytes(image: Image.Image, quality: int = 88) -> bytes:
    """Convert a PIL Image to compressed JPEG bytes in memory."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def process_user_uploaded_image(image_bytes: bytes, max_dim: int = 512) -> bytes:
    """Preprocess user-uploaded query image into standardized format for multimodal embedding."""
    with Image.open(io.BytesIO(image_bytes)) as img:
        img_rgb = img.convert("RGB")
        img_rgb.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img_rgb.save(buf, format="JPEG", quality=90)
        return buf.getvalue()

