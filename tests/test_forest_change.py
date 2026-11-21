"""Exercise the raster, GeoTIFF and feature contract on aligned test scenes."""
from base64 import b64decode
from datetime import datetime
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import rasterio
from PIL import Image
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from worker.forest_change import analyze_forest_change


class ForestChangeTests(TestCase):
    def test_georeferenced_outputs_and_candidate_attributes_use_the_same_pixels(self):
        polygon = {"type": "Polygon", "coordinates": [[[25.27, .55], [25.28, .55],
                   [25.28, .56], [25.27, .56], [25.27, .55]]]}
        with TemporaryDirectory() as work:
            scenes = {}
            for year, values in [(2023, (400, 3800, 800)), (2025, (1200, 2100, 1800))]:
                assets = {}
                for key, value in zip(("red", "nir", "swir22", "scl"), (*values, 4)):
                    path = Path(work) / f"{year}-{key}.tif"
                    with rasterio.open(path, "w", driver="GTiff", width=100, height=100, count=1,
                                       dtype="uint16", crs="EPSG:4326",
                                       transform=from_origin(25.26, .57, .0003, .0003)) as file:
                        file.write(np.full((100, 100), value, dtype="uint16"), 1)
                    assets[key] = SimpleNamespace(href=str(path), extra_fields={"raster:bands": [
                        {"scale": .0001, "offset": -.1},
                    ]})
                scenes[year] = [SimpleNamespace(id=f"{year}-{day}", datetime=datetime(year, 8, day),
                    properties={"eo:cloud_cover": 0, "earthsearch:boa_offset_applied": True},
                    bbox=(25.26, .54, 25.29, .57), assets=assets) for day in (12, 22)]

            def search(**kwargs):
                selected = scenes[int(kwargs["datetime"][:4])]
                return SimpleNamespace(items=lambda: iter(selected))

            with patch("pystac_client.Client.open", return_value=SimpleNamespace(search=search)):
                stats, (png, tiff, vectors) = analyze_forest_change(polygon, 2023, 2025)
        self.assertGreater(stats["candidate_ha"], 100)
        self.assertEqual(stats["candidate_count"], 1)
        props = vectors["geojson"]["features"][0]["properties"]
        self.assertEqual(props["target_id"], "V001")
        self.assertAlmostEqual(props["area_ha"], stats["candidate_ha"], places=2)
        with MemoryFile(b64decode(tiff["contentBase64"])) as file, file.open() as dataset:
            self.assertEqual(dataset.res, (20, 20))
            self.assertEqual(dataset.nodata, -9999)
            self.assertAlmostEqual(float(dataset.read(1, masked=True).mean()), .537, places=2)
        with Image.open(BytesIO(b64decode(png["contentBase64"]))) as image:
            self.assertEqual(image.mode, "RGBA")
            self.assertGreater(image.getchannel("A").getextrema()[1], 0)
