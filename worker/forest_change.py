"""Bounded Sentinel-2 change analysis for a drawn Web polygon."""
from base64 import b64encode
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from logging import getLogger
from math import ceil
from types import SimpleNamespace


logger = getLogger(__name__)

def analyze_forest_change(geometry, before_year, after_year):
    """Return scientific layers separately from the compact tool summary."""
    import numpy as np
    import rasterio as rio
    from PIL import Image
    from pystac_client import Client
    from pystac_client.stac_api_io import StacApiIO
    from rasterio.features import geometry_mask, geometry_window, shapes
    from rasterio.io import MemoryFile
    from rasterio.transform import from_origin
    from rasterio.vrt import WarpedVRT
    from rasterio.warp import Resampling, reproject, transform_bounds, transform_geom
    from rasterio.windows import transform as window_transform

    if not all(type(year) in (int, float) and year == int(year) for year in (before_year, after_year)):
        raise ValueError("Indiquez deux années entières.")
    before_year, after_year = int(before_year), int(after_year)
    if not 2018 <= before_year < after_year <= 2100:
        raise ValueError("Indiquez deux années croissantes à partir de 2018.")
    rings = geometry.get("coordinates", [])
    if geometry.get("type") != "Polygon" or not rings or not rings[0]:
        raise ValueError("Dessinez un polygone sur la carte.")
    lon, lat = zip(*rings[0])
    bbox = (min(lon), min(lat), max(lon), max(lat))
    if not (-180 <= bbox[0] < bbox[2] < 180 and -80 <= bbox[1] < bbox[3] <= 84):
        raise ValueError("Emprise géographique invalide pour l’analyse Sentinel-2.")
    zone = int(((bbox[0] + bbox[2]) / 2 + 180) / 6) + 1
    crs = f"EPSG:{32600 + zone if sum(lat) >= 0 else 32700 + zone}"
    west, south, east, north = transform_bounds("EPSG:4326", crs, *bbox)
    width, height = ceil((east - west) / 20), ceil((north - south) / 20)
    if max(width, height) > 256:
        raise ValueError("L’emprise doit tenir dans une grille de 256 × 256 pixels à 20 m.")
    grid = from_origin(west, north, 20, 20)
    region = geometry_mask([transform_geom("EPSG:4326", crs, geometry)], (height, width), grid, invert=True)
    if not region.any():
        raise ValueError("La zone est plus petite qu’un pixel de 20 m.")

    def composite(year):
        catalog = Client.open("https://earth-search.aws.element84.com/v1",
                              stac_io=StacApiIO(timeout=15, max_retries=1))
        items = catalog.search(method="GET", collections=["sentinel-2-l2a"], bbox=bbox,
                               datetime=f"{year}-08-01/{year}-09-30", max_items=100).items()
        samples, scenes = [], []
        def read(item, key):
            asset = item.assets[key]
            with rio.open(asset.href) as source, WarpedVRT(
                source, crs=crs, transform=grid, width=width, height=height,
                resampling=Resampling.nearest,
            ) as vrt:
                raw = vrt.read(1, masked=True).astype("float32").filled(np.nan)
            if key == "scl":
                return raw
            band = asset.extra_fields["raster:bands"][0]
            offset = 0 if item.properties.get("earthsearch:boa_offset_applied") else band.get("offset", 0)
            return raw * band["scale"] + offset

        def covers_region(item):
            if not item.bbox:
                return False
            left, bottom, right, top = item.bbox
            return left <= bbox[0] and bottom <= bbox[1] and right >= bbox[2] and top >= bbox[3]

        ordered = sorted(items, key=lambda item: (not covers_region(item), item.properties.get("eo:cloud_cover", 100)))
        read_errors = 0
        with rio.Env(
            GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
            GDAL_HTTP_MULTIRANGE="SERIAL",
            GDAL_HTTP_TIMEOUT="16",
            GDAL_HTTP_MAX_RETRY="1",
            GDAL_HTTP_RETRY_DELAY="1",
        ):
            for item in ordered[:20]:
                date = item.datetime.date().isoformat()
                if len(samples) == 4:
                    break
                if any(scene["date"] == date for scene in scenes):
                    continue
                try:
                    valid = region & np.isin(read(item, "scl"), [4, 5])
                    if valid.sum() < .70 * region.sum():
                        continue
                    red, nir, swir = [read(item, key) for key in ("red", "nir", "swir22")]
                except (rio.errors.RasterioError, KeyError) as exc:
                    logger.warning("Unable to read Sentinel-2 scene %s: %s", item.id, exc.__cause__ or exc)
                    read_errors += 1
                    continue
                valid &= (red >= 0) & (nir >= 0) & (swir >= 0) & ((nir + red) > .02) & ((nir + swir) > .02)
                if valid.sum() < .70 * region.sum():
                    continue
                samples.append([
                    np.divide(nir - red, nir + red, where=valid, out=np.full_like(nir, np.nan)),
                    np.divide(nir - swir, nir + swir, where=valid, out=np.full_like(nir, np.nan)),
                ])
                scenes.append({"id": item.id, "date": date})
        if len(samples) < 2:
            raise ValueError(f"{year} : moins de deux dates exploitables ({read_errors} erreurs de lecture).")
        stack = np.asarray(samples)
        covered = np.isfinite(stack[:, 0]).sum(axis=0) >= 2
        median = np.full(stack.shape[1:], np.nan, dtype=stack.dtype)
        median[:, covered] = np.nanmedian(stack[:, :, covered], axis=0)
        return median, scenes

    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = (pool.submit(composite, year) for year in (before_year, after_year))
        before, scenes_before = first.result()
        after, scenes_after = second.result()

    valid = np.isfinite(before).all(axis=0) & np.isfinite(after).all(axis=0)
    if not valid.any():
        raise ValueError("Aucun pixel exploitable en commun entre les deux périodes.")
    loss = before - after
    candidates = valid & (before[0] >= .65) & (loss[0] >= .20) & (loss[1] >= .20)
    features = []
    raster = SimpleNamespace(width=width, height=height, transform=grid)
    for polygon, _ in shapes(candidates.astype("uint8"), mask=candidates, transform=grid):
        window = geometry_window(raster, [polygon])
        rows, cols = window.toslices()
        patch = candidates[rows, cols]
        mask = geometry_mask([polygon], patch.shape, window_transform(window, grid), invert=True) & patch
        features.append({"type": "Feature", "geometry": transform_geom(crs, "EPSG:4326", polygon, precision=7),
                         "properties": {"area_ha": round(float(mask.sum()) * .04, 2),
                                        "ndvi_loss": round(float(loss[0][rows, cols][mask].mean()), 3)}})
    features.sort(key=lambda f: f["properties"]["area_ha"], reverse=True)
    for index, feature in enumerate(features, 1):
        feature["properties"]["target_id"] = f"V{index:03d}"
        if index <= 25:
            feature["properties"]["map_label"] = f"V{index:03d}"

    nodata = -9999
    score = np.where(valid, loss[0], nodata).astype("float32")
    with MemoryFile() as memory:
        with memory.open(driver="GTiff", width=width, height=height, count=1, dtype="float32",
                         transform=grid, crs=crs, nodata=nodata, compress="deflate") as dataset:
            dataset.write(score, 1)
            dataset.set_band_description(1, "NDVI avant moins NDVI après")
        tiff = memory.read()

    # The map image is projected into Web Mercator for Leaflet/MapLibre.
    x0, y0, x1, y1 = transform_bounds("EPSG:4326", "EPSG:3857", *bbox)
    preview = np.full_like(score, nodata)
    reproject(score, preview, src_crs=crs, src_transform=grid, src_nodata=nodata,
              dst_crs="EPSG:3857", dst_transform=from_origin(x0, y1, (x1-x0)/width, (y1-y0)/height),
              dst_nodata=nodata, resampling=Resampling.nearest)
    rgba = np.zeros((height, width, 4), dtype="uint8")
    shown = (preview > 0) & (preview != nodata)
    for channel, stops in enumerate(([35, 251, 225], [116, 191, 43], [220, 36, 30])):
        rgba[:, :, channel][shown] = np.interp(preview[shown], [0, .20, .45], stops)
    rgba[:, :, 3][shown] = 220
    image = BytesIO()
    Image.fromarray(rgba).save(image, format="PNG", optimize=True)

    def binary(kind, name, filename, data):
        return {"kind": kind, "analysisType": "deforestation", "name": name, "fileName": filename, "sizeBytes": len(data),
                "contentBase64": b64encode(data).decode("ascii")}

    stats = {"before_year": before_year, "after_year": after_year, "scenes_before": scenes_before,
             "scenes_after": scenes_after, "coverage_percent": round(100 * float(valid.sum() / region.sum()), 1),
             "candidate_ha": round(float(candidates.sum()) * .04, 2), "candidate_count": len(features)}
    series = {"title": "NDVI médian des pixels comparables", "points": [
        {"x": str(before_year), "y": round(float(np.median(before[0][valid])), 4)},
        {"x": str(after_year), "y": round(float(np.median(after[0][valid])), 4)},
    ]}
    return stats, [
        {**binary("raster_image", "Baisse du NDVI", "baisse-ndvi.png", image.getvalue()),
         "bbox": list(bbox), "series": series, "legend": [{"label": label, "color": color} for label, color in
                  [("0", "#2374dc"), ("0,20", "#fbbf24"), ("≥ 0,45", "#e12b1e")]]},
        binary("raster_geotiff", "Variation du NDVI · GeoTIFF", "variation-ndvi.tif", tiff),
        {"kind": "vector_geojson", "analysisType": "deforestation", "name": "Zones candidates de perte de végétation",
         "fileName": "zones-candidates.geojson", "geojson": {"type": "FeatureCollection", "features": features}},
    ]
