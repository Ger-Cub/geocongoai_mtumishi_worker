"""Measure an analysis area with system PyQGIS, using stdin and stdout only."""
import json
import math
import sys


def measure_area(geometry):
    if not isinstance(geometry, dict) or geometry.get("type") != "Polygon":
        raise ValueError("Dessinez un polygone sur la carte avant de mesurer sa superficie.")
    rings = geometry.get("coordinates")
    if not isinstance(rings, list) or not rings or len(rings) > 100:
        raise ValueError("Le polygone doit contenir entre 1 et 100 anneaux.")
    vertices = 0
    for ring in rings:
        if not isinstance(ring, list) or len(ring) < 4 or ring[0] != ring[-1]:
            raise ValueError("Chaque anneau doit être fermé et contenir au moins quatre positions.")
        vertices += len(ring)
        if vertices > 10_000:
            raise ValueError("Le polygone dépasse 10 000 positions.")
        for point in ring:
            if not isinstance(point, list) or len(point) != 2:
                raise ValueError("Les positions doivent contenir une longitude et une latitude.")
            for coordinate, limit in zip(point, (180, 90)):
                if type(coordinate) not in (int, float) or not -limit <= coordinate <= limit:
                    raise ValueError("Coordonnée hors des limites EPSG:4326.")

    from qgis.core import (
        QgsApplication,
        QgsCoordinateReferenceSystem,
        QgsCoordinateTransformContext,
        QgsDistanceArea,
        QgsGeometry,
        QgsPointXY,
    )

    app = QgsApplication([], False)
    app.initQgis()
    polygon = QgsGeometry.fromPolygonXY([[QgsPointXY(*point) for point in ring] for ring in rings])
    if not polygon.isGeosValid():
        raise ValueError("Le polygone est invalide. Vérifiez que ses contours ne se croisent pas.")
    distance = QgsDistanceArea()
    distance.setSourceCrs(QgsCoordinateReferenceSystem("EPSG:4326"), QgsCoordinateTransformContext())
    distance.setEllipsoid("WGS84")
    area_km2 = abs(distance.measureArea(polygon)) / 1_000_000
    if not math.isfinite(area_km2) or area_km2 <= 0:
        raise ValueError("La superficie du polygone doit être strictement positive.")
    return {
        "name": "Zone mesurée · QGIS",
        "area_km2": area_km2,
        "geojson": {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": json.loads(polygon.asJson(15)),
                "properties": {"area_km2": area_km2},
            }],
        },
    }


if __name__ == "__main__":
    try:
        print(json.dumps(measure_area(json.load(sys.stdin)), allow_nan=False))
    except ValueError as exc:
        print(json.dumps({"error": str(exc)}))
