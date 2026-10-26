"""Mtumishi runtime execution for leased jobs."""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from geocongoai import GeoSpatialContext, Mtumishi
from geocongoai.mtumishi.runtimes.native import _detect_conversational_intent

from worker.models import LeasedRun, WorkerConfig


def build_context(raw_context: Dict[str, Any]) -> GeoSpatialContext:
    """Build a `GeoSpatialContext` from the job payload."""
    return GeoSpatialContext.from_dict(raw_context or {})


def estimate_final_units(run: LeasedRun, result: Dict[str, Any]) -> int:
    """Basic billing fallback until token-based pricing is exposed by the runtime."""
    del result
    return max(1, run.units_reserved)


def execute_run(
    run: LeasedRun,
    config: WorkerConfig,
    on_event: Optional[Callable[[Any], None]] = None,
) -> Dict[str, Any]:
    """Execute a single Mtumishi job."""
    # Detect execution environment: Web vs QGIS
    client_env = (run.context_json or {}).get("environment") or "web"
    is_web = (client_env == "web")

    # Priority: if conversational intent (greetings, help, identity, etc.), route to native for instant response (< 50ms)
    prompt_str = run.prompt or ""
    is_conversational = False
    try:
        is_conversational = bool(_detect_conversational_intent(prompt_str))
    except Exception:
        pass

    if is_conversational:
        runtime_type = "native"
    else:
        runtime_type = run.requested_runtime or config.default_runtime

    if runtime_type == "native":
        runtime_kwargs = {
            "model_name": run.model_name or config.default_model_name,
        }
    else:
        # antigravity runtime — never provide directory workspaces to prevent dev_list_directory!
        runtime_kwargs = {
            "api_key": config.gemini_api_key,
            "model_name": run.model_name or config.default_model_name,
            "workspaces": [],
        }

    mtumishi = Mtumishi(
        runtime_type=runtime_type,
        qgis_rpc_port=config.qgis_rpc_port,
        runtime_kwargs=runtime_kwargs,
    )

    # When running on the Web: remove all desktop QGIS and file-dev tools
    # so the agent never attempts QGIS actions or dev tools
    if is_web and hasattr(mtumishi, "registry") and hasattr(mtumishi.registry, "_tools"):
        web_forbidden_prefixes = ("qgis_", "dev_")
        mtumishi.registry._tools = {
            name: tool for name, tool in mtumishi.registry._tools.items()
            if not any(name.startswith(p) for p in web_forbidden_prefixes)
        }

    context_override = build_context(run.context_json)

    # Add environment contextual hint for the agent prompt if in Web mode
    execution_prompt = run.prompt
    if is_web and not is_conversational:
        execution_prompt = (
            "[Contexte d'exécution : Application Web GeoCongo AI. Tu es exécuté sur l'interface cartographique web (pas dans QGIS Desktop). N'utilise aucun script ni outil QGIS. Concentre-toi sur l'imagerie satellitaire (STAC), la géologie (Pekua) et la détection d'indices minéraux/végétaux (Gundua) pour affichage sur la carte interactive.]\n\n"
            + run.prompt
        )

    if on_event and hasattr(mtumishi, "run_with_stream"):
        try:
            result = mtumishi.run_with_stream(
                prompt=execution_prompt,
                on_event=on_event,
                context_override=context_override,
            )
        except Exception:
            result = mtumishi.run(
                prompt=execution_prompt,
                context_override=context_override,
            )
    else:
        result = mtumishi.run(
            prompt=execution_prompt,
            context_override=context_override,
        )

    # Filter out any internal dev/workspace or QGIS tools from displayed steps
    cleaned_steps = [
        st for st in (result.steps or [])
        if not str(st.get("tool", "")).startswith("dev_")
        and not (is_web and str(st.get("tool", "")).startswith("qgis_"))
    ]

    # Automatically extract geospatial deliverables (GeoJSON and Raster Overlay)
    geojson_payload: Optional[Dict[str, Any]] = None
    raster_overlay_payload: Optional[Dict[str, Any]] = None

    bbox = context_override.bbox
    for st in cleaned_steps:
        tool_name = st.get("tool", "")
        res = st.get("result", {})
        if not isinstance(res, dict):
            continue

        # 1. STAC satellite imagery tools (stac_calc_indices, stac_search_scenes)
        if tool_name in ("stac_calc_indices", "stac_search_scenes"):
            step_bbox = res.get("bbox") or bbox
            scene_id = res.get("scene_id") or (res.get("best_scene") or {}).get("id")
            indices = res.get("indices", {})

            if step_bbox and len(step_bbox) == 4:
                min_lon, min_lat, max_lon, max_lat = (
                    float(step_bbox[0]),
                    float(step_bbox[1]),
                    float(step_bbox[2]),
                    float(step_bbox[3]),
                )

                preview_url = None
                overlay_name = "Imagerie Satellitaire Sentinel-2"

                if scene_id:
                    # If NDVI was computed, generate Planetary Computer viridis colorized NDVI preview
                    if "NDVI" in indices:
                        preview_url = (
                            f"https://planetarycomputer.microsoft.com/api/data/v1/item/preview.png?"
                            f"collection=sentinel-2-l2a&item={scene_id}&expression=(B08-B04)/(B08%2BB04)&colormap_name=viridis&rescale=-0.2,0.8"
                        )
                        overlay_name = f"NDVI Sentinel-2 ({scene_id[:20]}...)"
                    else:
                        # True color preview (B04, B03, B02)
                        preview_url = (
                            f"https://planetarycomputer.microsoft.com/api/data/v1/item/preview.png?"
                            f"collection=sentinel-2-l2a&item={scene_id}&assets=B04&assets=B03&assets=B02&color_formula=gamma+rgb+2.2+saturation+1.2"
                        )
                        overlay_name = f"Sentinel-2 ({scene_id[:20]}...)"

                if preview_url:
                    raster_overlay_payload = {
                        "url": preview_url,
                        # Leaflet bounds: [[South, West], [North, East]]
                        "bounds": [[min_lat, min_lon], [max_lat, max_lon]],
                        "name": overlay_name,
                        "scene_id": scene_id,
                        "opacity": 0.85,
                    }

                # Build GeoJSON feature representing the analyzed polygon
                geojson_payload = {
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [[
                                    [min_lon, min_lat],
                                    [max_lon, min_lat],
                                    [max_lon, max_lat],
                                    [min_lon, max_lat],
                                    [min_lon, min_lat],
                                ]],
                            },
                            "properties": {
                                "title": overlay_name,
                                "scene_id": scene_id,
                                "date": res.get("scene_date"),
                                "cloud_cover": res.get("cloud_cover"),
                                "indices": {
                                    k: v.get("stats")
                                    for k, v in indices.items()
                                    if isinstance(v, dict) and "stats" in v
                                },
                                "raster_url": preview_url,
                            },
                        }
                    ],
                }

        # 2. Gundua deterministic analysis
        elif tool_name == "gundua_analyse_deterministe":
            if res.get("geojson") and isinstance(res["geojson"], dict) and res["geojson"].get("type") in ("FeatureCollection", "Feature"):
                geojson_payload = res["geojson"]
            elif res.get("geojson_url"):
                raw_url = str(res["geojson_url"])
                full_url = raw_url if raw_url.startswith("http") else f"https://geocongo-solafune-greenfield-api.geocongoai.com/{raw_url.lstrip('/')}"
                # Download and inline GeoJSON directly for the web client
                try:
                    import requests
                    r = requests.get(full_url, timeout=6)
                    if r.ok:
                        geojson_payload = r.json()
                    else:
                        geojson_payload = {"geojson_url": full_url}
                except Exception:
                    geojson_payload = {"geojson_url": full_url}

            url = res.get("satellite_preview_url") or res.get("raster_url")
            if url and not str(url).startswith("http"):
                url = f"https://geocongo-solafune-greenfield-api.geocongoai.com/{str(url).lstrip('/')}"

            gundua_bbox = res.get("bbox") or bbox
            if url and gundua_bbox and len(gundua_bbox) == 4:
                raster_overlay_payload = {
                    "url": url,
                    "bounds": [[float(gundua_bbox[1]), float(gundua_bbox[0])], [float(gundua_bbox[3]), float(gundua_bbox[2])]],
                    "name": "Analyse Gundua",
                    "opacity": 0.85,
                }

    return {
        "status": result.status,
        "summary": result.summary,
        "steps": cleaned_steps,
        "qgis_scripts": [] if is_web else result.qgis_scripts,
        "geojson": geojson_payload,
        "raster_overlay": raster_overlay_payload,
        "context": result.context.to_dict() if result.context else None,
        "metadata": {
            "user_id": run.user_id,
            "run_id": run.id,
            "environment": client_env,
            "requested_runtime": runtime_type,
            "model_name": run.model_name or config.default_model_name,
            "worker_id": config.worker_id,
        },
    }
