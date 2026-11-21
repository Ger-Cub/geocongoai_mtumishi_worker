"""Offline regressions against the SDK result contract; no provider or Supabase calls."""
import subprocess
from base64 import b64decode
import unittest
import sys
import types
try:
    import supabase
except ImportError:
    # Unit tests mock the store; they never construct a network client.
    module = types.ModuleType("supabase")
    module.Client = object
    module.create_client = lambda *_args, **_kwargs: None
    sys.modules["supabase"] = module
from types import SimpleNamespace
from unittest.mock import Mock, patch

from geocongoai import Mtumishi
from geocongoai.mtumishi.events import GeoTraceEvent
from geocongoai.mtumishi.planner import MtumishiResult
from worker.models import LeasedRun
from worker.runtime import QGIS_AREA_TOOL, build_context, execute_run, collect_execution_steps
from geocongoai.mtumishi.memory import MtumishiMemory
from worker.realtime import MtumishiRealtimePublisher
from worker.supabase_ops import SupabaseRunStore


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        self.run = LeasedRun("run-a", "user-a", "running", "Analyse cette zone", {
            "environment": "web", "bbox": [25, -11, 26, -10],
            "geometry": {"type": "Polygon", "coordinates": []},
        }, "antigravity", "gemini-2.5-flash", 12)
        self.config = SimpleNamespace(default_runtime="antigravity", default_model_name="gemini-2.5-flash",
                                      gemini_api_key="test", qgis_rpc_port=65534, worker_id="test-worker")

    def execute(self, result):
        with patch("worker.runtime.Mtumishi") as cls:
            cls.return_value.run_with_stream.return_value = result
            return execute_run(self.run, self.config, on_event=lambda _: None)

    def test_real_sdk_greeting_uses_no_geospatial_tools(self):
        self.run.prompt = "Bonjour"
        result = execute_run(self.run, self.config, on_event=lambda _: None)
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["summary"])
        self.assertEqual(result["steps"], [])
        self.assertIsNone(result["geojson"])
        self.assertIsNone(result["raster_overlay"])

    def test_sdk_success_becomes_completed_and_preserves_all_tool_results(self):
        steps = [{"tool": "gundua_analyse_deterministe", "status": "success", "result": {
            "geojson_url": "outputs/targets.json", "raster_url": "outputs/score.tif",
        }}, {"tool": "stac_search_scenes", "status": "success", "result": {"scene_id": "scene-a"}}]
        result = self.execute(MtumishiResult(summary="Terminé", status="success", steps=steps))
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["steps"], steps)
        self.assertIsNone(result["raster_overlay"])
        self.assertIsNone(result["geojson"])

    def test_failed_sdk_or_tool_is_not_a_success(self):
        for status, steps in [("failed", []), ("success", [{"tool": "gundua_analyse_deterministe", "status": "error", "result": {"error": "timeout"}}])]:
            result = self.execute(MtumishiResult(summary="Échec", status=status, steps=steps))
            self.assertEqual(result["status"], "failed")
            self.assertEqual(len(result["steps"]), len(steps))
            if steps:
                self.assertEqual(result["steps"][0]["error"], "timeout")

    def test_tool_error_payload_overrides_sdk_success(self):
        result = self.execute(MtumishiResult(summary="Calcul partiel", status="success", steps=[{
            "tool": "stac_calc_indices", "status": "success",
            "result": {"status": "error", "message": "Lecture des pixels impossible"},
        }]))
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["steps"][0]["error"], "Lecture des pixels impossible")

    def test_missing_stream_results_are_recovered_from_real_sdk_memory(self):
        memory = MtumishiMemory()
        payload = {"satellite_preview_url": "outputs/preview.png", "geojson_url": "outputs/empty.json",
                   "details": {"bbox": [25, -11, 26, -10]}}
        memory.record_step("gundua_analyse_deterministe", {"analysis_type": "greenfield"}, payload)
        starts = [{"tool": "gundua_analyse_deterministe", "status": "running"}]
        with patch("worker.runtime.Mtumishi") as cls:
            cls.return_value.memory = memory
            cls.return_value.run_with_stream.return_value = MtumishiResult(summary="Fichiers produits", steps=[])
            result = execute_run(self.run, self.config, on_event=lambda _: None, steps_collector=starts)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(result["steps"]), 1)
        self.assertEqual(result["steps"][0]["status"], "success")
        self.assertEqual(result["steps"][0]["result"], payload)

    def test_real_sdk_stream_without_toolresult_chunks_preserves_artifacts(self):
        agent = Mtumishi(runtime_type="native")
        payload = {"satellite_preview_url": "outputs/image.png", "details": {"bbox": [25, -11, 26, -10]}}
        starts = []

        async def stream(prompt, context_override=None):
            yield GeoTraceEvent.tool_call("gundua_analyse_deterministe", {"analysis_type": "greenfield"})
            agent.memory.record_step("gundua_analyse_deterministe", {"analysis_type": "greenfield"}, payload)
            yield GeoTraceEvent.text("Résultats disponibles")

        def on_event(event):
            if event.type.value == "tool_call":
                starts.append({"tool": event.metadata["tool"], "status": "running"})

        agent.stream = stream
        with patch("worker.runtime.Mtumishi", return_value=agent):
            result = execute_run(self.run, self.config, on_event=on_event, steps_collector=starts)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["steps"][0]["result"], payload)
        self.assertEqual(result["steps"][0]["status"], "success")

    def test_web_agent_delivers_headless_qgis_polygon(self):
        available = subprocess.run(
            ["/usr/bin/python3", "-I", "-c", "import qgis.core"],
            capture_output=True, check=False, timeout=15,
        )
        if available.returncode:
            self.skipTest("System PyQGIS is not installed")
        polygon = {"type": "Polygon", "coordinates": [[[29, 0], [29.1, 0], [29.1, 0.1], [29, 0.1], [29, 0]]]}
        self.run.prompt = "Mesure la superficie de cette zone avec QGIS"
        self.run.context_json["geometry"] = polygon
        agent = Mtumishi(runtime_type="native")

        async def stream(prompt, context_override=None):
            self.assertIn(QGIS_AREA_TOOL, prompt)
            self.assertEqual(context_override.metadata["geometry"], polygon)
            tool = agent.registry.get_tool(QGIS_AREA_TOOL)
            yield GeoTraceEvent.tool_call(tool.name, {})
            result = tool.execute()
            agent.memory.record_step(tool.name, {}, result)
            yield GeoTraceEvent.text("Superficie mesurée avec QGIS.")

        agent.stream = stream
        with patch("worker.runtime.Mtumishi", return_value=agent):
            result = execute_run(self.run, self.config, on_event=lambda _: None)
        self.assertEqual(result["status"], "completed")
        payload = result["steps"][0]["result"]
        self.assertAlmostEqual(payload["area_km2"], 123.0907, places=3)
        self.assertEqual(payload["geojson"]["features"][0]["geometry"], polygon)

    def test_agent_keeps_raster_files_out_of_the_model_tool_response(self):
        agent = Mtumishi(runtime_type="native")
        stats = {"candidate_ha": 3.5}
        layers = [{"kind": "raster_image", "name": "Baisse du NDVI", "contentBase64": "iVBORw0KGgo="}]

        async def stream(prompt, context_override=None):
            tool = agent.registry.get_tool("sentinel2_forest_change")
            self.assertIsNotNone(tool)
            result = tool.execute(before_year=2023, after_year=2025)
            self.assertEqual(result, stats)
            agent.memory.record_step(tool.name, {}, result)
            yield GeoTraceEvent.text("Comparaison terminée.")

        agent.stream = stream
        with patch("worker.runtime.Mtumishi", return_value=agent), patch(
            "worker.runtime.analyze_forest_change", return_value=(stats, layers),
        ):
            result = execute_run(self.run, self.config, on_event=lambda _: None)
        self.assertEqual(result["status"], "completed")
        self.assertIn(layers[0], result["deliverables"])
        self.assertEqual(result["steps"][0]["result"], stats)

    def test_gundua_study_delivers_pdf_for_completed_and_incomplete_runs(self):
        self.run.context_json.update(gundua_study="greenfield", report_pdf=True)
        for tool_results in ([{"tool": "sentinel2_forest_change", "status": "success",
                               "result": {"coverage_percent": 58.4}}], []):
            report = self.execute(MtumishiResult(summary="Comparaison des images", status="success", steps=tool_results))
            self.assertEqual(report["status"], "completed" if tool_results else "failed")
            self.assertEqual(report["analysisType"], "greenfield")
            pdf, = [item for item in report["deliverables"] if item.get("fileName") == "rapport-gundua.pdf"]
            self.assertTrue(b64decode(pdf["contentBase64"]).startswith(b"%PDF-"))

    def test_forest_study_runs_without_constructing_a_model(self):
        self.run.context_json.update(gundua_study="deforestation", gundua_mode="direct", report_pdf=True,
                                     forest_before_year=2023, forest_after_year=2025)
        events = []
        stats = {"coverage_percent": 58.4, "candidate_count": 2, "candidate_ha": 3.5}
        with patch("worker.runtime.Mtumishi", side_effect=AssertionError("model must not start")), patch(
            "worker.runtime.analyze_forest_change", return_value=(stats, []),
        ) as analyze:
            result = execute_run(self.run, self.config, on_event=events.append)
        analyze.assert_called_once_with(self.run.context_json["geometry"], 2023, 2025)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["steps"][0]["result"], stats)
        self.assertEqual([event.type.value for event in events], ["tool_call", "tool_result"])
        self.assertTrue(any(item.get("fileName") == "rapport-gundua.pdf" for item in result["deliverables"]))

    def test_forest_study_returns_provider_independent_data_error(self):
        self.run.context_json.update(gundua_study="deforestation", report_pdf=True)
        with patch("worker.runtime.Mtumishi", side_effect=AssertionError("model must not start")), patch(
            "worker.runtime.analyze_forest_change", side_effect=ValueError("Moins de deux dates exploitables."),
        ):
            result = execute_run(self.run, self.config)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["steps"][0]["error"], "Moins de deux dates exploitables.")
        self.assertTrue(any(item.get("fileName") == "rapport-gundua.pdf" for item in result["deliverables"]))

    def test_forest_assistant_mode_uses_agent(self):
        self.run.context_json.update(gundua_study="deforestation", gundua_mode="assistant", report_pdf=True)
        with patch("worker.runtime.Mtumishi") as agent, patch("worker.runtime.analyze_forest_change") as direct:
            agent.return_value.run_with_stream.return_value = MtumishiResult(summary="Réponse", status="success")
            result = execute_run(self.run, self.config, on_event=lambda _: None)
        agent.return_value.run_with_stream.assert_called_once()
        direct.assert_not_called()
        self.assertEqual(result["metadata"]["requested_runtime"], "antigravity")

    def test_unsupported_direct_mode_never_starts_an_assistant(self):
        self.run.context_json.update(gundua_study="lithology", gundua_mode="direct", report_pdf=True)
        with patch("worker.runtime.Mtumishi", side_effect=AssertionError("model must not start")):
            result = execute_run(self.run, self.config)
        self.assertEqual(result["status"], "failed")
        self.assertIn("disponible avec l’assistant", result["summary"])

    def test_repeated_invocations_and_failures_are_preserved(self):
        memory = MtumishiMemory()
        memory.record_step("same", {"id": 1}, {"error": "first failed"}, status="error")
        memory.record_step("same", {"id": 2}, {"geojson_url": "outputs/ok.json"})
        starts = [{"tool": "same", "status": "running"}, {"tool": "same", "status": "running"}]
        steps = collect_execution_steps([], memory.steps, starts, True)
        self.assertEqual(len(steps), 2)
        self.assertEqual([step["status"] for step in steps], ["error", "success"])

    def test_unresolved_tool_starts_are_not_reported_as_completed(self):
        with patch("worker.runtime.Mtumishi") as cls:
            cls.return_value.memory.steps = []
            cls.return_value.run_with_stream.return_value = MtumishiResult(summary="Terminé", steps=[])
            result = execute_run(self.run, self.config, on_event=lambda _: None,
                                 steps_collector=[{"tool": "gundua_analyse_deterministe", "status": "running"}])
        self.assertEqual(result["status"], "failed")
        self.assertIn("n’ont pas pu être récupérés", result["summary"])

    def test_stream_exception_does_not_execute_analysis_twice(self):
        with patch("worker.runtime.Mtumishi") as cls:
            cls.return_value.run_with_stream.side_effect = RuntimeError("stream error")
            with self.assertRaisesRegex(RuntimeError, "stream error"):
                execute_run(self.run, self.config, on_event=lambda _: None)
            cls.return_value.run.assert_not_called()

    def test_exact_geometry_survives_sdk_context_conversion(self):
        context = build_context(self.run.context_json)
        self.assertEqual(context.metadata["geometry"], self.run.context_json["geometry"])

    def test_notification_failure_cannot_destroy_a_persisted_result(self):
        client = Mock()
        client.table.return_value.insert.return_value.execute.side_effect = RuntimeError("offline")
        client.table.return_value.upsert.return_value.execute.side_effect = RuntimeError("offline")
        publisher = MtumishiRealtimePublisher(client)
        with self.assertLogs("worker.realtime", level="ERROR"):
            publisher.emit("run-a", "run_completed", {})
            publisher.emit_geojson_layer("run-a", "targets", {"type": "FeatureCollection", "features": []}, "Targets")
        self.assertEqual(client.table.return_value.upsert.call_args.kwargs["on_conflict"], "run_id,layer_id")

    def test_failure_refunds_reservation_and_keeps_partial_results(self):
        store = SupabaseRunStore.__new__(SupabaseRunStore)
        store.client = Mock()
        store.config = self.config
        result = {"steps": [{"tool": "one", "result": {"geojson_url": "outputs/partial.json"}}]}
        store.mark_failed(self.run, "failed", result=result)
        name, args = store.client.rpc.call_args.args
        self.assertEqual(name, "complete_mtumishi_run")
        self.assertEqual(args["p_final"], 0)
        self.assertEqual(args["p_status"], "failed")
        self.assertEqual(args["p_steps"], result["steps"])
        self.assertEqual(args["p_result"]["error"]["message"], "failed")

    def test_completion_rpc_rejection_cannot_be_bypassed(self):
        store = SupabaseRunStore.__new__(SupabaseRunStore)
        store.config = self.config
        store.client = Mock()
        store.client.rpc.return_value.execute.side_effect = RuntimeError("Lease expired")
        with self.assertRaisesRegex(RuntimeError, "Lease expired"):
            store.mark_completed(self.run, {"summary": "done"}, 12)
        store.client.table.assert_not_called()


if __name__ == "__main__":
    unittest.main()
