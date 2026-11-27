"""Unit and integration tests for Level 1 monitoring, inference telemetry, and FastAPI endpoints."""

from __future__ import annotations

from io import BytesIO
import os
from pathlib import Path
import tempfile
import time
import unittest

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image

import app as app_module
from inference import (
    STAGE1_MODEL_PATH,
    STAGE2_MODEL_PATH,
    load_model,
    predict,
    predict_with_telemetry,
)
from monitoring import MetricsStore, PredictionTelemetryEvent


PROJECT_DIR = Path(__file__).resolve().parent.parent


def _find_sample_wound_image() -> Path | None:
    lutseg_dir = PROJECT_DIR / "data" / "LUTSeg"
    val_txt = lutseg_dir / "val.txt"
    if val_txt.exists():
        for line in val_txt.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            candidate = lutseg_dir / parts[0]
            if candidate.exists():
                return candidate
    return None


def _encode_png(image: Image.Image) -> bytes:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


class TestMetricsStore(unittest.TestCase):
    def test_summary_percentiles_drift_and_windows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "metrics.db"
            store = MetricsStore(db_path=db_path, json_logs=False)
            try:
                now = time.time()
                # Old event outside 24h window
                store.record(
                    PredictionTelemetryEvent(
                        request_id="old001",
                        model_version="V6-TwoStage",
                        status_code=200,
                        file_size_bytes=1024 * 100,
                        image_width=640,
                        image_height=480,
                        include_overlay=True,
                        decode_ms=5.0,
                        stage1_ms=150.0,
                        stage2_ms=200.0,
                        inference_ms=350.0,
                        overlay_ms=30.0,
                        total_ms=385.0,
                        wound_detected=True,
                        wound_coverage=12.5,
                        stage1_max_prob=0.95,
                        stage1_mean_prob=0.82,
                        stage2_confidence=0.88,
                        stage2_entropy=0.22,
                        epithelial_pct=10.0,
                        slough_pct=20.0,
                        granulation_pct=60.0,
                        necrotic_pct=10.0,
                        other_pct=0.0,
                        ts_epoch=now - (48 * 3600),
                    )
                )

                # Recent wound detected (high confidence)
                store.record(
                    PredictionTelemetryEvent(
                        request_id="rec001",
                        model_version="V6-TwoStage",
                        status_code=200,
                        file_size_bytes=1024 * 200,
                        image_width=1000,
                        image_height=1000,
                        include_overlay=True,
                        decode_ms=10.0,
                        stage1_ms=200.0,
                        stage2_ms=250.0,
                        inference_ms=450.0,
                        overlay_ms=40.0,
                        total_ms=500.0,
                        wound_detected=True,
                        wound_coverage=20.0,
                        stage1_max_prob=0.9123,
                        stage1_mean_prob=0.7654,
                        stage2_confidence=0.8123,
                        stage2_entropy=0.3100,
                        epithelial_pct=5.0,
                        slough_pct=25.0,
                        granulation_pct=50.0,
                        necrotic_pct=20.0,
                        other_pct=0.0,
                        ts_epoch=now - 60,
                    )
                )

                # Recent wound detected (low confidence -> should be flagged)
                store.record(
                    PredictionTelemetryEvent(
                        request_id="rec002",
                        model_version="V6-TwoStage",
                        status_code=200,
                        file_size_bytes=1024 * 300,
                        image_width=1000,
                        image_height=2000,
                        include_overlay=False,
                        decode_ms=20.0,
                        stage1_ms=220.0,
                        stage2_ms=280.0,
                        inference_ms=500.0,
                        overlay_ms=0.0,
                        total_ms=520.0,
                        wound_detected=True,
                        wound_coverage=10.0,
                        stage1_max_prob=0.6000,
                        stage1_mean_prob=0.4500,
                        stage2_confidence=0.5200,
                        stage2_entropy=0.6800,
                        epithelial_pct=15.0,
                        slough_pct=35.0,
                        granulation_pct=30.0,
                        necrotic_pct=10.0,
                        other_pct=10.0,
                        ts_epoch=now - 30,
                    )
                )

                # Recent healthy skin (no wound, stage2 skipped)
                store.record(
                    PredictionTelemetryEvent(
                        request_id="rec003",
                        model_version="V6-TwoStage",
                        status_code=200,
                        file_size_bytes=1024 * 100,
                        image_width=500,
                        image_height=500,
                        include_overlay=True,
                        decode_ms=4.0,
                        stage1_ms=180.0,
                        stage2_ms=0.0,
                        inference_ms=180.0,
                        overlay_ms=12.0,
                        total_ms=196.0,
                        wound_detected=False,
                        wound_coverage=0.0,
                        stage1_max_prob=0.0412,
                        stage1_mean_prob=None,
                        stage2_confidence=None,
                        stage2_entropy=None,
                        ts_epoch=now - 20,
                    )
                )

                # Recent client error
                store.record(
                    PredictionTelemetryEvent(
                        request_id="err001",
                        model_version="V6-TwoStage",
                        status_code=400,
                        error_type="invalid_image",
                        file_size_bytes=128,
                        include_overlay=True,
                        total_ms=2.5,
                        ts_epoch=now - 10,
                    )
                )

                summary_24h = store.get_summary(window_hours=24.0)
                self.assertEqual(summary_24h["traffic"]["total_requests"], 4)
                self.assertEqual(summary_24h["traffic"]["success_requests"], 3)
                self.assertEqual(summary_24h["traffic"]["error_requests"], 1)
                self.assertEqual(summary_24h["traffic"]["error_rate_pct"], 25.0)
                self.assertEqual(summary_24h["traffic"]["error_breakdown"], {"invalid_image": 1})

                # Stage 2 stats exclude no-wound request; overlay stats exclude include_overlay=False
                self.assertEqual(summary_24h["latency_ms"]["stage2_when_detected"]["count"], 2)
                self.assertEqual(summary_24h["latency_ms"]["stage2_when_detected"]["avg"], 265.0)
                self.assertEqual(summary_24h["latency_ms"]["overlay_when_enabled"]["count"], 2)
                self.assertEqual(summary_24h["latency_ms"]["overlay_when_enabled"]["avg"], 26.0)

                # Model behavior & probability precision (4 decimals preserved)
                mb = summary_24h["model_behavior"]
                self.assertEqual(mb["wound_detected_count"], 2)
                self.assertAlmostEqual(mb["wound_detection_rate_pct"], 66.67)
                self.assertEqual(mb["avg_wound_coverage_pct"], 15.0)
                self.assertEqual(mb["low_confidence_count"], 1)
                self.assertEqual(mb["low_confidence_rate_pct"], 50.0)
                self.assertEqual(mb["stage1_max_prob"]["p50"], 0.6)
                self.assertAlmostEqual(mb["stage1_mean_prob"]["avg"], 0.6077, places=4)
                self.assertEqual(mb["tissue_composition_avg_pct"]["granulation"], 40.0)
                self.assertEqual(mb["tissue_composition_avg_pct"]["slough"], 30.0)

                # All-time includes the 48h old event
                summary_all = store.get_summary(window_hours=0.0)
                self.assertEqual(summary_all["window"]["hours"], "all")
                self.assertEqual(summary_all["traffic"]["total_requests"], 5)

                # Recent filters
                errors = store.get_recent(limit=10, only_errors=True)
                self.assertEqual([e["request_id"] for e in errors], ["err001"])

                low_conf = store.get_recent(limit=10, low_confidence_only=True)
                self.assertEqual([e["request_id"] for e in low_conf], ["rec002"])
            finally:
                store.close()

    def test_retention_pruning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MetricsStore(
                db_path=Path(tmp) / "prune.db",
                json_logs=False,
                max_rows=3,
                retention_days=30,
                prune_every=1,
            )
            try:
                for i in range(6):
                    store.record(
                        PredictionTelemetryEvent(
                            request_id=f"req{i}",
                            model_version="V6-TwoStage",
                            status_code=200,
                            total_ms=10.0,
                        )
                    )
                recent = store.get_recent(limit=10)
                self.assertEqual(len(recent), 3)
                self.assertEqual([r["request_id"] for r in recent], ["req5", "req4", "req3"])
            finally:
                store.close()

    def test_failsafe_on_unwritable_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # Point db_path inside a regular file so directory creation fails
            blocker = Path(tmp) / "not_a_dir"
            blocker.write_text("x", encoding="utf-8")
            store = MetricsStore(db_path=blocker / "metrics.db", json_logs=False)
            self.assertFalse(store.db_ready)
            # Recording must never raise
            store.record(
                PredictionTelemetryEvent(
                    request_id="safe01",
                    model_version="V6-TwoStage",
                    status_code=200,
                    total_ms=12.3,
                )
            )
            summary = store.get_summary()
            self.assertFalse(summary["system"]["metrics_db_ready"])
            self.assertEqual(summary["traffic"]["total_requests"], 0)


@unittest.skipUnless(
    STAGE1_MODEL_PATH.exists() and STAGE2_MODEL_PATH.exists(),
    "Local V6 ONNX model files are required for integration tests",
)
class TestInferenceAndAPIIntegration(unittest.TestCase):
    def test_predict_pixel_parity_and_confidence(self) -> None:
        model = load_model()

        # 1. Plain image -> no wound detected, stage2 skipped
        blank = Image.new("RGB", (320, 240), (225, 195, 175))
        mask_legacy = predict(model, blank)
        res_blank = predict_with_telemetry(model, blank)
        np.testing.assert_array_equal(mask_legacy, res_blank.mask)
        self.assertGreaterEqual(res_blank.stage1_ms, 0.0)
        self.assertGreaterEqual(res_blank.stage1_max_prob, 0.0)
        self.assertLessEqual(res_blank.stage1_max_prob, 1.0)
        if not res_blank.stage2_ran:
            self.assertEqual(res_blank.stage2_ms, 0.0)
            self.assertIsNone(res_blank.stage2_confidence)
            self.assertIsNone(res_blank.stage2_entropy)

        # 2. Real LUTSeg validation image (if present) -> exact parity + valid confidence bounds
        sample_path = _find_sample_wound_image()
        if sample_path is not None:
            with Image.open(sample_path) as sample_img:
                rgb = sample_img.convert("RGB")
                mask_old = predict(model, rgb)
                res_wound = predict_with_telemetry(model, rgb)
                np.testing.assert_array_equal(mask_old, res_wound.mask)
                self.assertTrue(res_wound.stage2_ran)
                self.assertGreater(res_wound.stage1_ms, 0.0)
                self.assertGreater(res_wound.stage2_ms, 0.0)
                self.assertIsNotNone(res_wound.stage1_mean_prob)
                self.assertIsNotNone(res_wound.stage2_confidence)
                self.assertIsNotNone(res_wound.stage2_entropy)
                assert res_wound.stage1_mean_prob is not None
                assert res_wound.stage2_confidence is not None
                assert res_wound.stage2_entropy is not None
                self.assertTrue(0.0 <= res_wound.stage1_mean_prob <= 1.0)
                self.assertTrue(0.0 <= res_wound.stage2_confidence <= 1.0)
                self.assertTrue(0.0 <= res_wound.stage2_entropy <= 1.0)

    def test_fastapi_endpoints_and_metrics_auth(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "api_metrics.db"
            prev_env = {
                k: os.environ.get(k)
                for k in ("METRICS_DB_PATH", "ENABLE_JSON_LOGS", "API_KEY", "METRICS_API_KEY")
            }
            os.environ["METRICS_DB_PATH"] = str(db_path)
            os.environ["ENABLE_JSON_LOGS"] = "false"
            os.environ["API_KEY"] = ""
            os.environ["METRICS_API_KEY"] = "secret-metrics-key"

            try:
                with TestClient(app_module.app) as client:
                    # 1. Health endpoint
                    health_res = client.get("/health")
                    self.assertEqual(health_res.status_code, 200)
                    health_data = health_res.json()
                    self.assertEqual(health_data["status"], "healthy")
                    self.assertTrue(health_data["model_loaded"])
                    self.assertTrue(health_data["metrics_db_ready"])
                    self.assertIn("uptime_seconds", health_data)
                    self.assertIn("memory", health_data)

                    # 2. Predict on real sample wound image (or fallback synthetic image)
                    sample_path = _find_sample_wound_image()
                    if sample_path is not None:
                        img_bytes = sample_path.read_bytes()
                    else:
                        img_bytes = _encode_png(Image.new("RGB", (320, 320), (180, 50, 50)))

                    pred_res = client.post(
                        "/predict",
                        files={"file": ("wound.png", img_bytes, "image/png")},
                    )
                    self.assertEqual(pred_res.status_code, 200)
                    body = pred_res.json()
                    self.assertTrue(body["success"])
                    self.assertEqual(body["model_version"], "V6-TwoStage")
                    self.assertEqual(pred_res.headers.get("X-Request-ID"), body["request_id"])
                    for key in ("decode", "stage1", "stage2", "inference", "overlay", "total"):
                        self.assertIn(key, body["timings_ms"])
                    for key in (
                        "stage1_max_prob",
                        "stage1_mean_prob",
                        "stage2_confidence",
                        "stage2_entropy",
                    ):
                        self.assertIn(key, body["confidence"])
                    self.assertIsNotNone(body["overlay"])

                    # 3. Predict without overlay on blank skin
                    blank_bytes = _encode_png(Image.new("RGB", (240, 240), (225, 195, 175)))
                    pred_no_ov = client.post(
                        "/predict?include_overlay=false",
                        files={"file": ("skin.png", blank_bytes, "image/png")},
                    )
                    self.assertEqual(pred_no_ov.status_code, 200)
                    self.assertIsNone(pred_no_ov.json()["overlay"])
                    self.assertEqual(pred_no_ov.json()["timings_ms"]["overlay"], 0.0)

                    # 4. Predict invalid file -> 400 recorded
                    bad_res = client.post(
                        "/predict",
                        files={"file": ("bad.png", b"not-an-image", "image/png")},
                    )
                    self.assertEqual(bad_res.status_code, 400)

                    # 5. Predict empty file -> 400 recorded
                    empty_res = client.post(
                        "/predict",
                        files={"file": ("empty.png", b"", "image/png")},
                    )
                    self.assertEqual(empty_res.status_code, 400)

                    # 6. Metrics endpoints require METRICS_API_KEY
                    self.assertEqual(client.get("/metrics/summary").status_code, 401)
                    self.assertEqual(
                        client.get("/metrics/summary", headers={"X-API-Key": "wrong"}).status_code,
                        401,
                    )
                    self.assertEqual(client.get("/metrics/recent").status_code, 401)

                    # 7. Valid key returns accurate performance summary & recent events
                    headers = {"X-API-Key": "secret-metrics-key"}
                    sum_res = client.get("/metrics/summary?hours=24", headers=headers)
                    self.assertEqual(sum_res.status_code, 200)
                    summary = sum_res.json()
                    self.assertEqual(summary["traffic"]["total_requests"], 4)
                    self.assertEqual(summary["traffic"]["success_requests"], 2)
                    self.assertEqual(summary["traffic"]["error_requests"], 2)
                    self.assertEqual(
                        summary["traffic"]["error_breakdown"],
                        {"invalid_image": 1, "empty_file": 1},
                    )
                    self.assertEqual(summary["latency_ms"]["total"]["count"], 2)
                    self.assertEqual(summary["latency_ms"]["overlay_when_enabled"]["count"], 1)

                    rec_res = client.get("/metrics/recent?limit=10", headers=headers)
                    self.assertEqual(rec_res.status_code, 200)
                    self.assertEqual(rec_res.json()["count"], 4)

                    err_res = client.get("/metrics/recent?only_errors=true", headers=headers)
                    self.assertEqual(err_res.status_code, 200)
                    self.assertEqual(err_res.json()["count"], 2)
            finally:
                for k, v in prev_env.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v


if __name__ == "__main__":
    unittest.main()
