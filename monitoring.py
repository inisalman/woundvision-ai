"""Level 1 lightweight monitoring & telemetry for WoundVision AI.

Zero external dependencies (Python stdlib only):
- Structured one-line JSON logging to stdout (container-friendly).
- Compact SQLite WAL store for numeric prediction & error metadata (zero images/PII).
- Rolling aggregation for latency percentiles, model confidence/drift, input profile,
  and process memory/uptime.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
import resource
import sqlite3
import sys
import threading
import time
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_METRICS_DB_PATH = PROJECT_DIR / "telemetry" / "metrics.db"

LOW_STAGE1_CONFIDENCE_THRESHOLD = float(
    os.getenv("LOW_STAGE1_CONFIDENCE_THRESHOLD", "0.55")
)
LOW_STAGE2_CONFIDENCE_THRESHOLD = float(
    os.getenv("LOW_STAGE2_CONFIDENCE_THRESHOLD", "0.60")
)


def _env_bool(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _build_json_logger() -> logging.Logger:
    logger = logging.getLogger("woundvision.telemetry")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    return logger


_JSON_LOGGER = _build_json_logger()


def emit_json_log(event: str, payload: dict[str, Any], *, enabled: bool = True) -> None:
    """Emit a single-line structured JSON log record to stdout."""
    if not enabled:
        return
    record = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "event": event,
        **payload,
    }
    _JSON_LOGGER.info(json.dumps(record, separators=(",", ":"), ensure_ascii=True))


def get_memory_mb() -> dict[str, float | None]:
    """Return current RSS (Linux) and peak RSS (Linux/macOS) in MiB."""
    current_rss_mb: float | None = None
    proc_status = Path("/proc/self/status")
    if proc_status.exists():
        try:
            for line in proc_status.read_text(encoding="utf-8").splitlines():
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        current_rss_mb = round(int(parts[1]) / 1024.0, 2)
                    break
        except OSError:
            current_rss_mb = None

    peak_rss_mb: float | None = None
    try:
        ru_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports ru_maxrss in bytes; Linux reports kilobytes.
        divisor = (1024.0 * 1024.0) if sys.platform == "darwin" else 1024.0
        peak_rss_mb = round(ru_maxrss / divisor, 2)
    except Exception:
        peak_rss_mb = None

    return {
        "current_rss_mb": current_rss_mb,
        "peak_rss_mb": peak_rss_mb,
    }


def _percentile(sorted_values: list[float], p: float) -> float | None:
    """Linear-interpolated percentile on a pre-sorted list."""
    n = len(sorted_values)
    if n == 0:
        return None
    if n == 1:
        return float(sorted_values[0])
    k = (n - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(sorted_values[int(k)])
    return float(sorted_values[f] * (c - k) + sorted_values[c] * (k - f))


def _stats_summary(
    values: list[float],
    *,
    percentiles: tuple[int, ...] = (50, 95),
    include_max: bool = False,
    digits: int = 2,
) -> dict[str, float | int | None]:
    clean = sorted(v for v in values if v is not None)
    n = len(clean)
    if n == 0:
        out: dict[str, float | int | None] = {"count": 0, "avg": None}
        for p in percentiles:
            out[f"p{p}"] = None
        if include_max:
            out["max"] = None
        return out

    out = {
        "count": n,
        "avg": round(sum(clean) / n, digits),
    }
    for p in percentiles:
        val = _percentile(clean, float(p))
        out[f"p{p}"] = round(val, digits) if val is not None else None
    if include_max:
        out["max"] = round(clean[-1], digits)
    return out


@dataclass
class PredictionTelemetryEvent:
    request_id: str
    model_version: str
    status_code: int
    total_ms: float
    error_type: str | None = None
    file_size_bytes: int | None = None
    image_width: int | None = None
    image_height: int | None = None
    include_overlay: bool | None = None
    decode_ms: float | None = None
    stage1_ms: float | None = None
    stage2_ms: float | None = None
    inference_ms: float | None = None
    overlay_ms: float | None = None
    wound_detected: bool | None = None
    wound_coverage: float | None = None
    stage1_max_prob: float | None = None
    stage1_mean_prob: float | None = None
    stage2_confidence: float | None = None
    stage2_entropy: float | None = None
    epithelial_pct: float | None = None
    slough_pct: float | None = None
    granulation_pct: float | None = None
    necrotic_pct: float | None = None
    other_pct: float | None = None
    rss_mb: float | None = None
    timestamp: str | None = None
    ts_epoch: float | None = None


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS prediction_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    ts_epoch REAL NOT NULL,
    model_version TEXT NOT NULL,
    status_code INTEGER NOT NULL,
    error_type TEXT,
    file_size_bytes INTEGER,
    image_width INTEGER,
    image_height INTEGER,
    include_overlay INTEGER,
    decode_ms REAL,
    stage1_ms REAL,
    stage2_ms REAL,
    inference_ms REAL,
    overlay_ms REAL,
    total_ms REAL NOT NULL,
    wound_detected INTEGER,
    wound_coverage REAL,
    stage1_max_prob REAL,
    stage1_mean_prob REAL,
    stage2_confidence REAL,
    stage2_entropy REAL,
    epithelial_pct REAL,
    slough_pct REAL,
    granulation_pct REAL,
    necrotic_pct REAL,
    other_pct REAL,
    rss_mb REAL
);

CREATE INDEX IF NOT EXISTS idx_prediction_events_ts
    ON prediction_events (ts_epoch DESC);

CREATE INDEX IF NOT EXISTS idx_prediction_events_status_ts
    ON prediction_events (status_code, ts_epoch DESC);
"""


class MetricsStore:
    """Thread-safe, failsafe SQLite telemetry store with automatic retention pruning."""

    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        enabled: bool | None = None,
        json_logs: bool | None = None,
        retention_days: int | None = None,
        max_rows: int | None = None,
        prune_every: int = 200,
    ) -> None:
        raw_path = db_path if db_path is not None else os.getenv("METRICS_DB_PATH", "")
        self.db_path = Path(raw_path) if str(raw_path).strip() else DEFAULT_METRICS_DB_PATH
        self.enabled = _env_bool("METRICS_ENABLED", True) if enabled is None else enabled
        self.json_logs = _env_bool("ENABLE_JSON_LOGS", True) if json_logs is None else json_logs
        self.retention_days = (
            int(os.getenv("METRICS_RETENTION_DAYS", "90"))
            if retention_days is None
            else retention_days
        )
        self.max_rows = (
            int(os.getenv("METRICS_MAX_ROWS", "100000"))
            if max_rows is None
            else max_rows
        )
        self.prune_every = max(1, prune_every)
        self.started_at = time.time()

        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        self._write_count = 0
        self._init_error: str | None = None

        if self.enabled:
            self._init_db()

    @property
    def db_ready(self) -> bool:
        return self.enabled and self._conn is not None

    def _init_db(self) -> None:
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(
                str(self.db_path),
                timeout=3.0,
                check_same_thread=False,
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA busy_timeout=3000;")
            conn.executescript(_SCHEMA_SQL)
            conn.commit()
            self._conn = conn
            self._init_error = None
        except Exception as exc:
            self._conn = None
            self._init_error = str(exc)
            emit_json_log(
                "metrics_db_init_failed",
                {"db_path": str(self.db_path), "error": str(exc)},
                enabled=self.json_logs,
            )

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None

    def record(self, event: PredictionTelemetryEvent) -> None:
        """Persist one prediction/error event and emit structured JSON log. Never raises."""
        now_epoch = event.ts_epoch if event.ts_epoch is not None else time.time()
        ts_iso = (
            event.timestamp
            or datetime.fromtimestamp(now_epoch, tz=timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )
        if event.rss_mb is None:
            mem = get_memory_mb()
            event.rss_mb = mem["current_rss_mb"] if mem["current_rss_mb"] is not None else mem["peak_rss_mb"]

        event.timestamp = ts_iso
        event.ts_epoch = now_epoch

        # 1. Structured JSON log to stdout
        log_payload = {
            k: v
            for k, v in asdict(event).items()
            if k not in {"timestamp", "ts_epoch"} and v is not None
        }
        emit_json_log("predict", log_payload, enabled=self.json_logs)

        # 2. Persist numeric metadata to SQLite
        if not self.enabled or self._conn is None:
            return

        try:
            with self._lock:
                if self._conn is None:
                    return
                self._conn.execute(
                    """
                    INSERT INTO prediction_events (
                        request_id, timestamp, ts_epoch, model_version, status_code,
                        error_type, file_size_bytes, image_width, image_height,
                        include_overlay, decode_ms, stage1_ms, stage2_ms,
                        inference_ms, overlay_ms, total_ms, wound_detected,
                        wound_coverage, stage1_max_prob, stage1_mean_prob,
                        stage2_confidence, stage2_entropy, epithelial_pct,
                        slough_pct, granulation_pct, necrotic_pct, other_pct, rss_mb
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    (
                        event.request_id,
                        ts_iso,
                        now_epoch,
                        event.model_version,
                        event.status_code,
                        event.error_type,
                        event.file_size_bytes,
                        event.image_width,
                        event.image_height,
                        None if event.include_overlay is None else int(event.include_overlay),
                        event.decode_ms,
                        event.stage1_ms,
                        event.stage2_ms,
                        event.inference_ms,
                        event.overlay_ms,
                        event.total_ms,
                        None if event.wound_detected is None else int(event.wound_detected),
                        event.wound_coverage,
                        event.stage1_max_prob,
                        event.stage1_mean_prob,
                        event.stage2_confidence,
                        event.stage2_entropy,
                        event.epithelial_pct,
                        event.slough_pct,
                        event.granulation_pct,
                        event.necrotic_pct,
                        event.other_pct,
                        event.rss_mb,
                    ),
                )
                self._write_count += 1
                if self._write_count % self.prune_every == 0:
                    self._prune_locked(now_epoch)
                self._conn.commit()
        except Exception as exc:
            emit_json_log(
                "metrics_db_write_failed",
                {"request_id": event.request_id, "error": str(exc)},
                enabled=self.json_logs,
            )

    def _prune_locked(self, now_epoch: float) -> None:
        if self._conn is None:
            return
        if self.retention_days > 0:
            cutoff = now_epoch - (self.retention_days * 86400.0)
            self._conn.execute(
                "DELETE FROM prediction_events WHERE ts_epoch < ?",
                (cutoff,),
            )
        if self.max_rows > 0:
            self._conn.execute(
                """
                DELETE FROM prediction_events
                WHERE id NOT IN (
                    SELECT id FROM prediction_events
                    ORDER BY id DESC
                    LIMIT ?
                )
                """,
                (self.max_rows,),
            )

    def get_summary(self, window_hours: float = 24.0) -> dict[str, Any]:
        """Compute rolling traffic, latency, drift/confidence, input profile, and system stats."""
        now_epoch = time.time()
        generated_at = (
            datetime.fromtimestamp(now_epoch, tz=timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z")
        )
        since_epoch: float | None = None
        since_utc: str | None = None
        if window_hours and window_hours > 0:
            since_epoch = now_epoch - (window_hours * 3600.0)
            since_utc = (
                datetime.fromtimestamp(since_epoch, tz=timezone.utc)
                .isoformat(timespec="seconds")
                .replace("+00:00", "Z")
            )

        rows: list[sqlite3.Row] = []
        total_db_rows = 0
        if self.db_ready:
            with self._lock:
                assert self._conn is not None
                total_db_rows = int(
                    self._conn.execute("SELECT COUNT(*) FROM prediction_events").fetchone()[0]
                )
                if since_epoch is not None:
                    rows = self._conn.execute(
                        "SELECT * FROM prediction_events WHERE ts_epoch >= ? ORDER BY ts_epoch DESC",
                        (since_epoch,),
                    ).fetchall()
                else:
                    rows = self._conn.execute(
                        "SELECT * FROM prediction_events ORDER BY ts_epoch DESC"
                    ).fetchall()

        total_requests = len(rows)
        status_counts: dict[str, int] = {}
        error_breakdown: dict[str, int] = {}
        success_rows: list[sqlite3.Row] = []
        error_requests = 0

        for row in rows:
            code = int(row["status_code"])
            code_key = str(code)
            status_counts[code_key] = status_counts.get(code_key, 0) + 1
            if code == 200:
                success_rows.append(row)
            else:
                error_requests += 1
                err_key = row["error_type"] or f"http_{code}"
                error_breakdown[err_key] = error_breakdown.get(err_key, 0) + 1

        success_count = len(success_rows)
        error_rate_pct = (
            round((error_requests / total_requests) * 100.0, 2)
            if total_requests > 0
            else 0.0
        )

        # Latency breakdowns (only 200 OK requests)
        total_lat = [float(r["total_ms"]) for r in success_rows if r["total_ms"] is not None]
        inf_lat = [float(r["inference_ms"]) for r in success_rows if r["inference_ms"] is not None]
        decode_lat = [float(r["decode_ms"]) for r in success_rows if r["decode_ms"] is not None]
        s1_lat = [float(r["stage1_ms"]) for r in success_rows if r["stage1_ms"] is not None]
        s2_lat = [
            float(r["stage2_ms"])
            for r in success_rows
            if r["stage2_ms"] is not None and r["wound_detected"] == 1
        ]
        overlay_lat = [
            float(r["overlay_ms"])
            for r in success_rows
            if r["overlay_ms"] is not None and r["include_overlay"] == 1
        ]

        # Model behavior & drift signals
        detected_rows = [r for r in success_rows if r["wound_detected"] == 1]
        detected_count = len(detected_rows)
        detection_rate_pct = (
            round((detected_count / success_count) * 100.0, 2)
            if success_count > 0
            else None
        )

        coverages = [float(r["wound_coverage"]) for r in detected_rows if r["wound_coverage"] is not None]
        s1_max_probs = [float(r["stage1_max_prob"]) for r in success_rows if r["stage1_max_prob"] is not None]
        s1_mean_probs = [float(r["stage1_mean_prob"]) for r in detected_rows if r["stage1_mean_prob"] is not None]
        s2_confs = [float(r["stage2_confidence"]) for r in detected_rows if r["stage2_confidence"] is not None]
        s2_entropies = [float(r["stage2_entropy"]) for r in detected_rows if r["stage2_entropy"] is not None]

        low_conf_count = sum(
            1
            for r in detected_rows
            if (
                r["stage1_mean_prob"] is not None
                and float(r["stage1_mean_prob"]) < LOW_STAGE1_CONFIDENCE_THRESHOLD
            )
            or (
                r["stage2_confidence"] is not None
                and float(r["stage2_confidence"]) < LOW_STAGE2_CONFIDENCE_THRESHOLD
            )
        )
        low_conf_rate_pct = (
            round((low_conf_count / detected_count) * 100.0, 2)
            if detected_count > 0
            else None
        )

        tissue_avgs: dict[str, float | None] = {}
        for tissue_name, col in (
            ("epithelial", "epithelial_pct"),
            ("slough", "slough_pct"),
            ("granulation", "granulation_pct"),
            ("necrotic", "necrotic_pct"),
            ("other", "other_pct"),
        ):
            vals = [float(r[col]) for r in detected_rows if r[col] is not None]
            tissue_avgs[tissue_name] = round(sum(vals) / len(vals), 2) if vals else None

        # Input data profile
        file_sizes_kb = [
            float(r["file_size_bytes"]) / 1024.0
            for r in rows
            if r["file_size_bytes"] is not None
        ]
        megapixels = [
            (float(r["image_width"]) * float(r["image_height"])) / 1_000_000.0
            for r in success_rows
            if r["image_width"] is not None and r["image_height"] is not None
        ]

        # System & storage health
        mem = get_memory_mb()
        db_size_kb: float | None = None
        if self.db_path.exists():
            try:
                db_size_kb = round(self.db_path.stat().st_size / 1024.0, 2)
            except OSError:
                db_size_kb = None

        return {
            "window": {
                "hours": window_hours if (window_hours and window_hours > 0) else "all",
                "since_utc": since_utc,
                "generated_at_utc": generated_at,
            },
            "traffic": {
                "total_requests": total_requests,
                "success_requests": success_count,
                "error_requests": error_requests,
                "error_rate_pct": error_rate_pct,
                "status_counts": status_counts,
                "error_breakdown": error_breakdown,
            },
            "latency_ms": {
                "total": _stats_summary(total_lat, percentiles=(50, 95, 99), include_max=True),
                "inference": _stats_summary(inf_lat, percentiles=(50, 95, 99), include_max=True),
                "decode": _stats_summary(decode_lat, percentiles=(50, 95)),
                "stage1": _stats_summary(s1_lat, percentiles=(50, 95)),
                "stage2_when_detected": _stats_summary(s2_lat, percentiles=(50, 95)),
                "overlay_when_enabled": _stats_summary(overlay_lat, percentiles=(50, 95)),
            },
            "model_behavior": {
                "wound_detected_count": detected_count,
                "wound_detection_rate_pct": detection_rate_pct,
                "avg_wound_coverage_pct": (
                    round(sum(coverages) / len(coverages), 2) if coverages else None
                ),
                "stage1_max_prob": _stats_summary(
                    s1_max_probs, percentiles=(10, 50), digits=4
                ),
                "stage1_mean_prob": _stats_summary(
                    s1_mean_probs, percentiles=(10, 50), digits=4
                ),
                "stage2_confidence": _stats_summary(
                    s2_confs, percentiles=(10, 50), digits=4
                ),
                "stage2_entropy": _stats_summary(
                    s2_entropies, percentiles=(50, 90), digits=4
                ),
                "low_confidence_thresholds": {
                    "stage1_mean_prob_lt": LOW_STAGE1_CONFIDENCE_THRESHOLD,
                    "stage2_confidence_lt": LOW_STAGE2_CONFIDENCE_THRESHOLD,
                },
                "low_confidence_count": low_conf_count,
                "low_confidence_rate_pct": low_conf_rate_pct,
                "tissue_composition_avg_pct": tissue_avgs,
            },
            "input_profile": {
                "file_size_kb": _stats_summary(file_sizes_kb, percentiles=(50, 95), include_max=True),
                "megapixels": _stats_summary(megapixels, percentiles=(50, 95), include_max=True),
            },
            "system": {
                "uptime_seconds": round(time.time() - self.started_at, 1),
                "current_rss_mb": mem["current_rss_mb"],
                "peak_rss_mb": mem["peak_rss_mb"],
                "metrics_db_ready": self.db_ready,
                "metrics_db_error": self._init_error,
                "db_size_kb": db_size_kb,
                "db_total_rows": total_db_rows,
            },
        }

    def get_recent(
        self,
        *,
        limit: int = 20,
        only_errors: bool = False,
        low_confidence_only: bool = False,
    ) -> list[dict[str, Any]]:
        """Return recent prediction telemetry events for debugging & auditing."""
        if not self.db_ready:
            return []

        clauses: list[str] = []
        params: list[Any] = []
        if only_errors:
            clauses.append("status_code >= 400")
        if low_confidence_only:
            clauses.append(
                "(status_code = 200 AND wound_detected = 1 AND "
                "(stage1_mean_prob < ? OR stage2_confidence < ?))"
            )
            params.extend(
                [LOW_STAGE1_CONFIDENCE_THRESHOLD, LOW_STAGE2_CONFIDENCE_THRESHOLD]
            )

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"""
            SELECT
                request_id, timestamp, model_version, status_code, error_type,
                file_size_bytes, image_width, image_height, include_overlay,
                decode_ms, stage1_ms, stage2_ms, inference_ms, overlay_ms, total_ms,
                wound_detected, wound_coverage, stage1_max_prob, stage1_mean_prob,
                stage2_confidence, stage2_entropy,
                epithelial_pct, slough_pct, granulation_pct, necrotic_pct, other_pct,
                rss_mb
            FROM prediction_events
            {where_sql}
            ORDER BY id DESC
            LIMIT ?
        """
        params.append(limit)

        with self._lock:
            assert self._conn is not None
            rows = self._conn.execute(sql, params).fetchall()

        events: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            if item["include_overlay"] is not None:
                item["include_overlay"] = bool(item["include_overlay"])
            if item["wound_detected"] is not None:
                item["wound_detected"] = bool(item["wound_detected"])
            events.append(item)
        return events
