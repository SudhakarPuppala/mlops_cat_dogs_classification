"""
monitoring_utils.py
-------------------
Thread-safe in-process request counter and latency tracker.

Mirrors what Prometheus counters/histograms expose in production,
but works with zero external dependencies (no Prometheus push-gateway needed).

Usage
-----
    from src.monitoring_utils import monitor          # shared singleton

    with monitor.request():
        prediction = model.predict(image)

    # In a /metrics endpoint:
    return monitor.prometheus_text()

    # To persist a batch-performance report:
    monitor.save_batch_report(true_labels, pred_labels, confidences, path)
"""

import time
import threading
import statistics
import json
import os
import datetime
from collections import deque
from typing import List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
)


# ---------------------------------------------------------------------------
class ModelMonitor:
    """
    Thread-safe in-process monitor for request count and latency.

    Attributes
    ----------
    total_requests  : int   – total prediction requests seen
    success_count   : int   – successful predictions
    error_count     : int   – failed predictions (exceptions)
    """

    def __init__(self, window: int = 1_000):
        """
        Parameters
        ----------
        window : int
            Rolling window size for latency percentile calculations.
        """
        self._lock   = threading.Lock()
        self._window = window
        self.reset()

    # ── reset ─────────────────────────────────────────────────────────────
    def reset(self):
        """Reset all counters (useful between test runs)."""
        with self._lock:
            self.total_requests = 0
            self.success_count  = 0
            self.error_count    = 0
            self._latencies_ms  = deque(maxlen=self._window)
            self._started_at    = datetime.datetime.utcnow().isoformat()

    # ── context manager for a single request ──────────────────────────────
    class _RequestTimer:
        def __init__(self, monitor, label: str = ""):
            self._monitor = monitor
            self._label   = label
            self._t0: Optional[float] = None

        def __enter__(self):
            self._t0 = time.perf_counter()
            return self

        def __exit__(self, exc_type, *_):
            elapsed_ms = (time.perf_counter() - self._t0) * 1_000
            self._monitor._record(elapsed_ms, success=(exc_type is None))

    def request(self, label: str = "") -> "_RequestTimer":
        """Context-manager that times and records one prediction request.

        Example::

            with monitor.request():
                pred = model.predict(image)
        """
        return self._RequestTimer(self, label)

    # ── internal ──────────────────────────────────────────────────────────
    def _record(self, latency_ms: float, success: bool):
        with self._lock:
            self.total_requests += 1
            self._latencies_ms.append(latency_ms)
            if success:
                self.success_count += 1
            else:
                self.error_count += 1

    # ── latency statistics ────────────────────────────────────────────────
    @property
    def latency_stats(self) -> dict:
        """Return dict with mean/median/p95/p99/min/max in milliseconds."""
        with self._lock:
            lats = list(self._latencies_ms)
        if not lats:
            return {k: None for k in ("mean", "median", "p95", "p99", "min", "max")}
        s = sorted(lats)

        def pct(p: float) -> float:
            idx = max(0, int(len(s) * p / 100) - 1)
            return round(s[idx], 3)

        return {
            "mean":   round(statistics.mean(lats), 3),
            "median": round(statistics.median(lats), 3),
            "p95":    pct(95),
            "p99":    pct(99),
            "min":    round(s[0], 3),
            "max":    round(s[-1], 3),
        }

    # ── Prometheus text export ────────────────────────────────────────────
    def prometheus_text(self) -> str:
        """Return a Prometheus-compatible text exposition string."""
        stats = self.latency_stats
        lines = [
            "# HELP model_requests_total Total prediction requests received",
            "# TYPE model_requests_total counter",
            f"model_requests_total {self.total_requests}",
            "# HELP model_request_errors_total Failed prediction requests",
            "# TYPE model_request_errors_total counter",
            f"model_request_errors_total {self.error_count}",
            "# HELP model_request_duration_ms Request latency in milliseconds",
            "# TYPE model_request_duration_ms summary",
        ]
        for quantile, key in [("0.5", "median"), ("0.95", "p95"), ("0.99", "p99")]:
            val = stats[key] if stats[key] is not None else "NaN"
            lines.append(
                f'model_request_duration_ms{{quantile="{quantile}"}} {val}'
            )
        return "\n".join(lines)

    # ── console report ────────────────────────────────────────────────────
    def report(self):
        """Print a human-readable summary to stdout."""
        stats = self.latency_stats
        print(f"  Monitoring started : {self._started_at}")
        print(f"  Total requests     : {self.total_requests}")
        print(f"  Successes          : {self.success_count}")
        print(f"  Errors             : {self.error_count}")
        err_rate = self.error_count / max(self.total_requests, 1) * 100
        print(f"  Error rate         : {err_rate:.1f}%")
        print("  Latency (ms):")
        for k, v in stats.items():
            print(f"    {k:<8}: {v}")

    # ── post-deployment batch evaluation ──────────────────────────────────
    def save_batch_report(
        self,
        true_labels: List[int],
        pred_labels: List[int],
        confidences: Optional[List[float]] = None,
        path: str = "monitoring/batch_performance_report.json",
        accuracy_threshold: float = 0.70,
    ) -> dict:
        """
        Compute classification metrics on a batch of post-deployment predictions
        and persist the report as JSON.

        Parameters
        ----------
        true_labels          : ground-truth class indices (0/1)
        pred_labels          : model-predicted class indices (0/1)
        confidences          : optional list of max-softmax confidence scores
        path                 : output JSON path
        accuracy_threshold   : alert threshold for accuracy drift detection

        Returns
        -------
        dict with all computed metrics
        """
        acc  = accuracy_score(true_labels, pred_labels)
        prec = precision_score(true_labels, pred_labels, average="binary", zero_division=0)
        rec  = recall_score(true_labels, pred_labels,    average="binary", zero_division=0)
        f1   = f1_score(true_labels, pred_labels,        average="binary", zero_division=0)
        cm   = confusion_matrix(true_labels, pred_labels).tolist()

        report = {
            "timestamp":          datetime.datetime.utcnow().isoformat() + "Z",
            "batch_size":         len(true_labels),
            "accuracy":           round(acc,  4),
            "precision":          round(prec, 4),
            "recall":             round(rec,  4),
            "f1_score":           round(f1,   4),
            "mean_confidence":    round(float(np.mean(confidences)), 4) if confidences else None,
            "confusion_matrix":   cm,
            "accuracy_threshold": accuracy_threshold,
            "drift_alert":        bool(acc < accuracy_threshold),
            "request_metrics": {
                "total":      self.total_requests,
                "errors":     self.error_count,
                "error_pct":  round(self.error_count / max(self.total_requests, 1) * 100, 2),
                "latency_ms": self.latency_stats,
            },
        }

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fp:
            json.dump(report, fp, indent=2)

        return report


# ---------------------------------------------------------------------------
# Shared singleton — import this in inference_api.py and notebooks alike
# ---------------------------------------------------------------------------
monitor = ModelMonitor(window=1_000)
