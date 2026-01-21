"""
Real-time Statistics Collector

Tracks request/response metrics for live monitoring with minimal overhead.
"""

from __future__ import annotations
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Deque
import asyncio


# =============================================================================
# Stats Data Structures
# =============================================================================

@dataclass
class RequestMetric:
    """Single request metric"""
    timestamp: float
    provider_id: str
    model_alias: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    success: bool
    error_type: Optional[str] = None


@dataclass
class ProviderStats:
    """Real-time stats for a provider"""
    provider_id: str
    requests_total: int = 0
    requests_success: int = 0
    requests_error: int = 0
    tokens_input: int = 0
    tokens_output: int = 0
    latency_sum_ms: float = 0
    latency_count: int = 0

    # Rolling window data (last N requests)
    recent_latencies: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    recent_errors: Deque[tuple[float, str]] = field(default_factory=lambda: deque(maxlen=50))

    # Rate tracking (requests per second)
    request_timestamps: Deque[float] = field(default_factory=lambda: deque(maxlen=1000))

    def record_request(self, metric: RequestMetric):
        """Record a request metric"""
        self.requests_total += 1
        self.tokens_input += metric.input_tokens
        self.tokens_output += metric.output_tokens
        self.latency_sum_ms += metric.latency_ms
        self.latency_count += 1
        self.recent_latencies.append(metric.latency_ms)
        self.request_timestamps.append(metric.timestamp)

        if metric.success:
            self.requests_success += 1
        else:
            self.requests_error += 1
            if metric.error_type:
                self.recent_errors.append((metric.timestamp, metric.error_type))

    @property
    def avg_latency_ms(self) -> float:
        if self.latency_count == 0:
            return 0
        return self.latency_sum_ms / self.latency_count

    @property
    def p50_latency_ms(self) -> float:
        if not self.recent_latencies:
            return 0
        sorted_latencies = sorted(self.recent_latencies)
        idx = len(sorted_latencies) // 2
        return sorted_latencies[idx]

    @property
    def p95_latency_ms(self) -> float:
        if not self.recent_latencies:
            return 0
        sorted_latencies = sorted(self.recent_latencies)
        idx = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]

    @property
    def p99_latency_ms(self) -> float:
        if not self.recent_latencies:
            return 0
        sorted_latencies = sorted(self.recent_latencies)
        idx = int(len(sorted_latencies) * 0.99)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]

    @property
    def success_rate(self) -> float:
        if self.requests_total == 0:
            return 100.0
        return (self.requests_success / self.requests_total) * 100

    def get_rps(self, window_seconds: float = 60) -> float:
        """Get requests per second over window"""
        now = time.time()
        cutoff = now - window_seconds
        count = sum(1 for ts in self.request_timestamps if ts >= cutoff)
        return count / window_seconds

    def get_tps(self, window_seconds: float = 60) -> float:
        """Estimate tokens per second (rough)"""
        rps = self.get_rps(window_seconds)
        if self.requests_total == 0:
            return 0
        avg_tokens = (self.tokens_input + self.tokens_output) / self.requests_total
        return rps * avg_tokens

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "requests": {
                "total": self.requests_total,
                "success": self.requests_success,
                "error": self.requests_error,
                "success_rate": round(self.success_rate, 2)
            },
            "tokens": {
                "input": self.tokens_input,
                "output": self.tokens_output,
                "total": self.tokens_input + self.tokens_output
            },
            "latency_ms": {
                "avg": round(self.avg_latency_ms, 2),
                "p50": round(self.p50_latency_ms, 2),
                "p95": round(self.p95_latency_ms, 2),
                "p99": round(self.p99_latency_ms, 2)
            },
            "rates": {
                "rps_1m": round(self.get_rps(60), 2),
                "rps_5m": round(self.get_rps(300), 2),
                "tps_1m": round(self.get_tps(60), 2)
            },
            "recent_errors": [
                {"time": datetime.fromtimestamp(ts).isoformat(), "type": err}
                for ts, err in list(self.recent_errors)[-10:]
            ]
        }


@dataclass
class ModelStats:
    """Real-time stats for a model"""
    model_alias: str
    provider_id: str
    requests_total: int = 0
    tokens_input: int = 0
    tokens_output: int = 0
    latency_sum_ms: float = 0
    latency_count: int = 0

    def record_request(self, metric: RequestMetric):
        self.requests_total += 1
        self.tokens_input += metric.input_tokens
        self.tokens_output += metric.output_tokens
        self.latency_sum_ms += metric.latency_ms
        self.latency_count += 1

    @property
    def avg_latency_ms(self) -> float:
        if self.latency_count == 0:
            return 0
        return self.latency_sum_ms / self.latency_count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_alias": self.model_alias,
            "provider_id": self.provider_id,
            "requests": self.requests_total,
            "tokens": {
                "input": self.tokens_input,
                "output": self.tokens_output,
                "total": self.tokens_input + self.tokens_output
            },
            "avg_latency_ms": round(self.avg_latency_ms, 2)
        }


# =============================================================================
# Stats Collector
# =============================================================================

class StatsCollector:
    """
    Real-time statistics collector with minimal overhead.

    Thread-safe and designed for high-throughput scenarios.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._provider_stats: Dict[str, ProviderStats] = {}
        self._model_stats: Dict[str, ModelStats] = {}
        self._global_stats = {
            "start_time": time.time(),
            "requests_total": 0,
            "tokens_total": 0,
        }

        # Time series data (for charts)
        self._timeseries: Deque[Dict[str, Any]] = deque(maxlen=1440)  # 24h at 1min intervals
        self._last_timeseries_update = 0

    def record(
        self,
        provider_id: str,
        model_alias: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        success: bool,
        error_type: Optional[str] = None
    ):
        """Record a request metric (thread-safe)"""
        metric = RequestMetric(
            timestamp=time.time(),
            provider_id=provider_id,
            model_alias=model_alias,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            success=success,
            error_type=error_type
        )

        with self._lock:
            # Provider stats
            if provider_id not in self._provider_stats:
                self._provider_stats[provider_id] = ProviderStats(provider_id=provider_id)
            self._provider_stats[provider_id].record_request(metric)

            # Model stats
            if model_alias not in self._model_stats:
                self._model_stats[model_alias] = ModelStats(
                    model_alias=model_alias,
                    provider_id=provider_id
                )
            self._model_stats[model_alias].record_request(metric)

            # Global stats
            self._global_stats["requests_total"] += 1
            self._global_stats["tokens_total"] += input_tokens + output_tokens

            # Update timeseries (every minute)
            now = time.time()
            if now - self._last_timeseries_update >= 60:
                self._update_timeseries()
                self._last_timeseries_update = now

    def _update_timeseries(self):
        """Update timeseries data point"""
        now = datetime.utcnow()
        data_point = {
            "time": now.strftime("%Y-%m-%d %H:%M"),
            "requests": self._global_stats["requests_total"],
            "tokens": self._global_stats["tokens_total"],
            "providers": {}
        }

        for provider_id, stats in self._provider_stats.items():
            data_point["providers"][provider_id] = {
                "requests": stats.requests_total,
                "rps": round(stats.get_rps(60), 2),
                "errors": stats.requests_error
            }

        self._timeseries.append(data_point)

    def get_overview(self) -> Dict[str, Any]:
        """Get overall stats overview"""
        with self._lock:
            uptime = time.time() - self._global_stats["start_time"]

            total_requests = self._global_stats["requests_total"]
            total_tokens = self._global_stats["tokens_total"]

            # Calculate global RPS
            total_rps = sum(
                stats.get_rps(60) for stats in self._provider_stats.values()
            )

            # Calculate average latency
            total_latency = sum(
                stats.latency_sum_ms for stats in self._provider_stats.values()
            )
            total_count = sum(
                stats.latency_count for stats in self._provider_stats.values()
            )
            avg_latency = total_latency / max(total_count, 1)

            # Error count
            total_errors = sum(
                stats.requests_error for stats in self._provider_stats.values()
            )

            return {
                "uptime_seconds": round(uptime, 0),
                "uptime_formatted": self._format_uptime(uptime),
                "requests_total": total_requests,
                "tokens_total": total_tokens,
                "errors_total": total_errors,
                "current_rps": round(total_rps, 2),
                "avg_latency_ms": round(avg_latency, 2),
                "providers_active": len(self._provider_stats),
                "models_active": len(self._model_stats)
            }

    def get_provider_stats(self, provider_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get stats for providers"""
        with self._lock:
            if provider_id:
                stats = self._provider_stats.get(provider_id)
                return [stats.to_dict()] if stats else []
            return [stats.to_dict() for stats in self._provider_stats.values()]

    def get_model_stats(self, model_alias: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get stats for models"""
        with self._lock:
            if model_alias:
                stats = self._model_stats.get(model_alias)
                return [stats.to_dict()] if stats else []
            return [stats.to_dict() for stats in self._model_stats.values()]

    def get_timeseries(self, minutes: int = 60) -> List[Dict[str, Any]]:
        """Get timeseries data for charts"""
        with self._lock:
            return list(self._timeseries)[-minutes:]

    def get_live_metrics(self) -> Dict[str, Any]:
        """Get current live metrics for dashboard"""
        with self._lock:
            providers_data = []
            for provider_id, stats in self._provider_stats.items():
                providers_data.append({
                    "id": provider_id,
                    "rps": round(stats.get_rps(10), 2),  # Last 10 seconds
                    "tps": round(stats.get_tps(60), 0),
                    "latency_p50": round(stats.p50_latency_ms, 0),
                    "latency_p95": round(stats.p95_latency_ms, 0),
                    "error_rate": round(100 - stats.success_rate, 2),
                    "status": "healthy" if stats.success_rate >= 95 else (
                        "degraded" if stats.success_rate >= 80 else "unhealthy"
                    )
                })

            return {
                "timestamp": datetime.utcnow().isoformat(),
                "providers": providers_data
            }

    def reset(self):
        """Reset all stats"""
        with self._lock:
            self._provider_stats.clear()
            self._model_stats.clear()
            self._global_stats = {
                "start_time": time.time(),
                "requests_total": 0,
                "tokens_total": 0,
            }
            self._timeseries.clear()

    @staticmethod
    def _format_uptime(seconds: float) -> str:
        """Format uptime as human-readable string"""
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        minutes = int((seconds % 3600) // 60)

        parts = []
        if days > 0:
            parts.append(f"{days}d")
        if hours > 0:
            parts.append(f"{hours}h")
        parts.append(f"{minutes}m")

        return " ".join(parts)


# =============================================================================
# Global Access
# =============================================================================

_stats_collector: Optional[StatsCollector] = None


def get_stats_collector() -> StatsCollector:
    """Get the global stats collector"""
    global _stats_collector
    if _stats_collector is None:
        _stats_collector = StatsCollector()
    return _stats_collector


def set_stats_collector(collector: StatsCollector):
    """Set the global stats collector"""
    global _stats_collector
    _stats_collector = collector
