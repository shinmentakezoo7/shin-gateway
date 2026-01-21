"""
Shin Gateway Admin Module

Provides management UI for providers, API keys, and usage statistics.
"""

from admin.router import router as admin_router
from admin.stats import StatsCollector, get_stats_collector

__all__ = ["admin_router", "StatsCollector", "get_stats_collector"]
