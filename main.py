"""
Shin Gateway - Main Application

High-performance Anthropic-to-OpenAI protocol translation proxy
for agentic IDE tools like Claude Code, Roo Code, and Cline.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import load_settings, load_env_settings, Settings
from core.handlers.logger import setup_logger, set_logger
from core.handlers.circuit_breaker import CircuitBreaker, set_circuit_breaker, CircuitBreakerConfig
from core.handlers.rate_limiter import ProviderRateLimiter, set_rate_limiter, RateLimitConfig
from core.middleware.request_id import RequestIDMiddleware
from core.middleware.timing import TimingMiddleware
from core.middleware.error_handler import ExceptionHandlerMiddleware
from admin.models import get_admin_db, set_admin_db, AdminDatabase
from admin.stats import get_stats_collector, set_stats_collector, StatsCollector


# =============================================================================
# Application Lifespan
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    Manages startup and shutdown of:
    - HTTP client with connection pooling
    - Circuit breaker
    - Rate limiter
    - Logger configuration
    """
    # --- STARTUP ---

    # Load environment settings
    env = load_env_settings()

    # Configure logger
    json_format = not env.debug
    logger = setup_logger(
        name="shin-gateway",
        level=env.log_level,
        json_format=json_format
    )
    set_logger(logger)
    logger.info("Starting Shin Gateway...")

    # Load configuration
    settings = load_settings(env.config_path)
    logger.info(f"Loaded {len(settings.models)} model aliases")
    logger.info(f"Configured {len(settings.providers)} providers")

    # Initialize HTTP client with connection pooling
    http_client = httpx.AsyncClient(
        limits=httpx.Limits(
            max_keepalive_connections=20,
            max_connections=100,
            keepalive_expiry=30.0
        ),
        timeout=httpx.Timeout(
            connect=10.0,
            read=settings.gateway.request_timeout,
            write=30.0,
            pool=10.0
        ),
        http2=True,
        follow_redirects=True,
    )
    app.state.http_client = http_client
    app.state.settings = settings

    # Initialize circuit breaker
    circuit_breaker = CircuitBreaker(
        CircuitBreakerConfig(
            failure_threshold=settings.circuit_breaker.failure_threshold,
            success_threshold=settings.circuit_breaker.success_threshold,
            timeout=settings.circuit_breaker.timeout,
        )
    )
    set_circuit_breaker(circuit_breaker)
    app.state.circuit_breaker = circuit_breaker

    # Initialize rate limiter
    rate_limiter = ProviderRateLimiter()
    for name, provider in settings.providers.items():
        if provider.rate_limit:
            rate_limiter.configure(name, provider.rate_limit)
            logger.debug(f"Rate limit configured for {name}")
    set_rate_limiter(rate_limiter)
    app.state.rate_limiter = rate_limiter

    # Initialize admin database
    admin_db = AdminDatabase.get_instance()
    set_admin_db(admin_db)
    app.state.admin_db = admin_db

    # Initialize stats collector
    stats_collector = StatsCollector()
    set_stats_collector(stats_collector)
    app.state.stats_collector = stats_collector

    logger.info(
        f"Shin Gateway ready on {settings.gateway.host}:{settings.gateway.port}",
        extra={"models": list(settings.models.keys())}
    )

    yield  # Application runs here

    # --- SHUTDOWN ---
    logger.info("Shutting down Shin Gateway...")
    await http_client.aclose()
    logger.info("Shutdown complete")


# =============================================================================
# Application Factory
# =============================================================================

def create_app() -> FastAPI:
    """Create and configure FastAPI application"""

    app = FastAPI(
        title="Shin Gateway",
        description="Anthropic-to-OpenAI protocol translation proxy for agentic tools",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add custom middleware (order matters: last added = first executed)
    app.add_middleware(ExceptionHandlerMiddleware)
    app.add_middleware(TimingMiddleware)
    app.add_middleware(RequestIDMiddleware)

    # Health endpoints
    @app.get("/health")
    async def health():
        """Liveness probe"""
        return {"status": "ok"}

    @app.get("/ready")
    async def ready():
        """Readiness probe"""
        checks = {
            "http_client": hasattr(app.state, "http_client") and app.state.http_client is not None,
            "settings": hasattr(app.state, "settings") and app.state.settings is not None,
        }
        all_ready = all(checks.values())
        return {
            "status": "ready" if all_ready else "not_ready",
            "checks": checks
        }

    @app.get("/")
    async def root():
        """Root endpoint with gateway info"""
        return {
            "name": "Shin Gateway",
            "version": "1.0.0",
            "description": "Anthropic-to-OpenAI protocol translation proxy",
            "endpoints": {
                "messages": "/v1/messages",
                "models": "/v1/models",
                "health": "/health",
                "ready": "/ready",
                "admin": "/admin",
                "docs": "/docs",
            }
        }

    # Import and register routes
    from core.proxy import router as proxy_router
    from admin.router import router as admin_router

    app.include_router(proxy_router)
    app.include_router(admin_router)

    return app


# =============================================================================
# Application Instance
# =============================================================================

app = create_app()


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Run the gateway server"""
    import uvicorn

    env = load_env_settings()
    settings = load_settings(env.config_path)

    # Determine if we should use uvloop
    use_uvloop = sys.platform != "win32"

    uvicorn.run(
        "main:app",
        host=settings.gateway.host,
        port=settings.gateway.port,
        reload=env.debug,
        log_level=settings.gateway.log_level.lower(),
        access_log=env.debug,
        loop="uvloop" if use_uvloop else "asyncio",
        http="h11",
        ws="none",
    )


if __name__ == "__main__":
    main()
