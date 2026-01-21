"""
Request Timing Middleware

Tracks request duration and adds timing headers.
"""

from __future__ import annotations
import time
import logging
from typing import Callable
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


logger = logging.getLogger("shin-gateway")


class TimingMiddleware(BaseHTTPMiddleware):
    """
    Track request timing and add response headers.

    Adds:
    - X-Response-Time: Total request duration in milliseconds
    - X-Process-Time: Server processing time
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Get start time (may be set by RequestIDMiddleware)
        start_time = getattr(request.state, "start_time", None) or time.perf_counter()

        # Process request
        response = await call_next(request)

        # Calculate duration
        duration_ms = (time.perf_counter() - start_time) * 1000

        # Add timing headers
        response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"

        # Log timing for non-trivial requests
        if duration_ms > 100:  # Log requests taking > 100ms
            logger.info(
                f"Request completed in {duration_ms:.1f}ms",
                extra={
                    "path": request.url.path,
                    "method": request.method,
                    "duration_ms": round(duration_ms, 2),
                    "status_code": response.status_code,
                }
            )

        return response
