"""
Request ID Middleware

Injects unique request IDs for tracing and logging.
"""

from __future__ import annotations
import uuid
import time
from typing import Callable
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from core.handlers.logger import RequestContext, set_context


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Inject request ID into every request.

    - Uses X-Request-ID header if provided
    - Otherwise generates a new unique ID
    - Sets ID in response header
    - Sets logging context for request-scoped logging
    """

    def __init__(self, app, header_name: str = "X-Request-ID"):
        super().__init__(app)
        self.header_name = header_name

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Get from header or generate new
        request_id = request.headers.get(self.header_name)
        if not request_id:
            request_id = f"req_{uuid.uuid4().hex[:16]}"

        # Store in request state
        request.state.request_id = request_id
        request.state.start_time = time.perf_counter()

        # Set logging context
        ctx = RequestContext(
            request_id=request_id,
            start_time=request.state.start_time,
        )
        token = set_context(ctx)

        try:
            response = await call_next(request)

            # Add request ID to response header
            response.headers[self.header_name] = request_id

            return response

        finally:
            # Reset context
            from core.handlers.logger import request_context
            request_context.reset(token)
