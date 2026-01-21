"""
Global Exception Handler Middleware

Catches all exceptions and returns Anthropic-style error responses.
"""

from __future__ import annotations
import logging
from typing import Callable
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from core.handlers.circuit_breaker import CircuitOpenError


logger = logging.getLogger("shin-gateway")


class ExceptionHandlerMiddleware(BaseHTTPMiddleware):
    """
    Catch all exceptions and return Anthropic-style errors.

    Ensures consistent error response format across all endpoints.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> JSONResponse:
        try:
            return await call_next(request)

        except CircuitOpenError as e:
            # Circuit breaker open - return service unavailable
            return self._error_response(
                status_code=503,
                error_type="overloaded_error",
                message=f"Service temporarily unavailable: {e.provider}",
                request=request,
            )

        except Exception as e:
            # Get request context
            request_id = getattr(request.state, "request_id", "unknown")
            provider = getattr(request.state, "provider", "unknown")

            # Log the error
            logger.exception(
                f"Unhandled exception: {e}",
                extra={
                    "request_id": request_id,
                    "path": request.url.path,
                    "method": request.method,
                    "error_type": type(e).__name__,
                    "provider": provider,
                }
            )

            # Return Anthropic-style error
            return self._error_response(
                status_code=500,
                error_type="api_error",
                message="Internal server error",
                request=request,
            )

    def _error_response(
        self,
        status_code: int,
        error_type: str,
        message: str,
        request: Request,
    ) -> JSONResponse:
        """Create Anthropic-style error response"""
        request_id = getattr(request.state, "request_id", None)

        response = JSONResponse(
            status_code=status_code,
            content={
                "type": "error",
                "error": {
                    "type": error_type,
                    "message": message,
                }
            }
        )

        if request_id:
            response.headers["X-Request-ID"] = request_id

        return response
