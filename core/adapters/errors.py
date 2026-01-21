"""
Error Translator

Maps upstream provider errors to Anthropic-style error responses.
Ensures consistent error format across all providers.
"""

from __future__ import annotations
import logging
from typing import Dict, Any, Tuple, Optional
import httpx


logger = logging.getLogger("shin-gateway")


# =============================================================================
# Error Type Mappings
# =============================================================================

# HTTP status code to Anthropic error type mapping
STATUS_TO_ERROR_TYPE = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    409: "invalid_request_error",
    422: "invalid_request_error",
    429: "rate_limit_error",
    500: "api_error",
    502: "api_error",
    503: "overloaded_error",
    504: "api_error",
}


# =============================================================================
# Error Response Creation
# =============================================================================

def create_error_response(
    error_type: str,
    message: str
) -> Dict[str, Any]:
    """
    Create an Anthropic-style error response.

    Args:
        error_type: Anthropic error type
        message: Error message

    Returns:
        Error response dictionary
    """
    return {
        "type": "error",
        "error": {
            "type": error_type,
            "message": message
        }
    }


# =============================================================================
# Error Translation
# =============================================================================

def translate_error(
    error: Exception,
    provider: str,
    request_id: Optional[str] = None
) -> Tuple[int, Dict[str, Any]]:
    """
    Translate any error to Anthropic format.

    Args:
        error: The exception to translate
        provider: Provider name for context
        request_id: Optional request ID

    Returns:
        Tuple of (HTTP status code, error response dict)
    """
    if isinstance(error, httpx.HTTPStatusError):
        return _translate_http_error(error, provider)

    if isinstance(error, httpx.TimeoutException):
        return 504, create_error_response(
            "api_error",
            f"Request to {provider} timed out"
        )

    if isinstance(error, httpx.ConnectError):
        return 503, create_error_response(
            "api_error",
            f"Cannot connect to {provider}"
        )

    if isinstance(error, httpx.RequestError):
        return 502, create_error_response(
            "api_error",
            f"Request error from {provider}: {type(error).__name__}"
        )

    # Generic error
    logger.error(f"Unexpected error from {provider}: {error}", exc_info=error)
    return 500, create_error_response(
        "api_error",
        "Internal server error"
    )


def _translate_http_error(
    error: httpx.HTTPStatusError,
    provider: str
) -> Tuple[int, Dict[str, Any]]:
    """Translate HTTP status error"""
    status = error.response.status_code
    error_type = STATUS_TO_ERROR_TYPE.get(status, "api_error")

    # Try to extract error message from response
    message = _extract_error_message(error.response, provider, status)

    # Determine response status code
    # Some errors should be passed through, others mapped
    if status in (401, 403):
        response_status = 401
    elif status == 404:
        response_status = 404
    elif status == 429:
        response_status = 429
    elif status in (400, 422):
        response_status = 400
    elif status >= 500:
        response_status = 502
    else:
        response_status = status

    return response_status, create_error_response(error_type, message)


def _extract_error_message(
    response: httpx.Response,
    provider: str,
    status: int
) -> str:
    """Extract error message from response"""
    try:
        data = response.json()

        # OpenAI format
        if "error" in data:
            error_data = data["error"]
            if isinstance(error_data, dict):
                return error_data.get("message", str(error_data))
            return str(error_data)

        # Anthropic format
        if "message" in data:
            return data["message"]

        # Generic
        return str(data)

    except Exception:
        pass

    # Fallback messages
    if status == 401:
        return f"Invalid API key for {provider}"
    if status == 403:
        return f"Permission denied by {provider}"
    if status == 404:
        return f"Resource not found on {provider}"
    if status == 429:
        return f"Rate limited by {provider}"
    if status >= 500:
        return f"Upstream error from {provider}: {status}"

    return f"Error from {provider}: {status}"


# =============================================================================
# Validation Errors
# =============================================================================

def create_validation_error(message: str) -> Tuple[int, Dict[str, Any]]:
    """Create validation error response"""
    return 400, create_error_response("invalid_request_error", message)


def create_model_not_found_error(model: str) -> Tuple[int, Dict[str, Any]]:
    """Create model not found error"""
    return 404, create_error_response(
        "not_found_error",
        f"Model '{model}' not found. Check available models in configuration."
    )


def create_provider_error(provider: str, message: str) -> Tuple[int, Dict[str, Any]]:
    """Create provider-specific error"""
    return 502, create_error_response("api_error", f"{provider}: {message}")


def create_overloaded_error(provider: str) -> Tuple[int, Dict[str, Any]]:
    """Create overloaded error"""
    return 503, create_error_response(
        "overloaded_error",
        f"Provider {provider} is temporarily unavailable"
    )


def create_rate_limit_error(
    provider: str,
    retry_after: Optional[float] = None
) -> Tuple[int, Dict[str, Any]]:
    """Create rate limit error"""
    message = f"Rate limited by {provider}"
    if retry_after:
        message += f". Retry after {retry_after:.1f} seconds"

    return 429, create_error_response("rate_limit_error", message)


# =============================================================================
# Error Response Headers
# =============================================================================

def get_error_headers(
    status: int,
    retry_after: Optional[float] = None,
    request_id: Optional[str] = None
) -> Dict[str, str]:
    """Get headers for error response"""
    headers = {}

    if request_id:
        headers["x-request-id"] = request_id

    if status == 429 and retry_after:
        headers["retry-after"] = str(int(retry_after))

    return headers
