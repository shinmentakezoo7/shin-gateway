"""
Structured Logger with Request Context

Provides JSON-formatted logging with request-scoped context for debugging
and observability in agentic workflows.
"""

from __future__ import annotations
import logging
import sys
import json
from datetime import datetime, timezone
from contextvars import ContextVar
from typing import Any, Optional
from dataclasses import dataclass, field, asdict


# =============================================================================
# Context Variables
# =============================================================================

@dataclass
class RequestContext:
    """Request-scoped context data"""
    request_id: Optional[str] = None
    model: Optional[str] = None
    provider: Optional[str] = None
    start_time: Optional[float] = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for logging"""
        result = {}
        if self.request_id:
            result["request_id"] = self.request_id
        if self.model:
            result["model"] = self.model
        if self.provider:
            result["provider"] = self.provider
        result.update(self.extra)
        return result


# Context variable for request-scoped data
request_context: ContextVar[RequestContext] = ContextVar(
    "request_context",
    default=RequestContext()
)


def get_context() -> RequestContext:
    """Get current request context"""
    return request_context.get()


def set_context(ctx: RequestContext) -> Any:
    """Set request context, returns token for reset"""
    return request_context.set(ctx)


def update_context(**kwargs) -> None:
    """Update current context with additional data"""
    ctx = get_context()
    for key, value in kwargs.items():
        if hasattr(ctx, key):
            setattr(ctx, key, value)
        else:
            ctx.extra[key] = value


# =============================================================================
# JSON Formatter
# =============================================================================

class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging"""

    def __init__(self, include_extra: bool = True):
        super().__init__()
        self.include_extra = include_extra

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add request context
        ctx = request_context.get()
        ctx_dict = ctx.to_dict()
        if ctx_dict:
            log_data.update(ctx_dict)

        # Add extra fields from record
        if self.include_extra and hasattr(record, "extra_data"):
            log_data.update(record.extra_data)

        # Add location info for errors
        if record.levelno >= logging.ERROR:
            log_data["location"] = {
                "file": record.pathname,
                "line": record.lineno,
                "function": record.funcName,
            }

        # Add exception info
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info),
            }

        try:
            return json.dumps(log_data, default=str, ensure_ascii=False)
        except Exception:
            # Fallback to basic format
            return json.dumps({
                "timestamp": log_data["timestamp"],
                "level": log_data["level"],
                "message": str(log_data.get("message", ""))
            })


class PrettyFormatter(logging.Formatter):
    """Human-readable formatter for development"""

    COLORS = {
        "DEBUG": "\033[36m",    # Cyan
        "INFO": "\033[32m",     # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",    # Red
        "CRITICAL": "\033[35m", # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Format log record for human readability"""
        color = self.COLORS.get(record.levelname, "")
        reset = self.RESET

        # Build context string
        ctx = request_context.get()
        ctx_parts = []
        if ctx.request_id:
            ctx_parts.append(f"req={ctx.request_id[:12]}")
        if ctx.model:
            ctx_parts.append(f"model={ctx.model}")
        if ctx.provider:
            ctx_parts.append(f"provider={ctx.provider}")
        ctx_str = " ".join(ctx_parts)
        if ctx_str:
            ctx_str = f"[{ctx_str}] "

        # Format message
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        level = record.levelname.ljust(8)

        msg = f"{timestamp} {color}{level}{reset} {ctx_str}{record.getMessage()}"

        # Add extra data
        if hasattr(record, "extra_data") and record.extra_data:
            extra_str = " ".join(f"{k}={v}" for k, v in record.extra_data.items())
            msg += f" | {extra_str}"

        # Add exception
        if record.exc_info:
            msg += f"\n{self.formatException(record.exc_info)}"

        return msg


# =============================================================================
# Logger Setup
# =============================================================================

class ContextLogger(logging.Logger):
    """Logger that includes extra data in records"""

    def makeRecord(self, name, level, fn, lno, msg, args, exc_info,
                   func=None, extra=None, sinfo=None):
        """Create log record with extra data support"""
        record = super().makeRecord(name, level, fn, lno, msg, args, exc_info,
                                    func, extra, sinfo)
        if extra:
            record.extra_data = extra
        return record


def setup_logger(
    name: str = "shin-gateway",
    level: str = "INFO",
    json_format: bool = True,
    stream: Any = None
) -> logging.Logger:
    """
    Configure and return a structured logger.

    Args:
        name: Logger name
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_format: Use JSON format (True) or pretty format (False)
        stream: Output stream (default: stdout)

    Returns:
        Configured logger instance
    """
    # Set custom logger class
    logging.setLoggerClass(ContextLogger)

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    logger.handlers.clear()

    # Create handler
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setLevel(getattr(logging, level.upper()))

    # Set formatter
    if json_format:
        formatter = StructuredFormatter()
    else:
        formatter = PrettyFormatter()

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


# =============================================================================
# Logging Helpers
# =============================================================================

def log_request(
    logger: logging.Logger,
    method: str,
    path: str,
    model: Optional[str] = None,
    **extra
) -> None:
    """Log incoming request"""
    logger.info(
        f"Request: {method} {path}",
        extra={"method": method, "path": path, "model": model, **extra}
    )


def log_response(
    logger: logging.Logger,
    status_code: int,
    latency_ms: float,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    **extra
) -> None:
    """Log response with metrics"""
    logger.info(
        f"Response: {status_code} ({latency_ms:.1f}ms)",
        extra={
            "status_code": status_code,
            "latency_ms": round(latency_ms, 2),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            **extra
        }
    )


def log_upstream_call(
    logger: logging.Logger,
    provider: str,
    model: str,
    streaming: bool = False,
    **extra
) -> None:
    """Log upstream provider call"""
    logger.info(
        f"Upstream: {provider}/{model} (stream={streaming})",
        extra={"provider": provider, "upstream_model": model, "streaming": streaming, **extra}
    )


def log_error(
    logger: logging.Logger,
    error: Exception,
    message: Optional[str] = None,
    **extra
) -> None:
    """Log error with context"""
    logger.error(
        message or str(error),
        exc_info=error,
        extra={"error_type": type(error).__name__, **extra}
    )


# =============================================================================
# Default Logger
# =============================================================================

# Global logger instance
_logger: Optional[logging.Logger] = None


def get_logger() -> logging.Logger:
    """Get or create the default logger"""
    global _logger
    if _logger is None:
        _logger = setup_logger()
    return _logger


def set_logger(logger: logging.Logger) -> None:
    """Set the default logger"""
    global _logger
    _logger = logger
