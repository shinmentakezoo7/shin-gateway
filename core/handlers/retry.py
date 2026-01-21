"""
Retry Handler with Exponential Backoff

Provides automatic retry logic for transient failures with configurable
backoff strategies and jitter.
"""

from __future__ import annotations
import asyncio
import random
import logging
from typing import Callable, TypeVar, Set, Optional, Any, Awaitable
from functools import wraps
from dataclasses import dataclass, field
import httpx


T = TypeVar("T")
logger = logging.getLogger("shin-gateway")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class RetryConfig:
    """Retry configuration"""
    max_retries: int = 3
    base_delay: float = 0.5
    max_delay: float = 10.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_status_codes: Set[int] = field(
        default_factory=lambda: {429, 500, 502, 503, 504}
    )
    retryable_exceptions: tuple = field(
        default_factory=lambda: (
            httpx.ConnectError,
            httpx.ReadTimeout,
            httpx.WriteTimeout,
            httpx.ConnectTimeout,
            httpx.PoolTimeout,
            ConnectionError,
            TimeoutError,
        )
    )


# Default configuration
DEFAULT_RETRY_CONFIG = RetryConfig()


# =============================================================================
# Retry Logic
# =============================================================================

def calculate_delay(attempt: int, config: RetryConfig) -> float:
    """
    Calculate delay with exponential backoff and optional jitter.

    Args:
        attempt: Current attempt number (0-indexed)
        config: Retry configuration

    Returns:
        Delay in seconds
    """
    # Exponential backoff
    delay = min(
        config.base_delay * (config.exponential_base ** attempt),
        config.max_delay
    )

    # Add jitter to prevent thundering herd
    if config.jitter:
        delay = delay * (0.5 + random.random())

    return delay


def is_retryable_error(error: Exception, config: RetryConfig) -> bool:
    """
    Check if error is retryable.

    Args:
        error: The exception to check
        config: Retry configuration

    Returns:
        True if the error should be retried
    """
    # Check exception types
    if isinstance(error, config.retryable_exceptions):
        return True

    # Check HTTP status codes
    if isinstance(error, httpx.HTTPStatusError):
        return error.response.status_code in config.retryable_status_codes

    return False


def get_retry_after(error: Exception) -> Optional[float]:
    """
    Extract retry-after value from error response.

    Args:
        error: The exception to check

    Returns:
        Retry-after delay in seconds, or None
    """
    if isinstance(error, httpx.HTTPStatusError):
        retry_after = error.response.headers.get("retry-after")
        if retry_after:
            try:
                return float(retry_after)
            except ValueError:
                pass
    return None


# =============================================================================
# Async Retry Function
# =============================================================================

async def retry_async(
    func: Callable[..., Awaitable[T]],
    *args,
    config: Optional[RetryConfig] = None,
    on_retry: Optional[Callable[[int, Exception, float], Awaitable[None]]] = None,
    **kwargs
) -> T:
    """
    Execute async function with retry logic.

    Args:
        func: Async function to execute
        *args: Positional arguments for func
        config: Retry configuration (uses default if None)
        on_retry: Optional callback called before each retry
        **kwargs: Keyword arguments for func

    Returns:
        Result of func

    Raises:
        The last exception if all retries fail
    """
    config = config or DEFAULT_RETRY_CONFIG
    last_error: Optional[Exception] = None

    for attempt in range(config.max_retries + 1):
        try:
            return await func(*args, **kwargs)

        except Exception as e:
            last_error = e

            # Check if we should retry
            if attempt >= config.max_retries:
                logger.warning(
                    f"All {config.max_retries + 1} attempts failed",
                    extra={"error": str(e), "error_type": type(e).__name__}
                )
                raise

            if not is_retryable_error(e, config):
                logger.debug(f"Non-retryable error: {type(e).__name__}")
                raise

            # Calculate delay
            retry_after = get_retry_after(e)
            delay = retry_after if retry_after else calculate_delay(attempt, config)

            logger.warning(
                f"Retry {attempt + 1}/{config.max_retries} after {delay:.2f}s: {type(e).__name__}",
                extra={
                    "attempt": attempt + 1,
                    "max_retries": config.max_retries,
                    "delay": delay,
                    "error": str(e)
                }
            )

            # Call retry callback if provided
            if on_retry:
                await on_retry(attempt, e, delay)

            # Wait before retry
            await asyncio.sleep(delay)

    # This should never be reached, but just in case
    if last_error:
        raise last_error
    raise RuntimeError("Retry loop completed without result or error")


# =============================================================================
# Decorator
# =============================================================================

def with_retry(
    config: Optional[RetryConfig] = None,
    on_retry: Optional[Callable[[int, Exception, float], Awaitable[None]]] = None
):
    """
    Decorator for adding retry logic to async functions.

    Args:
        config: Retry configuration
        on_retry: Optional callback called before each retry

    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            return await retry_async(
                func, *args,
                config=config,
                on_retry=on_retry,
                **kwargs
            )
        return wrapper
    return decorator


# =============================================================================
# Retry Context Manager
# =============================================================================

class RetryContext:
    """
    Context manager for manual retry control.

    Example:
        async with RetryContext(config) as retry:
            while retry.should_continue:
                try:
                    result = await some_operation()
                    break
                except Exception as e:
                    await retry.handle_error(e)
    """

    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or DEFAULT_RETRY_CONFIG
        self.attempt = 0
        self.last_error: Optional[Exception] = None
        self._should_continue = True

    @property
    def should_continue(self) -> bool:
        """Check if we should continue retrying"""
        return self._should_continue

    async def handle_error(self, error: Exception) -> None:
        """
        Handle an error and prepare for retry if appropriate.

        Args:
            error: The exception that occurred

        Raises:
            The error if it's not retryable or max retries exceeded
        """
        self.last_error = error
        self.attempt += 1

        # Check if we've exceeded max retries
        if self.attempt > self.config.max_retries:
            self._should_continue = False
            raise error

        # Check if error is retryable
        if not is_retryable_error(error, self.config):
            self._should_continue = False
            raise error

        # Calculate and wait
        delay = calculate_delay(self.attempt - 1, self.config)
        logger.warning(
            f"Retry {self.attempt}/{self.config.max_retries} after {delay:.2f}s",
            extra={"attempt": self.attempt, "delay": delay}
        )
        await asyncio.sleep(delay)

    async def __aenter__(self) -> "RetryContext":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        return False  # Don't suppress exceptions
