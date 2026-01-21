"""
Rate Limiter with Token Bucket Algorithm

Provides client-side rate limiting to avoid hitting provider rate limits.
Implements token bucket algorithm with per-provider configuration.
"""

from __future__ import annotations
import asyncio
import time
import logging
from dataclasses import dataclass
from typing import Dict, Optional


logger = logging.getLogger("shin-gateway")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class RateLimitConfig:
    """Rate limit configuration for a provider"""
    requests_per_minute: int = 60
    requests_per_second: float = 10.0
    burst_size: int = 5
    tokens_per_minute: Optional[int] = None  # Optional token-based limiting


# =============================================================================
# Token Bucket Implementation
# =============================================================================

class TokenBucket:
    """
    Token bucket rate limiter.

    Tokens are added at a constant rate and consumed on each request.
    Burst capacity allows temporary spikes above the average rate.
    """

    def __init__(self, rate: float, capacity: int):
        """
        Initialize token bucket.

        Args:
            rate: Tokens per second to add
            capacity: Maximum tokens (burst capacity)
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = float(capacity)
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()

    async def _refill(self) -> None:
        """Refill tokens based on elapsed time"""
        now = time.monotonic()
        elapsed = now - self.last_update
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_update = now

    async def acquire(self, tokens: int = 1) -> float:
        """
        Try to acquire tokens.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            Wait time in seconds if rate limited, 0 if tokens available
        """
        async with self._lock:
            await self._refill()

            if self.tokens >= tokens:
                self.tokens -= tokens
                return 0.0

            # Calculate wait time
            tokens_needed = tokens - self.tokens
            wait_time = tokens_needed / self.rate
            return wait_time

    async def wait_and_acquire(self, tokens: int = 1) -> None:
        """
        Wait if necessary and acquire tokens.

        Args:
            tokens: Number of tokens to acquire
        """
        wait_time = await self.acquire(tokens)
        if wait_time > 0:
            await asyncio.sleep(wait_time)
            # Try again after waiting
            async with self._lock:
                await self._refill()
                self.tokens = max(0, self.tokens - tokens)

    async def try_acquire(self, tokens: int = 1) -> bool:
        """
        Try to acquire tokens without waiting.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            True if tokens were acquired, False otherwise
        """
        wait_time = await self.acquire(tokens)
        return wait_time == 0

    @property
    def available(self) -> float:
        """Get current available tokens (approximate)"""
        return self.tokens


# =============================================================================
# Sliding Window Rate Limiter
# =============================================================================

class SlidingWindowLimiter:
    """
    Sliding window rate limiter for more accurate rate limiting.

    Tracks individual request timestamps within a window.
    """

    def __init__(self, max_requests: int, window_seconds: float):
        """
        Initialize sliding window limiter.

        Args:
            max_requests: Maximum requests in window
            window_seconds: Window size in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: list[float] = []
        self._lock = asyncio.Lock()

    async def _cleanup(self) -> None:
        """Remove expired timestamps"""
        now = time.monotonic()
        cutoff = now - self.window_seconds
        self.requests = [t for t in self.requests if t > cutoff]

    async def acquire(self) -> float:
        """
        Try to acquire a request slot.

        Returns:
            Wait time in seconds if rate limited, 0 if allowed
        """
        async with self._lock:
            await self._cleanup()

            if len(self.requests) < self.max_requests:
                self.requests.append(time.monotonic())
                return 0.0

            # Calculate wait time until oldest request expires
            oldest = min(self.requests)
            wait_time = (oldest + self.window_seconds) - time.monotonic()
            return max(0, wait_time)

    async def wait_and_acquire(self) -> None:
        """Wait if necessary and acquire a slot"""
        wait_time = await self.acquire()
        if wait_time > 0:
            await asyncio.sleep(wait_time)
            async with self._lock:
                await self._cleanup()
                self.requests.append(time.monotonic())


# =============================================================================
# Provider Rate Limiter
# =============================================================================

class ProviderRateLimiter:
    """
    Per-provider rate limiting manager.

    Manages rate limiters for multiple providers with different configurations.
    """

    def __init__(self):
        self._limiters: Dict[str, TokenBucket] = {}
        self._configs: Dict[str, RateLimitConfig] = {}
        self._lock = asyncio.Lock()

    def configure(self, provider: str, config: RateLimitConfig) -> None:
        """
        Configure rate limit for a provider.

        Args:
            provider: Provider name
            config: Rate limit configuration
        """
        self._configs[provider] = config
        self._limiters[provider] = TokenBucket(
            rate=config.requests_per_second,
            capacity=config.burst_size
        )
        logger.debug(
            f"Rate limiter configured for {provider}: "
            f"{config.requests_per_second}/s, burst={config.burst_size}"
        )

    async def acquire(self, provider: str, tokens: int = 1) -> float:
        """
        Try to acquire tokens for a provider.

        Args:
            provider: Provider name
            tokens: Number of tokens to acquire

        Returns:
            Wait time in seconds if rate limited, 0 if allowed
        """
        if provider not in self._limiters:
            return 0.0
        return await self._limiters[provider].acquire(tokens)

    async def wait_and_acquire(self, provider: str, tokens: int = 1) -> None:
        """
        Wait if necessary and acquire tokens for a provider.

        Args:
            provider: Provider name
            tokens: Number of tokens to acquire
        """
        if provider not in self._limiters:
            return
        await self._limiters[provider].wait_and_acquire(tokens)

    async def try_acquire(self, provider: str, tokens: int = 1) -> bool:
        """
        Try to acquire tokens without waiting.

        Args:
            provider: Provider name
            tokens: Number of tokens to acquire

        Returns:
            True if tokens were acquired, False otherwise
        """
        if provider not in self._limiters:
            return True
        return await self._limiters[provider].try_acquire(tokens)

    def get_available(self, provider: str) -> float:
        """Get available tokens for provider"""
        if provider not in self._limiters:
            return float("inf")
        return self._limiters[provider].available

    def get_config(self, provider: str) -> Optional[RateLimitConfig]:
        """Get configuration for provider"""
        return self._configs.get(provider)

    def remove(self, provider: str) -> None:
        """Remove rate limiter for provider"""
        self._limiters.pop(provider, None)
        self._configs.pop(provider, None)


# =============================================================================
# Rate Limit Exception
# =============================================================================

class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded"""

    def __init__(self, provider: str, retry_after: float):
        self.provider = provider
        self.retry_after = retry_after
        super().__init__(
            f"Rate limit exceeded for {provider}. Retry after {retry_after:.2f}s"
        )


# =============================================================================
# Global Instance
# =============================================================================

_rate_limiter: Optional[ProviderRateLimiter] = None


def get_rate_limiter() -> ProviderRateLimiter:
    """Get or create global rate limiter"""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = ProviderRateLimiter()
    return _rate_limiter


def set_rate_limiter(limiter: ProviderRateLimiter) -> None:
    """Set global rate limiter"""
    global _rate_limiter
    _rate_limiter = limiter
