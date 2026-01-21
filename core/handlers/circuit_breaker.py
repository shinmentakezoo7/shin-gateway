"""
Circuit Breaker Pattern Implementation

Prevents cascading failures by temporarily blocking requests to failing
providers. Implements the standard CLOSED -> OPEN -> HALF-OPEN state machine.
"""

from __future__ import annotations
import asyncio
import time
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Optional, Callable, Awaitable, TypeVar


T = TypeVar("T")
logger = logging.getLogger("shin-gateway")


# =============================================================================
# State Definitions
# =============================================================================

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Failing, reject requests
    HALF_OPEN = "half_open" # Testing if recovered


@dataclass
class CircuitStats:
    """Statistics for a single circuit"""
    failures: int = 0
    successes: int = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_failure_time: float = 0.0
    last_success_time: float = 0.0
    state: CircuitState = CircuitState.CLOSED
    half_open_calls: int = 0

    def reset(self):
        """Reset statistics"""
        self.failures = 0
        self.successes = 0
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        self.half_open_calls = 0


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5        # Consecutive failures before opening
    success_threshold: int = 2        # Consecutive successes to close from half-open
    timeout: float = 30.0             # Seconds before transitioning to half-open
    half_open_max_calls: int = 1      # Max concurrent calls in half-open state
    failure_rate_threshold: float = 0.5  # Alternative: failure rate threshold
    minimum_calls: int = 10           # Minimum calls before rate-based evaluation


# =============================================================================
# Exceptions
# =============================================================================

class CircuitOpenError(Exception):
    """Raised when circuit is open and rejecting requests"""

    def __init__(self, provider: str, retry_after: float):
        self.provider = provider
        self.retry_after = retry_after
        super().__init__(
            f"Circuit breaker open for provider '{provider}'. "
            f"Retry after {retry_after:.1f}s"
        )


# =============================================================================
# Circuit Breaker Implementation
# =============================================================================

class CircuitBreaker:
    """
    Circuit breaker for provider resilience.

    Tracks failures per provider and temporarily blocks requests when
    a provider is experiencing issues.
    """

    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        self.config = config or CircuitBreakerConfig()
        self._circuits: Dict[str, CircuitStats] = {}
        self._lock = asyncio.Lock()

    def _get_circuit(self, provider: str) -> CircuitStats:
        """Get or create circuit for provider"""
        if provider not in self._circuits:
            self._circuits[provider] = CircuitStats()
        return self._circuits[provider]

    async def can_execute(self, provider: str) -> bool:
        """
        Check if a request can proceed for the given provider.

        Args:
            provider: Provider name

        Returns:
            True if request can proceed, False otherwise
        """
        async with self._lock:
            circuit = self._get_circuit(provider)
            now = time.time()

            if circuit.state == CircuitState.CLOSED:
                return True

            if circuit.state == CircuitState.OPEN:
                # Check if timeout has expired
                time_since_failure = now - circuit.last_failure_time
                if time_since_failure >= self.config.timeout:
                    # Transition to half-open
                    circuit.state = CircuitState.HALF_OPEN
                    circuit.half_open_calls = 0
                    circuit.consecutive_successes = 0
                    logger.info(
                        f"Circuit half-open for {provider}",
                        extra={"provider": provider, "state": "half_open"}
                    )
                    return True
                return False

            # HALF_OPEN state
            if circuit.half_open_calls < self.config.half_open_max_calls:
                circuit.half_open_calls += 1
                return True
            return False

    async def record_success(self, provider: str) -> None:
        """
        Record a successful request.

        Args:
            provider: Provider name
        """
        async with self._lock:
            circuit = self._get_circuit(provider)
            circuit.successes += 1
            circuit.consecutive_successes += 1
            circuit.consecutive_failures = 0
            circuit.last_success_time = time.time()

            if circuit.state == CircuitState.HALF_OPEN:
                if circuit.consecutive_successes >= self.config.success_threshold:
                    # Close the circuit
                    circuit.state = CircuitState.CLOSED
                    circuit.reset()
                    logger.info(
                        f"Circuit closed for {provider}",
                        extra={"provider": provider, "state": "closed"}
                    )
            elif circuit.state == CircuitState.CLOSED:
                # Reset failure count on success
                circuit.consecutive_failures = 0

    async def record_failure(self, provider: str, error: Optional[Exception] = None) -> None:
        """
        Record a failed request.

        Args:
            provider: Provider name
            error: Optional exception that caused the failure
        """
        async with self._lock:
            circuit = self._get_circuit(provider)
            circuit.failures += 1
            circuit.consecutive_failures += 1
            circuit.consecutive_successes = 0
            circuit.last_failure_time = time.time()

            if circuit.state == CircuitState.HALF_OPEN:
                # Immediately reopen on failure in half-open
                circuit.state = CircuitState.OPEN
                logger.warning(
                    f"Circuit reopened for {provider}",
                    extra={
                        "provider": provider,
                        "state": "open",
                        "error": str(error) if error else None
                    }
                )
            elif circuit.state == CircuitState.CLOSED:
                if circuit.consecutive_failures >= self.config.failure_threshold:
                    # Open the circuit
                    circuit.state = CircuitState.OPEN
                    logger.warning(
                        f"Circuit opened for {provider} after {circuit.consecutive_failures} failures",
                        extra={
                            "provider": provider,
                            "state": "open",
                            "failures": circuit.consecutive_failures
                        }
                    )

    def get_state(self, provider: str) -> CircuitState:
        """Get current state for provider"""
        return self._get_circuit(provider).state

    def get_stats(self, provider: str) -> dict:
        """Get statistics for provider"""
        circuit = self._get_circuit(provider)
        return {
            "state": circuit.state.value,
            "failures": circuit.failures,
            "successes": circuit.successes,
            "consecutive_failures": circuit.consecutive_failures,
            "consecutive_successes": circuit.consecutive_successes,
        }

    async def reset(self, provider: str) -> None:
        """Reset circuit for provider"""
        async with self._lock:
            if provider in self._circuits:
                self._circuits[provider] = CircuitStats()
                logger.info(f"Circuit reset for {provider}")

    async def reset_all(self) -> None:
        """Reset all circuits"""
        async with self._lock:
            self._circuits.clear()
            logger.info("All circuits reset")


# =============================================================================
# Context Manager
# =============================================================================

class CircuitBreakerContext:
    """
    Context manager for circuit breaker usage.

    Example:
        async with circuit_breaker.call(provider) as cb:
            try:
                result = await some_operation()
                await cb.success()
            except Exception as e:
                await cb.failure(e)
                raise
    """

    def __init__(self, breaker: CircuitBreaker, provider: str):
        self._breaker = breaker
        self._provider = provider
        self._completed = False

    async def __aenter__(self) -> "CircuitBreakerContext":
        can_execute = await self._breaker.can_execute(self._provider)
        if not can_execute:
            circuit = self._breaker._get_circuit(self._provider)
            retry_after = self._breaker.config.timeout - (
                time.time() - circuit.last_failure_time
            )
            raise CircuitOpenError(self._provider, max(0, retry_after))
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        if not self._completed:
            if exc_type is not None:
                await self._breaker.record_failure(self._provider, exc_val)
            else:
                await self._breaker.record_success(self._provider)
        return False

    async def success(self) -> None:
        """Manually record success"""
        await self._breaker.record_success(self._provider)
        self._completed = True

    async def failure(self, error: Optional[Exception] = None) -> None:
        """Manually record failure"""
        await self._breaker.record_failure(self._provider, error)
        self._completed = True


# =============================================================================
# Decorator
# =============================================================================

def with_circuit_breaker(
    breaker: CircuitBreaker,
    provider_getter: Callable[..., str]
):
    """
    Decorator for adding circuit breaker to async functions.

    Args:
        breaker: Circuit breaker instance
        provider_getter: Function to extract provider name from args/kwargs
    """
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        async def wrapper(*args, **kwargs) -> T:
            provider = provider_getter(*args, **kwargs)
            async with CircuitBreakerContext(breaker, provider):
                return await func(*args, **kwargs)
        return wrapper
    return decorator


# =============================================================================
# Global Instance
# =============================================================================

_circuit_breaker: Optional[CircuitBreaker] = None


def get_circuit_breaker() -> CircuitBreaker:
    """Get or create global circuit breaker"""
    global _circuit_breaker
    if _circuit_breaker is None:
        _circuit_breaker = CircuitBreaker()
    return _circuit_breaker


def set_circuit_breaker(breaker: CircuitBreaker) -> None:
    """Set global circuit breaker"""
    global _circuit_breaker
    _circuit_breaker = breaker
