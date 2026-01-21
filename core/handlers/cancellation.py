"""
Request Cancellation Handler

Provides clean cancellation of streaming requests when clients disconnect.
Critical for agentic workflows where users frequently cancel mid-generation.
"""

from __future__ import annotations
import asyncio
import logging
from typing import Optional, AsyncIterator, TypeVar, Generic, Dict
from contextlib import asynccontextmanager
import httpx


logger = logging.getLogger("shin-gateway")
T = TypeVar("T")


# =============================================================================
# Cancellation Token
# =============================================================================

class CancellationToken:
    """
    Token to signal cancellation across async boundaries.

    Thread-safe and can be shared across coroutines.
    """

    def __init__(self):
        self._cancelled = False
        self._event = asyncio.Event()
        self._callbacks: list = []

    def cancel(self) -> None:
        """Signal cancellation"""
        if not self._cancelled:
            self._cancelled = True
            self._event.set()
            # Call registered callbacks
            for callback in self._callbacks:
                try:
                    callback()
                except Exception as e:
                    logger.warning(f"Cancellation callback error: {e}")

    @property
    def is_cancelled(self) -> bool:
        """Check if cancelled"""
        return self._cancelled

    async def wait(self) -> None:
        """Wait until cancelled"""
        await self._event.wait()

    async def wait_timeout(self, timeout: float) -> bool:
        """
        Wait for cancellation with timeout.

        Returns:
            True if cancelled, False if timeout
        """
        try:
            await asyncio.wait_for(self._event.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            return False

    def on_cancel(self, callback) -> None:
        """Register callback to run on cancellation"""
        if self._cancelled:
            callback()
        else:
            self._callbacks.append(callback)

    def __bool__(self) -> bool:
        """Allow using in boolean context"""
        return not self._cancelled


# =============================================================================
# Cancellable Stream Reader
# =============================================================================

class CancellableStreamReader(Generic[T]):
    """
    Read from async stream with cancellation support.

    Wraps an async iterator and checks for cancellation between yields.
    """

    def __init__(
        self,
        stream: AsyncIterator[T],
        token: CancellationToken,
        cleanup: Optional[asyncio.coroutine] = None
    ):
        self.stream = stream
        self.token = token
        self.cleanup = cleanup
        self._exhausted = False

    async def __aiter__(self) -> AsyncIterator[T]:
        return self

    async def __anext__(self) -> T:
        if self.token.is_cancelled:
            await self._do_cleanup()
            raise StopAsyncIteration

        if self._exhausted:
            raise StopAsyncIteration

        try:
            # Create tasks for reading and cancellation
            read_task = asyncio.create_task(self.stream.__anext__())
            cancel_task = asyncio.create_task(self.token.wait())

            done, pending = await asyncio.wait(
                [read_task, cancel_task],
                return_when=asyncio.FIRST_COMPLETED
            )

            # Cancel pending tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            # Check if cancelled
            if cancel_task in done:
                await self._do_cleanup()
                raise StopAsyncIteration

            # Return result
            return read_task.result()

        except StopAsyncIteration:
            self._exhausted = True
            raise

        except Exception as e:
            await self._do_cleanup()
            raise

    async def _do_cleanup(self) -> None:
        """Run cleanup if provided"""
        if self.cleanup:
            try:
                await self.cleanup
            except Exception as e:
                logger.warning(f"Stream cleanup error: {e}")


# =============================================================================
# Cancellable HTTP Request
# =============================================================================

@asynccontextmanager
async def cancellable_stream_request(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    token: CancellationToken,
    **kwargs
) -> AsyncIterator[CancellableStreamReader[str]]:
    """
    Make a cancellable streaming HTTP request.

    Args:
        client: HTTP client
        method: HTTP method
        url: Request URL
        token: Cancellation token
        **kwargs: Additional request arguments

    Yields:
        CancellableStreamReader for response lines
    """
    response = None
    try:
        # Make streaming request
        async with client.stream(method, url, **kwargs) as response:
            # Create cancellable reader
            reader = CancellableStreamReader(
                stream=response.aiter_lines(),
                token=token,
                cleanup=None  # Response closes automatically
            )
            yield reader

    except httpx.RequestError as e:
        if token.is_cancelled:
            logger.debug("Request cancelled during connection")
        raise

    finally:
        # Ensure response is closed
        if response and not response.is_closed:
            await response.aclose()


# =============================================================================
# Request Cancellation Manager
# =============================================================================

class CancellationManager:
    """
    Manage active request cancellation tokens.

    Tracks requests by ID and provides cleanup on disconnect.
    """

    def __init__(self):
        self._tokens: Dict[str, CancellationToken] = {}

    def create_token(self, request_id: str) -> CancellationToken:
        """
        Create and register a cancellation token.

        Args:
            request_id: Unique request ID

        Returns:
            New CancellationToken
        """
        token = CancellationToken()
        self._tokens[request_id] = token
        return token

    def get_token(self, request_id: str) -> Optional[CancellationToken]:
        """Get token for request ID"""
        return self._tokens.get(request_id)

    def cancel(self, request_id: str) -> bool:
        """
        Cancel a request by ID.

        Args:
            request_id: Request ID to cancel

        Returns:
            True if request was found and cancelled
        """
        token = self._tokens.get(request_id)
        if token:
            token.cancel()
            return True
        return False

    def remove(self, request_id: str) -> None:
        """Remove token (call after request completes)"""
        self._tokens.pop(request_id, None)

    def cancel_all(self) -> int:
        """
        Cancel all active requests.

        Returns:
            Number of requests cancelled
        """
        count = 0
        for token in self._tokens.values():
            if not token.is_cancelled:
                token.cancel()
                count += 1
        return count

    @property
    def active_count(self) -> int:
        """Get count of active (non-cancelled) requests"""
        return sum(1 for t in self._tokens.values() if not t.is_cancelled)


# =============================================================================
# Client Disconnect Monitor
# =============================================================================

async def monitor_client_disconnect(
    is_disconnected,
    token: CancellationToken,
    check_interval: float = 0.1
) -> None:
    """
    Monitor for client disconnect and trigger cancellation.

    Args:
        is_disconnected: Async callable that returns True if client disconnected
        token: Cancellation token to trigger
        check_interval: How often to check (seconds)
    """
    try:
        while not token.is_cancelled:
            if await is_disconnected():
                logger.info("Client disconnected, cancelling request")
                token.cancel()
                break
            await asyncio.sleep(check_interval)
    except asyncio.CancelledError:
        pass


# =============================================================================
# Global Instance
# =============================================================================

_cancellation_manager: Optional[CancellationManager] = None


def get_cancellation_manager() -> CancellationManager:
    """Get or create global cancellation manager"""
    global _cancellation_manager
    if _cancellation_manager is None:
        _cancellation_manager = CancellationManager()
    return _cancellation_manager
