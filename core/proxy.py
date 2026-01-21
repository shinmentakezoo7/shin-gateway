"""
Proxy Router - Main Request Handler

Handles incoming Anthropic Messages API requests, translates them,
routes to providers, and returns translated responses.
"""

from __future__ import annotations
import asyncio
import json
import logging
from typing import Optional, AsyncIterator
import httpx
from fastapi import APIRouter, Request, HTTPException, Depends, Header
from fastapi.responses import JSONResponse, StreamingResponse

from config.settings import get_settings, Settings
from core.models.anthropic_types import MessagesRequest
from core.adapters.anthropic_to_openai import translate_request
from core.adapters.openai_to_anthropic import translate_response
from core.adapters.stream_translator import translate_stream, format_sse_event
from core.adapters.errors import (
    translate_error,
    create_model_not_found_error,
    create_validation_error,
)
from core.handlers.logger import get_context, update_context
from core.handlers.token_counter import get_token_counter
from core.handlers.capabilities import get_capability_detector
from core.handlers.context_manager import get_context_manager
from core.handlers.circuit_breaker import get_circuit_breaker, CircuitOpenError
from core.handlers.rate_limiter import get_rate_limiter
from core.handlers.retry import retry_async, RetryConfig
from core.handlers.cancellation import (
    CancellationToken,
    get_cancellation_manager,
    monitor_client_disconnect,
)
from core.handlers.beta_features import get_beta_handler, get_cache_handler
from core.handlers.protocol_features import (
    get_preflight_checker,
    get_stream_recovery,
    get_thinking_handler,
)
from admin.stats import get_stats_collector
from admin.models import get_admin_db


logger = logging.getLogger("shin-gateway")
router = APIRouter()


# =============================================================================
# Dependencies
# =============================================================================

async def get_http_client(request: Request) -> httpx.AsyncClient:
    """Get HTTP client from app state"""
    return request.app.state.http_client


async def verify_api_key(
    request: Request,
    x_api_key: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
) -> Optional[str]:
    """Verify API key if required"""
    settings = get_settings()

    if not settings.gateway.require_api_key:
        return None

    # Check x-api-key header
    if x_api_key and x_api_key in settings.gateway.api_keys:
        return x_api_key

    # Check Authorization header
    if authorization:
        if authorization.startswith("Bearer "):
            token = authorization[7:]
            if token in settings.gateway.api_keys:
                return token

    # No valid key found
    if settings.gateway.api_keys:
        raise HTTPException(
            status_code=401,
            detail={"type": "error", "error": {"type": "authentication_error", "message": "Invalid API key"}}
        )

    return None


# =============================================================================
# Messages Endpoint
# =============================================================================

@router.post("/v1/messages")
async def create_message(
    request: Request,
    body: MessagesRequest,
    client: httpx.AsyncClient = Depends(get_http_client),
    api_key: Optional[str] = Depends(verify_api_key),
    anthropic_beta: Optional[str] = Header(None, alias="anthropic-beta"),
    anthropic_version: Optional[str] = Header(None, alias="anthropic-version"),
):
    """
    Anthropic Messages API endpoint.

    Accepts Anthropic-format requests and routes them to configured providers.
    """
    settings = get_settings()
    request_id = getattr(request.state, "request_id", "unknown")

    # Parse beta features
    beta_handler = get_beta_handler()
    beta_features = set()
    if anthropic_beta:
        features = beta_handler.parse_header(anthropic_beta)
        supported, warnings, errors = beta_handler.validate_features(features)
        if errors:
            status, error_body = create_validation_error("; ".join(errors))
            return JSONResponse(status_code=status, content=error_body)
        beta_features = supported
        for warning in warnings:
            logger.warning(warning)

    # Resolve model alias
    model_alias = body.model
    provider_config, model_config = settings.resolve_model(model_alias)

    if not provider_config or not model_config:
        status, error_body = create_model_not_found_error(model_alias)
        return JSONResponse(status_code=status, content=error_body)

    provider_name = model_config.provider
    target_model = model_config.model

    # Update logging context
    update_context(model=model_alias, provider=provider_name)
    request.state.provider = provider_name

    logger.info(
        f"Request: {model_alias} -> {provider_name}/{target_model}",
        extra={"streaming": body.stream}
    )

    # Validate capabilities
    capability_detector = get_capability_detector()
    is_valid, error_msg = capability_detector.validate_request(
        body.model_dump(exclude_none=True),
        target_model
    )
    if not is_valid:
        status, error_body = create_validation_error(error_msg)
        return JSONResponse(status_code=status, content=error_body)

    # Validate context window (existing manager)
    context_manager = get_context_manager()
    request_dict = body.model_dump(exclude_none=True)
    is_valid, error_msg = context_manager.validate_request(request_dict, target_model)
    if not is_valid:
        status, error_body = create_validation_error(error_msg)
        return JSONResponse(status_code=status, content=error_body)

    # Token preflight check with detailed counting
    preflight_checker = get_preflight_checker()
    max_tokens = request_dict.get("max_tokens", 4096)
    is_valid, error_msg, token_counts = preflight_checker.validate_request(
        request_dict, target_model, max_tokens
    )
    if not is_valid:
        status, error_body = create_validation_error(error_msg)
        return JSONResponse(status_code=status, content=error_body)

    # Log token counts for monitoring
    logger.debug(
        f"Token preflight: {token_counts}",
        extra={"request_id": request_id}
    )

    # Check circuit breaker
    circuit_breaker = get_circuit_breaker()
    if not await circuit_breaker.can_execute(provider_name):
        raise CircuitOpenError(provider_name, 30.0)

    # Check if we should emulate thinking for non-Anthropic providers
    emulate_thinking = (
        "extended-thinking" in str(beta_features) and
        provider_config.type != "anthropic"
    )

    # Translate request
    strip_cache = "prompt-caching" not in str(provider_config.type)
    openai_request = translate_request(
        request_dict,
        target_model,
        strip_cache_control=strip_cache,
        emulate_thinking=emulate_thinking
    )

    # Apply model defaults
    if model_config.defaults:
        if model_config.defaults.temperature is not None and "temperature" not in openai_request:
            openai_request["temperature"] = model_config.defaults.temperature
        if model_config.defaults.max_tokens is not None and "max_tokens" not in openai_request:
            openai_request["max_tokens"] = model_config.defaults.max_tokens

    # Rate limiting
    rate_limiter = get_rate_limiter()
    await rate_limiter.wait_and_acquire(provider_name)

    # Determine endpoint
    if provider_config.type == "anthropic":
        # Passthrough to Anthropic
        return await _handle_anthropic_passthrough(
            request, body, provider_config, client
        )

    base_url = provider_config.base_url or "https://api.openai.com/v1"
    endpoint = f"{base_url.rstrip('/')}/chat/completions"

    # Prepare headers
    headers = {"Content-Type": "application/json"}
    provider_api_key = provider_config.get_api_key()
    if provider_api_key:
        headers["Authorization"] = f"Bearer {provider_api_key}"
    headers.update(provider_config.extra_headers)

    # Estimate input tokens for response
    token_counter = get_token_counter()
    input_tokens = token_counter.count_request(request_dict).input_tokens

    if body.stream:
        return await _handle_streaming_request(
            request=request,
            client=client,
            endpoint=endpoint,
            headers=headers,
            openai_request=openai_request,
            model_alias=model_alias,
            provider_name=provider_name,
            input_tokens=input_tokens,
            circuit_breaker=circuit_breaker,
        )
    else:
        return await _handle_non_streaming_request(
            client=client,
            endpoint=endpoint,
            headers=headers,
            openai_request=openai_request,
            model_alias=model_alias,
            provider_name=provider_name,
            provider_config=provider_config,
            circuit_breaker=circuit_breaker,
        )


# =============================================================================
# Non-Streaming Handler
# =============================================================================

async def _handle_non_streaming_request(
    client: httpx.AsyncClient,
    endpoint: str,
    headers: dict,
    openai_request: dict,
    model_alias: str,
    provider_name: str,
    provider_config,
    circuit_breaker,
    request_id: str = "unknown",
) -> JSONResponse:
    """Handle non-streaming request"""
    import time
    start_time = time.time()
    stats_collector = get_stats_collector()

    try:
        # Make request with retry
        retry_config = RetryConfig(max_retries=2, base_delay=0.5)

        async def make_request():
            response = await client.post(
                endpoint,
                headers=headers,
                json=openai_request,
                timeout=provider_config.timeout,
            )
            response.raise_for_status()
            return response.json()

        openai_response = await retry_async(make_request, config=retry_config)

        # Record success
        await circuit_breaker.record_success(provider_name)

        # Translate response
        anthropic_response = translate_response(openai_response, model_alias)

        # Calculate latency and record stats
        latency_ms = (time.time() - start_time) * 1000
        input_tokens = anthropic_response.get("usage", {}).get("input_tokens", 0)
        output_tokens = anthropic_response.get("usage", {}).get("output_tokens", 0)

        stats_collector.record(
            provider_id=provider_name,
            model_alias=model_alias,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            success=True
        )

        logger.info(
            "Request completed",
            extra={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "latency_ms": round(latency_ms, 2)
            }
        )

        # Record usage in DB
        try:
            get_admin_db().record_usage(
                provider_id=provider_name,
                model_alias=model_alias,
                request_id=request_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                status="success"
            )
        except Exception as e:
            logger.error(f"Failed to record usage: {e}")

        return JSONResponse(content=anthropic_response)

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        await circuit_breaker.record_failure(provider_name, e)

        stats_collector.record(
            provider_id=provider_name,
            model_alias=model_alias,
            input_tokens=0,
            output_tokens=0,
            latency_ms=latency_ms,
            success=False,
            error_type=type(e).__name__
        )

        # Record failure in DB
        try:
            get_admin_db().record_usage(
                provider_id=provider_name,
                model_alias=model_alias,
                request_id=request_id,
                input_tokens=0,
                output_tokens=0,
                latency_ms=latency_ms,
                status="error",
                error_type=type(e).__name__
            )
        except Exception as db_err:
            logger.error(f"Failed to record usage error: {db_err}")

        status, error_body = translate_error(e, provider_name)
        return JSONResponse(status_code=status, content=error_body)


# =============================================================================
# Streaming Handler
# =============================================================================

async def _handle_streaming_request(
    request: Request,
    client: httpx.AsyncClient,
    endpoint: str,
    headers: dict,
    openai_request: dict,
    model_alias: str,
    provider_name: str,
    input_tokens: int,
    circuit_breaker,
) -> StreamingResponse:
    """Handle streaming request"""
    import time
    start_time = time.time()
    stats_collector = get_stats_collector()

    # Create cancellation token
    cancellation_manager = get_cancellation_manager()
    request_id = getattr(request.state, "request_id", "unknown")
    cancel_token = cancellation_manager.create_token(request_id)

    # Track output tokens during streaming
    output_token_count = [0]  # Use list for mutable reference in closure
    stream_success = [True]

    async def stream_generator() -> AsyncIterator[str]:
        nonlocal output_token_count, stream_success
        try:
            # Start disconnect monitor
            disconnect_task = asyncio.create_task(
                monitor_client_disconnect(
                    request.is_disconnected,
                    cancel_token,
                    check_interval=0.5
                )
            )

            try:
                async with client.stream(
                    "POST",
                    endpoint,
                    headers=headers,
                    json=openai_request,
                    timeout=120.0,
                ) as response:
                    if response.status_code != 200:
                        # Handle error
                        error_body = await response.aread()
                        try:
                            error_data = json.loads(error_body)
                        except Exception:
                            error_data = {"message": error_body.decode()}

                        await circuit_breaker.record_failure(provider_name)
                        stream_success[0] = False
                        error_event = {
                            "type": "error",
                            "error": {
                                "type": "api_error",
                                "message": str(error_data.get("error", {}).get("message", error_data))
                            }
                        }
                        yield format_sse_event(error_event)
                        return

                    # Record success (connection established)
                    await circuit_breaker.record_success(provider_name)

                    # Create line iterator with cancellation check
                    async def line_iterator():
                        async for line in response.aiter_lines():
                            if cancel_token.is_cancelled:
                                break
                            yield line

                    # Translate and stream
                    async for event in translate_stream(
                        line_iterator(),
                        model_alias,
                        input_tokens
                    ):
                        if cancel_token.is_cancelled:
                            break
                        # Track output tokens from message_delta events
                        if event.get("type") == "message_delta":
                            usage = event.get("usage", {})
                            output_token_count[0] = usage.get("output_tokens", output_token_count[0])
                        yield format_sse_event(event)

            finally:
                disconnect_task.cancel()
                try:
                    await disconnect_task
                except asyncio.CancelledError:
                    pass

        except asyncio.CancelledError:
            logger.info("Stream cancelled by client")

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            await circuit_breaker.record_failure(provider_name, e)
            stream_success[0] = False
            error_event = {
                "type": "error",
                "error": {"type": "api_error", "message": str(e)}
            }
            yield format_sse_event(error_event)

        finally:
            cancellation_manager.remove(request_id)
            # Record stats at end of stream
            latency_ms = (time.time() - start_time) * 1000
            stats_collector.record(
                provider_id=provider_name,
                model_alias=model_alias,
                input_tokens=input_tokens,
                output_tokens=output_token_count[0],
                latency_ms=latency_ms,
                success=stream_success[0],
                error_type=None if stream_success[0] else "StreamError"
            )

            # Record usage in DB
            try:
                get_admin_db().record_usage(
                    provider_id=provider_name,
                    model_alias=model_alias,
                    request_id=request_id,
                    input_tokens=input_tokens,
                    output_tokens=output_token_count[0],
                    latency_ms=latency_ms,
                    status="success" if stream_success[0] else "error",
                    error_type=None if stream_success[0] else "StreamError"
                )
            except Exception as e:
                logger.error(f"Failed to record usage: {e}")

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


# =============================================================================
# Anthropic Passthrough
# =============================================================================

async def _handle_anthropic_passthrough(
    request: Request,
    body: MessagesRequest,
    provider_config,
    client: httpx.AsyncClient,
) -> StreamingResponse | JSONResponse:
    """Pass request through to Anthropic API directly"""
    api_key = provider_config.get_api_key()
    if not api_key:
        status, error = create_validation_error("Anthropic API key not configured")
        return JSONResponse(status_code=status, content=error)

    endpoint = "https://api.anthropic.com/v1/messages"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }

    request_body = body.model_dump(exclude_none=True)

    if body.stream:
        async def passthrough_stream():
            async with client.stream(
                "POST",
                endpoint,
                headers=headers,
                json=request_body,
                timeout=120.0,
            ) as response:
                async for line in response.aiter_lines():
                    yield line + "\n"

        return StreamingResponse(
            passthrough_stream(),
            media_type="text/event-stream",
        )
    else:
        response = await client.post(
            endpoint,
            headers=headers,
            json=request_body,
            timeout=provider_config.timeout,
        )
        return JSONResponse(
            status_code=response.status_code,
            content=response.json()
        )


# =============================================================================
# OpenAI-Compatible Chat Completions Endpoint
# =============================================================================

@router.post("/v1/chat/completions")
async def create_chat_completion(
    request: Request,
    client: httpx.AsyncClient = Depends(get_http_client),
    api_key: Optional[str] = Depends(verify_api_key),
):
    """
    OpenAI-compatible Chat Completions endpoint.

    Accepts OpenAI-format requests and routes them to configured providers.
    """
    import time
    from core.models.openai_types import ChatCompletionsRequest

    settings = get_settings()
    request_id = getattr(request.state, "request_id", "unknown")

    # Parse request body
    try:
        body_bytes = await request.body()
        body_json = json.loads(body_bytes)
        body = ChatCompletionsRequest(**body_json)
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": f"Invalid request: {str(e)}",
                    "type": "invalid_request_error",
                    "code": "invalid_request"
                }
            }
        )

    # Resolve model alias
    model_alias = body.model
    provider_config, model_config = settings.resolve_model(model_alias)

    if not provider_config or not model_config:
        return JSONResponse(
            status_code=404,
            content={
                "error": {
                    "message": f"Model '{model_alias}' not found",
                    "type": "invalid_request_error",
                    "code": "model_not_found"
                }
            }
        )

    provider_name = model_config.provider
    target_model = model_config.model

    # Update logging context
    update_context(model=model_alias, provider=provider_name)
    request.state.provider = provider_name

    logger.info(
        f"OpenAI Request: {model_alias} -> {provider_name}/{target_model}",
        extra={"streaming": body.stream}
    )

    # Check circuit breaker
    circuit_breaker = get_circuit_breaker()
    if not await circuit_breaker.can_execute(provider_name):
        raise CircuitOpenError(provider_name, 30.0)

    # Build the request for the provider
    # For OpenAI-compat providers, we can forward directly
    # For Anthropic providers, we need to translate
    if provider_config.type == "anthropic":
        return await _handle_openai_to_anthropic(
            request, body, provider_config, model_config, client
        )

    # Rate limiting
    rate_limiter = get_rate_limiter()
    await rate_limiter.wait_and_acquire(provider_name)

    # Build OpenAI request for the provider
    base_url = provider_config.base_url or "https://api.openai.com/v1"
    endpoint = f"{base_url.rstrip('/')}/chat/completions"

    # Prepare headers
    headers = {"Content-Type": "application/json"}
    provider_api_key = provider_config.get_api_key()
    if provider_api_key:
        headers["Authorization"] = f"Bearer {provider_api_key}"
    headers.update(provider_config.extra_headers)

    # Build request body - use target model name
    openai_request = body_json.copy()
    openai_request["model"] = target_model

    # Add stream_options for usage tracking
    if body.stream:
        openai_request["stream_options"] = {"include_usage": True}

    # Apply model defaults
    if model_config.defaults:
        if model_config.defaults.temperature is not None and "temperature" not in openai_request:
            openai_request["temperature"] = model_config.defaults.temperature
        if model_config.defaults.max_tokens is not None and "max_tokens" not in openai_request:
            openai_request["max_tokens"] = model_config.defaults.max_tokens

    if body.stream:
        return await _handle_openai_streaming_request(
            request=request,
            client=client,
            endpoint=endpoint,
            headers=headers,
            openai_request=openai_request,
            model_alias=model_alias,
            provider_name=provider_name,
            provider_config=provider_config,
            circuit_breaker=circuit_breaker,
        )
    else:
        return await _handle_openai_non_streaming_request(
            client=client,
            endpoint=endpoint,
            headers=headers,
            openai_request=openai_request,
            model_alias=model_alias,
            provider_name=provider_name,
            provider_config=provider_config,
            circuit_breaker=circuit_breaker,
            request_id=request_id,
        )


async def _handle_openai_non_streaming_request(
    client: httpx.AsyncClient,
    endpoint: str,
    headers: dict,
    openai_request: dict,
    model_alias: str,
    provider_name: str,
    provider_config,
    circuit_breaker,
    request_id: str = "unknown",
) -> JSONResponse:
    """Handle non-streaming OpenAI-compatible request"""
    import time
    start_time = time.time()
    stats_collector = get_stats_collector()

    try:
        # Make request with retry
        retry_config = RetryConfig(max_retries=2, base_delay=0.5)

        async def make_request():
            response = await client.post(
                endpoint,
                headers=headers,
                json=openai_request,
                timeout=provider_config.timeout,
            )
            response.raise_for_status()
            return response.json()

        provider_response = await retry_async(make_request, config=retry_config)

        # Record success
        await circuit_breaker.record_success(provider_name)

        # Update model name in response to use alias
        provider_response["model"] = model_alias

        # Calculate latency and record stats
        latency_ms = (time.time() - start_time) * 1000
        usage = provider_response.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        stats_collector.record(
            provider_id=provider_name,
            model_alias=model_alias,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            success=True
        )

        logger.info(
            "OpenAI Request completed",
            extra={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "latency_ms": round(latency_ms, 2)
            }
        )

        # Record usage in DB
        try:
            get_admin_db().record_usage(
                provider_id=provider_name,
                model_alias=model_alias,
                request_id=request_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                status="success"
            )
        except Exception as e:
            logger.error(f"Failed to record usage: {e}")

        return JSONResponse(content=provider_response)

    except httpx.HTTPStatusError as e:
        latency_ms = (time.time() - start_time) * 1000
        await circuit_breaker.record_failure(provider_name, e)

        # Try to get error details from response
        try:
            error_body = e.response.json()
        except Exception:
            error_body = {"error": {"message": str(e), "type": "api_error"}}

        stats_collector.record(
            provider_id=provider_name,
            model_alias=model_alias,
            input_tokens=0,
            output_tokens=0,
            latency_ms=latency_ms,
            success=False,
            error_type=type(e).__name__
        )

        return JSONResponse(
            status_code=e.response.status_code,
            content=error_body
        )

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        await circuit_breaker.record_failure(provider_name, e)

        stats_collector.record(
            provider_id=provider_name,
            model_alias=model_alias,
            input_tokens=0,
            output_tokens=0,
            latency_ms=latency_ms,
            success=False,
            error_type=type(e).__name__
        )

        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": str(e),
                    "type": "api_error",
                    "code": "internal_error"
                }
            }
        )


async def _handle_openai_streaming_request(
    request: Request,
    client: httpx.AsyncClient,
    endpoint: str,
    headers: dict,
    openai_request: dict,
    model_alias: str,
    provider_name: str,
    provider_config,
    circuit_breaker,
) -> StreamingResponse:
    """Handle streaming OpenAI-compatible request"""
    import time
    start_time = time.time()
    stats_collector = get_stats_collector()
    request_id = getattr(request.state, "request_id", "unknown")

    # Create cancellation token
    cancellation_manager = get_cancellation_manager()
    cancel_token = cancellation_manager.create_token(request_id)

    # Track tokens and success during streaming
    input_tokens = [0]
    output_tokens = [0]
    stream_success = [True]

    async def stream_generator() -> AsyncIterator[str]:
        nonlocal input_tokens, output_tokens, stream_success
        try:
            # Start disconnect monitor
            disconnect_task = asyncio.create_task(
                monitor_client_disconnect(
                    request.is_disconnected,
                    cancel_token,
                    check_interval=0.5
                )
            )

            try:
                async with client.stream(
                    "POST",
                    endpoint,
                    headers=headers,
                    json=openai_request,
                    timeout=provider_config.timeout,
                ) as response:
                    if response.status_code != 200:
                        # Handle error
                        error_body = await response.aread()
                        try:
                            error_data = json.loads(error_body)
                        except Exception:
                            error_data = {"error": {"message": error_body.decode(), "type": "api_error"}}

                        await circuit_breaker.record_failure(provider_name)
                        stream_success[0] = False
                        yield f"data: {json.dumps(error_data)}\n\n"
                        return

                    # Record success (connection established)
                    await circuit_breaker.record_success(provider_name)

                    async for line in response.aiter_lines():
                        if cancel_token.is_cancelled:
                            break

                        if not line:
                            continue

                        # Forward the SSE line directly, updating model name
                        if line.startswith("data: "):
                            data = line[6:]
                            if data.strip() == "[DONE]":
                                yield "data: [DONE]\n\n"
                                continue

                            try:
                                chunk = json.loads(data)
                                # Update model name to alias
                                chunk["model"] = model_alias

                                # Extract usage from final chunk
                                if chunk.get("usage"):
                                    input_tokens[0] = chunk["usage"].get("prompt_tokens", 0)
                                    output_tokens[0] = chunk["usage"].get("completion_tokens", 0)

                                yield f"data: {json.dumps(chunk)}\n\n"
                            except json.JSONDecodeError:
                                yield line + "\n"
                        else:
                            yield line + "\n"

            finally:
                disconnect_task.cancel()
                try:
                    await disconnect_task
                except asyncio.CancelledError:
                    pass

        except asyncio.CancelledError:
            logger.info("OpenAI stream cancelled by client")

        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            await circuit_breaker.record_failure(provider_name, e)
            stream_success[0] = False
            error_response = {
                "error": {
                    "message": str(e),
                    "type": "api_error",
                    "code": "stream_error"
                }
            }
            yield f"data: {json.dumps(error_response)}\n\n"

        finally:
            cancellation_manager.remove(request_id)
            # Record stats at end of stream
            latency_ms = (time.time() - start_time) * 1000
            stats_collector.record(
                provider_id=provider_name,
                model_alias=model_alias,
                input_tokens=input_tokens[0],
                output_tokens=output_tokens[0],
                latency_ms=latency_ms,
                success=stream_success[0],
                error_type=None if stream_success[0] else "StreamError"
            )

            # Record usage in DB
            try:
                get_admin_db().record_usage(
                    provider_id=provider_name,
                    model_alias=model_alias,
                    request_id=request_id,
                    input_tokens=input_tokens[0],
                    output_tokens=output_tokens[0],
                    latency_ms=latency_ms,
                    status="success" if stream_success[0] else "error",
                    error_type=None if stream_success[0] else "StreamError"
                )
            except Exception as e:
                logger.error(f"Failed to record usage: {e}")

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


async def _handle_openai_to_anthropic(
    request: Request,
    body,
    provider_config,
    model_config,
    client: httpx.AsyncClient,
) -> StreamingResponse | JSONResponse:
    """
    Handle OpenAI-format request to Anthropic provider.
    Translates OpenAI -> Anthropic for request, then Anthropic -> OpenAI for response.
    """
    import time

    # Get API key
    api_key = provider_config.get_api_key()
    if not api_key:
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": "Anthropic API key not configured",
                    "type": "server_error",
                    "code": "missing_api_key"
                }
            }
        )

    # Translate OpenAI request to Anthropic format
    anthropic_request = _translate_openai_to_anthropic_request(body, model_config.model)

    endpoint = "https://api.anthropic.com/v1/messages"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }

    if body.stream:
        return await _handle_openai_to_anthropic_streaming(
            request, client, endpoint, headers, anthropic_request, body.model
        )
    else:
        try:
            response = await client.post(
                endpoint,
                headers=headers,
                json=anthropic_request,
                timeout=provider_config.timeout,
            )
            response.raise_for_status()
            anthropic_response = response.json()

            # Translate Anthropic response to OpenAI format
            openai_response = _translate_anthropic_to_openai_response(anthropic_response, body.model)
            return JSONResponse(content=openai_response)

        except httpx.HTTPStatusError as e:
            try:
                error_body = e.response.json()
                error_msg = error_body.get("error", {}).get("message", str(e))
            except Exception:
                error_msg = str(e)

            return JSONResponse(
                status_code=e.response.status_code,
                content={
                    "error": {
                        "message": error_msg,
                        "type": "api_error",
                        "code": "anthropic_error"
                    }
                }
            )


async def _handle_openai_to_anthropic_streaming(
    request: Request,
    client: httpx.AsyncClient,
    endpoint: str,
    headers: dict,
    anthropic_request: dict,
    model_alias: str,
) -> StreamingResponse:
    """Handle streaming OpenAI request to Anthropic, translating response back to OpenAI format"""
    import time
    import uuid

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    async def stream_generator() -> AsyncIterator[str]:
        try:
            async with client.stream(
                "POST",
                endpoint,
                headers=headers,
                json=anthropic_request,
                timeout=120.0,
            ) as response:
                if response.status_code != 200:
                    error_body = await response.aread()
                    try:
                        error_data = json.loads(error_body)
                        error_msg = error_data.get("error", {}).get("message", str(error_body))
                    except Exception:
                        error_msg = error_body.decode()

                    yield f"data: {json.dumps({'error': {'message': error_msg, 'type': 'api_error'}})}\n\n"
                    return

                input_tokens = 0
                output_tokens = 0
                current_text = ""

                async for line in response.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue

                    data = line[6:]
                    if not data.strip():
                        continue

                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    event_type = event.get("type")

                    if event_type == "message_start":
                        # Send initial chunk with role
                        usage = event.get("message", {}).get("usage", {})
                        input_tokens = usage.get("input_tokens", 0)

                        chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_alias,
                            "choices": [{
                                "index": 0,
                                "delta": {"role": "assistant", "content": ""},
                                "finish_reason": None
                            }]
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"

                    elif event_type == "content_block_delta":
                        delta = event.get("delta", {})
                        if delta.get("type") == "text_delta":
                            text = delta.get("text", "")
                            current_text += text
                            chunk = {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model_alias,
                                "choices": [{
                                    "index": 0,
                                    "delta": {"content": text},
                                    "finish_reason": None
                                }]
                            }
                            yield f"data: {json.dumps(chunk)}\n\n"

                    elif event_type == "message_delta":
                        usage = event.get("usage", {})
                        output_tokens = usage.get("output_tokens", 0)
                        stop_reason = event.get("delta", {}).get("stop_reason", "stop")

                        # Map Anthropic stop_reason to OpenAI finish_reason
                        finish_reason = "stop"
                        if stop_reason == "max_tokens":
                            finish_reason = "length"
                        elif stop_reason == "tool_use":
                            finish_reason = "tool_calls"

                        chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_alias,
                            "choices": [{
                                "index": 0,
                                "delta": {},
                                "finish_reason": finish_reason
                            }],
                            "usage": {
                                "prompt_tokens": input_tokens,
                                "completion_tokens": output_tokens,
                                "total_tokens": input_tokens + output_tokens
                            }
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"

                yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Anthropic streaming error: {e}")
            yield f"data: {json.dumps({'error': {'message': str(e), 'type': 'api_error'}})}\n\n"

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


def _translate_openai_to_anthropic_request(body, target_model: str) -> dict:
    """Translate OpenAI Chat Completions request to Anthropic Messages request"""
    anthropic_request = {
        "model": target_model,
        "max_tokens": body.max_tokens or body.max_completion_tokens or 4096,
        "stream": body.stream or False,
    }

    # Extract system message and convert other messages
    messages = []
    system_content = None

    for msg in body.messages:
        if isinstance(msg, dict):
            role = msg.get("role")
            content = msg.get("content")
        else:
            role = msg.role
            content = msg.content if hasattr(msg, 'content') else None

        if role == "system":
            if isinstance(content, str):
                system_content = content
            elif isinstance(content, list):
                # Concatenate text parts
                system_content = " ".join(
                    p.get("text", "") if isinstance(p, dict) else str(p)
                    for p in content
                )
        elif role == "user":
            messages.append({
                "role": "user",
                "content": _translate_openai_content_to_anthropic(content)
            })
        elif role == "assistant":
            msg_content = []
            if content:
                if isinstance(content, str):
                    msg_content.append({"type": "text", "text": content})
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            msg_content.append({"type": "text", "text": part.get("text", "")})

            # Handle tool calls
            tool_calls = msg.get("tool_calls") if isinstance(msg, dict) else getattr(msg, 'tool_calls', None)
            if tool_calls:
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        tc_id = tc.get("id", "")
                        func = tc.get("function", {})
                        name = func.get("name", "")
                        args = func.get("arguments", "{}")
                    else:
                        tc_id = tc.id
                        name = tc.function.name
                        args = tc.function.arguments

                    try:
                        input_obj = json.loads(args) if isinstance(args, str) else args
                    except json.JSONDecodeError:
                        input_obj = {"_raw": args}

                    msg_content.append({
                        "type": "tool_use",
                        "id": tc_id,
                        "name": name,
                        "input": input_obj
                    })

            if msg_content:
                messages.append({"role": "assistant", "content": msg_content})
            else:
                messages.append({"role": "assistant", "content": ""})

        elif role == "tool":
            # Tool results in Anthropic are in user messages
            tool_call_id = msg.get("tool_call_id") if isinstance(msg, dict) else getattr(msg, 'tool_call_id', "")
            tool_content = msg.get("content") if isinstance(msg, dict) else getattr(msg, 'content', "")

            messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": str(tool_content)
                }]
            })

    if system_content:
        anthropic_request["system"] = system_content

    anthropic_request["messages"] = messages

    # Optional parameters
    if body.temperature is not None:
        # Anthropic uses 0-1 range, OpenAI uses 0-2
        anthropic_request["temperature"] = min(body.temperature, 1.0)

    if body.top_p is not None:
        anthropic_request["top_p"] = body.top_p

    if body.stop:
        if isinstance(body.stop, str):
            anthropic_request["stop_sequences"] = [body.stop]
        else:
            anthropic_request["stop_sequences"] = body.stop

    # Translate tools
    if body.tools:
        anthropic_tools = []
        for tool in body.tools:
            if isinstance(tool, dict):
                func = tool.get("function", {})
            else:
                func = tool.function

            if isinstance(func, dict):
                anthropic_tools.append({
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {"type": "object", "properties": {}})
                })
            else:
                anthropic_tools.append({
                    "name": func.name,
                    "description": func.description or "",
                    "input_schema": func.parameters or {"type": "object", "properties": {}}
                })

        anthropic_request["tools"] = anthropic_tools

    return anthropic_request


def _translate_openai_content_to_anthropic(content) -> str | list:
    """Translate OpenAI content to Anthropic format"""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        anthropic_content = []
        for part in content:
            if isinstance(part, dict):
                part_type = part.get("type")
                if part_type == "text":
                    anthropic_content.append({"type": "text", "text": part.get("text", "")})
                elif part_type == "image_url":
                    image_url = part.get("image_url", {})
                    url = image_url.get("url", "")

                    if url.startswith("data:"):
                        # Parse data URL
                        # Format: data:image/png;base64,<data>
                        try:
                            header, data = url.split(",", 1)
                            media_type = header.split(":")[1].split(";")[0]
                            anthropic_content.append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": data
                                }
                            })
                        except Exception:
                            # Fallback to URL type
                            anthropic_content.append({
                                "type": "image",
                                "source": {"type": "url", "url": url}
                            })
                    else:
                        anthropic_content.append({
                            "type": "image",
                            "source": {"type": "url", "url": url}
                        })
            elif isinstance(part, str):
                anthropic_content.append({"type": "text", "text": part})

        return anthropic_content if anthropic_content else ""

    return str(content) if content else ""


def _translate_anthropic_to_openai_response(anthropic_response: dict, model_alias: str) -> dict:
    """Translate Anthropic Messages response to OpenAI Chat Completions response"""
    import time
    import uuid

    # Extract content
    content_blocks = anthropic_response.get("content", [])
    text_parts = []
    tool_calls = []

    for idx, block in enumerate(content_blocks):
        if block.get("type") == "text":
            text_parts.append(block.get("text", ""))
        elif block.get("type") == "tool_use":
            tool_calls.append({
                "id": block.get("id", f"call_{uuid.uuid4().hex[:24]}"),
                "type": "function",
                "function": {
                    "name": block.get("name", ""),
                    "arguments": json.dumps(block.get("input", {}))
                }
            })

    # Build message
    message = {
        "role": "assistant",
        "content": " ".join(text_parts) if text_parts else None,
    }

    if tool_calls:
        message["tool_calls"] = tool_calls

    # Map stop_reason
    stop_reason = anthropic_response.get("stop_reason", "end_turn")
    finish_reason = "stop"
    if stop_reason == "max_tokens":
        finish_reason = "length"
    elif stop_reason == "tool_use":
        finish_reason = "tool_calls"

    # Extract usage
    usage = anthropic_response.get("usage", {})

    return {
        "id": f"chatcmpl-{anthropic_response.get('id', uuid.uuid4().hex[:24])}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_alias,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason
        }],
        "usage": {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        }
    }


# =============================================================================
# Health & Info Endpoints
# =============================================================================

@router.get("/v1/models")
async def list_models():
    """List available model aliases"""
    settings = get_settings()

    models = []
    for alias, config in settings.models.items():
        models.append({
            "id": alias,
            "object": "model",
            "created": 0,
            "owned_by": config.provider,
        })

    return {
        "object": "list",
        "data": models
    }
