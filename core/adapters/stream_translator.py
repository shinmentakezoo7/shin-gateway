"""
Streaming Translator

Translates OpenAI streaming chunks to Anthropic SSE events in real-time.
Handles text, tool calls, and all edge cases for agentic workflows.
"""

from __future__ import annotations
import json
import uuid
import logging
from typing import Dict, Any, List, Optional, AsyncIterator, Callable
from dataclasses import dataclass, field


logger = logging.getLogger("shin-gateway")


# Optional import for stream recovery
try:
    from core.handlers.protocol_features import get_stream_recovery, StreamCheckpoint
    HAS_STREAM_RECOVERY = True
except ImportError:
    HAS_STREAM_RECOVERY = False


# =============================================================================
# Streaming State
# =============================================================================

@dataclass
class ToolCallBuffer:
    """Buffer for accumulating tool call data during streaming"""
    id: str
    name: str
    arguments: str = ""
    index: int = 0
    block_started: bool = False
    block_stopped: bool = False


@dataclass
class StreamState:
    """State for stream translation"""
    message_id: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    block_index: int = 0
    text_started: bool = False
    text_stopped: bool = False
    text_buffer: str = ""
    tool_buffers: Dict[int, ToolCallBuffer] = field(default_factory=dict)
    current_tool_index: Optional[int] = None
    message_started: bool = False
    finished: bool = False


# =============================================================================
# Event Creators
# =============================================================================

def create_message_start_event(state: StreamState) -> Dict[str, Any]:
    """Create message_start SSE event"""
    return {
        "type": "message_start",
        "message": {
            "id": state.message_id,
            "type": "message",
            "role": "assistant",
            "model": state.model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {
                "input_tokens": state.input_tokens,
                "output_tokens": 0
            }
        }
    }


def create_content_block_start_event(
    index: int,
    block_type: str,
    **kwargs
) -> Dict[str, Any]:
    """Create content_block_start SSE event"""
    if block_type == "text":
        content_block = {"type": "text", "text": ""}
    elif block_type == "tool_use":
        content_block = {
            "type": "tool_use",
            "id": kwargs.get("id", ""),
            "name": kwargs.get("name", ""),
            "input": {}
        }
    else:
        content_block = {"type": block_type}

    return {
        "type": "content_block_start",
        "index": index,
        "content_block": content_block
    }


def create_text_delta_event(index: int, text: str) -> Dict[str, Any]:
    """Create text delta SSE event"""
    return {
        "type": "content_block_delta",
        "index": index,
        "delta": {
            "type": "text_delta",
            "text": text
        }
    }


def create_tool_input_delta_event(index: int, partial_json: str) -> Dict[str, Any]:
    """Create tool input JSON delta SSE event"""
    return {
        "type": "content_block_delta",
        "index": index,
        "delta": {
            "type": "input_json_delta",
            "partial_json": partial_json
        }
    }


def create_content_block_stop_event(index: int) -> Dict[str, Any]:
    """Create content_block_stop SSE event"""
    return {
        "type": "content_block_stop",
        "index": index
    }


def create_message_delta_event(
    stop_reason: str,
    output_tokens: int
) -> Dict[str, Any]:
    """Create message_delta SSE event"""
    return {
        "type": "message_delta",
        "delta": {
            "stop_reason": stop_reason,
            "stop_sequence": None
        },
        "usage": {
            "output_tokens": output_tokens
        }
    }


def create_message_stop_event() -> Dict[str, Any]:
    """Create message_stop SSE event"""
    return {"type": "message_stop"}


def create_ping_event() -> Dict[str, Any]:
    """Create ping SSE event for keep-alive"""
    return {"type": "ping"}


def create_error_event(error_type: str, message: str) -> Dict[str, Any]:
    """Create error SSE event"""
    return {
        "type": "error",
        "error": {
            "type": error_type,
            "message": message
        }
    }


# =============================================================================
# Finish Reason Mapping
# =============================================================================

FINISH_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "end_turn",
    None: "end_turn",
}


# =============================================================================
# Stream Translator
# =============================================================================

async def translate_stream(
    openai_stream: AsyncIterator[str],
    model: str,
    estimated_input_tokens: int = 0,
    enable_checkpoints: bool = True
) -> AsyncIterator[Dict[str, Any]]:
    """
    Translate OpenAI streaming chunks to Anthropic SSE events.

    Args:
        openai_stream: AsyncIterator of SSE data lines from OpenAI
        model: Model name to include in response
        estimated_input_tokens: Estimated input tokens for usage
        enable_checkpoints: Whether to create recovery checkpoints

    Yields:
        Anthropic SSE event dictionaries
    """
    state = StreamState(
        message_id=f"msg_{uuid.uuid4().hex[:24]}",
        model=model,
        input_tokens=estimated_input_tokens,
    )

    # Get stream recovery handler if available
    stream_recovery = None
    if enable_checkpoints and HAS_STREAM_RECOVERY:
        stream_recovery = get_stream_recovery()

    event_index = 0
    content_blocks = []
    tool_calls_accumulated = []

    try:
        async for line in openai_stream:
            # Skip empty lines and comments
            if not line or line.startswith(":"):
                continue

            # Parse SSE data
            if line.startswith("data: "):
                data = line[6:]

                # Handle [DONE] signal
                if data.strip() == "[DONE]":
                    # Remove checkpoint on successful completion
                    if stream_recovery:
                        stream_recovery.remove_checkpoint(state.message_id)
                    async for event in _finish_stream(state):
                        yield event
                    return

                # Parse JSON chunk
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse chunk: {data[:100]}")
                    continue

                # Process chunk and yield events
                # Note: _process_chunk is an async generator, iterate properly
                try:
                    async for event in _process_chunk(chunk, state):
                        if event is not None:
                            event_index += 1
                            yield event

                            # Track content for checkpoints
                            if event.get("type") == "content_block_delta":
                                delta = event.get("delta", {})
                                if delta.get("type") == "text_delta":
                                    # Accumulate text
                                    block_idx = event.get("index", 0)
                                    while len(content_blocks) <= block_idx:
                                        content_blocks.append({"type": "text", "text": ""})
                                    content_blocks[block_idx]["text"] += delta.get("text", "")

                            # Create checkpoint periodically
                            if stream_recovery and event_index % 10 == 0:
                                stream_recovery.create_checkpoint(
                                    message_id=state.message_id,
                                    content_blocks=content_blocks,
                                    tool_calls=tool_calls_accumulated,
                                    event_index=event_index,
                                    input_tokens=state.input_tokens,
                                    output_tokens=state.output_tokens
                                )
                except Exception as chunk_error:
                    logger.error(f"Error processing chunk: {chunk_error}")
                    continue

        # Handle stream end without [DONE]
        if not state.finished:
            if stream_recovery:
                stream_recovery.remove_checkpoint(state.message_id)
            async for event in _finish_stream(state):
                if event is not None:
                    yield event

    except Exception as e:
        logger.error(f"Stream translation error: {e}")
        # Keep checkpoint on error for potential recovery
        yield create_error_event("api_error", str(e))


async def _process_chunk(
    chunk: Dict[str, Any],
    state: StreamState
) -> AsyncIterator[Dict[str, Any]]:
    """Process a single OpenAI chunk and yield Anthropic events"""

    # Emit message_start on first chunk
    if not state.message_started:
        state.message_started = True
        yield create_message_start_event(state)

    # Get delta from chunk
    choices = chunk.get("choices", [])
    if not choices:
        # May have usage info in final chunk
        usage = chunk.get("usage")
        if usage:
            state.output_tokens = usage.get("completion_tokens", state.output_tokens)
            state.input_tokens = usage.get("prompt_tokens", state.input_tokens)
        # Return from generator (no more yields)
        return

    choice = choices[0]
    delta = choice.get("delta", {})
    finish_reason = choice.get("finish_reason")

    # Handle text content
    content = delta.get("content")
    if content:
        # Start text block if needed
        if not state.text_started:
            state.text_started = True
            yield create_content_block_start_event(state.block_index, "text")

        # Emit text delta
        state.text_buffer += content
        state.output_tokens += len(content) // 4  # Rough estimate
        yield create_text_delta_event(state.block_index, content)

    # Handle tool calls (can have multiple in parallel)
    tool_calls = delta.get("tool_calls", [])
    for tc in tool_calls:
        tc_index = tc.get("index", 0)

        # Check if this is a new tool call
        if tc_index not in state.tool_buffers:
            # Close text block if open and not already closed
            if state.text_started and not state.text_stopped:
                yield create_content_block_stop_event(state.block_index)
                state.block_index += 1
                state.text_stopped = True

            # Start new tool block
            tool_id = tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}")
            function = tc.get("function", {})
            tool_name = function.get("name", "")

            state.tool_buffers[tc_index] = ToolCallBuffer(
                id=tool_id,
                name=tool_name,
                arguments="",
                index=state.block_index,
                block_started=True,
                block_stopped=False
            )
            state.current_tool_index = tc_index

            yield create_content_block_start_event(
                state.block_index,
                "tool_use",
                id=tool_id,
                name=tool_name
            )
            state.block_index += 1

        # Stream tool arguments
        function = tc.get("function", {})
        arguments = function.get("arguments", "")
        if arguments:
            tool_buffer = state.tool_buffers[tc_index]
            tool_buffer.arguments += arguments
            yield create_tool_input_delta_event(tool_buffer.index, arguments)

        # Update tool name if streaming in chunks
        name = function.get("name", "")
        if name and tc_index in state.tool_buffers:
            state.tool_buffers[tc_index].name = name

    # Handle finish
    if finish_reason:
        async for event in _finish_stream(state, finish_reason):
            yield event


async def _finish_stream(
    state: StreamState,
    finish_reason: Optional[str] = None
) -> AsyncIterator[Dict[str, Any]]:
    """Finish the stream and emit final events"""
    if state.finished:
        # Already finished, yield nothing (empty async generator)
        if False:  # Never executed, but makes this a generator
            yield {}
        return

    state.finished = True

    # Close any open text block
    if state.text_started and not state.text_stopped:
        yield create_content_block_stop_event(state.block_index)
        state.block_index += 1

    # Close all open tool blocks
    for tc_index, tool_buffer in state.tool_buffers.items():
        if tool_buffer.block_started and not tool_buffer.block_stopped:
            yield create_content_block_stop_event(tool_buffer.index)
            tool_buffer.block_stopped = True

    # Map finish reason
    stop_reason = FINISH_REASON_MAP.get(finish_reason, "end_turn")

    # If we have tool calls, the stop reason should be tool_use
    if state.tool_buffers:
        stop_reason = "tool_use"

    # Emit message_delta with stop_reason
    yield create_message_delta_event(stop_reason, state.output_tokens)

    # Emit message_stop
    yield create_message_stop_event()


# =============================================================================
# SSE Formatting
# =============================================================================

def format_sse_event(event: Dict[str, Any]) -> str:
    """Format event dictionary as SSE string"""
    event_type = event.get("type", "message")
    data = json.dumps(event)
    return f"event: {event_type}\ndata: {data}\n\n"


def format_sse_comment(comment: str) -> str:
    """Format SSE comment (for keep-alive)"""
    return f": {comment}\n\n"
