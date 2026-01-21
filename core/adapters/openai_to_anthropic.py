"""
OpenAI to Anthropic Response Translator

Converts OpenAI Chat Completions responses to Anthropic Messages API format.
Handles both streaming and non-streaming responses.
"""

from __future__ import annotations
import json
import uuid
import logging
from typing import Dict, Any, List, Optional

from core.models.anthropic_types import (
    MessagesResponse, Usage, TextBlock, ToolUseBlock,
    ErrorResponse, ErrorDetail
)


logger = logging.getLogger("shin-gateway")


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


def _map_finish_reason(openai_reason: Optional[str]) -> str:
    """Map OpenAI finish_reason to Anthropic stop_reason"""
    return FINISH_REASON_MAP.get(openai_reason, "end_turn")


# =============================================================================
# Main Translation Function
# =============================================================================

def translate_response(
    openai_response: Dict[str, Any],
    original_model: str,
    request_model: Optional[str] = None
) -> Dict[str, Any]:
    """
    Translate OpenAI Chat Completions response to Anthropic Messages response.

    Args:
        openai_response: OpenAI-format response
        original_model: Original model alias from request
        request_model: Actual model used (if different)

    Returns:
        Anthropic-format response
    """
    # Extract choice
    choices = openai_response.get("choices", [])
    if not choices:
        return _create_empty_response(original_model)

    choice = choices[0]
    message = choice.get("message", {})

    # Build content blocks
    content = _translate_content(message)

    # Map finish reason
    finish_reason = choice.get("finish_reason")
    stop_reason = _map_finish_reason(finish_reason)

    # Extract usage
    usage = openai_response.get("usage", {})
    anthropic_usage = {
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
    }

    # Generate message ID
    openai_id = openai_response.get("id", "")
    message_id = f"msg_{openai_id}" if openai_id else f"msg_{uuid.uuid4().hex[:24]}"

    return {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "model": original_model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": anthropic_usage,
    }


# =============================================================================
# Content Translation
# =============================================================================

def _translate_content(message: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Translate OpenAI message content to Anthropic content blocks"""
    content = []

    # Add text content
    text = message.get("content")
    if text:
        content.append({
            "type": "text",
            "text": text
        })

    # Add tool calls
    tool_calls = message.get("tool_calls", [])
    for tc in tool_calls:
        tool_block = _translate_tool_call(tc)
        if tool_block:
            content.append(tool_block)

    # Handle legacy function_call
    function_call = message.get("function_call")
    if function_call:
        tool_block = _translate_function_call(function_call)
        if tool_block:
            content.append(tool_block)

    # Ensure at least empty text if no content
    if not content:
        content.append({"type": "text", "text": ""})

    return content


def _translate_tool_call(tool_call: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Translate OpenAI tool_call to Anthropic tool_use block"""
    if not isinstance(tool_call, dict):
        return None

    function = tool_call.get("function", {})
    name = function.get("name", "")
    arguments_str = function.get("arguments", "{}")

    # Parse arguments JSON
    try:
        arguments = json.loads(arguments_str) if arguments_str else {}
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse tool arguments: {arguments_str[:100]}")
        arguments = {"_raw": arguments_str}

    return {
        "type": "tool_use",
        "id": tool_call.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
        "name": name,
        "input": arguments
    }


def _translate_function_call(function_call: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Translate legacy function_call to tool_use block"""
    if not isinstance(function_call, dict):
        return None

    name = function_call.get("name", "")
    arguments_str = function_call.get("arguments", "{}")

    try:
        arguments = json.loads(arguments_str) if arguments_str else {}
    except json.JSONDecodeError:
        arguments = {"_raw": arguments_str}

    return {
        "type": "tool_use",
        "id": f"toolu_{uuid.uuid4().hex[:24]}",
        "name": name,
        "input": arguments
    }


# =============================================================================
# Helper Functions
# =============================================================================

def _create_empty_response(model: str) -> Dict[str, Any]:
    """Create an empty response when no choices available"""
    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": ""}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }


def create_error_response(
    error_type: str,
    message: str
) -> Dict[str, Any]:
    """Create an Anthropic error response"""
    return {
        "type": "error",
        "error": {
            "type": error_type,
            "message": message
        }
    }
