"""
Anthropic to OpenAI Request Translator

Converts incoming Anthropic Messages API requests to OpenAI Chat Completions format.
Handles all content types, tools, and parameters.
"""

from __future__ import annotations
import json
import logging
from typing import Dict, Any, List, Optional, Union

from core.handlers.beta_features import get_cache_handler
from core.handlers.protocol_features import (
    get_vision_handler,
    get_document_handler,
    get_thinking_handler,
    get_metadata_handler,
    get_stop_handler,
    ThinkingConfig,
)


logger = logging.getLogger("shin-gateway")


# =============================================================================
# Main Translation Function
# =============================================================================

def translate_request(
    anthropic_request: Dict[str, Any],
    target_model: str,
    strip_cache_control: bool = True,
    emulate_thinking: bool = False
) -> Dict[str, Any]:
    """
    Translate Anthropic Messages request to OpenAI Chat Completions request.

    Args:
        anthropic_request: Anthropic-format request
        target_model: Target model name for OpenAI
        strip_cache_control: Whether to strip cache_control blocks
        emulate_thinking: Whether to emulate extended thinking for OpenAI

    Returns:
        OpenAI-format request
    """
    # Get handlers
    thinking_handler = get_thinking_handler()
    metadata_handler = get_metadata_handler()
    stop_handler = get_stop_handler()

    openai_request: Dict[str, Any] = {
        "model": target_model,
        "stream": anthropic_request.get("stream", False),
    }

    # Build messages
    messages = []

    # Add system message
    system = anthropic_request.get("system")
    if system:
        system_content = _translate_system(system, strip_cache_control)
        messages.append({"role": "system", "content": system_content})

    # Translate conversation messages
    for msg in anthropic_request.get("messages", []):
        translated = _translate_message(msg, strip_cache_control)
        if isinstance(translated, list):
            messages.extend(translated)
        else:
            messages.append(translated)

    # Handle extended thinking emulation
    thinking_config = ThinkingConfig.from_request(anthropic_request)
    if thinking_config.enabled and emulate_thinking:
        messages = thinking_handler.emulate_thinking_for_openai(messages, thinking_config)

    openai_request["messages"] = messages

    # Translate parameters
    if "max_tokens" in anthropic_request:
        openai_request["max_tokens"] = anthropic_request["max_tokens"]

    if "temperature" in anthropic_request:
        openai_request["temperature"] = anthropic_request["temperature"]

    if "top_p" in anthropic_request:
        openai_request["top_p"] = anthropic_request["top_p"]

    # top_k is not supported by OpenAI - drop silently

    # Use stop handler for stop_sequences translation
    if "stop_sequences" in anthropic_request:
        openai_request = stop_handler.translate_to_openai({
            **openai_request,
            "stop_sequences": anthropic_request["stop_sequences"]
        })

    # Translate tools
    tools = anthropic_request.get("tools")
    if tools:
        openai_request["tools"] = _translate_tools(tools)

    # Translate tool_choice
    tool_choice = anthropic_request.get("tool_choice")
    if tool_choice:
        openai_request["tool_choice"] = _translate_tool_choice(tool_choice)

    # Handle stream options for usage in streaming
    if openai_request.get("stream"):
        openai_request["stream_options"] = {"include_usage": True}

    # Handle metadata using metadata handler
    openai_request = metadata_handler.translate_to_openai({
        **openai_request,
        "metadata": anthropic_request.get("metadata")
    })

    return openai_request


# =============================================================================
# System Message Translation
# =============================================================================

def _translate_system(
    system: Union[str, List[Dict]],
    strip_cache_control: bool = True
) -> str:
    """Translate system prompt to string"""
    if isinstance(system, str):
        return system

    if isinstance(system, list):
        parts = []
        for block in system:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)

    return str(system)


# =============================================================================
# Message Translation
# =============================================================================

def _translate_message(
    message: Dict[str, Any],
    strip_cache_control: bool = True
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Translate a single Anthropic message to OpenAI format.

    May return a list if tool_results need to be split into separate messages.
    """
    role = message.get("role")
    content = message.get("content")

    if role == "user":
        return _translate_user_message(content, strip_cache_control)

    elif role == "assistant":
        return _translate_assistant_message(content)

    else:
        # Unknown role - pass through
        return {"role": role, "content": str(content) if content else ""}


def _translate_user_message(
    content: Union[str, List[Dict]],
    strip_cache_control: bool = True
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """Translate user message"""
    if isinstance(content, str):
        return {"role": "user", "content": content}

    if not isinstance(content, list):
        return {"role": "user", "content": str(content) if content else ""}

    # Check for tool_results - need to split into separate messages
    tool_results = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_result"]
    other_content = [b for b in content if not (isinstance(b, dict) and b.get("type") == "tool_result")]

    messages = []

    # Add tool result messages first
    for result in tool_results:
        tool_message = _translate_tool_result(result)
        messages.append(tool_message)

    # Add remaining content as user message
    if other_content:
        translated_content = _translate_content_blocks(other_content, strip_cache_control)
        if translated_content:
            messages.append({"role": "user", "content": translated_content})
    elif not tool_results:
        # Empty content
        messages.append({"role": "user", "content": ""})

    return messages if len(messages) > 1 else messages[0] if messages else {"role": "user", "content": ""}


def _translate_assistant_message(
    content: Union[str, List[Dict]]
) -> Dict[str, Any]:
    """Translate assistant message"""
    if isinstance(content, str):
        return {"role": "assistant", "content": content}

    if not isinstance(content, list):
        return {"role": "assistant", "content": str(content) if content else None}

    # Separate text and tool_use blocks
    text_parts = []
    tool_calls = []

    for block in content:
        if not isinstance(block, dict):
            continue

        block_type = block.get("type")

        if block_type == "text":
            text_parts.append(block.get("text", ""))

        elif block_type == "tool_use":
            tool_calls.append({
                "id": block.get("id", ""),
                "type": "function",
                "function": {
                    "name": block.get("name", ""),
                    "arguments": json.dumps(block.get("input", {}))
                }
            })

        elif block_type in ("thinking", "redacted_thinking"):
            # Skip thinking blocks - not supported in OpenAI
            pass

    result: Dict[str, Any] = {"role": "assistant"}

    if text_parts:
        result["content"] = " ".join(text_parts)
    else:
        result["content"] = None

    if tool_calls:
        result["tool_calls"] = tool_calls

    return result


# =============================================================================
# Content Block Translation
# =============================================================================

def _translate_content_blocks(
    blocks: List[Dict],
    strip_cache_control: bool = True
) -> Union[str, List[Dict]]:
    """Translate content blocks to OpenAI format using protocol handlers"""
    # Get handlers
    vision_handler = get_vision_handler()
    document_handler = get_document_handler()

    # First, handle document blocks (PDF, etc.)
    processed_blocks = document_handler.strip_documents(blocks)

    # Check if we have any non-text content (images, etc.)
    has_complex_content = any(
        isinstance(b, dict) and b.get("type") in ("image", "image_url")
        for b in processed_blocks
    )

    if not has_complex_content:
        # Simple case: concatenate text
        text_parts = []
        for block in processed_blocks:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif isinstance(block, str):
                text_parts.append(block)
        return " ".join(text_parts)

    # Complex case: use VisionHandler for proper translation
    return vision_handler.translate_anthropic_to_openai(processed_blocks)


def _translate_image_block(block: Dict) -> Dict:
    """Translate Anthropic image block to OpenAI format"""
    source = block.get("source", {})
    source_type = source.get("type")

    if source_type == "base64":
        media_type = source.get("media_type", "image/png")
        data = source.get("data", "")
        url = f"data:{media_type};base64,{data}"
    elif source_type == "url":
        url = source.get("url", "")
    else:
        url = ""

    return {
        "type": "image_url",
        "image_url": {"url": url}
    }


# =============================================================================
# Tool Result Translation
# =============================================================================

def _translate_tool_result(result: Dict) -> Dict:
    """Translate tool_result block to OpenAI tool message"""
    tool_use_id = result.get("tool_use_id", "")
    content = result.get("content", "")
    is_error = result.get("is_error", False)

    # Content can be string or list of blocks
    if isinstance(content, list):
        # Concatenate text blocks and handle images
        text_parts = []
        for block in content:
            if isinstance(block, dict):
                block_type = block.get("type")
                if block_type == "text":
                    text_parts.append(block.get("text", ""))
                elif block_type == "image":
                    # Some providers may not support images in tool results
                    # Include a placeholder or base64 reference
                    source = block.get("source", {})
                    if source.get("type") == "base64":
                        text_parts.append(f"[Image: {source.get('media_type', 'image')}]")
                    else:
                        text_parts.append(f"[Image: {source.get('url', '')}]")
            elif isinstance(block, str):
                text_parts.append(block)
        content = "\n".join(text_parts) if text_parts else ""

    # For errors, prepend error indicator that some models recognize
    content_str = str(content)
    if is_error:
        content_str = f"Error: {content_str}"

    return {
        "role": "tool",
        "tool_call_id": tool_use_id,
        "content": content_str
    }


# =============================================================================
# Tools Translation
# =============================================================================

def _translate_tools(tools: List[Dict]) -> List[Dict]:
    """Translate Anthropic tools to OpenAI format"""
    translated = []
    for tool in tools:
        input_schema = tool.get("input_schema", {})

        # Ensure input_schema has proper structure
        if not input_schema:
            input_schema = {"type": "object", "properties": {}}
        elif input_schema.get("type") != "object":
            # Wrap non-object schemas
            input_schema = {"type": "object", "properties": {}}

        # Handle additionalProperties for OpenAI strict mode compatibility
        # Some providers require this to be explicitly set
        if "additionalProperties" not in input_schema:
            input_schema["additionalProperties"] = False

        function_def = {
            "name": tool.get("name", ""),
            "description": tool.get("description") or "",
            "parameters": input_schema
        }

        # Add strict mode if supported (OpenAI specific)
        # This ensures the model follows the schema exactly
        translated.append({
            "type": "function",
            "function": function_def
        })
    return translated


def _translate_tool_choice(choice: Union[Dict, str]) -> Union[str, Dict]:
    """Translate Anthropic tool_choice to OpenAI format"""
    if isinstance(choice, str):
        return choice

    if not isinstance(choice, dict):
        return "auto"

    choice_type = choice.get("type")

    if choice_type == "auto":
        return "auto"
    elif choice_type == "any":
        return "required"
    elif choice_type == "none":
        return "none"
    elif choice_type == "tool":
        return {
            "type": "function",
            "function": {"name": choice.get("name", "")}
        }

    return "auto"
