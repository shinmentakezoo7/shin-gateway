"""
Advanced Protocol Features Handler

Handles protocol translation features for Claude Code compatibility:
- Vision/Image support (base64 conversion)
- Extended Thinking content blocks
- PDF/Document support
- Token counting pre-flight
- Tool result validation
- Metadata passthrough
- Stop sequences translation
"""

from __future__ import annotations
import base64
import hashlib
import json
import logging
import re
from typing import Dict, Any, List, Optional, Union, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum

# Optional httpx for URL fetching
try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    httpx = None  # type: ignore
    HAS_HTTPX = False

logger = logging.getLogger("shin-gateway")


# =============================================================================
# Image/Vision Support
# =============================================================================

class ImageFormat(Enum):
    """Supported image formats"""
    JPEG = "image/jpeg"
    PNG = "image/png"
    GIF = "image/gif"
    WEBP = "image/webp"


@dataclass
class ImageData:
    """Parsed image data"""
    media_type: str
    data: str  # base64 encoded
    url: Optional[str] = None

    @property
    def data_url(self) -> str:
        """Get data URL format"""
        return f"data:{self.media_type};base64,{self.data}"

    @classmethod
    def from_data_url(cls, data_url: str) -> Optional["ImageData"]:
        """Parse data URL to ImageData"""
        try:
            # Format: data:image/png;base64,<data>
            match = re.match(r'data:([^;]+);base64,(.+)', data_url)
            if match:
                return cls(media_type=match.group(1), data=match.group(2))
        except Exception as e:
            logger.warning(f"Failed to parse data URL: {e}")
        return None

    @classmethod
    async def from_url(cls, url: str, client: Optional[Any] = None) -> Optional["ImageData"]:
        """Fetch image from URL and convert to base64"""
        if not HAS_HTTPX:
            logger.warning("httpx not available for URL fetching")
            return None
        try:
            if client is None:
                async with httpx.AsyncClient() as c:
                    response = await c.get(url, timeout=30.0)
            else:
                response = await client.get(url, timeout=30.0)

            response.raise_for_status()
            content_type = response.headers.get("content-type", "image/png")
            # Extract just the mime type without charset
            media_type = content_type.split(";")[0].strip()
            data = base64.b64encode(response.content).decode()
            return cls(media_type=media_type, data=data, url=url)
        except Exception as e:
            logger.warning(f"Failed to fetch image from URL {url}: {e}")
        return None


class VisionHandler:
    """Handle vision/image content translation between formats"""

    def translate_anthropic_to_openai(self, content: List[Dict]) -> List[Dict]:
        """
        Translate Anthropic image blocks to OpenAI format.

        Anthropic: {"type": "image", "source": {"type": "base64", "media_type": "...", "data": "..."}}
        OpenAI: {"type": "image_url", "image_url": {"url": "data:...;base64,..."}}
        """
        translated = []
        for block in content:
            if not isinstance(block, dict):
                translated.append({"type": "text", "text": str(block)})
                continue

            block_type = block.get("type")

            if block_type == "text":
                translated.append({"type": "text", "text": block.get("text", "")})

            elif block_type == "image":
                source = block.get("source", {})
                source_type = source.get("type")

                if source_type == "base64":
                    media_type = source.get("media_type", "image/png")
                    data = source.get("data", "")
                    url = f"data:{media_type};base64,{data}"
                elif source_type == "url":
                    url = source.get("url", "")
                else:
                    continue

                translated.append({
                    "type": "image_url",
                    "image_url": {"url": url, "detail": "auto"}
                })

            else:
                # Pass through unknown types
                translated.append(block)

        return translated

    def translate_openai_to_anthropic(self, content: List[Dict]) -> List[Dict]:
        """
        Translate OpenAI image blocks to Anthropic format.

        OpenAI: {"type": "image_url", "image_url": {"url": "data:...;base64,..."}}
        Anthropic: {"type": "image", "source": {"type": "base64", "media_type": "...", "data": "..."}}
        """
        translated = []
        for block in content:
            if not isinstance(block, dict):
                translated.append({"type": "text", "text": str(block)})
                continue

            block_type = block.get("type")

            if block_type == "text":
                translated.append({"type": "text", "text": block.get("text", "")})

            elif block_type == "image_url":
                image_url = block.get("image_url", {})
                url = image_url.get("url", "")

                # Check if it's a data URL
                image_data = ImageData.from_data_url(url)
                if image_data:
                    translated.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": image_data.media_type,
                            "data": image_data.data
                        }
                    })
                else:
                    # It's a regular URL - pass as URL source
                    translated.append({
                        "type": "image",
                        "source": {
                            "type": "url",
                            "url": url
                        }
                    })

            else:
                translated.append(block)

        return translated


# =============================================================================
# Extended Thinking Handler
# =============================================================================

@dataclass
class ThinkingConfig:
    """Extended thinking configuration"""
    enabled: bool = False
    budget_tokens: Optional[int] = None

    @classmethod
    def from_request(cls, request: Dict[str, Any]) -> "ThinkingConfig":
        """Parse thinking config from request"""
        thinking = request.get("thinking")
        if not thinking:
            return cls(enabled=False)

        if isinstance(thinking, dict):
            return cls(
                enabled=thinking.get("type") == "enabled",
                budget_tokens=thinking.get("budget_tokens")
            )
        return cls(enabled=False)


class ThinkingHandler:
    """Handle extended thinking content blocks"""

    def strip_thinking_from_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Strip thinking configuration for providers that don't support it.
        """
        modified = request.copy()
        modified.pop("thinking", None)
        return modified

    def translate_thinking_response(
        self,
        content: List[Dict],
        include_thinking: bool = False
    ) -> List[Dict]:
        """
        Handle thinking blocks in response.
        If include_thinking is False, strip thinking blocks.
        """
        if include_thinking:
            return content

        return [
            block for block in content
            if not isinstance(block, dict) or
            block.get("type") not in ("thinking", "redacted_thinking")
        ]

    def emulate_thinking_for_openai(
        self,
        messages: List[Dict],
        thinking_config: ThinkingConfig
    ) -> List[Dict]:
        """
        Add thinking instructions to system prompt for OpenAI models.
        This emulates extended thinking by asking the model to think step-by-step.
        """
        if not thinking_config.enabled:
            return messages

        thinking_instruction = (
            "\n\n<thinking_instruction>\n"
            "Before responding, think through the problem step by step in a <thinking> block. "
            "Show your reasoning process, consider different approaches, and work through the logic. "
            "After thinking, provide your final response outside the thinking block.\n"
            "</thinking_instruction>"
        )

        modified = []
        system_found = False

        for msg in messages:
            if msg.get("role") == "system":
                system_found = True
                new_msg = msg.copy()
                content = msg.get("content", "")
                if isinstance(content, str):
                    new_msg["content"] = content + thinking_instruction
                modified.append(new_msg)
            else:
                modified.append(msg)

        # Add system message if none exists
        if not system_found and thinking_config.enabled:
            modified.insert(0, {
                "role": "system",
                "content": thinking_instruction.strip()
            })

        return modified

    def extract_thinking_from_response(
        self,
        content: str
    ) -> Tuple[Optional[str], str]:
        """
        Extract thinking content from response text.
        Returns (thinking_content, main_content)
        """
        # Look for <thinking>...</thinking> blocks
        thinking_pattern = r'<thinking>(.*?)</thinking>'
        matches = re.findall(thinking_pattern, content, re.DOTALL)

        if matches:
            thinking = "\n".join(matches)
            main_content = re.sub(thinking_pattern, '', content, flags=re.DOTALL).strip()
            return thinking, main_content

        return None, content


# =============================================================================
# PDF/Document Handler
# =============================================================================

class DocumentHandler:
    """Handle PDF and document content"""

    SUPPORTED_TYPES = {
        "application/pdf": "pdf",
        "text/plain": "text",
        "text/markdown": "markdown",
        "text/html": "html",
    }

    def translate_document_block(self, block: Dict) -> List[Dict]:
        """
        Translate document block to text content.
        For PDFs, we extract text content or provide a placeholder.

        Anthropic document block:
        {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "..."}}
        """
        if block.get("type") != "document":
            return [block]

        source = block.get("source", {})
        media_type = source.get("media_type", "")

        if media_type == "application/pdf":
            # For PDFs, we need to extract text or notify that PDF processing is needed
            # This is a placeholder - real implementation would use pypdf or similar
            data = source.get("data", "")
            if data:
                # Return a text block indicating PDF content
                return [{
                    "type": "text",
                    "text": f"[PDF Document - {len(data)} bytes base64 encoded. PDF text extraction not available in this provider.]"
                }]

        elif media_type in ("text/plain", "text/markdown"):
            # Decode text content
            data = source.get("data", "")
            try:
                text = base64.b64decode(data).decode("utf-8")
                return [{"type": "text", "text": text}]
            except Exception as e:
                logger.warning(f"Failed to decode document: {e}")

        return [{"type": "text", "text": "[Document content not supported]"}]

    def strip_documents(self, content: List[Dict]) -> List[Dict]:
        """Strip document blocks from content, translating where possible"""
        result = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "document":
                result.extend(self.translate_document_block(block))
            else:
                result.append(block)
        return result


# =============================================================================
# Token Counter
# =============================================================================

class TokenCounter:
    """
    Token counting for pre-flight checks.
    Uses tiktoken for OpenAI models, approximation for others.
    """

    # Approximate tokens per character for different content types
    CHAR_RATIOS = {
        "text": 0.25,  # ~4 chars per token for English
        "code": 0.33,  # ~3 chars per token for code
        "json": 0.4,   # ~2.5 chars per token for JSON
    }

    def __init__(self):
        self._tiktoken = None
        self._encoders = {}

    def _get_encoder(self, model: str):
        """Get tiktoken encoder for model"""
        if self._tiktoken is None:
            try:
                import tiktoken
                self._tiktoken = tiktoken
            except ImportError:
                self._tiktoken = False

        if self._tiktoken is False:
            return None

        if model not in self._encoders:
            try:
                # Try to get encoding for model
                self._encoders[model] = self._tiktoken.encoding_for_model(model)
            except Exception:
                # Fallback to cl100k_base (GPT-4 encoding)
                try:
                    self._encoders[model] = self._tiktoken.get_encoding("cl100k_base")
                except Exception:
                    self._encoders[model] = None

        return self._encoders.get(model)

    def count_tokens(self, text: str, model: str = "gpt-4") -> int:
        """Count tokens in text"""
        encoder = self._get_encoder(model)
        if encoder:
            try:
                return len(encoder.encode(text))
            except Exception:
                pass

        # Fallback to character-based approximation
        return int(len(text) * self.CHAR_RATIOS["text"])

    def count_message_tokens(self, messages: List[Dict], model: str = "gpt-4") -> int:
        """Count tokens in messages"""
        total = 0

        for msg in messages:
            # Add overhead per message (role, etc.)
            total += 4  # Approximate message overhead

            content = msg.get("content", "")
            if isinstance(content, str):
                total += self.count_tokens(content, model)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            total += self.count_tokens(block.get("text", ""), model)
                        elif block.get("type") in ("image", "image_url"):
                            # Images are typically 85-1105 tokens depending on detail
                            total += 765  # Medium estimate

        # Add overhead for response format
        total += 3

        return total

    def count_tools_tokens(self, tools: List[Dict], model: str = "gpt-4") -> int:
        """Count tokens used by tool definitions"""
        if not tools:
            return 0

        # Serialize tools to JSON and count
        tools_json = json.dumps(tools)
        return self.count_tokens(tools_json, model)

    def estimate_request_tokens(
        self,
        request: Dict[str, Any],
        model: str = "gpt-4"
    ) -> Dict[str, int]:
        """
        Estimate total tokens for a request.
        Returns breakdown by component.
        """
        result = {
            "messages": 0,
            "system": 0,
            "tools": 0,
            "total": 0
        }

        # Count system prompt
        system = request.get("system", "")
        if system:
            if isinstance(system, str):
                result["system"] = self.count_tokens(system, model)
            elif isinstance(system, list):
                for block in system:
                    if isinstance(block, dict) and block.get("type") == "text":
                        result["system"] += self.count_tokens(block.get("text", ""), model)

        # Count messages
        messages = request.get("messages", [])
        result["messages"] = self.count_message_tokens(messages, model)

        # Count tools
        tools = request.get("tools", [])
        result["tools"] = self.count_tools_tokens(tools, model)

        result["total"] = result["messages"] + result["system"] + result["tools"]

        return result


class TokenPreflightChecker:
    """Pre-flight token validation"""

    # Default context windows for common models
    CONTEXT_WINDOWS = {
        "gpt-4": 8192,
        "gpt-4-turbo": 128000,
        "gpt-4o": 128000,
        "gpt-4o-mini": 128000,
        "gpt-3.5-turbo": 16385,
        "claude-3-opus": 200000,
        "claude-3-sonnet": 200000,
        "claude-3-haiku": 200000,
        "claude-3-5-sonnet": 200000,
        "llama-3.1-8b": 131072,
        "llama-3.1-70b": 131072,
        "llama-3.1-405b": 131072,
    }

    def __init__(self, token_counter: Optional[TokenCounter] = None):
        self.counter = token_counter or TokenCounter()

    def get_context_window(self, model: str) -> int:
        """Get context window for model"""
        # Check exact match
        if model in self.CONTEXT_WINDOWS:
            return self.CONTEXT_WINDOWS[model]

        # Check partial match
        model_lower = model.lower()
        for key, window in self.CONTEXT_WINDOWS.items():
            if key in model_lower:
                return window

        # Default to conservative estimate
        return 8192

    def validate_request(
        self,
        request: Dict[str, Any],
        model: str,
        max_tokens: int
    ) -> Tuple[bool, Optional[str], Dict[str, int]]:
        """
        Validate request won't exceed context window.

        Returns: (is_valid, error_message, token_counts)
        """
        context_window = self.get_context_window(model)
        token_counts = self.counter.estimate_request_tokens(request, model)

        # Check if input exceeds context
        if token_counts["total"] >= context_window:
            return (
                False,
                f"Input tokens ({token_counts['total']}) exceed context window ({context_window})",
                token_counts
            )

        # Check if input + max_tokens exceeds context
        if token_counts["total"] + max_tokens > context_window:
            available = context_window - token_counts["total"]
            return (
                False,
                f"Requested max_tokens ({max_tokens}) exceeds available space ({available})",
                token_counts
            )

        return True, None, token_counts


# =============================================================================
# Tool Result Validator
# =============================================================================

class ToolResultValidator:
    """Validate tool results against schemas"""

    def __init__(self, tools: Optional[List[Dict]] = None):
        self.tools = {t.get("name"): t for t in (tools or [])}

    def set_tools(self, tools: List[Dict]):
        """Update tool definitions"""
        self.tools = {t.get("name"): t for t in tools}

    def validate_tool_input(
        self,
        tool_name: str,
        input_data: Dict
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate tool input against schema.
        Returns (is_valid, error_message)
        """
        tool = self.tools.get(tool_name)
        if not tool:
            return True, None  # Can't validate unknown tools

        schema = tool.get("input_schema", {})
        required = schema.get("required", [])
        properties = schema.get("properties", {})

        # Check required fields
        for field in required:
            if field not in input_data:
                return False, f"Missing required field: {field}"

        # Check field types
        for field, value in input_data.items():
            if field in properties:
                expected_type = properties[field].get("type")
                if not self._check_type(value, expected_type):
                    return False, f"Invalid type for {field}: expected {expected_type}"

        return True, None

    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected JSON Schema type"""
        type_map = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None),
        }

        if expected_type not in type_map:
            return True

        return isinstance(value, type_map[expected_type])

    def validate_tool_result(
        self,
        tool_use_id: str,
        content: Any,
        is_error: bool = False
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate tool result format.
        Returns (is_valid, error_message)
        """
        # Tool results should have content
        if content is None:
            return False, "Tool result content cannot be null"

        # Content should be string or list of content blocks
        if isinstance(content, str):
            return True, None

        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type == "text" and not isinstance(block.get("text"), str):
                    return False, "Text block must have string text field"
            return True, None

        return False, "Invalid tool result content type"


# =============================================================================
# Metadata Handler
# =============================================================================

class MetadataHandler:
    """Handle metadata passthrough between formats"""

    # Fields to preserve in metadata
    PASSTHROUGH_FIELDS = [
        "user_id",
        "conversation_id",
        "session_id",
        "request_id",
        "trace_id",
        "span_id",
    ]

    def extract_metadata(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from Anthropic request"""
        metadata = request.get("metadata", {})
        if not isinstance(metadata, dict):
            return {}
        return metadata

    def translate_to_openai(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translate Anthropic metadata to OpenAI format.
        OpenAI uses 'user' field for user ID.
        """
        modified = request.copy()
        metadata = self.extract_metadata(request)

        if metadata.get("user_id"):
            modified["user"] = metadata["user_id"]

        # Remove metadata field (not supported in OpenAI)
        modified.pop("metadata", None)

        return modified

    def translate_to_anthropic(
        self,
        request: Dict[str, Any],
        original_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Translate OpenAI metadata to Anthropic format.
        """
        modified = request.copy()

        metadata = original_metadata.copy() if original_metadata else {}

        # Extract user field
        user = request.get("user")
        if user:
            metadata["user_id"] = user
            modified.pop("user", None)

        if metadata:
            modified["metadata"] = metadata

        return modified

    def add_gateway_metadata(
        self,
        response: Dict[str, Any],
        gateway_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Add gateway metadata to response for debugging"""
        modified = response.copy()

        # Add as a separate field or extend existing usage
        if "usage" in modified and isinstance(modified["usage"], dict):
            modified["usage"]["_gateway"] = gateway_info

        return modified


# =============================================================================
# Stop Sequences Handler
# =============================================================================

class StopSequencesHandler:
    """Handle stop sequences translation"""

    def translate_to_openai(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translate Anthropic stop_sequences to OpenAI stop parameter.
        Anthropic: stop_sequences (list)
        OpenAI: stop (string or list, max 4)
        """
        modified = request.copy()
        stop_sequences = modified.pop("stop_sequences", None)

        if stop_sequences:
            # OpenAI has max 4 stop sequences
            if len(stop_sequences) > 4:
                logger.warning(f"Truncating stop_sequences from {len(stop_sequences)} to 4")
                stop_sequences = stop_sequences[:4]
            modified["stop"] = stop_sequences

        return modified

    def translate_to_anthropic(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translate OpenAI stop to Anthropic stop_sequences.
        """
        modified = request.copy()
        stop = modified.pop("stop", None)

        if stop:
            if isinstance(stop, str):
                modified["stop_sequences"] = [stop]
            elif isinstance(stop, list):
                modified["stop_sequences"] = stop

        return modified


# =============================================================================
# Provider Fallback Chain
# =============================================================================

@dataclass
class FallbackProvider:
    """Fallback provider configuration"""
    provider: str
    model: str
    priority: int = 0
    conditions: Dict[str, Any] = field(default_factory=dict)


class FallbackChainHandler:
    """Handle provider fallback chains"""

    def __init__(self, fallbacks: Optional[List[FallbackProvider]] = None):
        self.fallbacks = sorted(fallbacks or [], key=lambda x: x.priority)

    def get_fallback_chain(
        self,
        primary_provider: str,
        error_type: Optional[str] = None
    ) -> List[FallbackProvider]:
        """Get ordered fallback chain for a provider"""
        chain = []

        for fb in self.fallbacks:
            # Skip the primary provider
            if fb.provider == primary_provider:
                continue

            # Check conditions
            if error_type and fb.conditions.get("on_error"):
                if error_type not in fb.conditions["on_error"]:
                    continue

            chain.append(fb)

        return chain

    def should_fallback(
        self,
        error: Exception,
        attempt: int,
        max_attempts: int = 3
    ) -> bool:
        """Determine if we should try fallback"""
        if attempt >= max_attempts:
            return False

        # Always fallback on rate limits and server errors
        error_str = str(error).lower()
        fallback_triggers = [
            "rate limit",
            "429",
            "500",
            "502",
            "503",
            "504",
            "timeout",
            "connection",
            "overloaded",
        ]

        return any(trigger in error_str for trigger in fallback_triggers)


# =============================================================================
# Streaming Error Recovery
# =============================================================================

@dataclass
class StreamCheckpoint:
    """Checkpoint for stream recovery"""
    message_id: str
    content_blocks: List[Dict]
    tool_calls: List[Dict]
    last_event_index: int
    input_tokens: int
    output_tokens: int
    timestamp: float


class StreamRecoveryHandler:
    """Handle streaming error recovery"""

    def __init__(self, max_checkpoints: int = 100):
        self.checkpoints: Dict[str, StreamCheckpoint] = {}
        self.max_checkpoints = max_checkpoints

    def create_checkpoint(
        self,
        message_id: str,
        content_blocks: List[Dict],
        tool_calls: List[Dict],
        event_index: int,
        input_tokens: int = 0,
        output_tokens: int = 0
    ) -> StreamCheckpoint:
        """Create a checkpoint for stream recovery"""
        import time

        checkpoint = StreamCheckpoint(
            message_id=message_id,
            content_blocks=content_blocks.copy(),
            tool_calls=tool_calls.copy(),
            last_event_index=event_index,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            timestamp=time.time()
        )

        # Store checkpoint
        self.checkpoints[message_id] = checkpoint

        # Clean up old checkpoints
        self._cleanup_old_checkpoints()

        return checkpoint

    def get_checkpoint(self, message_id: str) -> Optional[StreamCheckpoint]:
        """Get checkpoint for recovery"""
        return self.checkpoints.get(message_id)

    def remove_checkpoint(self, message_id: str):
        """Remove checkpoint after successful completion"""
        self.checkpoints.pop(message_id, None)

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to prevent memory leak"""
        import time

        if len(self.checkpoints) <= self.max_checkpoints:
            return

        # Remove checkpoints older than 5 minutes
        cutoff = time.time() - 300
        old_keys = [
            k for k, v in self.checkpoints.items()
            if v.timestamp < cutoff
        ]

        for key in old_keys:
            del self.checkpoints[key]

    def build_resume_response(
        self,
        checkpoint: StreamCheckpoint
    ) -> Dict[str, Any]:
        """Build partial response from checkpoint for resume"""
        content = []

        # Add accumulated text blocks
        for block in checkpoint.content_blocks:
            if block.get("type") == "text":
                content.append(block)

        # Add tool calls
        for tc in checkpoint.tool_calls:
            content.append({
                "type": "tool_use",
                "id": tc.get("id"),
                "name": tc.get("name"),
                "input": tc.get("input", {})
            })

        return {
            "id": checkpoint.message_id,
            "type": "message",
            "role": "assistant",
            "content": content,
            "stop_reason": None,  # Incomplete
            "usage": {
                "input_tokens": checkpoint.input_tokens,
                "output_tokens": checkpoint.output_tokens
            },
            "_checkpoint": {
                "last_event_index": checkpoint.last_event_index,
                "resumable": True
            }
        }


# =============================================================================
# Global Instances
# =============================================================================

_vision_handler: Optional[VisionHandler] = None
_thinking_handler: Optional[ThinkingHandler] = None
_document_handler: Optional[DocumentHandler] = None
_token_counter: Optional[TokenCounter] = None
_preflight_checker: Optional[TokenPreflightChecker] = None
_metadata_handler: Optional[MetadataHandler] = None
_stop_handler: Optional[StopSequencesHandler] = None
_stream_recovery: Optional[StreamRecoveryHandler] = None


def get_vision_handler() -> VisionHandler:
    global _vision_handler
    if _vision_handler is None:
        _vision_handler = VisionHandler()
    return _vision_handler


def get_thinking_handler() -> ThinkingHandler:
    global _thinking_handler
    if _thinking_handler is None:
        _thinking_handler = ThinkingHandler()
    return _thinking_handler


def get_document_handler() -> DocumentHandler:
    global _document_handler
    if _document_handler is None:
        _document_handler = DocumentHandler()
    return _document_handler


def get_token_counter() -> TokenCounter:
    global _token_counter
    if _token_counter is None:
        _token_counter = TokenCounter()
    return _token_counter


def get_preflight_checker() -> TokenPreflightChecker:
    global _preflight_checker
    if _preflight_checker is None:
        _preflight_checker = TokenPreflightChecker()
    return _preflight_checker


def get_metadata_handler() -> MetadataHandler:
    global _metadata_handler
    if _metadata_handler is None:
        _metadata_handler = MetadataHandler()
    return _metadata_handler


def get_stop_handler() -> StopSequencesHandler:
    global _stop_handler
    if _stop_handler is None:
        _stop_handler = StopSequencesHandler()
    return _stop_handler


def get_stream_recovery() -> StreamRecoveryHandler:
    global _stream_recovery
    if _stream_recovery is None:
        _stream_recovery = StreamRecoveryHandler()
    return _stream_recovery
