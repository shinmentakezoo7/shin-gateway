"""
Token Counter and Estimator

Provides token estimation when upstream providers don't return usage info.
Uses character-based estimation with model-specific adjustments.
"""

from __future__ import annotations
import re
import json
import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass


logger = logging.getLogger("shin-gateway")


# =============================================================================
# Token Estimation Constants
# =============================================================================

# Average characters per token for different content types
CHARS_PER_TOKEN = {
    "english": 4.0,
    "code": 3.5,
    "json": 3.0,
    "mixed": 3.75,
}

# Overhead tokens for various structures
OVERHEAD = {
    "message_base": 4,      # Base tokens per message
    "tool_call": 10,        # Overhead per tool call
    "tool_definition": 20,  # Overhead per tool definition
    "image_base": 85,       # Base tokens for image reference
    "system_prompt": 4,     # Overhead for system prompt
}


# =============================================================================
# Content Type Detection
# =============================================================================

def detect_content_type(text: str) -> str:
    """
    Detect content type for better token estimation.

    Args:
        text: Text to analyze

    Returns:
        Content type: 'code', 'json', 'english', or 'mixed'
    """
    if not text:
        return "english"

    # Check for JSON
    try:
        json.loads(text)
        return "json"
    except (json.JSONDecodeError, TypeError):
        pass

    # Check for code patterns
    code_patterns = [
        r'^\s*(def|class|function|const|let|var|import|from|if|for|while)\s',
        r'[{}\[\]();]',
        r'=>|->|::|\.\.|\+=|-=|\*=|/=',
        r'^\s*#.*$',  # Comments
        r'^\s*//.*$',
    ]

    code_matches = sum(
        1 for pattern in code_patterns
        if re.search(pattern, text, re.MULTILINE)
    )

    if code_matches >= 2:
        return "code"

    # Check character distribution for mixed content
    alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
    if alpha_ratio < 0.5:
        return "mixed"

    return "english"


# =============================================================================
# Token Counter Class
# =============================================================================

@dataclass
class TokenCount:
    """Token count result"""
    input_tokens: int
    output_tokens: int = 0
    cached_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


class TokenCounter:
    """
    Estimate token counts for messages and requests.

    Uses character-based estimation with adjustments for content type.
    """

    def __init__(
        self,
        default_chars_per_token: float = 4.0,
        auto_detect_content: bool = True
    ):
        self.default_chars_per_token = default_chars_per_token
        self.auto_detect = auto_detect_content

    def count_text(self, text: str, content_type: Optional[str] = None) -> int:
        """
        Estimate tokens in text.

        Args:
            text: Text to count
            content_type: Override content type detection

        Returns:
            Estimated token count
        """
        if not text:
            return 0

        if self.auto_detect and content_type is None:
            content_type = detect_content_type(text)

        chars_per_token = CHARS_PER_TOKEN.get(content_type, self.default_chars_per_token)
        return max(1, int(len(text) / chars_per_token))

    def count_content_block(self, block: Union[Dict[str, Any], str]) -> int:
        """
        Count tokens in a content block.

        Args:
            block: Content block (text, image, tool_use, tool_result)

        Returns:
            Estimated token count
        """
        if isinstance(block, str):
            return self.count_text(block)

        if not isinstance(block, dict):
            return 0

        block_type = block.get("type", "text")

        if block_type == "text":
            return self.count_text(block.get("text", ""))

        elif block_type == "image":
            # Images have base cost plus dimensions if available
            source = block.get("source", {})
            if source.get("type") == "base64":
                # Estimate based on data size
                data = source.get("data", "")
                # ~750 bytes per image token (rough estimate)
                return max(OVERHEAD["image_base"], len(data) // 750)
            return OVERHEAD["image_base"]

        elif block_type == "tool_use":
            tokens = OVERHEAD["tool_call"]
            tokens += self.count_text(block.get("name", ""))
            input_data = block.get("input", {})
            if isinstance(input_data, dict):
                tokens += self.count_text(json.dumps(input_data), "json")
            else:
                tokens += self.count_text(str(input_data))
            return tokens

        elif block_type == "tool_result":
            tokens = 4  # Base overhead
            content = block.get("content", "")
            if isinstance(content, list):
                for item in content:
                    tokens += self.count_content_block(item)
            else:
                tokens += self.count_text(str(content))
            return tokens

        elif block_type in ("thinking", "redacted_thinking"):
            return self.count_text(block.get("thinking", block.get("data", "")))

        return 0

    def count_message(self, message: Dict[str, Any]) -> int:
        """
        Count tokens in a message.

        Args:
            message: Message with role and content

        Returns:
            Estimated token count
        """
        tokens = OVERHEAD["message_base"]

        content = message.get("content")
        if isinstance(content, str):
            tokens += self.count_text(content)
        elif isinstance(content, list):
            for block in content:
                tokens += self.count_content_block(block)

        # Count tool calls (OpenAI format)
        tool_calls = message.get("tool_calls", [])
        for tc in tool_calls:
            tokens += OVERHEAD["tool_call"]
            if isinstance(tc, dict):
                func = tc.get("function", {})
                tokens += self.count_text(func.get("name", ""))
                tokens += self.count_text(func.get("arguments", ""), "json")

        return tokens

    def count_messages(self, messages: List[Dict[str, Any]]) -> int:
        """
        Count tokens in a list of messages.

        Args:
            messages: List of messages

        Returns:
            Estimated token count
        """
        total = 3  # Base conversation overhead
        for msg in messages:
            total += self.count_message(msg)
        return total

    def count_tools(self, tools: List[Dict[str, Any]]) -> int:
        """
        Count tokens in tool definitions.

        Args:
            tools: List of tool definitions

        Returns:
            Estimated token count
        """
        total = 0
        for tool in tools:
            total += OVERHEAD["tool_definition"]
            total += self.count_text(tool.get("name", ""))
            total += self.count_text(tool.get("description", ""))

            # Count input schema
            schema = tool.get("input_schema", tool.get("parameters", {}))
            if schema:
                total += self.count_text(json.dumps(schema), "json")

        return total

    def count_system(self, system: Union[str, List[Dict], None]) -> int:
        """
        Count tokens in system prompt.

        Args:
            system: System prompt (string or list of blocks)

        Returns:
            Estimated token count
        """
        if not system:
            return 0

        tokens = OVERHEAD["system_prompt"]

        if isinstance(system, str):
            tokens += self.count_text(system)
        elif isinstance(system, list):
            for block in system:
                tokens += self.count_content_block(block)

        return tokens

    def count_request(self, request: Dict[str, Any]) -> TokenCount:
        """
        Estimate total input tokens for a request.

        Args:
            request: Full request object

        Returns:
            TokenCount with input_tokens
        """
        tokens = 0

        # System prompt
        tokens += self.count_system(request.get("system"))

        # Messages
        messages = request.get("messages", [])
        tokens += self.count_messages(messages)

        # Tools
        tools = request.get("tools", [])
        if tools:
            tokens += self.count_tools(tools)

        return TokenCount(input_tokens=tokens)

    def count_response(
        self,
        content: Union[str, List[Dict[str, Any]]],
        tool_calls: Optional[List[Dict]] = None
    ) -> int:
        """
        Estimate output tokens for a response.

        Args:
            content: Response content (text or blocks)
            tool_calls: Optional tool calls in response

        Returns:
            Estimated output token count
        """
        tokens = 0

        if isinstance(content, str):
            tokens += self.count_text(content)
        elif isinstance(content, list):
            for block in content:
                tokens += self.count_content_block(block)

        if tool_calls:
            for tc in tool_calls:
                tokens += OVERHEAD["tool_call"]
                if isinstance(tc, dict):
                    func = tc.get("function", {})
                    tokens += self.count_text(func.get("name", ""))
                    tokens += self.count_text(func.get("arguments", ""), "json")

        return tokens


# =============================================================================
# Streaming Token Counter
# =============================================================================

class StreamingTokenCounter:
    """
    Track tokens during streaming.

    Accumulates text chunks and provides running token estimate.
    """

    def __init__(self, token_counter: Optional[TokenCounter] = None):
        self.counter = token_counter or TokenCounter()
        self.text_buffer = ""
        self.tool_buffers: Dict[str, str] = {}
        self._text_tokens = 0
        self._tool_tokens = 0

    def add_text(self, text: str) -> int:
        """
        Add text chunk and return incremental token count.

        Args:
            text: Text chunk

        Returns:
            Estimated tokens for this chunk
        """
        if not text:
            return 0

        self.text_buffer += text
        new_tokens = self.counter.count_text(text)
        self._text_tokens += new_tokens
        return new_tokens

    def add_tool_arguments(self, tool_id: str, arguments: str) -> int:
        """
        Add tool call arguments and return incremental token count.

        Args:
            tool_id: Tool call ID
            arguments: JSON arguments chunk

        Returns:
            Estimated tokens for this chunk
        """
        if not arguments:
            return 0

        if tool_id not in self.tool_buffers:
            self.tool_buffers[tool_id] = ""
            self._tool_tokens += OVERHEAD["tool_call"]

        self.tool_buffers[tool_id] += arguments
        new_tokens = self.counter.count_text(arguments, "json")
        self._tool_tokens += new_tokens
        return new_tokens

    @property
    def total_tokens(self) -> int:
        """Get total estimated output tokens"""
        return self._text_tokens + self._tool_tokens

    def reset(self) -> None:
        """Reset counters"""
        self.text_buffer = ""
        self.tool_buffers.clear()
        self._text_tokens = 0
        self._tool_tokens = 0


# =============================================================================
# Global Instance
# =============================================================================

# Singleton instance
_token_counter: Optional[TokenCounter] = None


def get_token_counter() -> TokenCounter:
    """Get or create global token counter"""
    global _token_counter
    if _token_counter is None:
        _token_counter = TokenCounter()
    return _token_counter


def count_tokens(request: Dict[str, Any]) -> TokenCount:
    """Convenience function to count request tokens"""
    return get_token_counter().count_request(request)
