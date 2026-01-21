"""
Anthropic Messages API Pydantic Models

Complete type definitions for Anthropic's Messages API protocol.
Used for request validation and response serialization.
"""

from __future__ import annotations
from pydantic import BaseModel, Field, field_validator
from typing import Literal, Optional, Union, Any
from enum import Enum
import uuid


# =============================================================================
# Content Block Types
# =============================================================================

class TextBlock(BaseModel):
    """Text content block"""
    type: Literal["text"] = "text"
    text: str
    cache_control: Optional[dict] = None


class ImageSource(BaseModel):
    """Image source (base64 or URL)"""
    type: Literal["base64", "url"]
    media_type: Optional[str] = None
    data: Optional[str] = None
    url: Optional[str] = None


class ImageBlock(BaseModel):
    """Image content block"""
    type: Literal["image"] = "image"
    source: ImageSource
    cache_control: Optional[dict] = None


class ToolUseBlock(BaseModel):
    """Tool use block (assistant requesting tool execution)"""
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict


class ToolResultContent(BaseModel):
    """Content within a tool result"""
    type: Literal["text", "image"] = "text"
    text: Optional[str] = None
    source: Optional[ImageSource] = None


class ToolResultBlock(BaseModel):
    """Tool result block (user providing tool output)"""
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: Union[str, list[Union[ToolResultContent, dict]]]
    is_error: Optional[bool] = False
    cache_control: Optional[dict] = None


class ThinkingBlock(BaseModel):
    """Thinking block (chain-of-thought)"""
    type: Literal["thinking"] = "thinking"
    thinking: str


class RedactedThinkingBlock(BaseModel):
    """Redacted thinking block"""
    type: Literal["redacted_thinking"] = "redacted_thinking"
    data: str


# Union of all content block types
ContentBlock = Union[
    TextBlock,
    ImageBlock,
    ToolUseBlock,
    ToolResultBlock,
    ThinkingBlock,
    RedactedThinkingBlock,
    dict  # Allow unknown block types for forward compatibility
]


# =============================================================================
# Message Types
# =============================================================================

class Message(BaseModel):
    """A single message in the conversation"""
    role: Literal["user", "assistant"]
    content: Union[str, list[ContentBlock]]

    @field_validator("content", mode="before")
    @classmethod
    def normalize_content(cls, v):
        """Normalize string content to list format internally"""
        if isinstance(v, str):
            return v  # Keep as string, adapter will handle
        return v


# =============================================================================
# Tool Definitions
# =============================================================================

class ToolInputSchema(BaseModel):
    """JSON Schema for tool input"""
    type: Literal["object"] = "object"
    properties: dict = Field(default_factory=dict)
    required: list[str] = Field(default_factory=list)
    additionalProperties: Optional[bool] = None

    class Config:
        extra = "allow"  # Allow additional schema fields


class Tool(BaseModel):
    """Tool definition"""
    name: str
    description: Optional[str] = None
    input_schema: ToolInputSchema
    cache_control: Optional[dict] = None


# =============================================================================
# Tool Choice
# =============================================================================

class ToolChoiceAuto(BaseModel):
    """Auto tool choice - model decides"""
    type: Literal["auto"] = "auto"
    disable_parallel_tool_use: Optional[bool] = None


class ToolChoiceAny(BaseModel):
    """Any tool choice - model must use a tool"""
    type: Literal["any"] = "any"
    disable_parallel_tool_use: Optional[bool] = None


class ToolChoiceTool(BaseModel):
    """Specific tool choice - model must use this tool"""
    type: Literal["tool"] = "tool"
    name: str
    disable_parallel_tool_use: Optional[bool] = None


class ToolChoiceNone(BaseModel):
    """No tool choice - model cannot use tools"""
    type: Literal["none"] = "none"


ToolChoice = Union[ToolChoiceAuto, ToolChoiceAny, ToolChoiceTool, ToolChoiceNone]


# =============================================================================
# Thinking Configuration
# =============================================================================

class ThinkingConfig(BaseModel):
    """Extended thinking configuration"""
    type: Literal["enabled", "disabled"] = "enabled"
    budget_tokens: Optional[int] = None


# =============================================================================
# Metadata
# =============================================================================

class RequestMetadata(BaseModel):
    """Request metadata"""
    user_id: Optional[str] = None

    class Config:
        extra = "allow"


# =============================================================================
# Request Model
# =============================================================================

class MessagesRequest(BaseModel):
    """Anthropic Messages API Request"""
    model: str
    messages: list[Message]
    max_tokens: int
    system: Optional[Union[str, list[dict]]] = None
    temperature: Optional[float] = Field(default=None, ge=0, le=1)
    top_p: Optional[float] = Field(default=None, ge=0, le=1)
    top_k: Optional[int] = Field(default=None, ge=0)
    stop_sequences: Optional[list[str]] = None
    stream: Optional[bool] = False
    tools: Optional[list[Tool]] = None
    tool_choice: Optional[ToolChoice] = None
    metadata: Optional[RequestMetadata] = None
    thinking: Optional[ThinkingConfig] = None

    class Config:
        extra = "allow"  # Allow unknown fields for forward compatibility


# =============================================================================
# Response Models
# =============================================================================

class Usage(BaseModel):
    """Token usage statistics"""
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: Optional[int] = None
    cache_read_input_tokens: Optional[int] = None


class MessagesResponse(BaseModel):
    """Anthropic Messages API Response"""
    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    model: str
    content: list[ContentBlock]
    stop_reason: Optional[str] = None
    stop_sequence: Optional[str] = None
    usage: Usage

    @classmethod
    def create(
        cls,
        content: list[ContentBlock],
        model: str,
        usage: Usage,
        stop_reason: str = "end_turn",
        stop_sequence: Optional[str] = None,
        message_id: Optional[str] = None
    ) -> "MessagesResponse":
        """Factory method to create a response"""
        return cls(
            id=message_id or f"msg_{uuid.uuid4().hex[:24]}",
            model=model,
            content=content,
            stop_reason=stop_reason,
            stop_sequence=stop_sequence,
            usage=usage
        )


# =============================================================================
# Error Models
# =============================================================================

class ErrorDetail(BaseModel):
    """Error detail"""
    type: str
    message: str


class ErrorResponse(BaseModel):
    """Anthropic error response"""
    type: Literal["error"] = "error"
    error: ErrorDetail

    @classmethod
    def create(cls, error_type: str, message: str) -> "ErrorResponse":
        """Factory method to create an error response"""
        return cls(error=ErrorDetail(type=error_type, message=message))


# =============================================================================
# Streaming Event Models
# =============================================================================

class MessageStartEvent(BaseModel):
    """message_start event"""
    type: Literal["message_start"] = "message_start"
    message: dict


class ContentBlockStartEvent(BaseModel):
    """content_block_start event"""
    type: Literal["content_block_start"] = "content_block_start"
    index: int
    content_block: dict


class ContentBlockDeltaEvent(BaseModel):
    """content_block_delta event"""
    type: Literal["content_block_delta"] = "content_block_delta"
    index: int
    delta: dict


class ContentBlockStopEvent(BaseModel):
    """content_block_stop event"""
    type: Literal["content_block_stop"] = "content_block_stop"
    index: int


class MessageDeltaEvent(BaseModel):
    """message_delta event"""
    type: Literal["message_delta"] = "message_delta"
    delta: dict
    usage: Optional[dict] = None


class MessageStopEvent(BaseModel):
    """message_stop event"""
    type: Literal["message_stop"] = "message_stop"


class PingEvent(BaseModel):
    """ping event for keep-alive"""
    type: Literal["ping"] = "ping"


class ErrorEvent(BaseModel):
    """error event"""
    type: Literal["error"] = "error"
    error: ErrorDetail


StreamEvent = Union[
    MessageStartEvent,
    ContentBlockStartEvent,
    ContentBlockDeltaEvent,
    ContentBlockStopEvent,
    MessageDeltaEvent,
    MessageStopEvent,
    PingEvent,
    ErrorEvent
]
