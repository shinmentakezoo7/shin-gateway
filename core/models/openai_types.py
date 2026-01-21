"""
OpenAI Chat Completions API Pydantic Models

Complete type definitions for OpenAI's Chat Completions API protocol.
Used for translating to/from Anthropic format.
"""

from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Literal, Optional, Union, Any


# =============================================================================
# Function/Tool Types
# =============================================================================

class FunctionDef(BaseModel):
    """Function definition for tools"""
    name: str
    description: Optional[str] = None
    parameters: Optional[dict] = None
    strict: Optional[bool] = None


class ToolDef(BaseModel):
    """Tool definition"""
    type: Literal["function"] = "function"
    function: FunctionDef


class FunctionCall(BaseModel):
    """Function call in a tool call"""
    name: str
    arguments: str  # JSON string


class ToolCall(BaseModel):
    """Tool call from assistant"""
    id: str
    type: Literal["function"] = "function"
    function: FunctionCall
    index: Optional[int] = None  # For streaming


# =============================================================================
# Message Content Types
# =============================================================================

class TextContent(BaseModel):
    """Text content part"""
    type: Literal["text"] = "text"
    text: str


class ImageURLDetail(BaseModel):
    """Image URL details"""
    url: str
    detail: Optional[Literal["auto", "low", "high"]] = "auto"


class ImageURLContent(BaseModel):
    """Image URL content part"""
    type: Literal["image_url"] = "image_url"
    image_url: ImageURLDetail


ContentPart = Union[TextContent, ImageURLContent, dict]


# =============================================================================
# Message Types
# =============================================================================

class SystemMessage(BaseModel):
    """System message"""
    role: Literal["system"] = "system"
    content: Union[str, list[ContentPart]]
    name: Optional[str] = None


class UserMessage(BaseModel):
    """User message"""
    role: Literal["user"] = "user"
    content: Union[str, list[ContentPart]]
    name: Optional[str] = None


class AssistantMessage(BaseModel):
    """Assistant message"""
    role: Literal["assistant"] = "assistant"
    content: Optional[Union[str, list[ContentPart]]] = None
    name: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None
    function_call: Optional[dict] = None  # Deprecated but still used


class ToolMessage(BaseModel):
    """Tool result message"""
    role: Literal["tool"] = "tool"
    tool_call_id: str
    content: str


class FunctionMessage(BaseModel):
    """Function result message (deprecated)"""
    role: Literal["function"] = "function"
    name: str
    content: str


OpenAIMessage = Union[
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ToolMessage,
    FunctionMessage,
    dict  # Allow raw dict for flexibility
]


# =============================================================================
# Tool Choice Types
# =============================================================================

class ToolChoiceFunction(BaseModel):
    """Specific function in tool choice"""
    name: str


class ToolChoiceObject(BaseModel):
    """Object form of tool choice"""
    type: Literal["function"] = "function"
    function: ToolChoiceFunction


# Tool choice can be string or object
OpenAIToolChoice = Union[
    Literal["none", "auto", "required"],
    ToolChoiceObject,
    dict
]


# =============================================================================
# Request Model
# =============================================================================

class ChatCompletionsRequest(BaseModel):
    """OpenAI Chat Completions Request"""
    model: str
    messages: list[OpenAIMessage]
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None  # New name
    temperature: Optional[float] = Field(default=None, ge=0, le=2)
    top_p: Optional[float] = Field(default=None, ge=0, le=1)
    n: Optional[int] = Field(default=1, ge=1)
    stream: Optional[bool] = False
    stream_options: Optional[dict] = None
    stop: Optional[Union[str, list[str]]] = None
    presence_penalty: Optional[float] = Field(default=None, ge=-2, le=2)
    frequency_penalty: Optional[float] = Field(default=None, ge=-2, le=2)
    logit_bias: Optional[dict[str, float]] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    user: Optional[str] = None
    tools: Optional[list[ToolDef]] = None
    tool_choice: Optional[OpenAIToolChoice] = None
    parallel_tool_calls: Optional[bool] = None
    response_format: Optional[dict] = None
    seed: Optional[int] = None

    class Config:
        extra = "allow"


# =============================================================================
# Response Models
# =============================================================================

class ChatMessage(BaseModel):
    """Message in response"""
    role: Literal["assistant"] = "assistant"
    content: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None
    function_call: Optional[dict] = None
    refusal: Optional[str] = None


class TopLogprob(BaseModel):
    """Top logprob entry"""
    token: str
    logprob: float
    bytes: Optional[list[int]] = None


class LogprobContent(BaseModel):
    """Logprob content entry"""
    token: str
    logprob: float
    bytes: Optional[list[int]] = None
    top_logprobs: Optional[list[TopLogprob]] = None


class Logprobs(BaseModel):
    """Logprobs container"""
    content: Optional[list[LogprobContent]] = None
    refusal: Optional[list[LogprobContent]] = None


class Choice(BaseModel):
    """Response choice"""
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None
    logprobs: Optional[Logprobs] = None


class OpenAIUsage(BaseModel):
    """Token usage statistics"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: Optional[dict] = None
    completion_tokens_details: Optional[dict] = None


class ChatCompletionsResponse(BaseModel):
    """OpenAI Chat Completions Response"""
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Optional[OpenAIUsage] = None
    system_fingerprint: Optional[str] = None
    service_tier: Optional[str] = None


# =============================================================================
# Streaming Response Models
# =============================================================================

class DeltaToolCall(BaseModel):
    """Tool call delta in streaming"""
    index: int
    id: Optional[str] = None
    type: Optional[Literal["function"]] = None
    function: Optional[dict] = None  # {name?: str, arguments?: str}


class DeltaContent(BaseModel):
    """Delta content in streaming"""
    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[list[DeltaToolCall]] = None
    function_call: Optional[dict] = None
    refusal: Optional[str] = None


class StreamChoice(BaseModel):
    """Choice in streaming response"""
    index: int
    delta: DeltaContent
    finish_reason: Optional[str] = None
    logprobs: Optional[Logprobs] = None


class ChatCompletionChunk(BaseModel):
    """Streaming chunk"""
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[StreamChoice]
    usage: Optional[OpenAIUsage] = None  # Only in final chunk with stream_options
    system_fingerprint: Optional[str] = None
    service_tier: Optional[str] = None


# =============================================================================
# Error Models
# =============================================================================

class OpenAIError(BaseModel):
    """OpenAI error detail"""
    message: str
    type: str
    param: Optional[str] = None
    code: Optional[str] = None


class OpenAIErrorResponse(BaseModel):
    """OpenAI error response"""
    error: OpenAIError
