# SHIN Gateway MVP – Protocol Adapters Plan
**Version:** 1.1.0
**Updated:** January 2026

---

## Goal
Build a local-first proxy that accepts **Anthropic Messages** requests at `POST /v1/messages` and can route to **OpenAI-compatible Chat Completions** upstreams (Ollama/OpenAI-compatible servers) and hosted providers (Groq/OpenAI) via `litellm`, then returns **Anthropic-shaped** responses (streaming + non-streaming).

---

## Protocols We Support (MVP)

### Inbound (Northbound) – Anthropic Messages API
- **Endpoint:** `POST /v1/messages`
- **Supports:** `stream: true|false`, `system`, `messages[]`, `tools[]`, `tool_choice`

### Outbound (Southbound) – OpenAI Chat Completions
- **Target Endpoint:** `POST /v1/chat/completions`
- **Supports:** `stream: true|false`, `messages[]`, `tools[]`, `tool_choice`

---

## Adapter/Translator Architecture (Detailed)

### Directory Structure for Adapters

```text
shin-gateway/
├── core/
│   ├── adapters/
│   │   ├── __init__.py
│   │   ├── base.py                      # Abstract base classes
│   │   ├── anthropic_to_openai.py       # Request translator
│   │   ├── openai_to_anthropic.py       # Response translator (non-stream)
│   │   ├── stream_translator.py         # Streaming event translator
│   │   └── errors.py                    # Error envelope adapter
│   ├── models/
│   │   ├── __init__.py
│   │   ├── anthropic_types.py           # Pydantic models for Anthropic protocol
│   │   └── openai_types.py              # Pydantic models for OpenAI protocol
│   └── ...
└── ...
```

---

## Adapter 1: Anthropic → OpenAI Request Translator

**File:** `core/adapters/anthropic_to_openai.py`

### Purpose
Converts incoming Anthropic `MessagesRequest` into OpenAI `ChatCompletionsRequest` format.

### Function Signature
```python
def translate_request(
    anthropic_request: AnthropicMessagesRequest,
    resolved_model: str,
    provider_config: ProviderConfig
) -> OpenAIChatCompletionsRequest:
    ...
```

### Field-by-Field Mapping Table

| Anthropic Field | OpenAI Field | Transformation Logic |
|-----------------|--------------|----------------------|
| `model` | `model` | Resolved via alias mapping from config |
| `system` | `messages[0]` | Insert as `{"role": "system", "content": system}` |
| `messages[]` | `messages[]` | Transform each message (see below) |
| `max_tokens` | `max_tokens` | Direct copy |
| `temperature` | `temperature` | Direct copy (default: 1.0) |
| `top_p` | `top_p` | Direct copy |
| `top_k` | - | **Not supported** in OpenAI, drop silently |
| `stop_sequences` | `stop` | Direct copy as array |
| `stream` | `stream` | Direct copy |
| `tools[]` | `tools[]` | Transform schema (see below) |
| `tool_choice` | `tool_choice` | Map values (see below) |
| `metadata` | - | Drop (not used by OpenAI) |

### Message Content Block Transformations

#### Text Content
```python
# Anthropic
{"role": "user", "content": [{"type": "text", "text": "Hello"}]}

# OpenAI
{"role": "user", "content": "Hello"}
```

#### Multiple Text Blocks (Concatenate)
```python
# Anthropic
{"role": "user", "content": [
    {"type": "text", "text": "First part. "},
    {"type": "text", "text": "Second part."}
]}

# OpenAI
{"role": "user", "content": "First part. Second part."}
```

#### Image Content (Vision)
```python
# Anthropic
{"role": "user", "content": [
    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "..."}}
]}

# OpenAI
{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
]}
```

#### Tool Use (Assistant Response)
```python
# Anthropic
{"role": "assistant", "content": [
    {"type": "tool_use", "id": "toolu_01abc", "name": "get_weather", "input": {"city": "Paris"}}
]}

# OpenAI
{
    "role": "assistant",
    "content": null,
    "tool_calls": [
        {"id": "toolu_01abc", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}
    ]
}
```

#### Tool Result (User Provides)
```python
# Anthropic
{"role": "user", "content": [
    {"type": "tool_result", "tool_use_id": "toolu_01abc", "content": "Sunny, 22C"}
]}

# OpenAI
{"role": "tool", "tool_call_id": "toolu_01abc", "content": "Sunny, 22C"}
```

### Tools Schema Transformation

```python
# Anthropic Tool
{
    "name": "get_weather",
    "description": "Get weather for a city",
    "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"]
    }
}

# OpenAI Tool
{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        }
    }
}
```

### Tool Choice Mapping

| Anthropic `tool_choice` | OpenAI `tool_choice` |
|-------------------------|----------------------|
| `{"type": "auto"}` | `"auto"` |
| `{"type": "any"}` | `"required"` |
| `{"type": "tool", "name": "X"}` | `{"type": "function", "function": {"name": "X"}}` |
| `{"type": "none"}` or omitted | `"none"` or omit |

---

## Adapter 2: OpenAI → Anthropic Response Translator (Non-Streaming)

**File:** `core/adapters/openai_to_anthropic.py`

### Purpose
Converts OpenAI `ChatCompletionsResponse` into Anthropic `MessagesResponse` format.

### Function Signature
```python
def translate_response(
    openai_response: OpenAIChatCompletionsResponse,
    original_request: AnthropicMessagesRequest
) -> AnthropicMessagesResponse:
    ...
```

### Field-by-Field Mapping Table

| OpenAI Field | Anthropic Field | Transformation Logic |
|--------------|-----------------|----------------------|
| `id` | `id` | Prefix with `msg_` if needed |
| `model` | `model` | Return original alias or actual model |
| `choices[0].message.content` | `content[]` | Wrap as `[{"type": "text", "text": ...}]` |
| `choices[0].message.tool_calls` | `content[]` | Append `tool_use` blocks |
| `choices[0].finish_reason` | `stop_reason` | Map values (see below) |
| `usage.prompt_tokens` | `usage.input_tokens` | Direct copy |
| `usage.completion_tokens` | `usage.output_tokens` | Direct copy |
| - | `type` | Always `"message"` |
| - | `role` | Always `"assistant"` |

### Finish Reason Mapping

| OpenAI `finish_reason` | Anthropic `stop_reason` |
|------------------------|-------------------------|
| `"stop"` | `"end_turn"` |
| `"length"` | `"max_tokens"` |
| `"tool_calls"` | `"tool_use"` |
| `"content_filter"` | `"end_turn"` (with warning log) |
| `null` / other | `"end_turn"` |

### Response Structure Example

```python
# OpenAI Response
{
    "id": "chatcmpl-123",
    "model": "gpt-4o-mini",
    "choices": [{
        "index": 0,
        "message": {
            "role": "assistant",
            "content": "Hello! How can I help?",
            "tool_calls": null
        },
        "finish_reason": "stop"
    }],
    "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18}
}

# Anthropic Response
{
    "id": "msg_chatcmpl-123",
    "type": "message",
    "role": "assistant",
    "model": "shin-coder",
    "content": [{"type": "text", "text": "Hello! How can I help?"}],
    "stop_reason": "end_turn",
    "stop_sequence": null,
    "usage": {"input_tokens": 10, "output_tokens": 8}
}
```

### Tool Calls Response Example

```python
# OpenAI Response with Tool Calls
{
    "id": "chatcmpl-456",
    "choices": [{
        "message": {
            "role": "assistant",
            "content": null,
            "tool_calls": [
                {"id": "call_abc", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}}
            ]
        },
        "finish_reason": "tool_calls"
    }],
    "usage": {...}
}

# Anthropic Response
{
    "id": "msg_chatcmpl-456",
    "type": "message",
    "role": "assistant",
    "model": "shin-coder",
    "content": [
        {"type": "tool_use", "id": "call_abc", "name": "get_weather", "input": {"city": "Paris"}}
    ],
    "stop_reason": "tool_use",
    "usage": {...}
}
```

---

## Adapter 3: Streaming Translator (SSE Events)

**File:** `core/adapters/stream_translator.py`

### Purpose
Translates OpenAI streaming chunks into Anthropic SSE event format in real-time.

### Function Signature
```python
async def translate_stream(
    openai_stream: AsyncIterator[OpenAIStreamChunk],
    original_request: AnthropicMessagesRequest
) -> AsyncIterator[AnthropicStreamEvent]:
    ...
```

### Anthropic Stream Event Types

| Event Type | When Emitted | Purpose |
|------------|--------------|---------|
| `message_start` | First chunk | Initialize message with metadata |
| `content_block_start` | New content block begins | Signal text or tool_use block |
| `content_block_delta` | Content arrives | Stream partial text or tool input |
| `content_block_stop` | Block complete | Finalize current block |
| `message_delta` | Final chunk | Send stop_reason and final usage |
| `message_stop` | Stream ends | Terminal event |

### Stream Translation State Machine

```
┌─────────────────────────────────────────────────────────────────┐
│                     STREAM TRANSLATOR STATE                      │
├─────────────────────────────────────────────────────────────────┤
│  State Variables:                                                │
│  - message_id: str (generated once)                              │
│  - current_block_index: int (starts at 0)                        │
│  - current_block_type: "text" | "tool_use" | None                │
│  - tool_call_buffers: Dict[str, ToolCallBuffer]                  │
│  - text_started: bool                                            │
└─────────────────────────────────────────────────────────────────┘

OpenAI Chunk                    │ Anthropic Event(s)
───────────────────────────────┼────────────────────────────────────
First chunk arrives            │ → message_start
                               │
delta.content = "Hello"        │ → content_block_start (type=text, if not started)
(text begins)                  │ → content_block_delta (text delta)
                               │
delta.content = " world"       │ → content_block_delta (text delta)
                               │
delta.tool_calls[0] starts     │ → content_block_stop (close text if open)
                               │ → content_block_start (type=tool_use)
                               │
delta.tool_calls[0].arguments  │ → content_block_delta (partial_json)
                               │
finish_reason = "stop"         │ → content_block_stop
                               │ → message_delta (stop_reason, usage)
                               │ → message_stop
```

### SSE Event Format Examples

#### message_start
```
event: message_start
data: {"type": "message_start", "message": {"id": "msg_01XYZ", "type": "message", "role": "assistant", "model": "shin-coder", "content": [], "stop_reason": null, "stop_sequence": null, "usage": {"input_tokens": 25, "output_tokens": 0}}}
```

#### content_block_start (text)
```
event: content_block_start
data: {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}
```

#### content_block_delta (text)
```
event: content_block_delta
data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}}
```

#### content_block_start (tool_use)
```
event: content_block_start
data: {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "toolu_01abc", "name": "get_weather", "input": {}}}
```

#### content_block_delta (tool input)
```
event: content_block_delta
data: {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{\"city\":"}}
```

#### content_block_stop
```
event: content_block_stop
data: {"type": "content_block_stop", "index": 0}
```

#### message_delta
```
event: message_delta
data: {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": null}, "usage": {"output_tokens": 42}}
```

#### message_stop
```
event: message_stop
data: {"type": "message_stop"}
```

### Streaming Implementation Pseudocode

```python
async def translate_stream(openai_stream, request):
    state = StreamState(
        message_id=generate_message_id(),
        model=request.model,
        input_tokens=estimate_input_tokens(request)
    )

    # Emit message_start
    yield create_message_start_event(state)

    async for chunk in openai_stream:
        delta = chunk.choices[0].delta if chunk.choices else None
        finish_reason = chunk.choices[0].finish_reason if chunk.choices else None

        if delta:
            # Handle text content
            if delta.content:
                if not state.text_block_started:
                    yield create_content_block_start(state.block_index, "text")
                    state.text_block_started = True
                yield create_text_delta(state.block_index, delta.content)

            # Handle tool calls
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    tc_index = tc.index
                    if tc_index not in state.tool_buffers:
                        # Close text block if open
                        if state.text_block_started:
                            yield create_content_block_stop(state.block_index)
                            state.block_index += 1
                            state.text_block_started = False

                        # Start new tool_use block
                        state.tool_buffers[tc_index] = ToolBuffer(
                            id=tc.id, name=tc.function.name, args=""
                        )
                        yield create_content_block_start(
                            state.block_index, "tool_use",
                            id=tc.id, name=tc.function.name
                        )

                    # Stream tool arguments
                    if tc.function and tc.function.arguments:
                        state.tool_buffers[tc_index].args += tc.function.arguments
                        yield create_input_json_delta(state.block_index, tc.function.arguments)

        # Handle finish
        if finish_reason:
            # Close any open blocks
            if state.text_block_started or state.tool_buffers:
                yield create_content_block_stop(state.block_index)

            # Emit final events
            yield create_message_delta(
                stop_reason=map_finish_reason(finish_reason),
                output_tokens=chunk.usage.completion_tokens if chunk.usage else state.estimated_tokens
            )
            yield create_message_stop()
```

---

## Adapter 4: Error Envelope Translator

**File:** `core/adapters/errors.py`

### Purpose
Maps upstream provider errors into consistent Anthropic-style error responses.

### Function Signature
```python
def translate_error(
    error: Exception,
    provider: str,
    request_id: str
) -> AnthropicErrorResponse:
    ...
```

### Error Mapping Table

| Upstream Status/Error | Anthropic Error Type | HTTP Status | Message Template |
|-----------------------|----------------------|-------------|------------------|
| 401, 403 | `authentication_error` | 401 | "Invalid API key for provider {provider}" |
| 404 (model) | `not_found_error` | 404 | "Model {model} not found on {provider}" |
| 404 (endpoint) | `not_found_error` | 404 | "Provider endpoint not available" |
| 422, 400 | `invalid_request_error` | 400 | "Upstream validation failed: {details}" |
| 429 | `rate_limit_error` | 429 | "Rate limited by {provider}" |
| 500, 502, 503 | `api_error` | 502 | "Upstream provider error: {provider}" |
| Timeout | `api_error` | 504 | "Upstream timeout: {provider}" |
| Connection Error | `api_error` | 503 | "Cannot reach {provider}" |

### Anthropic Error Response Format

```python
{
    "type": "error",
    "error": {
        "type": "invalid_request_error",
        "message": "max_tokens must be greater than 0"
    }
}
```

### Error Handler Implementation

```python
class UpstreamError(Exception):
    def __init__(self, status_code: int, provider: str, detail: str):
        self.status_code = status_code
        self.provider = provider
        self.detail = detail

def translate_error(error: Exception, provider: str, request_id: str) -> tuple[int, dict]:
    if isinstance(error, httpx.HTTPStatusError):
        status = error.response.status_code
        if status in (401, 403):
            return 401, {
                "type": "error",
                "error": {"type": "authentication_error", "message": f"Invalid credentials for {provider}"}
            }
        elif status == 404:
            return 404, {
                "type": "error",
                "error": {"type": "not_found_error", "message": f"Resource not found on {provider}"}
            }
        elif status == 429:
            return 429, {
                "type": "error",
                "error": {"type": "rate_limit_error", "message": f"Rate limited by {provider}"}
            }
        elif status == 422 or status == 400:
            return 400, {
                "type": "error",
                "error": {"type": "invalid_request_error", "message": str(error.response.text)}
            }
        else:
            return 502, {
                "type": "error",
                "error": {"type": "api_error", "message": f"Upstream error from {provider}: {status}"}
            }
    elif isinstance(error, httpx.TimeoutException):
        return 504, {
            "type": "error",
            "error": {"type": "api_error", "message": f"Request to {provider} timed out"}
        }
    elif isinstance(error, httpx.ConnectError):
        return 503, {
            "type": "error",
            "error": {"type": "api_error", "message": f"Cannot connect to {provider}"}
        }
    else:
        return 500, {
            "type": "error",
            "error": {"type": "api_error", "message": "Internal gateway error"}
        }
```

---

## Pydantic Models Specification

### File: `core/models/anthropic_types.py`

```python
from pydantic import BaseModel, Field
from typing import Literal, Optional, Union
from enum import Enum

# Content Block Types
class TextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str

class ImageSource(BaseModel):
    type: Literal["base64", "url"]
    media_type: Optional[str] = None
    data: Optional[str] = None
    url: Optional[str] = None

class ImageBlock(BaseModel):
    type: Literal["image"] = "image"
    source: ImageSource

class ToolUseBlock(BaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict

class ToolResultBlock(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: Union[str, list]
    is_error: Optional[bool] = False

ContentBlock = Union[TextBlock, ImageBlock, ToolUseBlock, ToolResultBlock]

# Message Types
class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: Union[str, list[ContentBlock]]

# Tool Definition
class ToolInputSchema(BaseModel):
    type: Literal["object"] = "object"
    properties: dict = Field(default_factory=dict)
    required: list[str] = Field(default_factory=list)

class Tool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: ToolInputSchema

# Tool Choice
class ToolChoiceAuto(BaseModel):
    type: Literal["auto"] = "auto"

class ToolChoiceAny(BaseModel):
    type: Literal["any"] = "any"

class ToolChoiceTool(BaseModel):
    type: Literal["tool"] = "tool"
    name: str

ToolChoice = Union[ToolChoiceAuto, ToolChoiceAny, ToolChoiceTool]

# Request
class MessagesRequest(BaseModel):
    model: str
    messages: list[Message]
    max_tokens: int
    system: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stop_sequences: Optional[list[str]] = None
    stream: Optional[bool] = False
    tools: Optional[list[Tool]] = None
    tool_choice: Optional[ToolChoice] = None
    metadata: Optional[dict] = None

# Response
class Usage(BaseModel):
    input_tokens: int
    output_tokens: int

class MessagesResponse(BaseModel):
    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    model: str
    content: list[ContentBlock]
    stop_reason: Optional[str] = None
    stop_sequence: Optional[str] = None
    usage: Usage
```

### File: `core/models/openai_types.py`

```python
from pydantic import BaseModel
from typing import Literal, Optional, Union

# Function/Tool Types
class FunctionDef(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[dict] = None

class ToolDef(BaseModel):
    type: Literal["function"] = "function"
    function: FunctionDef

class FunctionCall(BaseModel):
    name: str
    arguments: str

class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: FunctionCall

# Message Types
class SystemMessage(BaseModel):
    role: Literal["system"] = "system"
    content: str

class UserMessage(BaseModel):
    role: Literal["user"] = "user"
    content: Union[str, list]

class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None

class ToolMessage(BaseModel):
    role: Literal["tool"] = "tool"
    tool_call_id: str
    content: str

OpenAIMessage = Union[SystemMessage, UserMessage, AssistantMessage, ToolMessage]

# Request
class ChatCompletionsRequest(BaseModel):
    model: str
    messages: list[OpenAIMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stop: Optional[list[str]] = None
    stream: Optional[bool] = False
    tools: Optional[list[ToolDef]] = None
    tool_choice: Optional[Union[str, dict]] = None

# Response
class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None

class Choice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None

class OpenAIUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionsResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    model: str
    choices: list[Choice]
    usage: Optional[OpenAIUsage] = None

# Stream Chunk Types
class DeltaContent(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[list] = None

class StreamChoice(BaseModel):
    index: int
    delta: DeltaContent
    finish_reason: Optional[str] = None

class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    model: str
    choices: list[StreamChoice]
    usage: Optional[OpenAIUsage] = None
```

---

## Routing + Provider Execution Flow

### Sequence Diagram

```
┌──────────┐     ┌─────────────┐     ┌──────────────┐     ┌────────────┐     ┌──────────┐
│  Client  │     │   Gateway   │     │   Adapters   │     │  LiteLLM   │     │ Provider │
│ (Agent)  │     │  (FastAPI)  │     │              │     │            │     │          │
└────┬─────┘     └──────┬──────┘     └──────┬───────┘     └─────┬──────┘     └────┬─────┘
     │                  │                   │                   │                 │
     │ POST /v1/messages│                   │                   │                 │
     │ (Anthropic)      │                   │                   │                 │
     │─────────────────>│                   │                   │                 │
     │                  │                   │                   │                 │
     │                  │ 1. Validate Auth  │                   │                 │
     │                  │──────────────────>│                   │                 │
     │                  │                   │                   │                 │
     │                  │ 2. Parse Request  │                   │                 │
     │                  │ (Pydantic)        │                   │                 │
     │                  │──────────────────>│                   │                 │
     │                  │                   │                   │                 │
     │                  │ 3. Resolve Model  │                   │                 │
     │                  │ Alias → Provider  │                   │                 │
     │                  │──────────────────>│                   │                 │
     │                  │                   │                   │                 │
     │                  │ 4. Translate Req  │                   │                 │
     │                  │ Anthropic→OpenAI  │                   │                 │
     │                  │──────────────────>│                   │                 │
     │                  │                   │                   │                 │
     │                  │                   │ 5. Call Provider  │                 │
     │                  │                   │──────────────────>│                 │
     │                  │                   │                   │ HTTP Request    │
     │                  │                   │                   │────────────────>│
     │                  │                   │                   │                 │
     │                  │                   │                   │ HTTP Response   │
     │                  │                   │                   │<────────────────│
     │                  │                   │                   │                 │
     │                  │                   │ 6. Translate Resp │                 │
     │                  │                   │ OpenAI→Anthropic  │                 │
     │                  │                   │<──────────────────│                 │
     │                  │                   │                   │                 │
     │ Anthropic Response                   │                   │                 │
     │<─────────────────│                   │                   │                 │
     │                  │                   │                   │                 │
```

---

## Config Design (YAML) – Complete Schema

**File:** `config/config.yaml`

```yaml
gateway:
  host: "0.0.0.0"
  port: 8080
  require_api_key: true
  api_key_header: "x-api-key"  # or "Authorization" with Bearer prefix
  request_timeout: 60  # seconds
  log_level: "info"

providers:
  ollama_local:
    type: openai_compat
    base_url: "http://127.0.0.1:11434/v1"
    api_key: null  # Ollama doesn't require key
    timeout: 120

  groq:
    type: litellm
    api_key_env: GROQ_API_KEY
    timeout: 30

  openai:
    type: litellm
    api_key_env: OPENAI_API_KEY
    timeout: 60

models:
  # Fast local coding model
  shin-coder:
    provider: ollama_local
    model: "qwen2.5-coder:32b"
    defaults:
      temperature: 0.1
      max_tokens: 4096
    fallbacks:
      - provider: groq
        model: "llama-3.3-70b-versatile"
      - provider: openai
        model: "gpt-4o-mini"

  # Fast remote model
  shin-fast:
    provider: groq
    model: "llama-3.3-70b-versatile"
    defaults:
      temperature: 0.2
      max_tokens: 2048

  # High-quality model
  shin-quality:
    provider: openai
    model: "gpt-4o"
    defaults:
      temperature: 0.3
      max_tokens: 8192

  # Claude models (passthrough if needed)
  claude-3-5-sonnet:
    provider: anthropic
    model: "claude-3-5-sonnet-20241022"
```

---

## Performance Optimizations

### Sub-50ms Proxy Overhead Targets

| Component | Target Latency | Implementation |
|-----------|----------------|----------------|
| Request parsing | < 5ms | Pydantic v2 with `model_validate` |
| Auth validation | < 1ms | Simple header check + cache |
| Model resolution | < 1ms | In-memory dict lookup |
| Request translation | < 5ms | Pure Python dict transforms |
| Response translation | < 5ms | Pure Python dict transforms |
| Serialization | < 3ms | `orjson` for JSON encoding |
| **Total overhead** | **< 20ms** | Excluding network to provider |

### Implementation Requirements

```python
# requirements.txt additions for performance
orjson>=3.9.0        # Fast JSON serialization
uvloop>=0.19.0       # Fast event loop (Linux/macOS)
httpx[http2]>=0.25.0 # HTTP/2 connection pooling
```

### Connection Pooling

```python
# Shared async client with keep-alive
from httpx import AsyncClient, Limits

http_client = AsyncClient(
    limits=Limits(max_keepalive_connections=20, max_connections=100),
    timeout=60.0,
    http2=True
)
```

---

## Testing Plan

### Unit Tests for Adapters

```python
# tests/test_adapters.py

def test_anthropic_text_to_openai():
    """Simple text message translation"""

def test_anthropic_multiblock_to_openai():
    """Multiple content blocks concatenated"""

def test_anthropic_image_to_openai():
    """Vision content translation"""

def test_anthropic_tool_use_to_openai():
    """Tool use block → tool_calls"""

def test_anthropic_tool_result_to_openai():
    """Tool result → tool role message"""

def test_anthropic_tools_schema_to_openai():
    """Tools definition translation"""

def test_openai_text_to_anthropic():
    """Simple response translation"""

def test_openai_tool_calls_to_anthropic():
    """Tool calls → tool_use blocks"""

def test_stream_text_events():
    """Streaming text delta translation"""

def test_stream_tool_events():
    """Streaming tool call translation"""

def test_error_mapping():
    """All error types map correctly"""
```

### Integration Tests

```python
# tests/test_integration.py

async def test_full_request_cycle():
    """End-to-end with mock provider"""

async def test_streaming_full_cycle():
    """Streaming end-to-end with mock"""

async def test_fallback_on_error():
    """Primary fails, fallback succeeds"""
```

---

## Acceptance Criteria (MVP)

- [ ] `POST /v1/messages` accepts Anthropic-format requests
- [ ] Non-streaming responses return valid Anthropic JSON
- [ ] Streaming responses emit correct SSE event sequence
- [ ] Model aliases resolve to configured providers
- [ ] Tool definitions translate correctly both directions
- [ ] Tool calls and results work in conversation flow
- [ ] Errors return Anthropic-style error envelope
- [ ] Fallback providers activate on upstream failure
- [ ] Proxy overhead stays under 50ms
- [ ] IDE agents (Roo Code, Cline, Claude Code) can connect and operate

---

## Missing Components (Handlers & Utilities)

### Updated Directory Structure

```text
shin-gateway/
├── config/
│   ├── __init__.py
│   ├── config.yaml
│   └── settings.py              # Pydantic settings loader
├── core/
│   ├── __init__.py
│   ├── adapters/
│   │   ├── __init__.py
│   │   ├── anthropic_to_openai.py
│   │   ├── openai_to_anthropic.py
│   │   ├── stream_translator.py
│   │   └── errors.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── anthropic_types.py
│   │   └── openai_types.py
│   ├── handlers/                 # NEW: Handler components
│   │   ├── __init__.py
│   │   ├── logger.py             # Structured logging
│   │   ├── retry.py              # Retry with backoff
│   │   ├── circuit_breaker.py    # Circuit breaker pattern
│   │   ├── rate_limiter.py       # Client-side rate limiting
│   │   └── token_counter.py      # Token estimation
│   ├── middleware/               # NEW: FastAPI middleware
│   │   ├── __init__.py
│   │   ├── request_id.py         # Request ID injection
│   │   ├── timing.py             # Request timing
│   │   └── error_handler.py      # Global exception handler
│   ├── proxy.py                  # Main request handler
│   ├── security.py               # API key validation
│   └── utils.py                  # Helper functions
├── main.py                       # FastAPI app with lifespan
├── .env
├── requirements.txt
└── README.md
```

---

### Handler 1: Structured Logger

**File:** `core/handlers/logger.py`

#### Purpose
Provide structured JSON logging with request context for debugging and observability.

#### Implementation

```python
import logging
import sys
from contextvars import ContextVar
from typing import Optional
import orjson

# Context variable for request-scoped data
request_context: ContextVar[dict] = ContextVar("request_context", default={})

class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging"""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add request context if available
        ctx = request_context.get()
        if ctx:
            log_data["request_id"] = ctx.get("request_id")
            log_data["model"] = ctx.get("model")
            log_data["provider"] = ctx.get("provider")

        # Add extra fields
        if hasattr(record, "extra"):
            log_data.update(record.extra)

        # Add exception info
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return orjson.dumps(log_data).decode()

def setup_logger(name: str = "shin-gateway", level: str = "INFO") -> logging.Logger:
    """Configure structured logger"""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(StructuredFormatter())
    logger.addHandler(handler)

    return logger

def log_with_context(logger: logging.Logger, level: str, message: str, **extra):
    """Log with additional context"""
    record = logger.makeRecord(
        logger.name, getattr(logging, level.upper()),
        "", 0, message, (), None
    )
    record.extra = extra
    logger.handle(record)
```

#### Log Output Example
```json
{
  "timestamp": "2026-01-21 10:30:45",
  "level": "INFO",
  "logger": "shin-gateway",
  "message": "Request completed",
  "request_id": "req_abc123",
  "model": "shin-coder",
  "provider": "ollama_local",
  "latency_ms": 1250,
  "input_tokens": 150,
  "output_tokens": 89
}
```

---

### Handler 2: Retry with Exponential Backoff

**File:** `core/handlers/retry.py`

#### Purpose
Retry transient failures (network errors, 429, 503) with exponential backoff.

#### Implementation

```python
import asyncio
import random
from typing import Callable, TypeVar, Set
from functools import wraps
import httpx

T = TypeVar("T")

class RetryConfig:
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 0.5,
        max_delay: float = 10.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_status_codes: Set[int] = {429, 500, 502, 503, 504},
        retryable_exceptions: tuple = (
            httpx.ConnectError,
            httpx.ReadTimeout,
            httpx.WriteTimeout,
        ),
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_status_codes = retryable_status_codes
        self.retryable_exceptions = retryable_exceptions

def calculate_delay(attempt: int, config: RetryConfig) -> float:
    """Calculate delay with exponential backoff and optional jitter"""
    delay = min(
        config.base_delay * (config.exponential_base ** attempt),
        config.max_delay
    )
    if config.jitter:
        delay = delay * (0.5 + random.random())
    return delay

def is_retryable(error: Exception, config: RetryConfig) -> bool:
    """Check if error is retryable"""
    if isinstance(error, config.retryable_exceptions):
        return True
    if isinstance(error, httpx.HTTPStatusError):
        return error.response.status_code in config.retryable_status_codes
    return False

async def retry_async(
    func: Callable[..., T],
    *args,
    config: RetryConfig = None,
    logger = None,
    **kwargs
) -> T:
    """Execute async function with retry logic"""
    config = config or RetryConfig()
    last_error = None

    for attempt in range(config.max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_error = e

            if attempt >= config.max_retries or not is_retryable(e, config):
                raise

            delay = calculate_delay(attempt, config)

            if logger:
                logger.warning(
                    f"Retry attempt {attempt + 1}/{config.max_retries} "
                    f"after {delay:.2f}s due to: {type(e).__name__}"
                )

            await asyncio.sleep(delay)

    raise last_error

def with_retry(config: RetryConfig = None):
    """Decorator for retry logic"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await retry_async(func, *args, config=config, **kwargs)
        return wrapper
    return decorator
```

#### Usage Example
```python
@with_retry(RetryConfig(max_retries=3, base_delay=1.0))
async def call_provider(client: httpx.AsyncClient, request: dict):
    response = await client.post("/v1/chat/completions", json=request)
    response.raise_for_status()
    return response.json()
```

---

### Handler 3: Circuit Breaker

**File:** `core/handlers/circuit_breaker.py`

#### Purpose
Prevent cascading failures by temporarily blocking requests to failing providers.

#### States
- **CLOSED**: Normal operation, requests pass through
- **OPEN**: Provider is failing, requests fail fast
- **HALF-OPEN**: Testing if provider recovered

#### Implementation

```python
import asyncio
import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Optional

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class CircuitStats:
    failures: int = 0
    successes: int = 0
    last_failure_time: float = 0.0
    state: CircuitState = CircuitState.CLOSED

@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5          # Failures before opening
    success_threshold: int = 2          # Successes to close from half-open
    timeout: float = 30.0               # Seconds before half-open
    half_open_max_calls: int = 1        # Concurrent calls in half-open

class CircuitBreaker:
    """Circuit breaker for provider resilience"""

    def __init__(self, config: CircuitBreakerConfig = None):
        self.config = config or CircuitBreakerConfig()
        self._circuits: Dict[str, CircuitStats] = {}
        self._lock = asyncio.Lock()

    def _get_circuit(self, provider: str) -> CircuitStats:
        if provider not in self._circuits:
            self._circuits[provider] = CircuitStats()
        return self._circuits[provider]

    async def can_execute(self, provider: str) -> bool:
        """Check if request can proceed"""
        async with self._lock:
            circuit = self._get_circuit(provider)

            if circuit.state == CircuitState.CLOSED:
                return True

            if circuit.state == CircuitState.OPEN:
                # Check if timeout expired
                if time.time() - circuit.last_failure_time >= self.config.timeout:
                    circuit.state = CircuitState.HALF_OPEN
                    circuit.successes = 0
                    return True
                return False

            # HALF_OPEN: allow limited requests
            return True

    async def record_success(self, provider: str):
        """Record successful request"""
        async with self._lock:
            circuit = self._get_circuit(provider)

            if circuit.state == CircuitState.HALF_OPEN:
                circuit.successes += 1
                if circuit.successes >= self.config.success_threshold:
                    circuit.state = CircuitState.CLOSED
                    circuit.failures = 0
            else:
                circuit.failures = 0  # Reset on success

    async def record_failure(self, provider: str):
        """Record failed request"""
        async with self._lock:
            circuit = self._get_circuit(provider)
            circuit.failures += 1
            circuit.last_failure_time = time.time()

            if circuit.state == CircuitState.HALF_OPEN:
                # Immediately reopen on failure
                circuit.state = CircuitState.OPEN
            elif circuit.failures >= self.config.failure_threshold:
                circuit.state = CircuitState.OPEN

    def get_state(self, provider: str) -> CircuitState:
        """Get current circuit state"""
        return self._get_circuit(provider).state

class CircuitOpenError(Exception):
    """Raised when circuit is open"""
    def __init__(self, provider: str):
        self.provider = provider
        super().__init__(f"Circuit breaker open for provider: {provider}")
```

#### Usage Example
```python
circuit_breaker = CircuitBreaker()

async def call_with_circuit_breaker(provider: str, request_func):
    if not await circuit_breaker.can_execute(provider):
        raise CircuitOpenError(provider)

    try:
        result = await request_func()
        await circuit_breaker.record_success(provider)
        return result
    except Exception as e:
        await circuit_breaker.record_failure(provider)
        raise
```

---

### Handler 4: Rate Limiter

**File:** `core/handlers/rate_limiter.py`

#### Purpose
Client-side rate limiting to avoid hitting provider rate limits.

#### Implementation

```python
import asyncio
import time
from dataclasses import dataclass
from typing import Dict

@dataclass
class RateLimitConfig:
    requests_per_minute: int = 60
    requests_per_second: int = 10
    burst_size: int = 5

class TokenBucketLimiter:
    """Token bucket rate limiter"""

    def __init__(self, rate: float, capacity: int):
        self.rate = rate           # Tokens per second
        self.capacity = capacity   # Max tokens
        self.tokens = capacity
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> float:
        """
        Acquire tokens, returns wait time if rate limited.
        Returns 0 if tokens available immediately.
        """
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_update

            # Refill tokens
            self.tokens = min(
                self.capacity,
                self.tokens + elapsed * self.rate
            )
            self.last_update = now

            if self.tokens >= tokens:
                self.tokens -= tokens
                return 0.0

            # Calculate wait time
            tokens_needed = tokens - self.tokens
            wait_time = tokens_needed / self.rate
            return wait_time

    async def wait_and_acquire(self, tokens: int = 1):
        """Wait if necessary and acquire tokens"""
        wait_time = await self.acquire(tokens)
        if wait_time > 0:
            await asyncio.sleep(wait_time)
            await self.acquire(tokens)

class ProviderRateLimiter:
    """Per-provider rate limiting"""

    def __init__(self):
        self._limiters: Dict[str, TokenBucketLimiter] = {}

    def configure(self, provider: str, config: RateLimitConfig):
        """Configure rate limit for provider"""
        self._limiters[provider] = TokenBucketLimiter(
            rate=config.requests_per_second,
            capacity=config.burst_size
        )

    async def acquire(self, provider: str) -> float:
        """Acquire rate limit token for provider"""
        if provider not in self._limiters:
            return 0.0
        return await self._limiters[provider].acquire()

    async def wait_and_acquire(self, provider: str):
        """Wait and acquire for provider"""
        if provider in self._limiters:
            await self._limiters[provider].wait_and_acquire()
```

---

### Handler 5: Token Counter/Estimator

**File:** `core/handlers/token_counter.py`

#### Purpose
Estimate token counts when upstream provider doesn't return usage info (e.g., Ollama streaming).

#### Implementation

```python
from typing import List, Dict, Any, Optional
import re

# Rough token estimation (4 chars ≈ 1 token for English)
CHARS_PER_TOKEN = 4

class TokenCounter:
    """Estimate token counts for messages"""

    def __init__(self, chars_per_token: int = CHARS_PER_TOKEN):
        self.chars_per_token = chars_per_token

    def count_text(self, text: str) -> int:
        """Estimate tokens in text"""
        if not text:
            return 0
        return max(1, len(text) // self.chars_per_token)

    def count_message(self, message: Dict[str, Any]) -> int:
        """Count tokens in a single message"""
        tokens = 4  # Base overhead per message

        content = message.get("content")
        if isinstance(content, str):
            tokens += self.count_text(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        tokens += self.count_text(block.get("text", ""))
                    elif block.get("type") == "tool_use":
                        tokens += self.count_text(block.get("name", ""))
                        tokens += self.count_text(str(block.get("input", {})))
                    elif block.get("type") == "tool_result":
                        tokens += self.count_text(str(block.get("content", "")))
                    elif block.get("type") == "image":
                        tokens += 85  # Base cost for image reference

        # Tool calls
        tool_calls = message.get("tool_calls", [])
        for tc in tool_calls:
            tokens += 10  # Tool call overhead
            if tc.get("function"):
                tokens += self.count_text(tc["function"].get("name", ""))
                tokens += self.count_text(tc["function"].get("arguments", ""))

        return tokens

    def count_messages(self, messages: List[Dict[str, Any]]) -> int:
        """Count tokens in message list"""
        total = 3  # Base overhead
        for msg in messages:
            total += self.count_message(msg)
        return total

    def count_request(self, request: Dict[str, Any]) -> int:
        """Estimate input tokens for request"""
        tokens = 0

        # System message
        system = request.get("system")
        if system:
            tokens += 4 + self.count_text(system)

        # Messages
        messages = request.get("messages", [])
        tokens += self.count_messages(messages)

        # Tools
        tools = request.get("tools", [])
        for tool in tools:
            tokens += 20  # Tool definition overhead
            tokens += self.count_text(tool.get("name", ""))
            tokens += self.count_text(tool.get("description", ""))
            tokens += self.count_text(str(tool.get("input_schema", {})))

        return tokens

# Singleton instance
token_counter = TokenCounter()
```

---

### Middleware 1: Request ID

**File:** `core/middleware/request_id.py`

#### Purpose
Generate and inject unique request IDs for tracing.

```python
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from core.handlers.logger import request_context

class RequestIDMiddleware(BaseHTTPMiddleware):
    """Inject request ID into every request"""

    async def dispatch(self, request: Request, call_next):
        # Get from header or generate new
        request_id = request.headers.get("X-Request-ID") or f"req_{uuid.uuid4().hex[:12]}"

        # Store in request state
        request.state.request_id = request_id

        # Set context for logging
        token = request_context.set({"request_id": request_id})

        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            request_context.reset(token)
```

---

### Middleware 2: Request Timing

**File:** `core/middleware/timing.py`

```python
import time
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

class TimingMiddleware(BaseHTTPMiddleware):
    """Track request timing"""

    async def dispatch(self, request: Request, call_next):
        start_time = time.perf_counter()

        response = await call_next(request)

        duration_ms = (time.perf_counter() - start_time) * 1000
        response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"

        return response
```

---

### Middleware 3: Global Exception Handler

**File:** `core/middleware/error_handler.py`

```python
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from core.adapters.errors import translate_error
import logging

logger = logging.getLogger("shin-gateway")

class ExceptionHandlerMiddleware(BaseHTTPMiddleware):
    """Catch all exceptions and return Anthropic-style errors"""

    async def dispatch(self, request: Request, call_next):
        try:
            return await call_next(request)
        except Exception as e:
            request_id = getattr(request.state, "request_id", "unknown")
            provider = getattr(request.state, "provider", "unknown")

            logger.exception(f"Unhandled exception: {e}", extra={
                "request_id": request_id,
                "path": request.url.path,
            })

            status_code, error_body = translate_error(e, provider, request_id)

            return JSONResponse(
                status_code=status_code,
                content=error_body
            )
```

---

### Main App: Lifespan Handler

**File:** `main.py`

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
import httpx

from config.settings import load_settings
from core.handlers.logger import setup_logger
from core.handlers.circuit_breaker import CircuitBreaker
from core.handlers.rate_limiter import ProviderRateLimiter
from core.middleware.request_id import RequestIDMiddleware
from core.middleware.timing import TimingMiddleware
from core.middleware.error_handler import ExceptionHandlerMiddleware

# Global instances
http_client: httpx.AsyncClient = None
circuit_breaker: CircuitBreaker = None
rate_limiter: ProviderRateLimiter = None
settings = None
logger = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown"""
    global http_client, circuit_breaker, rate_limiter, settings, logger

    # --- STARTUP ---
    logger = setup_logger("shin-gateway", level="INFO")
    logger.info("Starting Shin Gateway...")

    # Load config
    settings = load_settings()
    logger.info(f"Loaded {len(settings.models)} model aliases")

    # Initialize HTTP client with connection pooling
    http_client = httpx.AsyncClient(
        limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
        timeout=httpx.Timeout(settings.gateway.request_timeout),
        http2=True
    )

    # Initialize handlers
    circuit_breaker = CircuitBreaker()
    rate_limiter = ProviderRateLimiter()

    # Configure rate limits per provider
    for name, provider in settings.providers.items():
        if hasattr(provider, "rate_limit"):
            rate_limiter.configure(name, provider.rate_limit)

    logger.info("Shin Gateway ready")

    yield  # Application runs here

    # --- SHUTDOWN ---
    logger.info("Shutting down Shin Gateway...")
    await http_client.aclose()
    logger.info("Shutdown complete")

# Create app
app = FastAPI(
    title="Shin Gateway",
    version="1.0.0",
    lifespan=lifespan
)

# Add middleware (order matters: last added = first executed)
app.add_middleware(ExceptionHandlerMiddleware)
app.add_middleware(TimingMiddleware)
app.add_middleware(RequestIDMiddleware)

# Health check endpoints
@app.get("/health")
async def health():
    """Liveness probe"""
    return {"status": "ok"}

@app.get("/ready")
async def ready():
    """Readiness probe"""
    # Check if critical dependencies are ready
    checks = {
        "http_client": http_client is not None,
        "settings": settings is not None,
    }
    all_ready = all(checks.values())
    return {
        "status": "ready" if all_ready else "not_ready",
        "checks": checks
    }

# Import and register routes
from core.proxy import router as proxy_router
app.include_router(proxy_router)
```

---

### Optional: Response Cache Handler

**File:** `core/handlers/cache.py`

```python
import hashlib
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass
import orjson

@dataclass
class CacheEntry:
    response: dict
    created_at: float
    ttl: float

class ResponseCache:
    """Simple in-memory response cache"""

    def __init__(self, default_ttl: float = 300.0, max_size: int = 1000):
        self.default_ttl = default_ttl
        self.max_size = max_size
        self._cache: Dict[str, CacheEntry] = {}

    def _make_key(self, request: dict) -> str:
        """Create cache key from request (excluding stream flag)"""
        # Only cache deterministic requests (temperature=0)
        if request.get("temperature", 1.0) != 0:
            return None

        cache_data = {
            "model": request.get("model"),
            "messages": request.get("messages"),
            "system": request.get("system"),
            "tools": request.get("tools"),
            "max_tokens": request.get("max_tokens"),
        }
        serialized = orjson.dumps(cache_data, option=orjson.OPT_SORT_KEYS)
        return hashlib.sha256(serialized).hexdigest()[:16]

    def get(self, request: dict) -> Optional[dict]:
        """Get cached response if available"""
        key = self._make_key(request)
        if not key or key not in self._cache:
            return None

        entry = self._cache[key]
        if time.time() - entry.created_at > entry.ttl:
            del self._cache[key]
            return None

        return entry.response

    def set(self, request: dict, response: dict, ttl: float = None):
        """Cache response"""
        key = self._make_key(request)
        if not key:
            return

        # Evict oldest if full
        if len(self._cache) >= self.max_size:
            oldest_key = min(self._cache, key=lambda k: self._cache[k].created_at)
            del self._cache[oldest_key]

        self._cache[key] = CacheEntry(
            response=response,
            created_at=time.time(),
            ttl=ttl or self.default_ttl
        )

    def clear(self):
        """Clear all cached responses"""
        self._cache.clear()
```

---

## Component Summary Table

| Component | File | Purpose | Priority |
|-----------|------|---------|----------|
| Structured Logger | `handlers/logger.py` | JSON logging with request context | **Required** |
| Retry Handler | `handlers/retry.py` | Exponential backoff for transient failures | **Required** |
| Circuit Breaker | `handlers/circuit_breaker.py` | Prevent cascading failures | **Required** |
| Rate Limiter | `handlers/rate_limiter.py` | Avoid provider rate limits | Recommended |
| Token Counter | `handlers/token_counter.py` | Estimate tokens when not provided | Recommended |
| Request ID Middleware | `middleware/request_id.py` | Request tracing | **Required** |
| Timing Middleware | `middleware/timing.py` | Performance monitoring | Recommended |
| Error Handler Middleware | `middleware/error_handler.py` | Consistent error responses | **Required** |
| Lifespan Handler | `main.py` | App startup/shutdown | **Required** |
| Health Endpoints | `main.py` | `/health`, `/ready` | **Required** |
| Response Cache | `handlers/cache.py` | Cache deterministic responses | Optional |

---

## Updated Acceptance Criteria (MVP)

- [ ] `POST /v1/messages` accepts Anthropic-format requests
- [ ] Non-streaming responses return valid Anthropic JSON
- [ ] Streaming responses emit correct SSE event sequence
- [ ] Model aliases resolve to configured providers
- [ ] Tool definitions translate correctly both directions
- [ ] Tool calls and results work in conversation flow
- [ ] Errors return Anthropic-style error envelope
- [ ] Fallback providers activate on upstream failure
- [ ] Proxy overhead stays under 50ms
- [ ] IDE agents (Roo Code, Cline, Claude Code) can connect and operate
- [ ] **Request IDs tracked through full request lifecycle**
- [ ] **Structured JSON logs with request context**
- [ ] **Retry logic handles transient failures**
- [ ] **Circuit breaker prevents cascading failures**
- [ ] **Health endpoints respond correctly**

---

---

## Agentic IDE Compatibility Components

These components are **CRITICAL** for Claude Code, Roo Code, Cline, Kilo Code, and similar agentic coding tools.

### Why Agentic Tools Are Special

Agentic IDE tools have unique requirements:

1. **High-frequency tool calls** - Agents make 10-50+ tool calls per task
2. **Long-running sessions** - Single conversation can last hours
3. **Streaming is mandatory** - Agents parse streaming tokens in real-time
4. **Cancellation support** - Users cancel mid-generation frequently
5. **Parallel tool calls** - Agents may request multiple tools at once
6. **Large context windows** - 100K+ tokens common in coding tasks
7. **Precise token counting** - Agents manage context window budget

---

### Component 1: Parallel Tool Calls Handler

**File:** `core/handlers/parallel_tools.py`

#### Problem
Anthropic supports `parallel_tool_use: true` where the model can request multiple tools in one response. OpenAI handles this differently.

#### Implementation

```python
from typing import List, Dict, Any

class ParallelToolHandler:
    """Handle parallel tool calls translation"""

    def translate_parallel_tool_use(
        self,
        anthropic_content: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Anthropic: Multiple tool_use blocks in content[]
        OpenAI: Multiple items in tool_calls[]
        """
        tool_calls = []
        text_content = []

        for block in anthropic_content:
            if block.get("type") == "tool_use":
                tool_calls.append({
                    "id": block["id"],
                    "type": "function",
                    "function": {
                        "name": block["name"],
                        "arguments": json.dumps(block["input"])
                    }
                })
            elif block.get("type") == "text":
                text_content.append(block["text"])

        return {
            "role": "assistant",
            "content": " ".join(text_content) if text_content else None,
            "tool_calls": tool_calls if tool_calls else None
        }

    def translate_parallel_tool_results(
        self,
        anthropic_message: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Anthropic: Single user message with multiple tool_result blocks
        OpenAI: Multiple separate tool role messages
        """
        openai_messages = []

        content = anthropic_message.get("content", [])
        if isinstance(content, str):
            return [{"role": "user", "content": content}]

        for block in content:
            if block.get("type") == "tool_result":
                # Handle content that can be string or list
                result_content = block.get("content", "")
                if isinstance(result_content, list):
                    # Concatenate text blocks
                    result_content = " ".join(
                        b.get("text", "") for b in result_content
                        if b.get("type") == "text"
                    )

                openai_messages.append({
                    "role": "tool",
                    "tool_call_id": block["tool_use_id"],
                    "content": str(result_content)
                })
            elif block.get("type") == "text":
                # Text alongside tool results (rare but possible)
                openai_messages.append({
                    "role": "user",
                    "content": block["text"]
                })

        return openai_messages
```

---

### Component 2: Request Cancellation Handler

**File:** `core/handlers/cancellation.py`

#### Problem
Users frequently cancel agent operations mid-stream. The gateway must:
1. Stop reading from upstream immediately
2. Close the SSE connection cleanly
3. Not leave orphaned connections

#### Implementation

```python
import asyncio
from typing import Optional
from contextlib import asynccontextmanager
import httpx

class CancellationToken:
    """Token to signal cancellation"""

    def __init__(self):
        self._cancelled = False
        self._event = asyncio.Event()

    def cancel(self):
        self._cancelled = True
        self._event.set()

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled

    async def wait(self):
        await self._event.wait()

class CancellableStreamReader:
    """Read upstream stream with cancellation support"""

    def __init__(
        self,
        response: httpx.Response,
        cancellation_token: CancellationToken
    ):
        self.response = response
        self.token = cancellation_token
        self._iterator = None

    async def __aiter__(self):
        self._iterator = self.response.aiter_lines()
        return self

    async def __anext__(self):
        if self.token.is_cancelled:
            # Clean up and stop
            await self.response.aclose()
            raise StopAsyncIteration

        try:
            # Race between next line and cancellation
            read_task = asyncio.create_task(self._iterator.__anext__())
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

            if cancel_task in done:
                await self.response.aclose()
                raise StopAsyncIteration

            return read_task.result()

        except StopAsyncIteration:
            raise
        except Exception as e:
            await self.response.aclose()
            raise

@asynccontextmanager
async def cancellable_request(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    cancellation_token: CancellationToken,
    **kwargs
):
    """Make a cancellable streaming request"""
    response = None
    try:
        response = await client.stream(method, url, **kwargs).__aenter__()
        yield CancellableStreamReader(response, cancellation_token)
    finally:
        if response and not response.is_closed:
            await response.aclose()
```

#### FastAPI Integration

```python
from fastapi import Request
from starlette.responses import StreamingResponse

# Store active cancellation tokens
active_requests: Dict[str, CancellationToken] = {}

@app.post("/v1/messages")
async def messages(request: Request, body: MessagesRequest):
    request_id = request.state.request_id
    cancellation_token = CancellationToken()
    active_requests[request_id] = cancellation_token

    async def cleanup():
        active_requests.pop(request_id, None)

    # Handle client disconnect
    async def check_disconnect():
        while not cancellation_token.is_cancelled:
            if await request.is_disconnected():
                cancellation_token.cancel()
                break
            await asyncio.sleep(0.1)

    asyncio.create_task(check_disconnect())

    try:
        # ... streaming logic with cancellation_token
        pass
    finally:
        await cleanup()
```

---

### Component 3: Long Context Window Support

**File:** `core/handlers/context_manager.py`

#### Problem
Agentic tools often work with 100K+ token contexts. Need to:
1. Handle large payloads efficiently
2. Track context usage accurately
3. Warn before hitting limits

#### Implementation

```python
from dataclasses import dataclass
from typing import Dict, Any, Optional
import tiktoken

@dataclass
class ContextLimits:
    """Model context window limits"""
    model: str
    max_input_tokens: int
    max_output_tokens: int
    total_context: int

# Known model limits
MODEL_LIMITS: Dict[str, ContextLimits] = {
    # Anthropic models
    "claude-3-5-sonnet": ContextLimits("claude-3-5-sonnet", 200000, 8192, 200000),
    "claude-3-opus": ContextLimits("claude-3-opus", 200000, 4096, 200000),

    # OpenAI models
    "gpt-4o": ContextLimits("gpt-4o", 128000, 16384, 128000),
    "gpt-4o-mini": ContextLimits("gpt-4o-mini", 128000, 16384, 128000),

    # Open models via Ollama
    "qwen2.5-coder:32b": ContextLimits("qwen2.5-coder:32b", 32768, 8192, 32768),
    "llama-3.3-70b": ContextLimits("llama-3.3-70b", 128000, 8192, 128000),

    # Groq
    "llama-3.3-70b-versatile": ContextLimits("llama-3.3-70b-versatile", 128000, 8192, 128000),
}

class ContextManager:
    """Manage context window for agentic workflows"""

    def __init__(self):
        # Cache tokenizers
        self._tokenizers: Dict[str, Any] = {}

    def get_limits(self, model: str) -> ContextLimits:
        """Get context limits for model"""
        # Try exact match first
        if model in MODEL_LIMITS:
            return MODEL_LIMITS[model]

        # Try prefix match
        for key, limits in MODEL_LIMITS.items():
            if model.startswith(key) or key in model:
                return limits

        # Default conservative limits
        return ContextLimits(model, 8192, 2048, 8192)

    def estimate_tokens(self, request: Dict[str, Any]) -> int:
        """Estimate total input tokens"""
        from core.handlers.token_counter import token_counter
        return token_counter.count_request(request)

    def validate_request(
        self,
        request: Dict[str, Any],
        model: str
    ) -> tuple[bool, Optional[str]]:
        """
        Validate request fits in context window.
        Returns (is_valid, error_message)
        """
        limits = self.get_limits(model)
        estimated_input = self.estimate_tokens(request)
        requested_output = request.get("max_tokens", 4096)

        total_needed = estimated_input + requested_output

        if estimated_input > limits.max_input_tokens:
            return False, (
                f"Estimated input tokens ({estimated_input}) exceeds "
                f"model limit ({limits.max_input_tokens})"
            )

        if requested_output > limits.max_output_tokens:
            return False, (
                f"Requested max_tokens ({requested_output}) exceeds "
                f"model limit ({limits.max_output_tokens})"
            )

        if total_needed > limits.total_context:
            return False, (
                f"Total tokens needed ({total_needed}) exceeds "
                f"context window ({limits.total_context})"
            )

        return True, None

    def get_available_tokens(
        self,
        request: Dict[str, Any],
        model: str
    ) -> int:
        """Calculate available tokens for output"""
        limits = self.get_limits(model)
        used = self.estimate_tokens(request)
        return max(0, limits.total_context - used)

context_manager = ContextManager()
```

---

### Component 4: Thinking/Extended Thinking Support

**File:** `core/handlers/thinking.py`

#### Problem
Claude supports `thinking` blocks for chain-of-thought reasoning. Some agents use extended thinking. Need to handle this in translation.

#### Implementation

```python
from typing import Dict, Any, List, Optional

class ThinkingHandler:
    """Handle thinking blocks translation"""

    def extract_thinking_config(
        self,
        anthropic_request: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Extract thinking configuration from request"""
        # Anthropic thinking config
        thinking = anthropic_request.get("thinking")
        if thinking:
            return {
                "type": thinking.get("type", "enabled"),
                "budget_tokens": thinking.get("budget_tokens", 10000)
            }
        return None

    def translate_thinking_to_openai(
        self,
        thinking_config: Optional[Dict[str, Any]],
        openai_request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Translate thinking config for OpenAI-compatible models.
        Most don't support it, so we add system prompt guidance.
        """
        if not thinking_config:
            return openai_request

        # Add thinking guidance to system message
        thinking_prompt = (
            "\n\n<thinking_mode>\n"
            "Before responding, think through your approach step by step. "
            "Show your reasoning process.\n"
            "</thinking_mode>"
        )

        messages = openai_request.get("messages", [])
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] += thinking_prompt
        else:
            messages.insert(0, {
                "role": "system",
                "content": thinking_prompt.strip()
            })

        openai_request["messages"] = messages
        return openai_request

    def translate_thinking_response(
        self,
        openai_response: Dict[str, Any],
        had_thinking_config: bool
    ) -> List[Dict[str, Any]]:
        """
        Extract thinking from response if present.
        Returns list of content blocks.
        """
        content = openai_response.get("choices", [{}])[0].get("message", {}).get("content", "")

        if not content or not had_thinking_config:
            return [{"type": "text", "text": content}] if content else []

        # Try to extract thinking blocks from response
        # Look for <thinking> tags or similar patterns
        import re

        thinking_pattern = r'<thinking>(.*?)</thinking>'
        thinking_match = re.search(thinking_pattern, content, re.DOTALL)

        if thinking_match:
            thinking_content = thinking_match.group(1).strip()
            remaining_content = re.sub(thinking_pattern, '', content, flags=re.DOTALL).strip()

            blocks = []
            if thinking_content:
                blocks.append({
                    "type": "thinking",
                    "thinking": thinking_content
                })
            if remaining_content:
                blocks.append({
                    "type": "text",
                    "text": remaining_content
                })
            return blocks

        return [{"type": "text", "text": content}]

thinking_handler = ThinkingHandler()
```

---

### Component 5: Beta Headers Handler

**File:** `core/handlers/beta_features.py`

#### Problem
Anthropic uses `anthropic-beta` headers for experimental features. Agents may send:
- `prompt-caching-2024-07-31`
- `max-tokens-3-5-sonnet-2024-07-15`
- `computer-use-2024-10-22`
- `token-counting-2024-11-01`

#### Implementation

```python
from typing import Dict, List, Optional, Set
from dataclasses import dataclass

@dataclass
class BetaFeature:
    name: str
    supported: bool
    fallback_behavior: str  # "ignore", "error", "emulate"

# Known beta features and how to handle them
BETA_FEATURES: Dict[str, BetaFeature] = {
    "prompt-caching-2024-07-31": BetaFeature(
        "prompt-caching", False, "ignore"  # No caching in gateway
    ),
    "max-tokens-3-5-sonnet-2024-07-15": BetaFeature(
        "extended-output", True, "emulate"  # Just pass through max_tokens
    ),
    "computer-use-2024-10-22": BetaFeature(
        "computer-use", False, "error"  # Not supported
    ),
    "token-counting-2024-11-01": BetaFeature(
        "token-counting", True, "emulate"  # Use our token counter
    ),
    "message-batches-2024-09-24": BetaFeature(
        "batches", False, "error"  # Not supported
    ),
}

class BetaFeaturesHandler:
    """Handle Anthropic beta features"""

    def parse_beta_header(self, header_value: Optional[str]) -> List[str]:
        """Parse anthropic-beta header"""
        if not header_value:
            return []
        return [f.strip() for f in header_value.split(",")]

    def validate_features(
        self,
        features: List[str]
    ) -> tuple[Set[str], List[str]]:
        """
        Validate requested beta features.
        Returns (supported_features, error_messages)
        """
        supported = set()
        errors = []

        for feature in features:
            if feature in BETA_FEATURES:
                config = BETA_FEATURES[feature]
                if config.fallback_behavior == "error" and not config.supported:
                    errors.append(f"Beta feature not supported: {feature}")
                elif config.supported or config.fallback_behavior != "error":
                    supported.add(feature)
            else:
                # Unknown feature - log warning but don't fail
                pass

        return supported, errors

    def apply_feature_defaults(
        self,
        request: Dict,
        features: Set[str]
    ) -> Dict:
        """Apply any default behaviors for enabled features"""

        # Extended output tokens
        if "max-tokens-3-5-sonnet-2024-07-15" in features:
            # Allow higher max_tokens
            if request.get("max_tokens", 0) > 8192:
                # Model supports extended output
                pass

        return request

beta_handler = BetaFeaturesHandler()
```

---

### Component 6: System Prompt Caching Emulation

**File:** `core/handlers/prompt_cache.py`

#### Problem
Anthropic has prompt caching (`cache_control` blocks). For non-Anthropic providers, we need to at least not break, and optionally emulate with our own cache.

#### Implementation

```python
from typing import Dict, Any, List, Optional
import hashlib
import time
from dataclasses import dataclass

@dataclass
class CachedPromptEntry:
    hash: str
    tokens: int
    created_at: float
    ttl: float = 300.0  # 5 minutes

class PromptCacheHandler:
    """Handle Anthropic cache_control blocks"""

    def __init__(self):
        self._cache: Dict[str, CachedPromptEntry] = {}

    def strip_cache_control(
        self,
        messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Remove cache_control from messages for non-Anthropic providers.
        These providers don't understand cache_control.
        """
        cleaned = []
        for msg in messages:
            cleaned_msg = msg.copy()

            content = msg.get("content")
            if isinstance(content, list):
                cleaned_content = []
                for block in content:
                    if isinstance(block, dict):
                        cleaned_block = {
                            k: v for k, v in block.items()
                            if k != "cache_control"
                        }
                        cleaned_content.append(cleaned_block)
                    else:
                        cleaned_content.append(block)
                cleaned_msg["content"] = cleaned_content

            cleaned.append(cleaned_msg)

        return cleaned

    def extract_cacheable_prefix(
        self,
        messages: List[Dict[str, Any]]
    ) -> tuple[List[Dict], List[Dict]]:
        """
        Split messages into cacheable prefix and dynamic suffix.
        Cacheable: everything up to and including last cache_control ephemeral
        """
        last_cache_idx = -1

        for i, msg in enumerate(messages):
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("cache_control"):
                        last_cache_idx = i

        if last_cache_idx == -1:
            return [], messages

        return messages[:last_cache_idx + 1], messages[last_cache_idx + 1:]

    def compute_prefix_hash(self, prefix: List[Dict]) -> str:
        """Compute hash of cacheable prefix"""
        import orjson
        serialized = orjson.dumps(prefix, option=orjson.OPT_SORT_KEYS)
        return hashlib.sha256(serialized).hexdigest()[:16]

    def get_cache_stats(
        self,
        request: Dict[str, Any]
    ) -> Dict[str, int]:
        """
        Return simulated cache statistics.
        For non-Anthropic providers, always returns 0 cache hits.
        """
        return {
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }

prompt_cache = PromptCacheHandler()
```

---

### Component 7: Multi-Turn Tool Loop Handler

**File:** `core/handlers/tool_loop.py`

#### Problem
Agentic tools often run in a loop:
1. Send request with tools
2. Model returns tool_use
3. Agent executes tool
4. Agent sends tool_result
5. Repeat until model returns text

Need to track and manage these loops properly.

#### Implementation

```python
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import uuid

@dataclass
class ToolExecution:
    tool_use_id: str
    tool_name: str
    input: Dict[str, Any]
    started_at: datetime
    completed_at: Optional[datetime] = None
    result: Optional[str] = None
    is_error: bool = False

@dataclass
class ToolLoopSession:
    session_id: str
    started_at: datetime
    turns: int = 0
    max_turns: int = 50  # Safety limit
    tool_executions: List[ToolExecution] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0

class ToolLoopHandler:
    """Track and manage agentic tool loops"""

    def __init__(self):
        self._sessions: Dict[str, ToolLoopSession] = {}

    def get_or_create_session(
        self,
        conversation_id: Optional[str] = None
    ) -> ToolLoopSession:
        """Get existing session or create new one"""
        if conversation_id and conversation_id in self._sessions:
            return self._sessions[conversation_id]

        session_id = conversation_id or f"session_{uuid.uuid4().hex[:8]}"
        session = ToolLoopSession(
            session_id=session_id,
            started_at=datetime.now()
        )
        self._sessions[session_id] = session
        return session

    def record_turn(
        self,
        session: ToolLoopSession,
        input_tokens: int,
        output_tokens: int,
        tool_uses: Optional[List[Dict]] = None
    ):
        """Record a conversation turn"""
        session.turns += 1
        session.total_input_tokens += input_tokens
        session.total_output_tokens += output_tokens

        if tool_uses:
            for tool in tool_uses:
                execution = ToolExecution(
                    tool_use_id=tool.get("id", ""),
                    tool_name=tool.get("name", ""),
                    input=tool.get("input", {}),
                    started_at=datetime.now()
                )
                session.tool_executions.append(execution)

    def record_tool_result(
        self,
        session: ToolLoopSession,
        tool_use_id: str,
        result: str,
        is_error: bool = False
    ):
        """Record tool execution result"""
        for execution in reversed(session.tool_executions):
            if execution.tool_use_id == tool_use_id:
                execution.completed_at = datetime.now()
                execution.result = result
                execution.is_error = is_error
                break

    def check_loop_limits(
        self,
        session: ToolLoopSession
    ) -> tuple[bool, Optional[str]]:
        """Check if loop should continue"""
        if session.turns >= session.max_turns:
            return False, f"Maximum turns ({session.max_turns}) exceeded"

        # Could add more checks:
        # - Token budget exceeded
        # - Time limit
        # - Error rate too high

        return True, None

    def get_session_stats(
        self,
        session: ToolLoopSession
    ) -> Dict[str, Any]:
        """Get session statistics"""
        return {
            "session_id": session.session_id,
            "turns": session.turns,
            "total_input_tokens": session.total_input_tokens,
            "total_output_tokens": session.total_output_tokens,
            "tool_calls": len(session.tool_executions),
            "duration_seconds": (datetime.now() - session.started_at).total_seconds()
        }

tool_loop_handler = ToolLoopHandler()
```

---

### Component 8: Model Capability Detection

**File:** `core/handlers/capabilities.py`

#### Problem
Different models have different capabilities. Agents need to know what's supported.

#### Implementation

```python
from typing import Dict, Set, Optional
from dataclasses import dataclass

@dataclass
class ModelCapabilities:
    supports_tools: bool = True
    supports_vision: bool = False
    supports_streaming: bool = True
    supports_system_prompt: bool = True
    supports_parallel_tools: bool = True
    supports_json_mode: bool = False
    supports_thinking: bool = False
    max_tools: int = 128
    tool_choice_modes: Set[str] = None  # auto, any, tool, none

    def __post_init__(self):
        if self.tool_choice_modes is None:
            self.tool_choice_modes = {"auto", "any", "tool", "none"}

# Model capabilities database
MODEL_CAPABILITIES: Dict[str, ModelCapabilities] = {
    # Claude models
    "claude-3-5-sonnet": ModelCapabilities(
        supports_tools=True,
        supports_vision=True,
        supports_streaming=True,
        supports_parallel_tools=True,
        supports_thinking=True,
        max_tools=128
    ),
    "claude-3-opus": ModelCapabilities(
        supports_tools=True,
        supports_vision=True,
        supports_streaming=True,
        supports_thinking=True
    ),

    # GPT models
    "gpt-4o": ModelCapabilities(
        supports_tools=True,
        supports_vision=True,
        supports_streaming=True,
        supports_json_mode=True,
        supports_parallel_tools=True
    ),
    "gpt-4o-mini": ModelCapabilities(
        supports_tools=True,
        supports_vision=True,
        supports_streaming=True,
        supports_json_mode=True
    ),

    # Qwen models (via Ollama)
    "qwen2.5-coder": ModelCapabilities(
        supports_tools=True,
        supports_vision=False,
        supports_streaming=True,
        supports_parallel_tools=True,
        max_tools=64
    ),

    # Llama models
    "llama-3.3-70b": ModelCapabilities(
        supports_tools=True,
        supports_vision=False,
        supports_streaming=True,
        max_tools=32
    ),

    # Deepseek
    "deepseek-coder": ModelCapabilities(
        supports_tools=True,
        supports_vision=False,
        supports_streaming=True,
        max_tools=32
    ),
}

class CapabilityDetector:
    """Detect and validate model capabilities"""

    def get_capabilities(self, model: str) -> ModelCapabilities:
        """Get capabilities for model"""
        # Exact match
        if model in MODEL_CAPABILITIES:
            return MODEL_CAPABILITIES[model]

        # Prefix match
        for key, caps in MODEL_CAPABILITIES.items():
            if model.startswith(key) or key in model.lower():
                return caps

        # Default conservative capabilities
        return ModelCapabilities(
            supports_tools=True,
            supports_vision=False,
            supports_streaming=True,
            max_tools=16
        )

    def validate_request(
        self,
        request: Dict,
        model: str
    ) -> tuple[bool, Optional[str]]:
        """Validate request against model capabilities"""
        caps = self.get_capabilities(model)

        # Check tools
        tools = request.get("tools", [])
        if tools and not caps.supports_tools:
            return False, f"Model {model} does not support tools"

        if len(tools) > caps.max_tools:
            return False, f"Too many tools ({len(tools)}), max is {caps.max_tools}"

        # Check vision
        has_images = self._has_images(request)
        if has_images and not caps.supports_vision:
            return False, f"Model {model} does not support vision"

        # Check streaming
        if request.get("stream") and not caps.supports_streaming:
            return False, f"Model {model} does not support streaming"

        return True, None

    def _has_images(self, request: Dict) -> bool:
        """Check if request contains images"""
        for msg in request.get("messages", []):
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "image":
                        return True
        return False

capability_detector = CapabilityDetector()
```

---

### Component 9: Streaming Heartbeat / Keep-Alive

**File:** `core/handlers/heartbeat.py`

#### Problem
Long-running generations may timeout. Need to send periodic keep-alive signals.

```python
import asyncio
from typing import AsyncIterator, TypeVar

T = TypeVar("T")

async def with_heartbeat(
    stream: AsyncIterator[T],
    heartbeat_interval: float = 15.0,
    heartbeat_value: T = None
) -> AsyncIterator[T]:
    """
    Wrap stream with periodic heartbeat.
    Emits heartbeat_value if no data received within interval.
    """
    async def heartbeat_generator():
        while True:
            await asyncio.sleep(heartbeat_interval)
            yield heartbeat_value

    heartbeat = heartbeat_generator()
    stream_iter = stream.__aiter__()
    heartbeat_iter = heartbeat.__aiter__()

    pending_stream = asyncio.create_task(stream_iter.__anext__())
    pending_heartbeat = asyncio.create_task(heartbeat_iter.__anext__())

    try:
        while True:
            done, _ = await asyncio.wait(
                [pending_stream, pending_heartbeat],
                return_when=asyncio.FIRST_COMPLETED
            )

            if pending_stream in done:
                try:
                    value = pending_stream.result()
                    yield value
                    pending_stream = asyncio.create_task(stream_iter.__anext__())
                except StopAsyncIteration:
                    break

            if pending_heartbeat in done:
                # Send heartbeat (SSE comment for keep-alive)
                if heartbeat_value is not None:
                    yield heartbeat_value
                pending_heartbeat = asyncio.create_task(heartbeat_iter.__anext__())

    finally:
        pending_stream.cancel()
        pending_heartbeat.cancel()
```

#### SSE Keep-Alive

```python
# In streaming response
async def stream_with_keepalive():
    async for event in translate_stream(upstream_stream, request):
        yield format_sse(event)

    # SSE comment for keep-alive (every 15s if no data)
    # ": keep-alive\n\n"
```

---

## Updated Component Summary Table

| Component | File | Purpose | Priority |
|-----------|------|---------|----------|
| **Agentic Components** | | | |
| Parallel Tool Handler | `handlers/parallel_tools.py` | Multiple tool calls in one turn | **Required** |
| Cancellation Handler | `handlers/cancellation.py` | Stop mid-stream cleanly | **Required** |
| Context Manager | `handlers/context_manager.py` | Large context window support | **Required** |
| Thinking Handler | `handlers/thinking.py` | Chain-of-thought blocks | Recommended |
| Beta Features Handler | `handlers/beta_features.py` | Handle anthropic-beta headers | **Required** |
| Prompt Cache Handler | `handlers/prompt_cache.py` | Handle cache_control blocks | Recommended |
| Tool Loop Handler | `handlers/tool_loop.py` | Track multi-turn tool loops | Recommended |
| Capability Detector | `handlers/capabilities.py` | Model capability validation | **Required** |
| Streaming Heartbeat | `handlers/heartbeat.py` | Keep-alive for long generations | Recommended |
| **Infrastructure** | | | |
| Structured Logger | `handlers/logger.py` | JSON logging with context | **Required** |
| Retry Handler | `handlers/retry.py` | Exponential backoff | **Required** |
| Circuit Breaker | `handlers/circuit_breaker.py` | Prevent cascading failures | **Required** |
| Rate Limiter | `handlers/rate_limiter.py` | Avoid rate limits | Recommended |
| Token Counter | `handlers/token_counter.py` | Estimate tokens | Recommended |

---

## Final Acceptance Criteria (Agentic MVP)

### Core Functionality
- [ ] `POST /v1/messages` accepts Anthropic-format requests
- [ ] Non-streaming responses return valid Anthropic JSON
- [ ] Streaming responses emit correct SSE event sequence
- [ ] Model aliases resolve to configured providers
- [ ] Errors return Anthropic-style error envelope
- [ ] Fallback providers activate on upstream failure

### Tool Calling (Critical for Agents)
- [ ] Single tool calls translate correctly
- [ ] **Parallel tool calls work (multiple tools in one response)**
- [ ] Tool results translate back correctly
- [ ] **Multi-turn tool loops work without state corruption**
- [ ] Tool choice modes (auto, any, tool, none) translate correctly

### Streaming (Critical for Agents)
- [ ] Text streaming works with proper SSE events
- [ ] **Tool call streaming works with partial JSON**
- [ ] **Client disconnect stops upstream request**
- [ ] Keep-alive prevents timeout on long generations

### Context & Tokens
- [ ] **Large contexts (100K+) handled efficiently**
- [ ] Token estimation works when provider doesn't return usage
- [ ] Context limit validation before sending to provider

### Compatibility Headers
- [ ] **anthropic-beta headers parsed and handled**
- [ ] cache_control blocks stripped for non-Anthropic providers
- [ ] Request-ID headers propagated through

### IDE Agent Verification
- [ ] **Claude Code can connect and run agentic tasks**
- [ ] **Roo Code can connect and run agentic tasks**
- [ ] **Cline can connect and run agentic tasks**
- [ ] Agents can execute multi-step file editing workflows

---

## Next Steps

1. **Create project skeleton** with directory structure
2. **Implement Pydantic models** for both protocols
3. **Build handlers** (logger, retry, circuit breaker)
4. **Build agentic handlers** (parallel tools, cancellation, context)
5. **Build middleware** (request ID, timing, error handler)
6. **Build adapters** as pure functions with unit tests
7. **Create FastAPI endpoint** with streaming support
8. **Add config loading** and model alias resolution
9. **Integration testing** with Ollama and Groq
10. **Agent compatibility testing** with Claude Code, Roo Code, Cline
11. **Performance benchmarking** and optimization
