# Usage Guide

Learn how to use Shin Gateway with your agentic IDE tools and applications.

## Table of Contents

- [Basic Usage](#basic-usage)
- [IDE Tool Configuration](#ide-tool-configuration)
- [Making API Requests](#making-api-requests)
- [Streaming Responses](#streaming-responses)
- [Tool/Function Calling](#toolfunction-calling)
- [Working with Images](#working-with-images)
- [Model Selection](#model-selection)
- [Authentication](#authentication)

---

## Basic Usage

### Starting the Gateway

```bash
# Standard start
python main.py

# Using the convenience script
./start.sh

# With uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8082

# Development mode with auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8082
```

### Stopping the Gateway

```bash
# If running in foreground: Ctrl+C

# Using the convenience script
./stop.sh

# Restart
./restart.sh
```

---

## IDE Tool Configuration

### Claude Code

Set the Anthropic base URL to point to Shin Gateway:

```bash
export ANTHROPIC_BASE_URL=http://localhost:8082
```

Or in your shell configuration (`~/.bashrc`, `~/.zshrc`):

```bash
echo 'export ANTHROPIC_BASE_URL=http://localhost:8082' >> ~/.bashrc
source ~/.bashrc
```

### Roo Code

In your Roo Code settings:

```json
{
  "anthropic.baseUrl": "http://localhost:8082"
}
```

### Cline (VS Code Extension)

1. Open VS Code Settings
2. Search for "Cline"
3. Set "Base URL" to `http://localhost:8082`

### Kilo Code

Configure via environment variable or settings file:

```bash
export ANTHROPIC_API_BASE=http://localhost:8082
```

### Generic OpenAI-Compatible Tools

For tools expecting OpenAI API:

```bash
export OPENAI_BASE_URL=http://localhost:8082/v1
export OPENAI_API_KEY=your-gateway-key  # If gateway auth is enabled
```

---

## Making API Requests

### Anthropic Messages API Format

```bash
curl -X POST http://localhost:8082/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-api-key" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 1024,
    "messages": [
      {"role": "user", "content": "Explain quantum computing in simple terms."}
    ]
  }'
```

### OpenAI Chat Completions Format

```bash
curl -X POST http://localhost:8082/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 1024,
    "messages": [
      {"role": "user", "content": "Explain quantum computing in simple terms."}
    ]
  }'
```

### Python Example

```python
import httpx

# Using Anthropic format
response = httpx.post(
    "http://localhost:8082/v1/messages",
    headers={
        "Content-Type": "application/json",
        "x-api-key": "your-api-key",
        "anthropic-version": "2023-06-01"
    },
    json={
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": "Hello!"}
        ]
    }
)

print(response.json())
```

### Using Anthropic Python SDK

```python
from anthropic import Anthropic

client = Anthropic(
    base_url="http://localhost:8082",
    api_key="your-api-key"  # Or gateway key if auth enabled
)

message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)

print(message.content)
```

### Using OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8082/v1",
    api_key="your-api-key"
)

response = client.chat.completions.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)

print(response.choices[0].message.content)
```

---

## Streaming Responses

### Anthropic Streaming

```bash
curl -X POST http://localhost:8082/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 1024,
    "stream": true,
    "messages": [{"role": "user", "content": "Count from 1 to 10"}]
  }'
```

### SSE Event Types

When streaming, you'll receive these event types:

| Event | Description |
|-------|-------------|
| `message_start` | Initial message with metadata |
| `content_block_start` | Start of text or tool_use block |
| `content_block_delta` | Incremental content updates |
| `content_block_stop` | End of content block |
| `message_delta` | Stop reason and final usage |
| `message_stop` | Stream complete |

### Python Streaming Example

```python
from anthropic import Anthropic

client = Anthropic(base_url="http://localhost:8082")

with client.messages.stream(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Write a haiku"}]
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

---

## Tool/Function Calling

Shin Gateway fully supports tool calling with parallel tool execution.

### Defining Tools

```json
{
  "model": "claude-3-5-sonnet-20241022",
  "max_tokens": 1024,
  "tools": [
    {
      "name": "get_weather",
      "description": "Get current weather for a location",
      "input_schema": {
        "type": "object",
        "properties": {
          "location": {
            "type": "string",
            "description": "City name"
          },
          "unit": {
            "type": "string",
            "enum": ["celsius", "fahrenheit"]
          }
        },
        "required": ["location"]
      }
    }
  ],
  "messages": [
    {"role": "user", "content": "What's the weather in Tokyo?"}
  ]
}
```

### Handling Tool Results

```json
{
  "model": "claude-3-5-sonnet-20241022",
  "max_tokens": 1024,
  "messages": [
    {"role": "user", "content": "What's the weather in Tokyo?"},
    {
      "role": "assistant",
      "content": [
        {
          "type": "tool_use",
          "id": "tool_123",
          "name": "get_weather",
          "input": {"location": "Tokyo", "unit": "celsius"}
        }
      ]
    },
    {
      "role": "user",
      "content": [
        {
          "type": "tool_result",
          "tool_use_id": "tool_123",
          "content": "Sunny, 22°C"
        }
      ]
    }
  ]
}
```

---

## Working with Images

### Sending Images (Base64)

```json
{
  "model": "claude-3-5-sonnet-20241022",
  "max_tokens": 1024,
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "image",
          "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
          }
        },
        {
          "type": "text",
          "text": "What is in this image?"
        }
      ]
    }
  ]
}
```

### Sending Images (URL)

```json
{
  "model": "claude-3-5-sonnet-20241022",
  "max_tokens": 1024,
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "image",
          "source": {
            "type": "url",
            "url": "https://example.com/image.png"
          }
        },
        {
          "type": "text",
          "text": "Describe this image"
        }
      ]
    }
  ]
}
```

---

## Model Selection

### Using Model Aliases

Model aliases defined in `config/config.yaml` allow you to use familiar model names:

```yaml
models:
  claude-3-5-sonnet-20241022:
    provider: groq
    model: llama-3.3-70b-versatile

  shin-coder:
    provider: ollama_local
    model: qwen2.5-coder:32b
```

Then use in requests:

```json
{"model": "claude-3-5-sonnet-20241022", ...}
{"model": "shin-coder", ...}
```

### Listing Available Models

```bash
curl http://localhost:8082/v1/models
```

Response:

```json
{
  "object": "list",
  "data": [
    {"id": "claude-3-5-sonnet-20241022", "object": "model", ...},
    {"id": "shin-coder", "object": "model", ...}
  ]
}
```

---

## Authentication

### Gateway API Keys

If gateway authentication is enabled, include your API key:

**Anthropic format:**
```bash
-H "x-api-key: sk-shin-your-key"
```

**OpenAI format:**
```bash
-H "Authorization: Bearer sk-shin-your-key"
```

### Creating API Keys

Via Admin Dashboard at `http://localhost:8082/admin/api-keys`

Or via API:

```bash
curl -X POST http://localhost:8082/admin/api-keys \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My App Key",
    "rate_limit_rpm": 100,
    "rate_limit_tpm": 100000
  }'
```

---

## Next Steps

- [Configuration Guide](./configuration.md) - Detailed configuration options
- [API Reference](./api-reference.md) - Complete API documentation
- [Troubleshooting](./troubleshooting.md) - Common issues and solutions
