# API Reference

Complete API documentation for Shin Gateway.

## Table of Contents

- [Overview](#overview)
- [Authentication](#authentication)
- [Core Endpoints](#core-endpoints)
- [Admin Endpoints](#admin-endpoints)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)

---

## Overview

Shin Gateway provides two primary API interfaces:

| Format | Base Path | Description |
|--------|-----------|-------------|
| Anthropic | `/v1/messages` | Anthropic Messages API compatible |
| OpenAI | `/v1/chat/completions` | OpenAI Chat Completions API compatible |

### Base URL

```
http://localhost:8082
```

### Content Type

All requests should use:
```
Content-Type: application/json
```

---

## Authentication

### Anthropic Format

```bash
-H "x-api-key: your-api-key"
-H "anthropic-version: 2023-06-01"
```

### OpenAI Format

```bash
-H "Authorization: Bearer your-api-key"
```

---

## Core Endpoints

### POST /v1/messages

Anthropic Messages API endpoint.

#### Request

```bash
curl -X POST http://localhost:8082/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-key" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 1024,
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

#### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model` | string | Yes | Model name or alias |
| `max_tokens` | integer | Yes | Maximum tokens in response |
| `messages` | array | Yes | Conversation messages |
| `system` | string | No | System prompt |
| `temperature` | number | No | Sampling temperature (0-1) |
| `top_p` | number | No | Nucleus sampling parameter |
| `top_k` | integer | No | Top-k sampling parameter |
| `stop_sequences` | array | No | Stop sequences |
| `stream` | boolean | No | Enable streaming (default: false) |
| `tools` | array | No | Tool definitions |
| `tool_choice` | object | No | Tool selection preference |
| `metadata` | object | No | Request metadata |

#### Message Object

```json
{
  "role": "user",  // "user" or "assistant"
  "content": "Hello!"  // string or array of content blocks
}
```

#### Content Block Types

**Text:**
```json
{"type": "text", "text": "Hello!"}
```

**Image (base64):**
```json
{
  "type": "image",
  "source": {
    "type": "base64",
    "media_type": "image/png",
    "data": "base64-encoded-data"
  }
}
```

**Tool Use:**
```json
{
  "type": "tool_use",
  "id": "tool_123",
  "name": "get_weather",
  "input": {"location": "Tokyo"}
}
```

**Tool Result:**
```json
{
  "type": "tool_result",
  "tool_use_id": "tool_123",
  "content": "Sunny, 22°C"
}
```

#### Response

```json
{
  "id": "msg_abc123",
  "type": "message",
  "role": "assistant",
  "content": [
    {
      "type": "text",
      "text": "Hello! How can I help you today?"
    }
  ],
  "model": "claude-3-5-sonnet-20241022",
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "usage": {
    "input_tokens": 10,
    "output_tokens": 25
  }
}
```

---

### POST /v1/chat/completions

OpenAI Chat Completions API endpoint.

#### Request

```bash
curl -X POST http://localhost:8082/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-key" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 1024,
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

#### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model` | string | Yes | Model name or alias |
| `messages` | array | Yes | Conversation messages |
| `max_tokens` | integer | No | Maximum tokens in response |
| `temperature` | number | No | Sampling temperature (0-2) |
| `top_p` | number | No | Nucleus sampling parameter |
| `n` | integer | No | Number of completions |
| `stream` | boolean | No | Enable streaming (default: false) |
| `stop` | array | No | Stop sequences |
| `functions` | array | No | Function definitions (legacy) |
| `tools` | array | No | Tool definitions |
| `tool_choice` | string/object | No | Tool selection |
| `user` | string | No | User identifier |

#### Response

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "claude-3-5-sonnet-20241022",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help you today?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 25,
    "total_tokens": 35
  }
}
```

---

### GET /v1/models

List available model aliases.

#### Request

```bash
curl http://localhost:8082/v1/models
```

#### Response

```json
{
  "object": "list",
  "data": [
    {
      "id": "claude-3-5-sonnet-20241022",
      "object": "model",
      "created": 1234567890,
      "owned_by": "shin-gateway"
    },
    {
      "id": "shin-coder",
      "object": "model",
      "created": 1234567890,
      "owned_by": "shin-gateway"
    }
  ]
}
```

---

### GET /health

Liveness probe endpoint.

#### Request

```bash
curl http://localhost:8082/health
```

#### Response

```json
{
  "status": "healthy"
}
```

---

### GET /ready

Readiness probe endpoint with component status.

#### Request

```bash
curl http://localhost:8082/ready
```

#### Response

```json
{
  "status": "ready",
  "components": {
    "database": "healthy",
    "providers": {
      "groq": "healthy",
      "ollama_local": "healthy"
    }
  }
}
```

---

## Admin Endpoints

### Providers

#### GET /admin/providers

List all providers.

```bash
curl http://localhost:8082/admin/providers
```

#### POST /admin/providers

Create a new provider.

```bash
curl -X POST http://localhost:8082/admin/providers \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my_provider",
    "type": "openai",
    "base_url": "https://api.example.com/v1",
    "api_key": "sk-xxx",
    "enabled": true
  }'
```

#### GET /admin/providers/{id}

Get provider details.

#### PATCH /admin/providers/{id}

Update a provider.

#### DELETE /admin/providers/{id}

Delete a provider.

#### POST /admin/providers/{id}/toggle

Enable/disable a provider.

#### GET /admin/providers/{id}/models

Fetch available models from provider.

---

### Models

#### GET /admin/models

List all model aliases.

```bash
curl http://localhost:8082/admin/models
```

#### POST /admin/models

Create a new model alias.

```bash
curl -X POST http://localhost:8082/admin/models \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my-model",
    "provider_id": 1,
    "model": "llama-3.3-70b",
    "enabled": true,
    "defaults": {
      "temperature": 0.7
    }
  }'
```

#### GET /admin/models/{id}

Get model details.

#### PATCH /admin/models/{id}

Update a model alias.

#### DELETE /admin/models/{id}

Delete a model alias.

---

### API Keys

#### GET /admin/api-keys

List all API keys (keys are masked).

```bash
curl http://localhost:8082/admin/api-keys
```

#### POST /admin/api-keys

Create a new API key.

```bash
curl -X POST http://localhost:8082/admin/api-keys \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Production Key",
    "rate_limit_rpm": 100,
    "rate_limit_tpm": 100000
  }'
```

**Response:**
```json
{
  "api_key": {...},
  "key": "sk-shin-abc123...",
  "message": "API key created. Save this key - it won't be shown again!"
}
```

#### GET /admin/api-keys/{id}

Get API key details.

#### PATCH /admin/api-keys/{id}

Update an API key.

#### DELETE /admin/api-keys/{id}

Delete an API key.

#### POST /admin/api-keys/{id}/toggle

Enable/disable an API key.

---

### Statistics

#### GET /admin/stats/overview

Get overall statistics.

```bash
curl http://localhost:8082/admin/stats/overview
```

#### GET /admin/stats/live

Get live metrics (for dashboard).

#### GET /admin/stats/providers

Get statistics by provider.

```bash
curl "http://localhost:8082/admin/stats/providers?period=24h"
```

#### GET /admin/stats/models

Get statistics by model.

```bash
curl "http://localhost:8082/admin/stats/models?period=7d"
```

#### GET /admin/stats/usage

Get historical usage data.

```bash
curl "http://localhost:8082/admin/stats/usage?start=2024-01-01&end=2024-01-31"
```

---

## Streaming

### SSE Event Format

When `stream: true`, responses are Server-Sent Events:

```
event: message_start
data: {"type":"message_start","message":{"id":"msg_123",...}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":25}}

event: message_stop
data: {"type":"message_stop"}
```

---

## Error Handling

### Error Response Format

```json
{
  "type": "error",
  "error": {
    "type": "invalid_request_error",
    "message": "Description of the error"
  }
}
```

### Error Types

| Type | HTTP Status | Description |
|------|-------------|-------------|
| `invalid_request_error` | 400 | Malformed request |
| `authentication_error` | 401 | Invalid or missing API key |
| `permission_error` | 403 | Insufficient permissions |
| `not_found_error` | 404 | Resource not found |
| `rate_limit_error` | 429 | Rate limit exceeded |
| `api_error` | 500 | Internal server error |
| `overloaded_error` | 529 | Service overloaded |

---

## Rate Limiting

### Rate Limit Headers

Responses include rate limit information:

```
X-RateLimit-Limit-Requests: 100
X-RateLimit-Limit-Tokens: 100000
X-RateLimit-Remaining-Requests: 95
X-RateLimit-Remaining-Tokens: 98500
X-RateLimit-Reset-Requests: 60
X-RateLimit-Reset-Tokens: 60
```

### Rate Limit Error

```json
{
  "type": "error",
  "error": {
    "type": "rate_limit_error",
    "message": "Rate limit exceeded. Please retry after 60 seconds."
  }
}
```

---

## Beta Features

### Anthropic Beta Header

Enable beta features via header:

```bash
-H "anthropic-beta: prompt-caching-2024-07-31,extended-thinking-2025-01-24"
```

### Supported Beta Features

| Feature | Description |
|---------|-------------|
| `prompt-caching-2024-07-31` | Enable prompt caching |
| `extended-thinking-2025-01-24` | Enable thinking blocks |
| `tools-2024-04-04` | Tool use support |
| `output-128k-2025-02-19` | Extended output tokens |
| `token-counting-2024-11-01` | Token counting |

---

## Next Steps

- [Usage Guide](./usage.md) - Practical usage examples
- [Configuration](./configuration.md) - Configuration options
- [Troubleshooting](./troubleshooting.md) - Common issues
