# Shin Gateway

High-performance bidirectional API translation proxy for agentic IDE tools like Claude Code, Roo Code, Cline, and Kilo Code.

## Overview

Shin Gateway provides both Anthropic Messages API and OpenAI Chat Completions API endpoints, with full bidirectional translation. This enables you to:

- Use any OpenAI-compatible model with tools that expect the Anthropic API
- Use the gateway as an OpenAI-compatible endpoint that routes to any backend
- Mix and match providers seamlessly with a unified interface

### Key Features

- **Bidirectional Protocol Translation**: Full Anthropic ↔ OpenAI translation
- **Dual API Endpoints**: Both `/v1/messages` (Anthropic) and `/v1/chat/completions` (OpenAI)
- **Streaming Support**: Real-time SSE event translation with proper event sequences
- **Tool Calling**: Complete tool/function calling translation including parallel tools
- **Vision Support**: Image translation between Anthropic and OpenAI formats
- **Extended Thinking**: Emulation of Anthropic's thinking blocks for other providers
- **Multi-Provider**: Route to Ollama, Groq, OpenAI, NVIDIA, xAI, or any OpenAI-compatible endpoint
- **Model Aliasing**: Define custom model names that map to specific providers
- **Resilience**: Circuit breaker, rate limiting, retry with exponential backoff
- **Agentic Optimized**: Parallel tool calls, request cancellation, context window management
- **Token Preflight**: Validate requests won't exceed context window before sending
- **Stream Recovery**: Checkpoint system for recovering from streaming errors
- **Admin Dashboard**: Web UI for managing providers, models, API keys, and viewing real-time stats
- **Usage Analytics**: Track RPM, TPM, TPS, latency percentiles, and error rates per provider/model

## Quick Start

### 1. Install Dependencies

```bash
cd shin-gateway
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Configure Providers

Edit `config/config.yaml` to set up your providers and model aliases:

```yaml
providers:
  ollama_local:
    type: openai
    base_url: "http://localhost:11434/v1"

  groq:
    type: openai
    base_url: "https://api.groq.com/openai/v1"
    api_key_env: GROQ_API_KEY

models:
  claude-3-5-sonnet-20241022:
    provider: groq
    model: llama-3.3-70b-versatile
```

### 4. Start the Gateway

```bash
python main.py
```

Or with uvicorn directly:

```bash
uvicorn main:app --host 0.0.0.0 --port 8082
```

### 5. Configure Your IDE Tool

Point your agentic tool to the gateway:

```
ANTHROPIC_BASE_URL=http://localhost:8082
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SHIN_CONFIG_PATH` | Path to config file | `config/config.yaml` |
| `SHIN_LOG_LEVEL` | Log level (DEBUG, INFO, WARNING, ERROR) | `INFO` |
| `SHIN_DEBUG` | Enable debug mode | `false` |
| `OPENAI_API_KEY` | OpenAI API key | - |
| `GROQ_API_KEY` | Groq API key | - |
| `ANTHROPIC_API_KEY` | Anthropic API key (for passthrough) | - |

### Provider Configuration

```yaml
providers:
  provider_name:
    type: openai           # Provider type (openai, anthropic)
    base_url: "..."        # API base URL
    api_key: "..."         # Direct API key (not recommended)
    api_key_env: "..."     # Environment variable for API key
    timeout: 120           # Request timeout in seconds
    extra_headers: {}      # Additional headers
    rate_limit:            # Optional rate limiting
      requests_per_minute: 100
      tokens_per_minute: 100000
```

### Model Aliases

Map Anthropic model names to provider models:

```yaml
models:
  # When client requests claude-3-5-sonnet-20241022, route to Groq
  claude-3-5-sonnet-20241022:
    provider: groq
    model: llama-3.3-70b-versatile
    defaults:
      temperature: 0.7
      max_tokens: 8192

  # Local Ollama for coding tasks
  shin-coder:
    provider: ollama_local
    model: qwen2.5-coder:32b
```

## API Endpoints

### POST /v1/messages (Anthropic Format)

Main Anthropic Messages API endpoint. Accepts Anthropic-format requests and returns Anthropic-format responses.

```bash
curl -X POST http://localhost:8082/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-key" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### POST /v1/chat/completions (OpenAI Format)

OpenAI-compatible Chat Completions endpoint. Accepts OpenAI-format requests and returns OpenAI-format responses.

```bash
curl -X POST http://localhost:8082/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-key" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

This endpoint supports:
- Streaming with `"stream": true`
- Tool/function calling
- Vision (images in messages)
- All OpenAI parameters (temperature, top_p, stop, etc.)

### GET /v1/models

List available model aliases:

```bash
curl http://localhost:8082/v1/models
```

### GET /health

Liveness probe:

```bash
curl http://localhost:8082/health
```

### GET /ready

Readiness probe with component status:

```bash
curl http://localhost:8082/ready
```

## Admin Dashboard

Access the web-based admin dashboard at `http://localhost:8082/admin`

### Features

- **Overview**: Real-time stats, provider status, RPS/TPS metrics
- **Providers**: Add, edit, enable/disable providers with rate limits
- **Models**: Create and manage model aliases mapping to providers
- **API Keys**: Generate and manage gateway API keys with per-key rate limits
- **Statistics**: Usage analytics by provider, model, time range

### Admin API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/admin/` | GET | Admin dashboard UI |
| `/admin/providers` | GET/POST | List/create providers |
| `/admin/providers/{id}` | GET/PATCH/DELETE | Manage provider |
| `/admin/providers/{id}/toggle` | POST | Enable/disable provider |
| `/admin/models` | GET/POST | List/create model aliases |
| `/admin/models/{id}` | GET/PATCH/DELETE | Manage model |
| `/admin/api-keys` | GET/POST | List/create API keys |
| `/admin/api-keys/{id}` | GET/PATCH/DELETE | Manage API key |
| `/admin/api-keys/{id}/toggle` | POST | Enable/disable API key |
| `/admin/stats/overview` | GET | Overall statistics |
| `/admin/stats/live` | GET | Live metrics for dashboard |
| `/admin/stats/providers` | GET | Stats by provider |
| `/admin/stats/models` | GET | Stats by model |
| `/admin/stats/usage` | GET | Historical usage stats |

### Creating an API Key via API

```bash
curl -X POST http://localhost:8082/admin/api-keys \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Production Key",
    "rate_limit_rpm": 100,
    "rate_limit_tpm": 100000
  }'
```

Response includes the API key (shown only once):
```json
{
  "api_key": {...},
  "key": "sk-shin-abc123...",
  "message": "API key created. Save this key - it won't be shown again!"
}
```

## Streaming

The gateway fully supports Server-Sent Events (SSE) streaming with proper Anthropic event translation:

```bash
curl -X POST http://localhost:8082/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 1024,
    "stream": true,
    "messages": [{"role": "user", "content": "Count to 5"}]
  }'
```

Events are translated from OpenAI format to Anthropic format:
- `message_start` - Initial message with metadata
- `content_block_start` - Start of text or tool_use block
- `content_block_delta` - Incremental content (text_delta, input_json_delta)
- `content_block_stop` - End of content block
- `message_delta` - Stop reason and final usage
- `message_stop` - Stream complete

## Protocol Features

Shin Gateway includes advanced protocol translation features for full Claude Code compatibility:

### Vision/Image Support

Automatically translates image formats between Anthropic and OpenAI:

```json
// Anthropic format (input)
{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "..."}}

// OpenAI format (translated)
{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
```

### Extended Thinking Emulation

When using the `anthropic-beta: extended-thinking-2025-01-24` header with non-Anthropic providers, the gateway emulates thinking by adding instructions to the system prompt that encourage step-by-step reasoning.

### Token Preflight Checking

Before sending requests, the gateway validates that the input tokens + max_tokens won't exceed the model's context window, returning a clear error if the request would fail.

### Beta Features Support

Supported beta features via `anthropic-beta` header:
- `prompt-caching-2024-07-31` - Cache control blocks (stripped for non-Anthropic)
- `extended-thinking-2025-01-24` - Thinking blocks (emulated for OpenAI)
- `tools-2024-04-04` - Tool use support
- `output-128k-2025-02-19` - Extended output tokens
- `token-counting-2024-11-01` - Token counting

### Metadata Passthrough

User metadata is preserved across translations:
- Anthropic `metadata.user_id` ↔ OpenAI `user` field
- Request IDs and trace context preserved for debugging

## Tool Calling

Full support for parallel tool calls:

```json
{
  "model": "claude-3-5-sonnet-20241022",
  "max_tokens": 1024,
  "tools": [
    {
      "name": "get_weather",
      "description": "Get weather for a location",
      "input_schema": {
        "type": "object",
        "properties": {
          "location": {"type": "string"}
        },
        "required": ["location"]
      }
    }
  ],
  "messages": [{"role": "user", "content": "What's the weather in Tokyo?"}]
}
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Agentic IDE Tool                        │
│              (Claude Code / Roo Code / Cline)               │
└─────────────────────────┬───────────────────────────────────┘
                          │ Anthropic Messages API
                          │ or OpenAI Chat Completions API
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                      SHIN GATEWAY                           │
│  ┌────────────┐  ┌────────────┐  ┌────────────────────────┐ │
│  │ Middleware │  │  Protocol  │  │      Handlers          │ │
│  │            │  │  Adapters  │  │                        │ │
│  │ • RequestID│  │            │  │ • Circuit Breaker      │ │
│  │ • Timing   │  │ Anthropic  │  │ • Rate Limiter         │ │
│  │ • Errors   │  │    ↕       │  │ • Retry                │ │
│  │ • Auth     │  │  OpenAI    │  │ • Cancellation         │ │
│  │            │  │            │  │ • Token Preflight      │ │
│  │            │  │  Protocol  │  │ • Stream Recovery      │ │
│  │            │  │  Features  │  │ • Vision Handler       │ │
│  └────────────┘  └────────────┘  └────────────────────────┘ │
└─────────────────────────┬───────────────────────────────────┘
                          │ OpenAI Chat Completions API
          ┌───────────────┼───────────────┬───────────────┐
          ▼               ▼               ▼               ▼
   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
   │  Ollama  │    │   Groq   │    │  OpenAI  │    │  NVIDIA  │
   └──────────┘    └──────────┘    └──────────┘    └──────────┘
```

## Production Deployment

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8082
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8082"]
```

### Docker Compose

```yaml
version: '3.8'
services:
  shin-gateway:
    build: .
    ports:
      - "8082:8082"
    environment:
      - GROQ_API_KEY=${GROQ_API_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./config:/app/config:ro
    restart: unless-stopped
```

### Performance Tuning

For production, use uvloop on Linux:

```bash
pip install uvloop
uvicorn main:app --host 0.0.0.0 --port 8082 --loop uvloop --http h11
```

## Troubleshooting

### Model Not Found

Ensure the model alias is defined in `config/config.yaml`:

```yaml
models:
  your-model-alias:
    provider: your_provider
    model: actual-model-name
```

### Connection Errors

Check provider connectivity:

```bash
# For Ollama
curl http://localhost:11434/v1/models

# For Groq
curl -H "Authorization: Bearer $GROQ_API_KEY" https://api.groq.com/openai/v1/models
```

### Rate Limiting

Configure per-provider rate limits:

```yaml
providers:
  groq:
    rate_limit:
      requests_per_minute: 30
      tokens_per_minute: 50000
```

## License

MIT License
