# Architecture Overview

Understanding the internal architecture of Shin Gateway.

## Table of Contents

- [High-Level Architecture](#high-level-architecture)
- [Request Flow](#request-flow)
- [Core Components](#core-components)
- [Protocol Translation](#protocol-translation)
- [Resilience Patterns](#resilience-patterns)
- [Database Schema](#database-schema)
- [Directory Structure](#directory-structure)

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Agentic IDE Tools                                │
│              (Claude Code / Roo Code / Cline / Kilo Code)               │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                │ Anthropic Messages API
                                │ or OpenAI Chat Completions API
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           SHIN GATEWAY                                  │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                        FastAPI Application                       │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │   │
│  │  │ Middleware  │  │   Router    │  │     Admin Router        │  │   │
│  │  │             │  │             │  │                         │  │   │
│  │  │ • RequestID │  │ /v1/messages│  │ /admin/providers        │  │   │
│  │  │ • Timing    │  │ /v1/chat/.. │  │ /admin/models           │  │   │
│  │  │ • Errors    │  │ /v1/models  │  │ /admin/api-keys         │  │   │
│  │  │ • CORS      │  │ /health     │  │ /admin/stats            │  │   │
│  │  └─────────────┘  └──────┬──────┘  └─────────────────────────┘  │   │
│  └──────────────────────────┼──────────────────────────────────────┘   │
│                              │                                          │
│  ┌──────────────────────────┼──────────────────────────────────────┐   │
│  │                    Core Proxy Layer                              │   │
│  │                          │                                       │   │
│  │  ┌─────────────┐  ┌──────┴──────┐  ┌─────────────────────────┐  │   │
│  │  │  Protocol   │  │   Proxy     │  │      Handlers           │  │   │
│  │  │  Adapters   │  │   Handler   │  │                         │  │   │
│  │  │             │  │             │  │ • Circuit Breaker       │  │   │
│  │  │ Anthropic   │  │ • Route     │  │ • Rate Limiter          │  │   │
│  │  │    ↕        │  │ • Translate │  │ • Retry Logic           │  │   │
│  │  │  OpenAI     │  │ • Stream    │  │ • Token Counter         │  │   │
│  │  │             │  │ • Validate  │  │ • Cancellation          │  │   │
│  │  │ • Messages  │  │             │  │ • Context Manager       │  │   │
│  │  │ • Tools     │  │             │  │ • Capabilities          │  │   │
│  │  │ • Images    │  │             │  │                         │  │   │
│  │  │ • Stream    │  │             │  │                         │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│  ┌──────────────────────────┼──────────────────────────────────────┐   │
│  │                    Data Layer                                    │   │
│  │                          │                                       │   │
│  │  ┌─────────────┐  ┌──────┴──────┐  ┌─────────────────────────┐  │   │
│  │  │   Config    │  │   SQLite    │  │    Stats Collector      │  │   │
│  │  │   (YAML)    │  │  Database   │  │                         │  │   │
│  │  │             │  │             │  │ • Request metrics       │  │   │
│  │  │ • Providers │  │ • Providers │  │ • Token usage           │  │   │
│  │  │ • Models    │  │ • Models    │  │ • Latency tracking      │  │   │
│  │  │ • Settings  │  │ • API Keys  │  │ • Error rates           │  │   │
│  │  │             │  │ • Usage     │  │                         │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                │
                                │ OpenAI Chat Completions API
                                │ (or Anthropic for passthrough)
                                ▼
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
        ▼                       ▼                       ▼
┌──────────────┐       ┌──────────────┐       ┌──────────────┐
│    Ollama    │       │     Groq     │       │   OpenAI     │
│   (Local)    │       │   (Cloud)    │       │   (Cloud)    │
└──────────────┘       └──────────────┘       └──────────────┘
```

---

## Request Flow

### Anthropic → OpenAI Provider Flow

```
1. Request Received at /v1/messages
           │
           ▼
2. Middleware Processing
   • Add request ID
   • Start timing
   • CORS handling
           │
           ▼
3. Authentication
   • Validate x-api-key header
   • Check API key in database
   • Verify rate limits
           │
           ▼
4. Request Validation
   • Parse Anthropic request body
   • Validate required fields
   • Parse beta headers
           │
           ▼
5. Model Resolution
   • Look up model alias
   • Get provider configuration
   • Apply default parameters
           │
           ▼
6. Capability Check
   • Verify model supports requested features
   • Check context window limits
   • Token preflight validation
           │
           ▼
7. Circuit Breaker Check
   • Is circuit open? → Return 503
   • Is circuit closed? → Continue
   • Is circuit half-open? → Limited requests
           │
           ▼
8. Protocol Translation
   • Convert Anthropic messages → OpenAI format
   • Translate tools/functions
   • Convert images
   • Handle system prompt
           │
           ▼
9. Rate Limiter
   • Check RPM/TPM limits
   • Wait if necessary
   • Update counters
           │
           ▼
10. HTTP Request to Provider
    • Send translated request
    • Handle streaming if enabled
    • Retry on failure (with backoff)
           │
           ▼
11. Response Translation
    • Convert OpenAI response → Anthropic format
    • Translate tool calls
    • Map finish reasons
           │
           ▼
12. Stats Recording
    • Record request metrics
    • Update usage counters
    • Log latency
           │
           ▼
13. Response Returned
```

### Streaming Flow

```
Client                    Gateway                     Provider
  │                          │                           │
  │──── POST /v1/messages ──►│                           │
  │     stream: true         │                           │
  │                          │──── POST /chat/completions ►│
  │                          │     stream: true          │
  │                          │                           │
  │                          │◄──── SSE: chunk ──────────│
  │◄── SSE: message_start ───│                           │
  │                          │                           │
  │                          │◄──── SSE: chunk ──────────│
  │◄── SSE: content_delta ───│      (translated)        │
  │                          │                           │
  │        ...               │         ...               │
  │                          │                           │
  │                          │◄──── SSE: [DONE] ─────────│
  │◄── SSE: message_stop ────│                           │
  │                          │                           │
  │◄── Connection closed ────│                           │
```

---

## Core Components

### Proxy Handler (`core/proxy.py`)

The central request handling logic:

- **Route resolution**: Maps model aliases to providers
- **Request translation**: Converts between API formats
- **Provider communication**: HTTP client with connection pooling
- **Response handling**: Streaming and non-streaming responses

### Protocol Adapters (`core/adapters/`)

| Adapter | Purpose |
|---------|---------|
| `anthropic_to_openai.py` | Convert Anthropic requests to OpenAI |
| `openai_to_anthropic.py` | Convert OpenAI responses to Anthropic |
| `stream_translator.py` | Translate streaming SSE events |
| `errors.py` | Error format translation |

### Handlers (`core/handlers/`)

| Handler | Purpose |
|---------|---------|
| `circuit_breaker.py` | Prevent cascading failures |
| `rate_limiter.py` | Per-provider request limiting |
| `retry.py` | Exponential backoff retry |
| `token_counter.py` | Count tokens in requests |
| `cancellation.py` | Detect client disconnects |
| `capabilities.py` | Model feature detection |
| `context_manager.py` | Context window validation |

### Middleware (`core/middleware/`)

| Middleware | Purpose |
|------------|---------|
| `request_id.py` | Generate unique request IDs |
| `timing.py` | Track request duration |
| `error_handler.py` | Global error handling |

### Admin Module (`admin/`)

| Component | Purpose |
|-----------|---------|
| `router.py` | Admin REST API endpoints |
| `models.py` | SQLite ORM and CRUD operations |
| `stats.py` | Real-time statistics collection |

---

## Protocol Translation

### Message Translation

**Anthropic → OpenAI:**

```python
# Anthropic format
{
  "role": "user",
  "content": [
    {"type": "text", "text": "Hello"},
    {"type": "image", "source": {"type": "base64", "data": "..."}}
  ]
}

# OpenAI format (translated)
{
  "role": "user",
  "content": [
    {"type": "text", "text": "Hello"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
  ]
}
```

### Tool Translation

**Anthropic → OpenAI:**

```python
# Anthropic tool definition
{
  "name": "get_weather",
  "description": "Get weather",
  "input_schema": {"type": "object", "properties": {...}}
}

# OpenAI tool definition (translated)
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "Get weather",
    "parameters": {"type": "object", "properties": {...}}
  }
}
```

### Stop Reason Mapping

| Anthropic | OpenAI |
|-----------|--------|
| `end_turn` | `stop` |
| `tool_use` | `tool_calls` |
| `max_tokens` | `length` |
| `stop_sequence` | `stop` |

---

## Resilience Patterns

### Circuit Breaker

```
         ┌─────────────────────────────────┐
         │            CLOSED               │
         │      (normal operation)         │
         │                                 │
         │  failure_count < threshold      │
         └───────────────┬─────────────────┘
                         │
                         │ failure_count >= threshold
                         ▼
         ┌─────────────────────────────────┐
         │             OPEN                │
         │    (fail fast, no requests)     │
         │                                 │
         │      wait for timeout           │
         └───────────────┬─────────────────┘
                         │
                         │ timeout elapsed
                         ▼
         ┌─────────────────────────────────┐
         │          HALF-OPEN              │
         │   (limited probe requests)      │
         │                                 │
         │  success_count >= threshold     │──────► CLOSED
         │  failure occurs                 │──────► OPEN
         └─────────────────────────────────┘
```

### Retry with Exponential Backoff

```
Attempt 1 ──► Fail ──► Wait 1s
Attempt 2 ──► Fail ──► Wait 2s (+ jitter)
Attempt 3 ──► Fail ──► Wait 4s (+ jitter)
Attempt 4 ──► Success ──► Return response
```

### Rate Limiting

Token bucket algorithm with per-provider limits:

```
Bucket capacity: 100 requests
Refill rate: 100 requests/minute

Request arrives:
├── Bucket has tokens? → Consume token, proceed
└── Bucket empty? → Wait for refill or return 429
```

---

## Database Schema

### Entity Relationship

```
┌───────────────┐       ┌───────────────┐
│   providers   │       │    models     │
├───────────────┤       ├───────────────┤
│ id (PK)       │◄──────│ id (PK)       │
│ name          │       │ name          │
│ type          │       │ provider_id(FK)│
│ base_url      │       │ model         │
│ api_key_hash  │       │ defaults      │
│ enabled       │       │ enabled       │
│ rate_limit    │       │ context_window│
│ created_at    │       │ created_at    │
└───────────────┘       └───────────────┘
                               │
                               │
┌───────────────┐       ┌──────┴────────┐
│   api_keys    │       │ usage_records │
├───────────────┤       ├───────────────┤
│ id (PK)       │       │ id (PK)       │
│ name          │       │ timestamp     │
│ key_hash      │       │ provider      │
│ key_prefix    │       │ model         │
│ rate_limit_rpm│       │ input_tokens  │
│ rate_limit_tpm│       │ output_tokens │
│ enabled       │       │ latency_ms    │
│ created_at    │       │ status        │
│ last_used_at  │       │ error_type    │
└───────────────┘       └───────────────┘
```

---

## Directory Structure

```
shin-gateway/
├── main.py                     # Application entry, FastAPI factory
│
├── config/
│   ├── config.yaml            # Main YAML configuration
│   └── settings.py            # Pydantic settings models
│
├── core/
│   ├── proxy.py               # Main proxy handler (~1500 lines)
│   │
│   ├── adapters/              # Protocol translation
│   │   ├── __init__.py
│   │   ├── anthropic_to_openai.py
│   │   ├── openai_to_anthropic.py
│   │   ├── stream_translator.py
│   │   └── errors.py
│   │
│   ├── handlers/              # Request handling
│   │   ├── __init__.py
│   │   ├── circuit_breaker.py
│   │   ├── rate_limiter.py
│   │   ├── retry.py
│   │   ├── token_counter.py
│   │   ├── cancellation.py
│   │   ├── capabilities.py
│   │   ├── context_manager.py
│   │   ├── logger.py
│   │   ├── beta_features.py
│   │   └── protocol_features.py
│   │
│   ├── middleware/            # FastAPI middleware
│   │   ├── __init__.py
│   │   ├── request_id.py
│   │   ├── timing.py
│   │   └── error_handler.py
│   │
│   └── models/                # Pydantic types
│       ├── __init__.py
│       ├── anthropic_types.py
│       └── openai_types.py
│
├── admin/
│   ├── router.py              # Admin API endpoints
│   ├── models.py              # SQLite database layer
│   ├── stats.py               # Statistics collector
│   ├── templates/             # Jinja2 templates
│   └── static/                # Static assets
│
├── admin-ui/                   # Next.js dashboard
│   ├── package.json
│   ├── next.config.js
│   ├── tailwind.config.js
│   └── src/
│       ├── app/               # Next.js App Router
│       ├── components/        # React components
│       ├── hooks/             # Custom hooks
│       └── lib/               # Utilities
│
├── data/                       # SQLite database
│   └── admin.db
│
├── tests/                      # Test files
│
├── requirements.txt           # Python dependencies
├── .env                       # Environment variables
└── .env.example               # Example environment file
```

---

## Next Steps

- [Configuration Guide](./configuration.md) - Configure the gateway
- [Deployment](./deployment.md) - Production deployment
- [API Reference](./api-reference.md) - Complete API docs
