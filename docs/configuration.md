# Configuration Guide

Complete reference for configuring Shin Gateway.

## Table of Contents

- [Configuration Files](#configuration-files)
- [Environment Variables](#environment-variables)
- [Gateway Settings](#gateway-settings)
- [Provider Configuration](#provider-configuration)
- [Model Aliases](#model-aliases)
- [Rate Limiting](#rate-limiting)
- [Circuit Breaker](#circuit-breaker)
- [Retry Settings](#retry-settings)
- [Logging](#logging)

---

## Configuration Files

Shin Gateway uses two main configuration sources:

| File | Purpose |
|------|---------|
| `.env` | Environment variables and secrets |
| `config/config.yaml` | Main configuration (providers, models, settings) |

### Configuration Loading Order

1. Default values from code
2. `config/config.yaml`
3. Environment variables (override YAML)
4. Admin dashboard settings (stored in SQLite)

---

## Environment Variables

### Core Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `SHIN_CONFIG_PATH` | Path to config.yaml | `config/config.yaml` |
| `SHIN_LOG_LEVEL` | Log level (DEBUG, INFO, WARNING, ERROR) | `INFO` |
| `SHIN_DEBUG` | Enable debug mode | `false` |
| `SHIN_HOST` | Server bind address | `0.0.0.0` |
| `SHIN_PORT` | Server port | `8082` |

### API Keys (Secrets)

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key |
| `GROQ_API_KEY` | Groq API key |
| `ANTHROPIC_API_KEY` | Anthropic API key (for passthrough) |
| `NVIDIA_API_KEY` | NVIDIA NIM API key |
| `XAI_API_KEY` | xAI (Grok) API key |

### Example .env File

```env
# Server settings
SHIN_HOST=0.0.0.0
SHIN_PORT=8082
SHIN_LOG_LEVEL=INFO
SHIN_DEBUG=false

# Provider API keys
GROQ_API_KEY=gsk_xxxxxxxxxxxxx
OPENAI_API_KEY=sk-xxxxxxxxxxxxx
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxx

# Optional
NVIDIA_API_KEY=nvapi-xxxxxxxxxxxxx
XAI_API_KEY=xai-xxxxxxxxxxxxx
```

---

## Gateway Settings

### Basic Gateway Configuration

```yaml
gateway:
  host: "0.0.0.0"
  port: 8082

  # API key requirement for gateway access
  require_api_key: false

  # CORS settings
  cors:
    enabled: true
    allow_origins: ["*"]
    allow_methods: ["GET", "POST", "OPTIONS"]
    allow_headers: ["*"]

  # Request timeouts
  timeout:
    connect: 10.0      # Connection timeout (seconds)
    read: 120.0        # Read timeout (seconds)
    write: 120.0       # Write timeout (seconds)
    pool: 5.0          # Connection pool timeout (seconds)
```

### CORS Configuration Examples

**Development (allow all):**
```yaml
cors:
  enabled: true
  allow_origins: ["*"]
```

**Production (specific origins):**
```yaml
cors:
  enabled: true
  allow_origins:
    - "https://myapp.com"
    - "https://admin.myapp.com"
  allow_credentials: true
```

---

## Provider Configuration

### Provider Structure

```yaml
providers:
  provider_name:
    type: openai              # Provider type (openai, anthropic)
    base_url: "..."           # API base URL
    api_key: "..."            # Direct API key (not recommended)
    api_key_env: "..."        # Environment variable for API key
    enabled: true             # Enable/disable provider
    timeout: 120              # Request timeout (seconds)
    extra_headers: {}         # Additional headers to send
    rate_limit:               # Optional rate limiting
      requests_per_minute: 100
      tokens_per_minute: 100000
```

### Provider Examples

#### Ollama (Local)

```yaml
providers:
  ollama_local:
    type: openai
    base_url: "http://localhost:11434/v1"
    enabled: true
    timeout: 300  # Longer timeout for local models
```

#### Groq

```yaml
providers:
  groq:
    type: openai
    base_url: "https://api.groq.com/openai/v1"
    api_key_env: GROQ_API_KEY
    enabled: true
    rate_limit:
      requests_per_minute: 30
      tokens_per_minute: 50000
```

#### OpenAI

```yaml
providers:
  openai:
    type: openai
    base_url: "https://api.openai.com/v1"
    api_key_env: OPENAI_API_KEY
    enabled: true
```

#### Anthropic (Passthrough)

```yaml
providers:
  anthropic:
    type: anthropic
    base_url: "https://api.anthropic.com"
    api_key_env: ANTHROPIC_API_KEY
    enabled: true
```

#### NVIDIA NIM

```yaml
providers:
  nvidia:
    type: openai
    base_url: "https://integrate.api.nvidia.com/v1"
    api_key_env: NVIDIA_API_KEY
    enabled: true
```

#### xAI (Grok)

```yaml
providers:
  xai:
    type: openai
    base_url: "https://api.x.ai/v1"
    api_key_env: XAI_API_KEY
    enabled: true
```

#### Azure OpenAI

```yaml
providers:
  azure_openai:
    type: openai
    base_url: "https://your-resource.openai.azure.com/openai/deployments/your-deployment"
    api_key_env: AZURE_OPENAI_API_KEY
    extra_headers:
      api-version: "2024-02-15-preview"
    enabled: true
```

---

## Model Aliases

Model aliases map familiar model names to specific providers and their models.

### Basic Model Alias

```yaml
models:
  claude-3-5-sonnet-20241022:
    provider: groq
    model: llama-3.3-70b-versatile
```

### Model with Defaults

```yaml
models:
  shin-coder:
    provider: ollama_local
    model: qwen2.5-coder:32b
    defaults:
      temperature: 0.1
      max_tokens: 8192
      top_p: 0.95
```

### Model with Fallbacks

```yaml
models:
  claude-3-5-sonnet-20241022:
    provider: groq
    model: llama-3.3-70b-versatile
    fallbacks:
      - provider: ollama_local
        model: qwen2.5-coder:32b
      - provider: openai
        model: gpt-4o
```

### Context Window Override

```yaml
models:
  custom-model:
    provider: ollama_local
    model: my-fine-tuned-model
    context_window: 32768  # Override detected context window
```

### Complete Model Example

```yaml
models:
  shin-opus:
    provider: groq
    model: llama-3.3-70b-versatile
    enabled: true
    context_window: 128000
    defaults:
      temperature: 0.7
      max_tokens: 4096
      top_p: 0.9
    fallbacks:
      - provider: openai
        model: gpt-4o
    metadata:
      description: "Primary model for complex tasks"
      cost_tier: "medium"
```

---

## Rate Limiting

### Provider-Level Rate Limits

```yaml
providers:
  groq:
    type: openai
    base_url: "https://api.groq.com/openai/v1"
    api_key_env: GROQ_API_KEY
    rate_limit:
      requests_per_minute: 30    # Max requests per minute
      tokens_per_minute: 50000   # Max tokens per minute
      concurrent_requests: 5     # Max concurrent requests
```

### API Key Rate Limits

Set per-key rate limits when creating API keys:

```bash
curl -X POST http://localhost:8082/admin/api-keys \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Limited Key",
    "rate_limit_rpm": 10,
    "rate_limit_tpm": 10000
  }'
```

---

## Circuit Breaker

The circuit breaker prevents cascading failures when providers are unhealthy.

```yaml
circuit_breaker:
  enabled: true
  failure_threshold: 5       # Failures before opening
  success_threshold: 3       # Successes to close
  timeout: 60                # Seconds to wait before half-open
  half_open_requests: 3      # Requests allowed in half-open state
```

### Circuit Breaker States

| State | Description |
|-------|-------------|
| **Closed** | Normal operation, requests pass through |
| **Open** | Requests immediately fail, no backend calls |
| **Half-Open** | Limited requests allowed to test recovery |

---

## Retry Settings

Configure automatic retry behavior for failed requests.

```yaml
retry:
  enabled: true
  max_retries: 3              # Maximum retry attempts
  initial_delay: 1.0          # Initial delay (seconds)
  max_delay: 30.0             # Maximum delay (seconds)
  exponential_base: 2         # Exponential backoff multiplier
  jitter: true                # Add random jitter

  # HTTP status codes to retry
  retry_on_status:
    - 429  # Rate limited
    - 500  # Server error
    - 502  # Bad gateway
    - 503  # Service unavailable
    - 504  # Gateway timeout
```

---

## Logging

### Log Levels

| Level | Description |
|-------|-------------|
| `DEBUG` | Detailed debugging information |
| `INFO` | General operational information |
| `WARNING` | Warning messages |
| `ERROR` | Error messages only |

### Configuration

```yaml
logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

  # File logging (optional)
  file:
    enabled: false
    path: "logs/shin-gateway.log"
    max_size: 10485760  # 10 MB
    backup_count: 5
```

### Environment Override

```bash
export SHIN_LOG_LEVEL=DEBUG
```

---

## Complete Configuration Example

```yaml
# config/config.yaml

gateway:
  host: "0.0.0.0"
  port: 8082
  require_api_key: false
  cors:
    enabled: true
    allow_origins: ["*"]
  timeout:
    connect: 10.0
    read: 120.0

retry:
  enabled: true
  max_retries: 3
  initial_delay: 1.0
  max_delay: 30.0
  exponential_base: 2
  jitter: true

circuit_breaker:
  enabled: true
  failure_threshold: 5
  success_threshold: 3
  timeout: 60

providers:
  ollama_local:
    type: openai
    base_url: "http://localhost:11434/v1"
    enabled: true
    timeout: 300

  groq:
    type: openai
    base_url: "https://api.groq.com/openai/v1"
    api_key_env: GROQ_API_KEY
    enabled: true
    rate_limit:
      requests_per_minute: 30
      tokens_per_minute: 50000

  openai:
    type: openai
    base_url: "https://api.openai.com/v1"
    api_key_env: OPENAI_API_KEY
    enabled: true

models:
  claude-3-5-sonnet-20241022:
    provider: groq
    model: llama-3.3-70b-versatile
    defaults:
      temperature: 0.7
      max_tokens: 4096
    fallbacks:
      - provider: ollama_local
        model: qwen2.5-coder:32b

  shin-coder:
    provider: ollama_local
    model: qwen2.5-coder:32b
    defaults:
      temperature: 0.1
      max_tokens: 8192
```

---

## Next Steps

- [API Reference](./api-reference.md) - Complete API documentation
- [Admin UI Guide](./admin-ui.md) - Manage configuration via web UI
- [Deployment](./deployment.md) - Production deployment guide
