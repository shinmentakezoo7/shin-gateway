# Shin Gateway

<div align="center">

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.11+-green.svg)
![License](https://img.shields.io/badge/license-MIT-purple.svg)

**High-performance bidirectional API translation proxy for agentic IDE tools**

[Installation](./docs/installation.md) | [Usage Guide](./docs/usage.md) | [Configuration](./docs/configuration.md) | [API Reference](./docs/api-reference.md)

</div>

---

## What is Shin Gateway?

Shin Gateway is a powerful API translation proxy designed for agentic IDE tools like **Claude Code**, **Roo Code**, **Cline**, and **Kilo Code**. It provides seamless bidirectional translation between Anthropic Messages API and OpenAI Chat Completions API formats.

### Why Use Shin Gateway?

- **Use Any Model with Any Tool**: Run Claude Code with Ollama, Groq, or any OpenAI-compatible provider
- **Cost Optimization**: Route requests to cheaper providers while maintaining API compatibility
- **Local Development**: Use local models (via Ollama) with tools designed for cloud APIs
- **Provider Flexibility**: Switch providers without changing your application code

## Key Features

| Feature | Description |
|---------|-------------|
| **Bidirectional Translation** | Full Anthropic ↔ OpenAI protocol translation |
| **Dual API Endpoints** | Both `/v1/messages` (Anthropic) and `/v1/chat/completions` (OpenAI) |
| **Streaming Support** | Real-time SSE event translation with proper event sequences |
| **Tool Calling** | Complete tool/function calling translation including parallel tools |
| **Vision Support** | Image translation between Anthropic and OpenAI formats |
| **Extended Thinking** | Emulation of Anthropic's thinking blocks for other providers |
| **Multi-Provider** | Route to Ollama, Groq, OpenAI, NVIDIA, xAI, or any OpenAI-compatible endpoint |
| **Model Aliasing** | Define custom model names that map to specific providers |
| **Resilience** | Circuit breaker, rate limiting, retry with exponential backoff |
| **Admin Dashboard** | Web UI for managing providers, models, API keys, and viewing stats |

## Tech Stack

### Backend
- **FastAPI** - High-performance async Python web framework
- **Uvicorn** - Lightning-fast ASGI server
- **httpx** - Modern async HTTP client with HTTP/2 support
- **Pydantic** - Data validation and settings management
- **SQLite** - Lightweight database for admin data

### Frontend (Admin UI)
- **Next.js 16** - React framework with App Router
- **TypeScript** - Type-safe JavaScript
- **Tailwind CSS** - Utility-first CSS framework
- **Radix UI** - Accessible component primitives
- **Recharts** - Composable charting library

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/shinmentakezoo7/shin-gateway.git
cd shin-gateway
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env with your API keys (GROQ_API_KEY, OPENAI_API_KEY, etc.)
```

### 3. Start the Gateway

```bash
python main.py
# Or use the convenience script
./start.sh
```

### 4. Point Your IDE Tool

```bash
export ANTHROPIC_BASE_URL=http://localhost:8082
```

That's it! Your agentic tool will now route through Shin Gateway.

## Documentation

| Document | Description |
|----------|-------------|
| [Installation Guide](./docs/installation.md) | Detailed installation instructions for all platforms |
| [Usage Guide](./docs/usage.md) | How to use Shin Gateway with your tools |
| [Configuration](./docs/configuration.md) | Complete configuration reference |
| [API Reference](./docs/api-reference.md) | Full API endpoint documentation |
| [Architecture](./docs/architecture.md) | System architecture and design |
| [Admin UI Guide](./docs/admin-ui.md) | Managing the gateway via web interface |
| [Deployment](./docs/deployment.md) | Production deployment guide |
| [Troubleshooting](./docs/troubleshooting.md) | Common issues and solutions |

## Project Structure

```
shin-gateway/
├── main.py                 # Application entry point
├── requirements.txt        # Python dependencies
├── config/
│   ├── config.yaml        # Main configuration file
│   └── settings.py        # Pydantic settings models
├── core/                   # Core proxy functionality
│   ├── proxy.py           # Main request handler
│   ├── adapters/          # Protocol translation
│   ├── handlers/          # Request handling (retry, rate limit, etc.)
│   ├── middleware/        # FastAPI middleware
│   └── models/            # Pydantic request/response models
├── admin/                  # Admin API and dashboard
│   ├── router.py          # Admin REST API
│   ├── models.py          # Database models
│   └── stats.py           # Statistics collector
├── admin-ui/              # Next.js admin dashboard
│   └── src/app/           # React pages and components
├── data/                   # SQLite database
└── docs/                   # Documentation
```

## Supported Providers

| Provider | Type | Description |
|----------|------|-------------|
| **Ollama** | Local | Run open-source models locally |
| **Groq** | Cloud | Ultra-fast inference |
| **OpenAI** | Cloud | GPT models |
| **Anthropic** | Cloud | Claude models (passthrough) |
| **NVIDIA** | Cloud | NVIDIA NIM endpoints |
| **xAI** | Cloud | Grok models |
| **Any OpenAI-compatible** | Various | Any API following OpenAI spec |

## Example Configuration

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

  shin-coder:
    provider: ollama_local
    model: qwen2.5-coder:32b
```

## Admin Dashboard

Access the web-based admin dashboard at `http://localhost:8082/admin`

- **Overview**: Real-time stats, provider status, RPS/TPS metrics
- **Providers**: Manage provider configurations
- **Models**: Create and edit model aliases
- **API Keys**: Generate and manage API keys
- **Statistics**: Usage analytics and charts

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting PRs.

## License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">

**[Documentation](./docs/)** | **[Report Bug](https://github.com/shinmentakezoo7/shin-gateway/issues)** | **[Request Feature](https://github.com/shinmentakezoo7/shin-gateway/issues)**

</div>
