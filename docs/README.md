# Shin Gateway Documentation

Welcome to the Shin Gateway documentation. This guide will help you install, configure, and use Shin Gateway.

## Quick Links

| Document | Description |
|----------|-------------|
| [Installation](./installation.md) | How to install Shin Gateway |
| [Usage Guide](./usage.md) | How to use the gateway with your tools |
| [Configuration](./configuration.md) | Complete configuration reference |
| [API Reference](./api-reference.md) | API endpoint documentation |
| [Architecture](./architecture.md) | System architecture overview |
| [Admin UI](./admin-ui.md) | Admin dashboard guide |
| [Deployment](./deployment.md) | Production deployment guide |
| [Troubleshooting](./troubleshooting.md) | Common issues and solutions |

## What is Shin Gateway?

Shin Gateway is a high-performance bidirectional API translation proxy for agentic IDE tools. It enables tools like Claude Code, Roo Code, Cline, and Kilo Code to work with any OpenAI-compatible LLM provider.

### Key Features

- **Bidirectional Translation**: Full Anthropic ↔ OpenAI protocol translation
- **Multi-Provider Support**: Route to Ollama, Groq, OpenAI, NVIDIA, xAI, and more
- **Streaming**: Real-time SSE event translation
- **Tool Calling**: Complete function/tool calling support
- **Resilience**: Circuit breaker, rate limiting, and retry logic
- **Admin Dashboard**: Web UI for management and monitoring

## Getting Started

### 1. Install

```bash
git clone https://github.com/shinmentakezoo7/shin-gateway.git
cd shin-gateway
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Run

```bash
python main.py
```

### 4. Use

```bash
export ANTHROPIC_BASE_URL=http://localhost:8082
# Your IDE tool will now use Shin Gateway
```

## Documentation Structure

```
docs/
├── README.md           # This file
├── installation.md     # Installation guide
├── usage.md           # Usage guide
├── configuration.md   # Configuration reference
├── api-reference.md   # API documentation
├── architecture.md    # Architecture overview
├── admin-ui.md        # Admin dashboard guide
├── deployment.md      # Deployment guide
└── troubleshooting.md # Troubleshooting guide
```

## Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/shinmentakezoo7/shin-gateway/issues)
- **Documentation**: You're reading it!

## License

MIT License
