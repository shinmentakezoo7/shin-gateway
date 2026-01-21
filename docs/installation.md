# Installation Guide

This guide covers how to install and set up Shin Gateway on your system.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation Methods](#installation-methods)
  - [Standard Installation](#standard-installation)
  - [Docker Installation](#docker-installation)
  - [Development Installation](#development-installation)
- [Platform-Specific Instructions](#platform-specific-instructions)
- [Verifying Installation](#verifying-installation)

---

## Prerequisites

### System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.11+ | 3.12+ |
| RAM | 512 MB | 1 GB+ |
| Disk Space | 100 MB | 500 MB |
| Node.js (for Admin UI) | 18+ | 20+ |

### Required Software

- **Python 3.11 or higher** - [Download Python](https://python.org)
- **pip** - Python package manager (included with Python)
- **Git** - For cloning the repository

### Optional Software

- **Docker** - For containerized deployment
- **Node.js 18+** - For building the Admin UI from source
- **Ollama** - For running local models

---

## Installation Methods

### Standard Installation

This is the recommended method for most users.

#### Step 1: Clone the Repository

```bash
git clone https://github.com/shinmentakezoo7/shin-gateway.git
cd shin-gateway
```

#### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate on Linux/macOS
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

#### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

#### Step 4: Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit with your API keys
nano .env  # or use your preferred editor
```

Add your API keys:

```env
# Required for cloud providers
GROQ_API_KEY=your-groq-key
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key

# Gateway settings
SHIN_LOG_LEVEL=INFO
SHIN_DEBUG=false
```

#### Step 5: Start the Gateway

```bash
python main.py
```

Or use the convenience script:

```bash
./start.sh
```

---

### Docker Installation

For containerized deployment:

#### Step 1: Build the Image

```bash
docker build -t shin-gateway .
```

#### Step 2: Run the Container

```bash
docker run -d \
  --name shin-gateway \
  -p 8082:8082 \
  -e GROQ_API_KEY=your-key \
  -e OPENAI_API_KEY=your-key \
  -v $(pwd)/config:/app/config:ro \
  shin-gateway
```

#### Using Docker Compose

Create a `docker-compose.yml`:

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
      - ./data:/app/data
    restart: unless-stopped
```

Then run:

```bash
docker-compose up -d
```

---

### Development Installation

For contributing or modifying the code:

#### Step 1: Clone and Setup Backend

```bash
git clone https://github.com/shinmentakezoo7/shin-gateway.git
cd shin-gateway

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies with dev extras
pip install -r requirements.txt
pip install pytest pytest-asyncio httpx[cli]
```

#### Step 2: Setup Admin UI (Optional)

```bash
cd admin-ui

# Install Node.js dependencies
npm install

# Start development server
npm run dev
```

#### Step 3: Run in Development Mode

```bash
# Backend with auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8082

# Admin UI development server (in separate terminal)
cd admin-ui && npm run dev
```

---

## Platform-Specific Instructions

### Linux (Ubuntu/Debian)

```bash
# Install Python 3.11+
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip

# Clone and install
git clone https://github.com/shinmentakezoo7/shin-gateway.git
cd shin-gateway
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### macOS

```bash
# Install Python via Homebrew
brew install python@3.12

# Clone and install
git clone https://github.com/shinmentakezoo7/shin-gateway.git
cd shin-gateway
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Windows

```powershell
# Install Python from python.org or via winget
winget install Python.Python.3.12

# Clone and install
git clone https://github.com/shinmentakezoo7/shin-gateway.git
cd shin-gateway
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Windows (WSL2)

Follow the Linux instructions within your WSL2 distribution.

---

## Verifying Installation

### Check Gateway is Running

```bash
# Health check
curl http://localhost:8082/health
# Expected: {"status": "healthy"}

# Readiness check
curl http://localhost:8082/ready
# Expected: {"status": "ready", "components": {...}}
```

### Test API Endpoint

```bash
curl -X POST http://localhost:8082/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 100,
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Access Admin Dashboard

Open your browser to: `http://localhost:8082/admin`

---

## Installing Ollama (Optional)

If you want to use local models:

### Linux

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### macOS

```bash
brew install ollama
```

### Windows

Download from [ollama.com](https://ollama.com)

### Pull a Model

```bash
ollama pull qwen2.5-coder:32b
# Or a smaller model
ollama pull llama3.2:3b
```

---

## Next Steps

- [Configuration Guide](./configuration.md) - Configure providers and models
- [Usage Guide](./usage.md) - Learn how to use with your IDE tools
- [Troubleshooting](./troubleshooting.md) - Common issues and solutions
