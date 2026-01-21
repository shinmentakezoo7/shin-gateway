# Troubleshooting Guide

Solutions to common issues when using Shin Gateway.

## Table of Contents

- [Connection Issues](#connection-issues)
- [Authentication Errors](#authentication-errors)
- [Model and Provider Errors](#model-and-provider-errors)
- [Streaming Issues](#streaming-issues)
- [Performance Problems](#performance-problems)
- [Admin Dashboard Issues](#admin-dashboard-issues)
- [Database Issues](#database-issues)
- [Logging and Debugging](#logging-and-debugging)

---

## Connection Issues

### Gateway Not Starting

**Symptoms:**
- `uvicorn` exits immediately
- Port already in use error

**Solutions:**

```bash
# Check if port is in use
lsof -i :8082

# Kill existing process
kill -9 $(lsof -t -i :8082)

# Or use a different port
uvicorn main:app --port 8083
```

### Cannot Connect to Gateway

**Symptoms:**
- `Connection refused` error
- Requests timeout

**Solutions:**

1. Check gateway is running:
```bash
curl http://localhost:8082/health
```

2. Check firewall:
```bash
# Ubuntu/Debian
sudo ufw allow 8082

# CentOS/RHEL
sudo firewall-cmd --add-port=8082/tcp --permanent
sudo firewall-cmd --reload
```

3. Check binding address:
```yaml
# config/config.yaml
gateway:
  host: "0.0.0.0"  # Not "127.0.0.1" for external access
  port: 8082
```

### Provider Connection Errors

**Symptoms:**
- `Failed to connect to provider`
- `Connection timed out`

**Solutions:**

1. Test provider connectivity:
```bash
# For Ollama
curl http://localhost:11434/v1/models

# For Groq
curl -H "Authorization: Bearer $GROQ_API_KEY" \
  https://api.groq.com/openai/v1/models

# For OpenAI
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
  https://api.openai.com/v1/models
```

2. Check provider URL in config:
```yaml
providers:
  groq:
    base_url: "https://api.groq.com/openai/v1"  # Note: /v1 suffix
```

3. Increase timeout:
```yaml
providers:
  ollama_local:
    timeout: 300  # Increase for slow models
```

---

## Authentication Errors

### Invalid API Key (401)

**Symptoms:**
- `{"error": {"type": "authentication_error"}}`
- `Invalid API key`

**Solutions:**

1. Check API key format:
```bash
# Anthropic format
-H "x-api-key: your-key"

# OpenAI format
-H "Authorization: Bearer your-key"
```

2. Verify key in database:
```bash
curl http://localhost:8082/admin/api-keys
```

3. Check if key is enabled in Admin UI

4. Regenerate key if needed:
```bash
curl -X POST http://localhost:8082/admin/api-keys \
  -H "Content-Type: application/json" \
  -d '{"name": "New Key"}'
```

### Provider API Key Invalid

**Symptoms:**
- `401 Unauthorized` from provider
- `Invalid API key` from upstream

**Solutions:**

1. Check environment variable is set:
```bash
echo $GROQ_API_KEY
```

2. Check key is correct in `.env`:
```bash
# .env
GROQ_API_KEY=gsk_xxxxx  # No quotes needed
```

3. Verify key works directly:
```bash
curl -H "Authorization: Bearer $GROQ_API_KEY" \
  https://api.groq.com/openai/v1/models
```

---

## Model and Provider Errors

### Model Not Found (404)

**Symptoms:**
- `Model 'xxx' not found`
- `Unknown model`

**Solutions:**

1. Check model alias exists:
```bash
curl http://localhost:8082/v1/models
```

2. Add model alias in config:
```yaml
models:
  claude-3-5-sonnet-20241022:
    provider: groq
    model: llama-3.3-70b-versatile
```

3. Verify in Admin UI under Models

### Provider Not Found

**Symptoms:**
- `Provider 'xxx' not configured`
- `No provider available`

**Solutions:**

1. Check provider exists and is enabled:
```bash
curl http://localhost:8082/admin/providers
```

2. Enable provider in config:
```yaml
providers:
  groq:
    enabled: true  # Must be true
```

### All Providers Failed

**Symptoms:**
- `All providers failed`
- Request fails after retries

**Solutions:**

1. Check provider health:
```bash
curl http://localhost:8082/ready
```

2. Check circuit breaker status in Admin UI

3. Reset circuit breaker by restarting gateway or waiting for timeout

4. Configure fallbacks:
```yaml
models:
  claude-3-5-sonnet:
    provider: groq
    model: llama-3.3-70b
    fallbacks:
      - provider: openai
        model: gpt-4o
```

---

## Streaming Issues

### Stream Stops Unexpectedly

**Symptoms:**
- Streaming stops mid-response
- Connection closed prematurely

**Solutions:**

1. Increase timeout:
```yaml
gateway:
  timeout:
    read: 300.0  # Increase for long responses
```

2. Check proxy timeouts (if using nginx):
```nginx
proxy_read_timeout 300s;
proxy_send_timeout 300s;
```

3. Disable response buffering:
```nginx
proxy_buffering off;
```

### SSE Events Not Received

**Symptoms:**
- No streaming events
- Response comes all at once

**Solutions:**

1. Check `stream: true` in request:
```json
{
  "stream": true,
  "messages": [...]
}
```

2. Check reverse proxy SSE configuration:
```nginx
proxy_set_header Accept-Encoding "";
chunked_transfer_encoding on;
```

3. Test direct connection (bypass proxy):
```bash
curl http://localhost:8082/v1/messages \
  -d '{"stream": true, ...}'
```

---

## Performance Problems

### High Latency

**Symptoms:**
- Slow response times
- Requests taking too long

**Solutions:**

1. Check provider latency:
```bash
# Time a request
time curl http://localhost:8082/v1/messages -d '{...}'
```

2. Use faster providers:
```yaml
models:
  fast-model:
    provider: groq  # Groq is typically faster
    model: llama-3.3-70b-versatile
```

3. Enable connection pooling (default):
```yaml
gateway:
  connection_pool:
    max_connections: 100
    max_keepalive: 20
```

4. Use uvloop:
```bash
pip install uvloop
uvicorn main:app --loop uvloop
```

### Rate Limiting

**Symptoms:**
- `429 Too Many Requests`
- Requests being throttled

**Solutions:**

1. Check rate limits in Admin UI

2. Increase limits:
```yaml
providers:
  groq:
    rate_limit:
      requests_per_minute: 60  # Increase
      tokens_per_minute: 100000
```

3. Spread requests across providers with fallbacks

4. Implement client-side rate limiting

### Memory Usage High

**Symptoms:**
- OOM errors
- Gateway crashes under load

**Solutions:**

1. Limit concurrent requests:
```yaml
providers:
  ollama:
    rate_limit:
      concurrent_requests: 2  # Limit concurrency
```

2. Reduce workers:
```bash
uvicorn main:app --workers 2  # Reduce from 4
```

3. Set container memory limits:
```yaml
deploy:
  resources:
    limits:
      memory: 1G
```

---

## Admin Dashboard Issues

### Dashboard Not Loading

**Symptoms:**
- Blank page at /admin
- 404 errors

**Solutions:**

1. Check admin router is mounted:
```python
# main.py
app.include_router(admin_router, prefix="/admin")
```

2. Check Next.js build (if using separate UI):
```bash
cd admin-ui
npm run build
```

3. Check browser console for errors

### API Key Not Showing

**Symptoms:**
- Created key not visible
- Key shown as masked

**Solutions:**

API keys are shown only once at creation for security.

1. Create a new key:
```bash
curl -X POST http://localhost:8082/admin/api-keys \
  -H "Content-Type: application/json" \
  -d '{"name": "New Key"}'
```

2. Save the returned key immediately

---

## Database Issues

### Database Locked

**Symptoms:**
- `database is locked`
- Write operations failing

**Solutions:**

1. Check for multiple processes:
```bash
lsof data/admin.db
```

2. Set WAL mode:
```python
# Already default, but verify
conn.execute("PRAGMA journal_mode=WAL")
```

3. Restart the gateway

### Corrupted Database

**Symptoms:**
- `database disk image is malformed`
- Startup failures

**Solutions:**

1. Check database integrity:
```bash
sqlite3 data/admin.db "PRAGMA integrity_check"
```

2. Restore from backup:
```bash
cp backup/admin.db data/admin.db
```

3. Recreate database:
```bash
rm data/admin.db
python main.py  # Will recreate
```

---

## Logging and Debugging

### Enable Debug Logging

```bash
# Environment variable
export SHIN_LOG_LEVEL=DEBUG

# Or in config
logging:
  level: DEBUG
```

### View Logs

```bash
# Docker
docker logs -f shin-gateway

# Systemd
journalctl -u shin-gateway -f

# File logs
tail -f logs/shin-gateway.log
```

### Debug a Specific Request

1. Add request ID header:
```bash
curl -H "X-Request-ID: test-123" ...
```

2. Search logs:
```bash
grep "test-123" logs/shin-gateway.log
```

### Common Log Messages

| Message | Meaning | Action |
|---------|---------|--------|
| `Circuit breaker opened` | Provider failing | Check provider health |
| `Rate limit exceeded` | Too many requests | Reduce request rate |
| `Request cancelled` | Client disconnected | Normal for aborted requests |
| `Token preflight failed` | Input too large | Reduce input or increase limit |
| `Fallback to provider` | Primary failed | Check primary provider |

---

## Getting Help

If you're still experiencing issues:

1. **Search existing issues**: [GitHub Issues](https://github.com/shinmentakezoo7/shin-gateway/issues)

2. **Create a new issue** with:
   - Shin Gateway version
   - Python version
   - Full error message
   - Steps to reproduce
   - Relevant config (sanitized)

3. **Check the documentation**:
   - [Configuration Guide](./configuration.md)
   - [API Reference](./api-reference.md)
   - [Architecture](./architecture.md)

---

## Quick Reference

### Health Check Commands

```bash
# Gateway health
curl http://localhost:8082/health

# Gateway readiness
curl http://localhost:8082/ready

# List models
curl http://localhost:8082/v1/models

# Test request
curl -X POST http://localhost:8082/v1/messages \
  -H "Content-Type: application/json" \
  -d '{"model": "claude-3-5-sonnet-20241022", "max_tokens": 10, "messages": [{"role": "user", "content": "Hi"}]}'
```

### Reset Commands

```bash
# Restart gateway
./restart.sh

# Or with Docker
docker-compose restart

# Or with systemd
sudo systemctl restart shin-gateway

# Clear database (will lose all admin data!)
rm data/admin.db
```
