# Admin UI Guide

Guide to using the Shin Gateway Admin Dashboard.

## Table of Contents

- [Accessing the Dashboard](#accessing-the-dashboard)
- [Overview Page](#overview-page)
- [Providers Management](#providers-management)
- [Models Management](#models-management)
- [API Keys Management](#api-keys-management)
- [Statistics](#statistics)
- [Settings](#settings)

---

## Accessing the Dashboard

### Web Interface

Open your browser to:

```
http://localhost:8082/admin
```

### Next.js Development Server

If running the admin UI separately:

```bash
cd admin-ui
npm run dev
```

Access at: `http://localhost:3000`

---

## Overview Page

The overview page provides a real-time snapshot of your gateway.

### Key Metrics

| Metric | Description |
|--------|-------------|
| **Total Requests** | Total API requests processed |
| **Active Providers** | Number of enabled providers |
| **Active Models** | Number of configured model aliases |
| **Active API Keys** | Number of enabled API keys |

### Real-Time Stats

- **RPS (Requests Per Second)**: Current request throughput
- **TPS (Tokens Per Second)**: Token processing rate
- **Avg Latency**: Average response time
- **Error Rate**: Percentage of failed requests

### Provider Status

Quick view of all providers with their current status:

- **Healthy** (green): Provider is responding normally
- **Degraded** (yellow): Provider experiencing issues
- **Unhealthy** (red): Provider is down or circuit is open

### Recent Activity

Live feed of recent requests showing:
- Timestamp
- Model used
- Provider
- Latency
- Status (success/error)

---

## Providers Management

### Viewing Providers

Navigate to **Providers** in the sidebar to see all configured providers.

Each provider shows:
- Name and type
- Base URL
- Status (enabled/disabled)
- Rate limits
- Request/error counts

### Adding a Provider

1. Click **Add Provider**
2. Fill in the form:

| Field | Description | Example |
|-------|-------------|---------|
| Name | Unique identifier | `my_groq` |
| Type | Provider type | `openai` or `anthropic` |
| Base URL | API endpoint | `https://api.groq.com/openai/v1` |
| API Key | Direct key (optional) | `gsk_xxx` |
| API Key Env | Environment variable | `GROQ_API_KEY` |
| Timeout | Request timeout (seconds) | `120` |

3. Configure rate limits (optional):
   - Requests per minute
   - Tokens per minute
   - Concurrent requests

4. Click **Save**

### Editing a Provider

1. Click the **Edit** icon on a provider row
2. Modify the fields
3. Click **Save Changes**

### Enabling/Disabling a Provider

Toggle the switch in the **Status** column to enable or disable a provider.

Disabled providers:
- Won't receive any requests
- Are skipped in fallback chains
- Appear grayed out in the list

### Fetching Available Models

1. Click the **Fetch Models** button on a provider
2. The gateway queries the provider's `/models` endpoint
3. Available models are displayed for reference

### Deleting a Provider

1. Click the **Delete** icon
2. Confirm the deletion
3. Note: Models using this provider will become orphaned

---

## Models Management

### Viewing Models

Navigate to **Models** in the sidebar to see all model aliases.

Each model shows:
- Alias name
- Mapped provider
- Actual model name
- Status (enabled/disabled)
- Default parameters

### Adding a Model Alias

1. Click **Add Model**
2. Fill in the form:

| Field | Description | Example |
|-------|-------------|---------|
| Alias Name | Name clients will use | `claude-3-5-sonnet-20241022` |
| Provider | Select from list | `groq` |
| Model | Actual model name | `llama-3.3-70b-versatile` |
| Context Window | Max tokens (optional) | `128000` |

3. Configure defaults (optional):
   - Temperature
   - Max tokens
   - Top P
   - Top K

4. Click **Save**

### Setting Up Fallbacks

1. Edit a model alias
2. Scroll to **Fallbacks** section
3. Click **Add Fallback**
4. Select provider and model
5. Arrange fallback priority (drag to reorder)
6. Click **Save Changes**

Fallback behavior:
- Primary provider fails → Try first fallback
- First fallback fails → Try second fallback
- All fail → Return error to client

### Editing a Model

1. Click the **Edit** icon on a model row
2. Modify the configuration
3. Click **Save Changes**

### Deleting a Model

1. Click the **Delete** icon
2. Confirm the deletion
3. Requests for this model will return 404

---

## API Keys Management

### Viewing API Keys

Navigate to **API Keys** in the sidebar.

Each key shows:
- Name
- Key prefix (e.g., `sk-shin-abc...`)
- Rate limits
- Status (enabled/disabled)
- Last used timestamp
- Request count

### Creating an API Key

1. Click **Generate New Key**
2. Fill in the form:

| Field | Description | Example |
|-------|-------------|---------|
| Name | Descriptive name | `Production App` |
| Rate Limit (RPM) | Requests per minute | `100` |
| Rate Limit (TPM) | Tokens per minute | `100000` |

3. Click **Generate**
4. **Important**: Copy the displayed key immediately - it won't be shown again!

### API Key Security

- Keys are hashed before storage (SHA256)
- Only the prefix is stored for identification
- Full key is shown only once at creation
- Rotate keys regularly for security

### Managing API Keys

**Enable/Disable:**
- Toggle the switch to temporarily disable a key
- Disabled keys receive 401 errors

**Edit Rate Limits:**
1. Click **Edit** on a key
2. Modify rate limits
3. Click **Save**

**Delete:**
1. Click **Delete**
2. Confirm deletion
3. Key immediately becomes invalid

### Viewing Key Usage

Click on a key to see detailed usage:
- Requests by day/hour
- Token consumption
- Error breakdown
- Top models used

---

## Statistics

### Overview Stats

Navigate to **Statistics** for analytics.

### Time Range Selection

Select the period to analyze:
- Last hour
- Last 24 hours
- Last 7 days
- Last 30 days
- Custom range

### Charts Available

**Request Volume:**
- Total requests over time
- Grouped by provider/model
- Success vs failure breakdown

**Token Usage:**
- Input tokens
- Output tokens
- Total tokens over time

**Latency Distribution:**
- P50 (median)
- P90
- P99
- Max latency

**Error Analysis:**
- Error rate over time
- Errors by type
- Errors by provider

### Provider Comparison

Compare providers side by side:
- Request volume
- Average latency
- Error rate
- Token throughput

### Model Analytics

Per-model statistics:
- Usage frequency
- Average response time
- Token consumption patterns
- Error rates

### Export Data

Click **Export** to download statistics:
- CSV format
- JSON format
- PDF report (with charts)

---

## Settings

### Gateway Settings

Configure gateway behavior:

| Setting | Description |
|---------|-------------|
| Require API Key | Force authentication for all requests |
| Default Timeout | Default request timeout |
| Max Retries | Default retry count |
| Log Level | Logging verbosity |

### CORS Settings

Configure Cross-Origin Resource Sharing:

- Enable/disable CORS
- Allowed origins
- Allowed methods
- Allowed headers

### Circuit Breaker Settings

Configure resilience behavior:

| Setting | Description |
|---------|-------------|
| Enabled | Toggle circuit breaker |
| Failure Threshold | Failures before opening |
| Success Threshold | Successes to close |
| Timeout | Wait time before half-open |

### Retry Settings

Configure retry behavior:

| Setting | Description |
|---------|-------------|
| Max Retries | Maximum attempts |
| Initial Delay | First retry delay |
| Max Delay | Maximum delay cap |
| Exponential Base | Backoff multiplier |

### Database Maintenance

- **Clear Old Records**: Delete usage records older than X days
- **Vacuum Database**: Reclaim disk space
- **Export Config**: Download current configuration
- **Import Config**: Upload configuration file

---

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+/` or `Cmd+/` | Open search |
| `g` then `o` | Go to Overview |
| `g` then `p` | Go to Providers |
| `g` then `m` | Go to Models |
| `g` then `k` | Go to API Keys |
| `g` then `s` | Go to Statistics |
| `?` | Show all shortcuts |

---

## Next Steps

- [Configuration Guide](./configuration.md) - YAML configuration
- [API Reference](./api-reference.md) - Admin API endpoints
- [Deployment](./deployment.md) - Production deployment
