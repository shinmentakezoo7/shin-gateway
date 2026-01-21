# Deployment Guide

Guide to deploying Shin Gateway in production environments.

## Table of Contents

- [Deployment Options](#deployment-options)
- [Docker Deployment](#docker-deployment)
- [Docker Compose](#docker-compose)
- [Kubernetes](#kubernetes)
- [Systemd Service](#systemd-service)
- [Reverse Proxy Setup](#reverse-proxy-setup)
- [SSL/TLS Configuration](#ssltls-configuration)
- [Performance Tuning](#performance-tuning)
- [Monitoring](#monitoring)
- [Backup and Recovery](#backup-and-recovery)

---

## Deployment Options

| Method | Best For |
|--------|----------|
| Docker | Single server, easy deployment |
| Docker Compose | Development, small production |
| Kubernetes | Large scale, high availability |
| Systemd | Bare metal, Linux servers |

---

## Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create data directory
RUN mkdir -p /app/data

# Expose port
EXPOSE 8082

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8082/health || exit 1

# Run with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8082"]
```

### Build and Run

```bash
# Build image
docker build -t shin-gateway:latest .

# Run container
docker run -d \
  --name shin-gateway \
  --restart unless-stopped \
  -p 8082:8082 \
  -e GROQ_API_KEY=${GROQ_API_KEY} \
  -e OPENAI_API_KEY=${OPENAI_API_KEY} \
  -v $(pwd)/config:/app/config:ro \
  -v shin-gateway-data:/app/data \
  shin-gateway:latest
```

### Production Docker Run

```bash
docker run -d \
  --name shin-gateway \
  --restart unless-stopped \
  --memory=1g \
  --cpus=2 \
  -p 8082:8082 \
  --env-file .env \
  -v $(pwd)/config:/app/config:ro \
  -v shin-gateway-data:/app/data \
  --log-driver json-file \
  --log-opt max-size=100m \
  --log-opt max-file=3 \
  shin-gateway:latest
```

---

## Docker Compose

### Basic docker-compose.yml

```yaml
version: '3.8'

services:
  shin-gateway:
    build: .
    container_name: shin-gateway
    restart: unless-stopped
    ports:
      - "8082:8082"
    environment:
      - SHIN_LOG_LEVEL=INFO
      - GROQ_API_KEY=${GROQ_API_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    volumes:
      - ./config:/app/config:ro
      - gateway-data:/app/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8082/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s

volumes:
  gateway-data:
```

### Production docker-compose.yml

```yaml
version: '3.8'

services:
  shin-gateway:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: shin-gateway
    restart: unless-stopped
    ports:
      - "8082:8082"
    environment:
      - SHIN_LOG_LEVEL=INFO
      - SHIN_DEBUG=false
    env_file:
      - .env
    volumes:
      - ./config:/app/config:ro
      - gateway-data:/app/data
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 1G
        reservations:
          cpus: '0.5'
          memory: 256M
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8082/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    logging:
      driver: json-file
      options:
        max-size: "100m"
        max-file: "3"

  # Optional: Nginx reverse proxy
  nginx:
    image: nginx:alpine
    container_name: shin-nginx
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./certs:/etc/nginx/certs:ro
    depends_on:
      - shin-gateway

volumes:
  gateway-data:
```

### Run

```bash
# Start
docker-compose up -d

# View logs
docker-compose logs -f shin-gateway

# Stop
docker-compose down

# Rebuild and restart
docker-compose up -d --build
```

---

## Kubernetes

### Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: shin-gateway
  labels:
    app: shin-gateway
spec:
  replicas: 3
  selector:
    matchLabels:
      app: shin-gateway
  template:
    metadata:
      labels:
        app: shin-gateway
    spec:
      containers:
      - name: shin-gateway
        image: shin-gateway:latest
        ports:
        - containerPort: 8082
        env:
        - name: SHIN_LOG_LEVEL
          value: "INFO"
        - name: GROQ_API_KEY
          valueFrom:
            secretKeyRef:
              name: shin-gateway-secrets
              key: groq-api-key
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: shin-gateway-secrets
              key: openai-api-key
        resources:
          limits:
            cpu: "1"
            memory: "512Mi"
          requests:
            cpu: "250m"
            memory: "256Mi"
        livenessProbe:
          httpGet:
            path: /health
            port: 8082
          initialDelaySeconds: 10
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8082
          initialDelaySeconds: 5
          periodSeconds: 10
        volumeMounts:
        - name: config
          mountPath: /app/config
          readOnly: true
        - name: data
          mountPath: /app/data
      volumes:
      - name: config
        configMap:
          name: shin-gateway-config
      - name: data
        persistentVolumeClaim:
          claimName: shin-gateway-pvc
```

### Service

```yaml
apiVersion: v1
kind: Service
metadata:
  name: shin-gateway
spec:
  selector:
    app: shin-gateway
  ports:
  - port: 80
    targetPort: 8082
  type: ClusterIP
```

### Ingress

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: shin-gateway
  annotations:
    nginx.ingress.kubernetes.io/proxy-body-size: "50m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - gateway.example.com
    secretName: shin-gateway-tls
  rules:
  - host: gateway.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: shin-gateway
            port:
              number: 80
```

### Secrets

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: shin-gateway-secrets
type: Opaque
stringData:
  groq-api-key: "gsk_xxx"
  openai-api-key: "sk-xxx"
```

### ConfigMap

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: shin-gateway-config
data:
  config.yaml: |
    gateway:
      host: "0.0.0.0"
      port: 8082
    providers:
      groq:
        type: openai
        base_url: "https://api.groq.com/openai/v1"
        api_key_env: GROQ_API_KEY
    models:
      claude-3-5-sonnet-20241022:
        provider: groq
        model: llama-3.3-70b-versatile
```

---

## Systemd Service

### Create Service File

```bash
sudo nano /etc/systemd/system/shin-gateway.service
```

```ini
[Unit]
Description=Shin Gateway API Proxy
After=network.target

[Service]
Type=simple
User=shin-gateway
Group=shin-gateway
WorkingDirectory=/opt/shin-gateway
Environment="PATH=/opt/shin-gateway/venv/bin"
EnvironmentFile=/opt/shin-gateway/.env
ExecStart=/opt/shin-gateway/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8082
Restart=always
RestartSec=10

# Security
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/shin-gateway/data

[Install]
WantedBy=multi-user.target
```

### Enable and Start

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable service
sudo systemctl enable shin-gateway

# Start service
sudo systemctl start shin-gateway

# Check status
sudo systemctl status shin-gateway

# View logs
sudo journalctl -u shin-gateway -f
```

---

## Reverse Proxy Setup

### Nginx Configuration

```nginx
upstream shin_gateway {
    server 127.0.0.1:8082;
    keepalive 32;
}

server {
    listen 80;
    server_name gateway.example.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name gateway.example.com;

    ssl_certificate /etc/nginx/certs/fullchain.pem;
    ssl_certificate_key /etc/nginx/certs/privkey.pem;

    # SSL settings
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256;
    ssl_prefer_server_ciphers off;

    # Timeouts for streaming
    proxy_read_timeout 300s;
    proxy_send_timeout 300s;
    proxy_connect_timeout 60s;

    # Buffer settings
    proxy_buffering off;
    proxy_request_buffering off;

    location / {
        proxy_pass http://shin_gateway;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Connection "";

        # SSE support
        proxy_set_header Accept-Encoding "";
        chunked_transfer_encoding on;
    }
}
```

### Caddy Configuration

```
gateway.example.com {
    reverse_proxy localhost:8082 {
        flush_interval -1
        transport http {
            read_timeout 300s
        }
    }
}
```

---

## SSL/TLS Configuration

### Let's Encrypt with Certbot

```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Get certificate
sudo certbot --nginx -d gateway.example.com

# Auto-renewal is set up automatically
# Test renewal
sudo certbot renew --dry-run
```

### Self-Signed Certificate (Development)

```bash
# Generate self-signed certificate
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout privkey.pem \
  -out fullchain.pem \
  -subj "/CN=localhost"
```

---

## Performance Tuning

### Uvicorn Workers

```bash
# Production with multiple workers
uvicorn main:app \
  --host 0.0.0.0 \
  --port 8082 \
  --workers 4 \
  --loop uvloop \
  --http h11
```

### Gunicorn with Uvicorn Workers

```bash
gunicorn main:app \
  --bind 0.0.0.0:8082 \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --timeout 300 \
  --keep-alive 5 \
  --max-requests 1000 \
  --max-requests-jitter 100
```

### System Tuning

```bash
# Increase file descriptors
echo "* soft nofile 65535" | sudo tee -a /etc/security/limits.conf
echo "* hard nofile 65535" | sudo tee -a /etc/security/limits.conf

# TCP tuning
sudo sysctl -w net.core.somaxconn=65535
sudo sysctl -w net.ipv4.tcp_max_syn_backlog=65535
```

---

## Monitoring

### Health Endpoints

```bash
# Liveness
curl http://localhost:8082/health

# Readiness
curl http://localhost:8082/ready
```

### Prometheus Metrics (if enabled)

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'shin-gateway'
    static_configs:
      - targets: ['localhost:8082']
    metrics_path: '/metrics'
```

### Log Aggregation

Configure structured JSON logging:

```yaml
logging:
  format: json
  level: INFO
```

Parse with Elasticsearch, Loki, or similar.

---

## Backup and Recovery

### Database Backup

```bash
# Backup SQLite database
cp /app/data/admin.db /backup/admin.db.$(date +%Y%m%d)

# Or use sqlite3
sqlite3 /app/data/admin.db ".backup '/backup/admin.db'"
```

### Configuration Backup

```bash
# Backup config
cp -r /app/config /backup/config.$(date +%Y%m%d)
```

### Automated Backup Script

```bash
#!/bin/bash
BACKUP_DIR="/backup/shin-gateway"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR

# Backup database
sqlite3 /app/data/admin.db ".backup '$BACKUP_DIR/admin_$DATE.db'"

# Backup config
cp /app/config/config.yaml $BACKUP_DIR/config_$DATE.yaml

# Keep only last 7 days
find $BACKUP_DIR -type f -mtime +7 -delete
```

### Recovery

```bash
# Stop gateway
docker-compose down

# Restore database
cp /backup/admin.db /app/data/admin.db

# Restore config
cp /backup/config.yaml /app/config/config.yaml

# Start gateway
docker-compose up -d
```

---

## Next Steps

- [Configuration Guide](./configuration.md) - Configuration options
- [Troubleshooting](./troubleshooting.md) - Common issues
- [Architecture](./architecture.md) - System design
