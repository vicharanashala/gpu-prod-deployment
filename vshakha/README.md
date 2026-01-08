# FAQ MCP Server - vShakha Deployment

## Overview

This directory contains the Docker Compose configuration for deploying the FAQ MCP Server (vShakha) from Docker Hub.

## Quick Start

### 1. Configure Environment

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your MongoDB URI and other settings
nano .env
```

### 2. Deploy

```bash
# Pull the latest image and start the service
docker-compose pull
docker-compose up -d

# View logs
docker-compose logs -f

# Check status
docker-compose ps
```

### 3. Verify

```bash
# Check if the service is healthy
curl http://localhost:9010/health

# Or check container health
docker ps
```

## Configuration

### Required Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `MONGODB_URI` | MongoDB connection string | `mongodb://localhost:27017/faq_bootcamp` |

### Optional Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DB_NAME` | `faq_bootcamp` | Database name |
| `COLLECTION_NAME` | `questions` | Collection name |
| `EMBEDDING_PROVIDER` | `local` | Embedding provider (local/openai/anthropic) |
| `EMBEDDING_MODEL` | `BAAI/bge-large-en-v1.5` | Embedding model |
| `EMBEDDING_DIMENSION` | `1024` | Embedding dimension |
| `TFIDF_WEIGHT` | `0.3` | TF-IDF weight (0-1) |
| `EMBEDDING_WEIGHT` | `0.7` | Embedding weight (0-1) |
| `SERVER_PORT` | `9010` | Server port |

## Docker Image

**Image**: `vicharanashala/faq_vsakha:latest`

**Tags**:
- `latest` - Most recent build
- `<git-sha>` - Specific commit builds

**Platform**: `linux/amd64`

## Service Details

- **Container Name**: `faq-mcp-server`
- **Port**: `9010:9010`
- **Network**: `vshakha-network`
- **Restart Policy**: `unless-stopped`
- **Health Check**: HTTP check on `/health` endpoint every 30s

## Management Commands

### Start Service
```bash
docker-compose up -d
```

### Stop Service
```bash
docker-compose down
```

### Restart Service
```bash
docker-compose restart
```

### View Logs
```bash
# Follow logs
docker-compose logs -f

# Last 100 lines
docker-compose logs --tail=100

# Specific service
docker-compose logs -f faq-mcp-server
```

### Update to Latest Image
```bash
# Pull latest image
docker-compose pull

# Restart with new image
docker-compose up -d
```

### Check Status
```bash
# Container status
docker-compose ps

# Health check status
docker inspect faq-mcp-server --format='{{.State.Health.Status}}'
```

## Troubleshooting

### Container Won't Start

1. **Check logs**:
   ```bash
   docker-compose logs faq-mcp-server
   ```

2. **Verify environment variables**:
   ```bash
   docker-compose config
   ```

3. **Check MongoDB connection**:
   - Ensure `MONGODB_URI` is correct
   - Verify MongoDB is accessible from the container

### Health Check Failing

1. **Check if port is accessible**:
   ```bash
   curl http://localhost:9010/health
   ```

2. **Verify container is running**:
   ```bash
   docker ps | grep faq-mcp-server
   ```

3. **Check container logs for errors**:
   ```bash
   docker-compose logs --tail=50 faq-mcp-server
   ```

### Port Already in Use

If port 9010 is already in use, modify `docker-compose.yml`:

```yaml
ports:
  - "9011:9010"  # Use different host port
```

## Integration

### LibreChat Integration

Configure in `librechat.yaml`:

```yaml
mcpServers:
  faq-server:
    type: streamable-http
    url: http://faq-mcp-server:9010/mcp
```

### Direct API Access

```bash
# Example API call
curl -X POST http://localhost:9010/search \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I register for bootcamp?", "top_k": 3}'
```

## Maintenance

### Backup

No persistent data in the container. FAQ data is stored in MongoDB.

### Updates

1. Pull latest image: `docker-compose pull`
2. Restart service: `docker-compose up -d`
3. Verify: `docker-compose logs -f`

### Monitoring

- **Health endpoint**: `http://localhost:9010/health`
- **Container logs**: `docker-compose logs -f`
- **Resource usage**: `docker stats faq-mcp-server`

## Support

- **Docker Hub**: https://hub.docker.com/r/vicharanashala/faq_vsakha
- **GitHub**: https://github.com/vicharanashala/faq-mcp-server
- **Issues**: https://github.com/vicharanashala/faq-mcp-server/issues
