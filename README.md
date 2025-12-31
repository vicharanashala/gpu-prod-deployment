# vLLM Production Monitoring Stack

Complete monitoring solution for vLLM with Prometheus and Grafana.

## Stack Components

- **vLLM**: LLM inference engine with Qwen 2.5-3B model
- **DCGM Exporter**: NVIDIA GPU metrics
- **Node Exporter**: System metrics (CPU, RAM, disk)
- **Prometheus**: Metrics collection and storage
- **Grafana**: Visualization dashboards

## Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with drivers installed
- NVIDIA Container Toolkit

## Deployment

### Option 1: Docker Compose CLI
```bash
docker-compose up -d
```

### Option 2: Portainer

1. Go to Portainer UI → Stacks → Add stack
2. Paste the contents of `docker-compose.yml`
3. Deploy the stack

## Access URLs

- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **vLLM API**: http://localhost:8001
- **vLLM Metrics**: http://localhost:8001/metrics
- **DCGM Exporter**: http://localhost:9401/metrics
- **Node Exporter**: http://localhost:9100/metrics

## Grafana Dashboards

### Import Pre-built Dashboards

1. **NVIDIA DCGM Dashboard**
   - ID: 12239
   - Shows GPU utilization, memory, temperature, power

2. **Node Exporter Full**
   - ID: 1860
   - Shows CPU, memory, disk, network metrics

3. **vLLM Monitoring Dashboard**
   - Import from `grafana-dashboards/vllm-dashboard.json`
   - Shows request queue, token throughput, latency, cache usage

## Testing vLLM

Send a test request:
```bash
curl http://localhost:8001/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-3B-Instruct",
    "prompt": "Hello, how are you?",
    "max_tokens": 50
  }'
```

## Configuration

### vLLM Settings

Modify in docker-compose.yml under vllm service command:
- `--gpu-memory-utilization`: GPU memory to use (default: 0.90)
- `--max-model-len`: Maximum context length (default: 4096)
- `--model`: Model to load

### Prometheus Settings

- **Scrape interval**: 15s
- **Retention**: 15 days
- Config is embedded in docker-compose.yml

## Data Persistence

All data is stored in Docker volumes:
- `prometheus_data`: Prometheus metrics
- `grafana_data`: Grafana dashboards and settings
- `./hf_cache`: HuggingFace model cache

## Troubleshooting

### Check if all services are running
```bash
docker ps
```

### Check logs
```bash
docker logs vllm-qwen3
docker logs prometheus
docker logs grafana
```

### Verify metrics
```bash
# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# Check vLLM metrics
curl http://localhost:8001/metrics
```

## Stopping the Stack
```bash
docker-compose down
```

To remove volumes as well:
```bash
docker-compose down -v
```
EOF
