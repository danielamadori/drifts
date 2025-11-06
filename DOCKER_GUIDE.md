# Docker Setup for DRIFTS

## Overview

DRIFTS includes Docker support for isolated environment with Redis pre-configured.

## Prerequisites

**Docker Desktop** must be installed:
- Windows: https://www.docker.com/products/docker-desktop
- Linux/macOS: https://docs.docker.com/get-docker/

## Quick Start

### Option 1: Using Batch Script (Windows)

```cmd
# Start container (build + run)
run.bat

# Or explicitly
run.bat start
```

### Option 2: Using Shell Script (Linux/macOS)

```bash
# Make executable
chmod +x run.sh

# Start container
./run.sh start
```

## Available Commands

### Windows (run.bat)
```cmd
run.bat start      # Build and start container
run.bat stop       # Stop container
run.bat restart    # Restart container
run.bat shell      # Open bash shell in container
run.bat logs       # View container logs
run.bat help       # Show help
```

### Linux/macOS (run.sh)
```bash
./run.sh start     # Build and start container
./run.sh stop      # Stop container
./run.sh restart   # Restart container
./run.sh shell     # Open bash shell in container
./run.sh logs      # View container logs
./run.sh help      # Show help
```

## What Gets Created

The container includes:
- ✅ **Redis Server** on port 6379
- ✅ **Python 3.11** with all dependencies
- ✅ **Supervisor** to manage services
- ✅ **Volume mounts** for:
  - `logs/` - Worker logs
  - `workers/` - Worker PID tracking
  - `results/` - Test results
  - `fig/` - Figures/plots

## Inside the Container

Once started, access the container:

```cmd
# Windows
run.bat shell

# Linux/macOS
./run.sh shell
```

Inside the container you can:

```bash
# Check Redis
redis-cli ping
# Should respond: PONG

# Run init on a dataset
python init_aeon_univariate.py Coffee --class-label 0 --optimize

# Start workers
python enhanced_launch_workers.py start

# Run tests
python test_datasets_with_workers.py --max-datasets 5 --worker-duration 20

# Check worker status
python enhanced_launch_workers.py status

# Stop workers
python enhanced_launch_workers.py stop
```

## Container Details

### Image Info
- **Name**: `drifts:latest`
- **Container Name**: `drifts-container`
- **Base Image**: `python:3.11-slim`

### Ports
- **6379**: Redis (mapped to host)

### Services (managed by Supervisor)
- **Redis**: Automatically started
- **Init Message**: Shows startup info

### Health Check
The container includes a health check that pings Redis every 30 seconds.

## Volume Persistence

Data is persisted in these directories:
- `./logs` → `/app/logs` (worker logs)
- `./workers` → `/app/workers` (PID files)
- `./results` → `/app/results` (test results)
- `./fig` → `/app/fig` (figures)

This means your data persists even if you stop/remove the container.

## Docker Compose (Alternative)

If you prefer Docker Compose, create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  drifts:
    build: .
    container_name: drifts-container
    ports:
      - "6379:6379"
    volumes:
      - ./logs:/app/logs
      - ./workers:/app/workers
      - ./results:/app/results
      - ./fig:/app/fig
    environment:
      - PYTHONUNBUFFERED=1
      - REDIS_HOST=localhost
      - REDIS_PORT=6379
```

Then use:
```bash
docker-compose up -d    # Start
docker-compose down     # Stop
docker-compose exec drifts bash  # Shell
```

## Troubleshooting

### Docker not found
Install Docker Desktop from: https://www.docker.com/products/docker-desktop

### Permission denied (Linux)
```bash
sudo usermod -aG docker $USER
# Then log out and back in
```

### Container won't start
```bash
# Check logs
docker logs drifts-container

# Remove and rebuild
docker stop drifts-container
docker rm drifts-container
docker rmi drifts:latest
# Then run.bat/run.sh again
```

### Redis connection refused
```bash
# Inside container, check Redis status
docker exec -it drifts-container redis-cli ping

# Check supervisor logs
docker exec -it drifts-container cat /var/log/supervisor/redis.log
```

### Can't access from host
Make sure Redis is bound to 0.0.0.0 in the container (it is by default).

Check port mapping:
```bash
docker ps
# Should show: 0.0.0.0:6379->6379/tcp
```

## Advantages of Docker

✅ **Isolated Environment**: No conflicts with system packages
✅ **Pre-configured Redis**: No separate Redis installation needed
✅ **Reproducible**: Same environment everywhere
✅ **Easy Cleanup**: Just remove container
✅ **Cross-platform**: Works on Windows, Linux, macOS

## Native vs Docker

### Use Native (Without Docker) if:
- You already have Redis installed
- You prefer direct system access
- You want faster I/O for large datasets

### Use Docker if:
- You want isolated environment
- You don't want to install Redis separately
- You want reproducible setup
- You're sharing the project with others

---

**Both approaches work perfectly with the DRIFTS test system!**

