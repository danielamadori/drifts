# icde

This repository contains utilities for converting scikit-learn tree ensembles
into the internal ICDE representation and for preparing time-series datasets
from the [Aeon](https://www.aeon-toolkit.org/) collection. The tooling spans
from dataset initialisation scripts to helpers that persist forests and samples
into Redis-backed caches.

## Installation

### Option 1: Docker (Recommended)

Docker provides an isolated environment with Redis pre-configured.

**Requirements:**
- Docker Desktop installed and running
  - Windows: https://www.docker.com/products/docker-desktop
  - Linux/macOS: https://docs.docker.com/get-docker/

**Quick Start:**

```bash
# Windows
run.bat

# Linux/macOS
chmod +x run.sh  # First time only
./run.sh
```

This will build the Docker image and start a container with Redis on `localhost:6379`.

**Available Commands:**

| Command | Windows | Linux/macOS | Description |
|---------|---------|-------------|-------------|
| Start | `run.bat` or `run.bat start` | `./run.sh` or `./run.sh start` | Build and start container |
| Stop | `run.bat stop` | `./run.sh stop` | Stop container |
| Shell | `run.bat shell` | `./run.sh shell` | Open bash shell in container |
| Logs | `run.bat logs` | `./run.sh logs` | View container logs |
| Restart | `run.bat restart` | `./run.sh restart` | Restart container |
| Help | `run.bat help` | `./run.sh help` | Show help |

**Using the container:**

```bash
# Open shell in container
run.bat shell  # Windows
./run.sh shell # Linux/macOS

# Inside the container, run any script:
python init_aeon_univariate.py Coffee --class-label 0 --optimize
python enhanced_launch_workers.py start --profile development
```

The following directories are automatically mounted and accessible from your host:
- `./logs` - Application logs
- `./workers` - Workers configuration
- `./results` - Experiment results
- `./fig` - Plots and visualizations

### Option 2: Local Installation

1. **Create a virtual environment (recommended)**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. **Install the Python dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

   The `requirements.txt` file includes all necessary dependencies.
3. **Start Redis**

   ```bash
   redis-server
   ```

## Usage

### Aeon dataset initialisation

`init_aeon_univariate.py` exposes a command-line utility for initialising the
Redis caches with samples and optimised forests for a single dataset. Examples:

```bash
# List supported datasets
python init_aeon_univariate.py --list-datasets

# Optimise a Random Forest for the ECG200 dataset using Bayesian search
python init_aeon_univariate.py ECG200 --class-label "1" --optimize

```


#### Core arguments

- `dataset_name` — Dataset to load
- `--class-label` — Class label whose samples will be processed
- `--list-datasets` — Print the curated catalogue of supported Aeon datasets and exit.
- `--info` — Display dataset metadata without performing any processing.

#### Bayesian optimisation controls

- `--optimize` — Enable Bayesian optimisation via `scikit-optimize` to tune the Random Forest hyper-parameters.
- `--redis-port` *(int, default: 6379)* — Port of the Redis or KeyDB instance used by the workers.

Run `python init_aeon_univariate.py --help` to view the auto-generated help message with the latest defaults.

---

## Quick Start Tutorial

> **Note:** If using Docker, run `run.bat shell` (Windows) or `./run.sh shell` (Linux/macOS) first to enter the container, then execute the commands below.

### Step 1: Test with a Single Dataset (5 minutes)

```bash
# Initialize Coffee dataset with Bayesian optimization
python3 init_aeon_univariate.py Coffee --class-label "0" --optimize
```

### Step 2: Start the Worker Algorithm

```bash
# Launch workers to process the initialized dataset (default: 1 worker)
python3 enhanced_launch_workers.py start

# Or use development profile (4 workers with logging)
python3 enhanced_launch_workers.py start --profile development

# Or use production profile (4 workers with logging)
python3 enhanced_launch_workers.py start --profile production
```

Edit the file `worker_config.yaml` to customize worker settings, e.g., increase the number of workers.

**Key parameters:**
- `start` — Start workers using configuration
- `--profile {default|development|production}` — Use predefined worker profiles
  - `default`: 1 worker with `worker_cache.py`
  - `development`: 4 workers with `worker_cache_logged.py`
  - `production`: 4 workers with `worker_cache_logged.py`
- `--config FILE` — Use custom YAML configuration file

**Other useful commands:**
```bash
# Check worker status
python3 enhanced_launch_workers.py status

# View logs for a specific worker
python3 enhanced_launch_workers.py logs 1

# Stop all workers
python3 enhanced_launch_workers.py stop

# Clean restart (stop + clean + start fresh)
python3 enhanced_launch_workers.py clean-restart
```

**Expected output:**
```
Starting 1 worker processes...
Worker 1 started (PID: 12345)
Workers running. Press Ctrl+C to monitor or stop.
```

**Monitor progress:**
```bash
# Check Redis databases for candidate reasons and confirmed reasons
redis-cli -n 1 DBSIZE  # CAN database (candidates)
redis-cli -n 2 DBSIZE  # R database (confirmed reasons)

# Or use the status command
python3 enhanced_launch_workers.py status
```

### Step 3: Analyze Results in Jupyter Notebook

```bash
# Open the analysis notebook
jupyter notebook models_analysis.ipynb
```

---

## Automated Testing

### Test All Datasets with Optimization and Workers

The `test_all_optimize_with_workers.py` script automatically tests all datasets with Bayesian optimization and optionally runs workers to verify the setup works correctly.

**Basic usage (optimize only):**
```bash
python test_all_optimize_with_workers.py
```

**With worker execution (20 seconds per dataset):**
```bash
python test_all_optimize_with_workers.py --worker-duration 20
```

**Advanced options:**
```bash
# Test only first 3 datasets with workers
python test_all_optimize_with_workers.py --worker-duration 20 --max-datasets 3

# Use production profile for workers
python test_all_optimize_with_workers.py --worker-duration 20 --worker-profile production

# Continue on errors instead of stopping
python test_all_optimize_with_workers.py --worker-duration 20 --continue-on-error

# Skip workers explicitly
python test_all_optimize_with_workers.py --skip-workers
```

**Results:**
- Log file: `test_all_optimize_results.txt` - Human-readable log
- JSON file: `test_all_optimize_results.json` - Machine-readable results

**What it does:**
1. For each dataset: Load and identify all classes
2. Run `init_aeon_univariate.py` with `--optimize` for the first class
3. If `--worker-duration` > 0: Launch workers for specified seconds
4. Verify no errors occurred in both optimization and worker execution
5. Move to next dataset (or stop on first error if `--continue-on-error` not set)

---
