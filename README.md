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

### Quick Test (Single Dataset)

Verify everything works with a quick test:

```bash
python test_datasets_with_workers.py --max-datasets 1 --worker-duration 10
```

This completes in 2-5 minutes and confirms:
- ✅ Dataset initialization works
- ✅ Workers start correctly
- ✅ Processing completes
- ✅ Error detection functions

### Test All Datasets

Run comprehensive tests on all available datasets (88 datasets from UCR Time Series Archive):

```bash
# Windows
RUN_NOW.bat

# Or directly with Python
python test_datasets_with_workers.py --worker-duration 20
```

**What it does for each dataset:**
1. Initialize with Bayesian optimization (`--optimize`)
2. Start workers using the 'default' profile (1 worker)
3. Let workers process for 20 seconds
4. Stop workers and check for errors
5. **Stops immediately if any error occurs**
6. Moves to next dataset if successful

**Command options:**
```bash
# Test first 5 datasets only
python test_datasets_with_workers.py --max-datasets 5 --worker-duration 20

# Test specific datasets
python test_datasets_with_workers.py --datasets Coffee Wine ECG200 --worker-duration 20

# Continue even if errors occur (don't stop)
python test_datasets_with_workers.py --worker-duration 20 --continue-on-error

# Use different worker profile
python test_datasets_with_workers.py --worker-profile production --worker-duration 20
```

**Monitor progress in real-time:**

Open a second terminal and run:
```bash
# Windows
MONITOR.bat

# Or with Python
python monitor_live.py
```

The monitor shows:
- ⏱️ Elapsed time
- 📊 Progress (completed/failed/total)
- 📝 Last 25 lines of execution log
- ❌ Failed datasets with error details
- 🔄 Auto-refresh every 5 seconds

**Output files:**
- `test_datasets_workers.log` - Detailed execution log
- `test_datasets_workers.json` - Results summary in JSON format

**Example results:**
```json
{
  "success": ["Coffee", "ECG200", "GunPoint"],
  "failed": {
    "ProblematicDataset": {
      "status": "init_failed",
      "error": "Error message..."
    }
  },
  "config": {
    "worker_profile": "default",
    "worker_duration": 20,
    "total_datasets": 88
  }
}
```

**Estimated time:**
- Per dataset: ~2-10 minutes (init + optimize) + 20 seconds (workers)
- All 88 datasets: **4-13 hours**

---

## Project Structure

```
drifts/
├── README.md                          # Main documentation
├── DOCKER_GUIDE.md                    # Docker setup guide
├── requirements.txt                   # Python dependencies
│
├── init_aeon_univariate.py           # Dataset initialization script
├── enhanced_launch_workers.py        # Worker management system
├── test_datasets_with_workers.py     # Automated testing script
├── monitor_live.py                   # Live test monitoring
├── worker_config.yaml                # Worker configuration
│
├── worker_cache.py                   # Worker script (simple)
├── worker_cache_logged.py            # Worker script (with logging)
├── ar_check_cache.py                 # Alternative worker script
├── rcheck_cache.py                   # R-check worker script
│
├── tree.py                           # Tree utilities
├── forest.py                         # Forest utilities
├── cost_function.py                  # Cost function implementation
├── helpers.py                        # Helper functions
│
├── docker/                           # Docker configuration
│   └── supervisord.conf              # Supervisor config for container
├── Dockerfile                        # Docker image definition
├── run.bat / run.sh                  # Docker management scripts
│
├── scripts/                          # Utility and helper scripts
│   ├── utilities/                    # Data conversion utilities
│   ├── monitoring/                   # Monitoring scripts
│   ├── testing/                      # Test scripts
│   ├── redis_tools/                  # Redis management tools
│   └── README.md                     # Scripts documentation
│
├── notebooks/                        # Jupyter notebooks
│   ├── models_analysis.ipynb         # Model analysis
│   ├── redis_*.ipynb                 # Redis management notebooks
│   └── README.md                     # Notebooks documentation
│
├── redis_helpers/                    # Redis helper modules
│   ├── endpoints.py
│   ├── forest.py
│   ├── icf.py
│   ├── preferred.py
│   ├── samples.py
│   └── utils.py
│
├── etl/                              # ETL and reporting tools
│
├── logs/                             # Worker logs (auto-generated)
├── workers/                          # Worker PID files (auto-generated)
├── results/                          # Test results (auto-generated)
└── temp/                             # Temporary files (auto-generated)
```

---

## Documentation

- **README.md** (this file) - Complete project documentation
- **DOCKER_GUIDE.md** - Docker setup and usage guide
- **scripts/README.md** - Documentation for utility scripts
- **notebooks/README.md** - Documentation for Jupyter notebooks

---
