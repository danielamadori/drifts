# icde

This repository contains utilities for converting scikit-learn tree ensembles
into the internal ICDE representation and for preparing time-series datasets
from the [Aeon](https://www.aeon-toolkit.org/) collection. The tooling spans
from dataset initialisation scripts to helpers that persist forests and samples
into Redis-backed caches.

## Installation

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

Key options exposed by `init_aeon_univariate.py`:

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

### Step 1: Test with a Single Dataset (5 minutes)

```bash
# Initialize Coffee dataset with Bayesian optimization
python3 init_aeon_univariate.py Coffee --class-label "0" --optimize
```

### Step 2: Start the Worker Algorithm

```bash
# Launch workers to process the initialized dataset (default: 1 worker)
python3 enhanced_launch_workers.py start

# Or use production profile (8 cache workers + 4 rcheck workers)
python3 enhanced_launch_workers.py start --profile production
```

**Key parameters:**
- `start` — Start workers using configuration
- `--profile {development|production}` — Use predefined worker profiles
  - `development`: 2 cache workers with verbose logging
  - `production`: 8 cache workers + 4 rcheck workers
- `--config FILE` — Use custom YAML configuration file
- `--groups GROUP [GROUP ...]` — Start specific worker groups only

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
jupyter notebook dataset_complexity_analysis.ipynb
```

**What you'll see:**
- Summary table with all datasets ranked by complexity
- EU statistics (MIN/MAX/MEAN/STD for endpoint counts)
- Bar charts showing smallest datasets by estimator count

---

### Dataset Complexity Metrics

The system tracks multiple complexity indicators:

- **n_estimators**: Number of trees in the optimized Random Forest
- **EU complexity**: Product of mean endpoint count × number of features
- **series_length**: Original time series length

Use `dataset_complexity_analysis.ipynb` to explore correlations and generate visualizations.
