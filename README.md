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

3. **(Optional) Start Redis**
   Some scripts (e.g., `init_aeon_univariate.py`) push forests, samples, and
   metadata into Redis databases. Launch a Redis server locally if you intend
   to use those features:
   ```bash
   redis-server
   ```

## Usage

### Dataset forest report

Use `dataset_forest_report.py` to benchmark the Random Forest optimisation
pipeline across Aeon datasets and gather structural statistics:

```bash
python dataset_forest_report.py --output results/forest_report.json
```

If the path supplied via `--output` already exists as a directory, the report
is written to `dataset_forest_report.json` within that folder.

The script:

- loads the listed datasets (or all available datasets when `--datasets` is
  omitted),
- skips datasets where the time series have inconsistent lengths, reporting a
  series length of `0` for those entries,
- runs Bayesian optimisation of a Random Forest classifier via
  `scikit-optimize`,
- converts the best estimator into the internal `Forest` representation to
  inspect the tree structure,
- reports the cross-validated accuracy used during optimisation so you can
  compare datasets on a consistent footing,
- prints a summary table with dataset sizes, class counts, and forest
  statistics, and optionally writes the full JSON report to the path provided
  via `--output`.

### Aeon dataset initialisation

`init_aeon_univariate.py` exposes a command-line utility for initialising the
Redis caches with samples and optimised forests for a single dataset. Examples:

```bash
# List supported datasets
python init_aeon_univariate.py --list-datasets

# Optimise a Random Forest for the ECG200 dataset using Bayesian search
python init_aeon_univariate.py ECG200 --class-label "1" --optimize-rf
```

Key options exposed by `init_aeon_univariate.py`:

#### Core arguments

- `dataset_name` *(positional, optional)* — Dataset to load. Required unless
  `--list-datasets` is used.
- `--class-label` *(str)* — Class label whose samples will be processed. Required
  when `dataset_name` is supplied.
- `--list-datasets` — Print the curated catalogue of supported Aeon datasets
  and exit.
- `--info` — Display dataset metadata without performing any processing.

#### Random Forest parameters

- `--n-estimators` *(int, default: 50)* — Number of trees in the forest when
  manual hyper-parameters are supplied instead of optimisation.
- `--criterion` *(gini | entropy)* — Split quality measure.
- `--max-depth` *(int)* — Maximum tree depth.
- `--min-samples-split` *(int)* — Minimum samples required to split an internal
  node.
- `--min-samples-leaf` *(int)* — Minimum samples required at a leaf node.
- `--max-features` *(str)* — Feature subset size (e.g., `sqrt`, `log2`, or a
  float fraction).
- `--max-leaf-nodes` *(int)* — Cap on the number of leaf nodes.
- `--min-impurity-decrease` *(float)* — Minimum impurity reduction needed to
  accept a split.
- `--bootstrap` *(True | False)* — Whether bootstrap sampling is enabled.
- `--max-samples` *(float)* — Fraction of the training set drawn for each tree
  when bootstrapping is active.
- `--ccp-alpha` *(float)* — Complexity pruning parameter.

#### Bayesian optimisation controls

- `--optimize-rf` — Enable Bayesian optimisation via `scikit-optimize` to tune
  the Random Forest hyper-parameters.
- `--opt-n-iter` *(int, default: 50)* — Number of optimisation iterations.
- `--opt-cv` *(int, default: 5)* — Cross-validation folds for scoring.
- `--opt-n-jobs` *(int, default: -1)* — Parallel jobs; `-1` uses all CPU cores.
- `--opt-use-test` — Validate candidate models on the Aeon test split instead
  of cross-validation (use cautiously to avoid overfitting).

#### Sampling and infrastructure

- `--sample-percentage` *(float, default: 100.0)* — Percentage of the selected
  dataset to process.
- `--test-split` *(float)* — Combine the Aeon train/test splits and create a
  custom train/test partition with the given test fraction (e.g., `0.3`).
- `--random-state` *(int, default: 42)* — Seed for reproducibility.
- `--feature-prefix` *(str, default: `t`)* — Prefix used when naming features
  in the exported samples.
- `--no-clean` — Skip the initial wipe of Redis databases before inserting new
  data.
- `--redis-port` *(int, default: 6379)* — Port of the Redis or KeyDB instance
  used by the workers.

Run `python init_aeon_univariate.py --help` to view the auto-generated help
message with the latest defaults.
