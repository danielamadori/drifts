#!/usr/bin/env python3
"""Generate an alphabetical summary for every available univariate dataset.

For each dataset listed in ``init_aeon_univariate.AVAILABLE_DATASETS`` the
script gathers basic metadata (dataset size, number of channels, length,
class information) and, whenever the series length is consistent, mimics the
behaviour of running::

    python ./init_aeon_univariate.py <dataset> --class <first_class> --optimize

The optimisation stage relies exclusively on the ``--optimize`` workflow of
``init_aeon_univariate``: the first class discovered in the dataset is used as
target label and Bayesian optimisation is executed to obtain a tuned Random
Forest model. Structural statistics (number of trees, depth, nodes, leaves)
from the optimised forest are recorded so that the size of the produced forests
can be compared across datasets.

Datasets containing time series with inconsistent lengths are reported with a
``series_length`` of ``0`` and the optimisation stage is skipped for them.
Results are printed as a table and can optionally be written to disk as JSON.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
from aeon.datasets import load_classification

from init_aeon_univariate import (
    AVAILABLE_DATASETS,
    convert_numpy_types,
    get_rf_search_space,
    optimize_rf_hyperparameters,
)
from skforest_to_forest import sklearn_forest_to_forest


try:  # pragma: no cover - handled at runtime
    import skopt  # type: ignore  # noqa: F401
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise SystemExit(
        "scikit-optimize is required. Install it with 'pip install scikit-optimize'."
    ) from exc


@dataclass
class DatasetMetadata:
    """Basic metadata gathered for each dataset."""

    dataset: str
    train_size: int = 0
    test_size: int = 0
    n_channels: int = 0
    series_length: int = 0
    n_classes: int = 0
    classes: Sequence[Any] = ()
    length_consistent: bool = False
    first_class: Optional[str] = None


@dataclass
class ForestStatistics:
    """Structural information extracted from a fitted Random Forest."""

    n_estimators: int
    min_depth: int
    max_depth: int
    avg_depth: float
    avg_nodes: float
    avg_leaves: float


@dataclass
class DatasetReport:
    """Full report for a dataset, including metadata and training stats."""

    dataset: str
    status: str
    metadata: DatasetMetadata
    best_params: Optional[Dict[str, Any]] = None
    validation_score: Optional[float] = None
    test_score: Optional[float] = None
    forest_statistics: Optional[ForestStatistics] = None
    endpoints_universe: Optional[Dict[str, List[float]]] = None
    error: Optional[str] = None

    def to_json_ready(self) -> Dict[str, Any]:
        """Convert the report into a JSON-safe dictionary."""

        payload: Dict[str, Any] = {
            "dataset": self.dataset,
            "status": self.status,
            "metadata": asdict(self.metadata),
        }

        if self.best_params is not None:
            payload["best_params"] = convert_numpy_types(self.best_params)
        if self.validation_score is not None:
            payload["validation_score"] = self.validation_score
        if self.test_score is not None:
            payload["test_score"] = self.test_score
        if self.forest_statistics is not None:
            payload["forest_statistics"] = asdict(self.forest_statistics)
        if self.endpoints_universe is not None:
            payload["endpoints_universe"] = convert_numpy_types(self.endpoints_universe)
        if self.error is not None:
            payload["error"] = self.error

        return payload


def _flatten_dataset(X: np.ndarray) -> np.ndarray:
    """Reshape time-series arrays into a 2D feature matrix."""

    if X.dtype == object:
        stacked = np.stack([np.asarray(sample).reshape(-1) for sample in X])
        return stacked

    if X.ndim == 3:
        return X.reshape(X.shape[0], -1)

    if X.ndim == 2:
        return X

    raise ValueError(
        "Unsupported array shape for flattening: "
        f"expected 2D or 3D data but received {X.shape}."
    )


def _extract_series_length(samples: np.ndarray) -> (int, bool):
    """Determine whether the provided samples share the same length."""

    if samples.dtype == object:
        lengths = {np.asarray(item).shape[-1] for item in samples}
        if len(lengths) == 1:
            return next(iter(lengths)), True
        return 0, False

    if samples.ndim >= 3:
        return int(samples.shape[-1]), True

    if samples.ndim == 2:
        return int(samples.shape[-1]), True

    return 0, False


def _gather_metadata(dataset: str) -> DatasetMetadata:
    """Load the dataset and collect metadata about it."""

    X_train, y_train = load_classification(dataset, split="train")
    X_test, y_test = load_classification(dataset, split="test")

    train_length, train_consistent = _extract_series_length(X_train)
    test_length, test_consistent = _extract_series_length(X_test)
    consistent = train_consistent and test_consistent and train_length == test_length

    classes = np.unique(np.concatenate([y_train, y_test])).astype(str)
    classes_sorted = np.sort(classes)
    first_class = str(classes_sorted[0]) if classes_sorted.size else None

    if X_train.dtype == object and consistent:
        sample_shape = np.asarray(X_train[0]).shape
        n_channels = sample_shape[0] if len(sample_shape) == 2 else 1
    elif X_train.ndim >= 3:
        n_channels = int(X_train.shape[1])
    else:
        n_channels = 1

    metadata = DatasetMetadata(
        dataset=dataset,
        train_size=int(X_train.shape[0]),
        test_size=int(X_test.shape[0]),
        n_channels=n_channels,
        series_length=train_length if consistent else 0,
        n_classes=int(len(classes_sorted)),
        classes=classes_sorted.tolist(),
        length_consistent=bool(consistent),
        first_class=first_class,
    )
    return metadata


def _summarise_forest(estimators: Sequence[Any]) -> ForestStatistics:
    """Compute aggregated statistics from fitted sklearn estimators."""

    depths = [int(estimator.tree_.max_depth) for estimator in estimators]
    node_counts = [int(estimator.tree_.node_count) for estimator in estimators]
    leaf_counts = [int(estimator.tree_.n_leaves) for estimator in estimators]

    return ForestStatistics(
        n_estimators=len(estimators),
        min_depth=min(depths),
        max_depth=max(depths),
        avg_depth=float(statistics.fmean(depths)),
        avg_nodes=float(statistics.fmean(node_counts)),
        avg_leaves=float(statistics.fmean(leaf_counts)),
    )


DEFAULT_OPTIMIZATION_ITERATIONS = 25
DEFAULT_OPTIMIZATION_CV = 5
DEFAULT_OPTIMIZATION_N_JOBS = -1
DEFAULT_RANDOM_STATE = 42
INCLUDE_BOOTSTRAP_IN_SEARCH = True


def generate_report(
    dataset: str,
    metadata: DatasetMetadata,
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
    n_iter: int = DEFAULT_OPTIMIZATION_ITERATIONS,
    cv: int = DEFAULT_OPTIMIZATION_CV,
    n_jobs: int = DEFAULT_OPTIMIZATION_N_JOBS,
    include_bootstrap: bool = INCLUDE_BOOTSTRAP_IN_SEARCH,
) -> DatasetReport:
    """Generate the optimisation report for a single dataset."""

    if not metadata.length_consistent:
        return DatasetReport(
            dataset=dataset,
            status="skipped_variable_length",
            metadata=metadata,
        )

    try:
        X_train, y_train = load_classification(dataset, split="train")
        X_test, y_test = load_classification(dataset, split="test")
    except Exception as exc:  # pragma: no cover - runtime safety
        return DatasetReport(
            dataset=dataset,
            status="failed",
            metadata=metadata,
            error=f"Failed to reload dataset: {exc}",
        )

    X_train_2d = _flatten_dataset(X_train)
    X_test_2d = _flatten_dataset(X_test)

    feature_count = X_train_2d.shape[1]
    padding_width = max(1, int(math.log10(feature_count)) + 1)
    feature_names = [f"t_{i:0{padding_width}d}" for i in range(feature_count)]

    search_space = get_rf_search_space(include_bootstrap=include_bootstrap)

    try:
        best_params, best_score, test_score, optimizer = optimize_rf_hyperparameters(
            X_train_2d,
            y_train,
            search_space,
            n_iter=n_iter,
            cv=cv,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=0,
            X_test=X_test_2d,
            y_test=y_test,
            use_test_for_validation=False,
        )
    except Exception as exc:  # pragma: no cover - runtime safety
        return DatasetReport(
            dataset=dataset,
            status="failed",
            metadata=metadata,
            error=f"Optimisation failed: {exc}",
        )

    if hasattr(optimizer, "best_estimator_"):
        best_estimator = optimizer.best_estimator_
    else:
        best_estimator = optimizer

    forest_stats = _summarise_forest(best_estimator.estimators_)

    # Convert to internal forest and extract endpoints universe
    internal_forest = sklearn_forest_to_forest(
        best_estimator,
        feature_names=feature_names,
        class_names=metadata.classes,
    )
    
    # Extract endpoints universe (EU) from the forest
    try:
        endpoints_universe = internal_forest.extract_feature_thresholds()
    except Exception:
        endpoints_universe = None

    return DatasetReport(
        dataset=dataset,
        status="optimized",
        metadata=metadata,
        best_params=best_params,
        validation_score=float(best_score) if best_score is not None else None,
        test_score=float(test_score) if test_score is not None else None,
        forest_statistics=forest_stats,
        endpoints_universe=endpoints_universe,
    )


def _format_table(reports: Iterable[DatasetReport]) -> str:
    """Create a readable summary table for the console."""

    headers = [
        "Dataset",
        "Status",
        "Train",
        "Test",
        "Series length",
        "Classes",
        "CV score",
        "First class",
        "Trees",
        "Avg depth",
    ]

    rows: List[List[str]] = []
    for report in reports:
        meta = report.metadata
        stats = report.forest_statistics
        rows.append(
            [
                report.dataset,
                report.status,
                str(meta.train_size),
                str(meta.test_size),
                str(meta.series_length),
                str(meta.n_classes),
                f"{report.validation_score:.3f}" if report.validation_score is not None else "-",
                meta.first_class or "-",
                f"{stats.n_estimators}" if stats else "-",
                f"{stats.avg_depth:.2f}" if stats else "-",
            ]
        )

    col_widths = [max(len(row[i]) for row in ([headers] + rows)) for i in range(len(headers))]

    def _fmt(row: Sequence[str]) -> str:
        return " | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row))

    separator = "-+-".join("-" * width for width in col_widths)

    output_lines = [_fmt(headers), separator]
    output_lines.extend(_fmt(row) for row in rows)
    return "\n".join(output_lines)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to write the JSON report. If omitted, the report is not written to disk.",
    )
    return parser.parse_args(argv)


def _prepare_output_path(path: Path) -> Path:
    """Normalise the output path provided via the CLI.

    When users pass an existing directory we drop the JSON file inside it.
    Otherwise the supplied path is treated as the file destination.  Any
    missing parent directories are created automatically.
    """

    if path.exists() and path.is_dir():
        path = path / "dataset_forest_report.json"

    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write_reports(output_path: Path, reports: Sequence[DatasetReport]) -> Path:
    """Persist the collected reports to disk."""

    payload = [report.to_json_ready() for report in reports]
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    return output_path


def _filter_dataclass_kwargs(
    data: Mapping[str, Any], cls: type
) -> Dict[str, Any]:
    """Return a dictionary restricted to the fields accepted by ``cls``."""

    allowed = {field.name for field in fields(cls)}
    return {key: value for key, value in data.items() if key in allowed}


def _load_existing_reports(output_path: Path) -> Tuple[List[DatasetReport], Set[str]]:
    """Load reports from an existing JSON file if it is available.

    Returns both the parsed reports and the set of dataset names that were
    already processed. Any issue while reading or parsing the file results in an
    empty collection so that the caller can safely resume from scratch.
    """

    if not output_path.exists():
        return [], set()

    try:
        raw_content = output_path.read_text()
    except OSError as exc:  # pragma: no cover - runtime safety
        print(f"Warning: failed to read existing report '{output_path}': {exc}")
        return [], set()

    raw_content = raw_content.strip()
    if not raw_content:
        return [], set()

    try:
        payload = json.loads(raw_content)
    except json.JSONDecodeError as exc:  # pragma: no cover - runtime safety
        print(
            f"Warning: could not parse JSON report '{output_path}'; "
            "ignoring previous results."
        )
        print(f"  Parsing error: {exc}")
        return [], set()

    if not isinstance(payload, list):
        print(
            f"Warning: unexpected JSON structure in '{output_path}'; "
            "ignoring previous results."
        )
        return [], set()

    existing_reports: List[DatasetReport] = []
    completed_datasets: Set[str] = set()

    for item in payload:
        if not isinstance(item, dict):
            continue

        dataset_name = item.get("dataset")
        if not isinstance(dataset_name, str) or not dataset_name:
            continue

        completed_datasets.add(dataset_name)

        metadata_payload = item.get("metadata")
        metadata_dict = (
            _filter_dataclass_kwargs(dict(metadata_payload), DatasetMetadata)
            if isinstance(metadata_payload, dict)
            else {}
        )
        metadata_dict.setdefault("dataset", dataset_name)
        metadata = DatasetMetadata(**metadata_dict)

        forest_payload = item.get("forest_statistics")
        forest_statistics = None
        if isinstance(forest_payload, dict):
            forest_kwargs = _filter_dataclass_kwargs(
                dict(forest_payload), ForestStatistics
            )
            if forest_kwargs:
                try:
                    forest_statistics = ForestStatistics(**forest_kwargs)
                except TypeError:
                    forest_statistics = None

        report = DatasetReport(
            dataset=dataset_name,
            status=str(item.get("status") or "unknown"),
            metadata=metadata,
            best_params=item.get("best_params"),
            validation_score=item.get("validation_score"),
            test_score=item.get("test_score"),
            forest_statistics=forest_statistics,
            error=item.get("error"),
        )
        existing_reports.append(report)

    return existing_reports, completed_datasets


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    datasets = sorted(set(AVAILABLE_DATASETS))

    output_path: Optional[Path] = None
    if args.output:
        output_path = _prepare_output_path(args.output)

    existing_reports: List[DatasetReport] = []
    completed_datasets: Set[str] = set()
    if output_path is not None:
        existing_reports, completed_datasets = _load_existing_reports(output_path)

    reports: List[DatasetReport] = list(existing_reports)
    interrupted = False
    new_reports_generated = False

    try:
        for dataset in datasets:
            if dataset in completed_datasets:
                print(f"Skipping dataset already present in report: {dataset}")
                continue

            print(f"Processing dataset: {dataset}")
            try:
                metadata = _gather_metadata(dataset)
            except Exception as exc:  # pragma: no cover - runtime safety
                reports.append(
                    DatasetReport(
                        dataset=dataset,
                        status="failed",
                        metadata=DatasetMetadata(dataset=dataset),
                        error=f"Failed to load dataset metadata: {exc}",
                    )
                )
                completed_datasets.add(dataset)
                new_reports_generated = True
                if output_path is not None:
                    _write_reports(output_path, reports)
                continue

            if metadata.first_class:
                print(f"  Target class: {metadata.first_class}")

            report = generate_report(dataset, metadata)
            reports.append(report)
            completed_datasets.add(dataset)
            new_reports_generated = True

            if output_path is not None:
                _write_reports(output_path, reports)
    except KeyboardInterrupt:  # pragma: no cover - runtime safety
        interrupted = True
        print("\nInterrupted by user. Writing partial results...")

    if reports:
        print("\n" + _format_table(reports))
    else:
        print("\nNo reports were generated.")

    if output_path is not None and (new_reports_generated or not output_path.exists()):
        _write_reports(output_path, reports)
        print(f"\nReport written to {output_path}")
    elif output_path is not None:
        print(f"\nReport already up-to-date at {output_path}")

    return 130 if interrupted else 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
