"""Compatibility shim for older imports.

This file re-exports the primary functions from `scripts/analyze_results.py`
so notebooks and other code that do `from drifts_results import ...` keep working.

It also provides small helper functions used in the notebooks.
"""
from pathlib import Path
from typing import List, Optional

# Try multiple import paths so the shim works both when running from the
# repository root (scripts/analyze_results.py) and when installed as a module.
try:
    # Preferred: import from the scripts module under the project root
    from scripts.analyze_results import (
        compute_counts_from_results,
        list_missing_manifests,
    )
except Exception:
    try:
        # Fallback: import analyze_results as a top-level module
        from analyze_results import (
            compute_counts_from_results,
            list_missing_manifests,
        )
    except Exception:
        # Final fallback: provide stubs that raise helpful errors at runtime
        def compute_counts_from_results(results_dir: Path, verbose: bool = False):
            raise RuntimeError(
                "compute_counts_from_results is not available; make sure scripts/analyze_results.py is present and importable"
            )
        def list_missing_manifests(results_dir: Path) -> List[str]:
            raise RuntimeError(
                "list_missing_manifests is not available; make sure scripts/analyze_results.py is present and importable"
            )
        )

# Small helpers that are convenient in notebooks
try:
    import pandas as pd
except Exception:  # pragma: no cover - pandas should be present for notebook work
    pd = None


def load_analyzed_df(fr_csv: Path):
    """Load an existing analyzed CSV and prefer rows marked as analyzed.

    If the CSV has an "analyzed" column this function returns only rows where
    analyzed == "YES" (unless that would be empty, in that case returns all rows).
    If the CSV doesn't exist, returns an empty DataFrame with expected columns.
    """
    if pd is None:
        raise RuntimeError("pandas is required to use load_analyzed_df")
    if fr_csv.exists():
        base = pd.read_csv(fr_csv)
        if "analyzed" in base.columns:
            analyzed = base[base["analyzed"] == "YES"].copy()
            if analyzed.empty:
                analyzed = base.copy()
        else:
            analyzed = base.copy()
            analyzed["analyzed"] = "YES"
        return analyzed
    return pd.DataFrame(columns=["dataset", "analyzed"])


def cast_dataset_str(df):
    """Ensure the `dataset` column is present and is of string dtype."""
    if pd is None:
        raise RuntimeError("pandas is required to use cast_dataset_str")
    if "dataset" in df.columns:
        df = df.dropna(subset=["dataset"]).copy()
        df["dataset"] = df["dataset"].astype(str)
    return df
