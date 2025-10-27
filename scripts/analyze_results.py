"""Compatibilita CLI: delega tutte le funzioni a ``drifts_results``.

Questo modulo viene mantenuto per chi invoca ancora
``python scripts/analyze_results.py`` oppure importa le funzioni da qui.
La logica effettiva vive ora in ``drifts_results.py``.
"""

from pathlib import Path
import sys

if __package__ in (None, "", "__main__"):
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from etl.drifts_results import (  # noqa: F401 re-export
    CAT_LIST,
    DB_TO_CAT,
    CATEGORY_FULL_NAMES,
    DISPLAY_CATEGORIES,
    DISPLAY_NAMES,
    DISPLAY_LABELS,
    SUMMARY_EXTRA_COLUMNS,
    cast_dataset_str,
    compute_counts_from_results,
    fix_notebook_calls,
    list_missing_manifests,
    load_analyzed_df,
    main as _main,
    patch_models_analysis_enriched,
    print_counts_summary,
    summarize_counts,
    summarize_counts_from_df,
)

main = _main

__all__ = [
    "CAT_LIST",
    "DB_TO_CAT",
    "CATEGORY_FULL_NAMES",
    "DISPLAY_CATEGORIES",
    "DISPLAY_NAMES",
    "DISPLAY_LABELS",
    "SUMMARY_EXTRA_COLUMNS",
    "cast_dataset_str",
    "compute_counts_from_results",
    "fix_notebook_calls",
    "list_missing_manifests",
    "load_analyzed_df",
    "main",
    "patch_models_analysis_enriched",
    "print_counts_summary",
    "summarize_counts",
    "summarize_counts_from_df",
]


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

