import pandas as pd
from typing import Any

# Column -> colormap mapping consistent with the notebook legend
COLUMN_COLORMAPS = {
    'n_estimators': 'Reds',
    'eu_complexity': 'Oranges',
    'series_length': 'YlOrBr',
    'n_features': 'YlGn',
    'mean_eu': 'Greens',
    'eu_min': 'Greens',
    'eu_max': 'Greens',
    'eu_std_dev': 'Greens',
    'Candidate': 'Purples',
    'Reason': 'Greens',
    'Non-reason': 'Blues',
    'Candidate Anti-reason': 'Oranges',
    'Anti-reason': 'Reds',
    'Good profile': 'BuGn',
    'Bad profile': 'PuRd',
    'Preferred reason': 'PuBu',
    'Anti-reason profile': 'YlGnBu',
    'Total': 'Greys',
    'Worker span (s)': 'PuBuGn',
    'avg_depth': 'GnBu',
    'avg_leaves': 'Blues',
    'avg_nodes': 'BuPu',
    'train_size': 'Purples',
    'test_size': 'RdPu',
    'Total time (s) max': 'YlOrBr',
    'Total time (s) mean': 'YlOrBr',
    'ICF checks': 'PuBuGn',
    'Reason check iteration total': 'Purples',
    'IterGoodRatio': 'Greens',
    'IterBadRatio': 'Reds',
    'Earlystop Good total': 'PuBu',
    'ESG': 'BuGn',
    'ESB': 'Oranges',
    'Filtrered rate': 'YlGnBu',
}
DEFAULT_CMAP = 'Greys'

ANALYZED_COLORS = {
    'YES': 'background-color: #b6e192',
    'NO': 'background-color: #f6583e',
    'N/A': '',
}


def style_summary_table(df: pd.DataFrame) -> Any:
    styled = df.style
    numeric_cols = df.select_dtypes(include='number').columns
    for col in numeric_cols:
        col_min = df[col].min(skipna=True)
        col_max = df[col].max(skipna=True)
        if pd.isna(col_min) or pd.isna(col_max):
            continue
        cmap = COLUMN_COLORMAPS.get(col, DEFAULT_CMAP)
        if col_min == col_max:
            styled = styled.background_gradient(subset=[col], cmap=cmap)
        else:
            styled = styled.background_gradient(
                subset=[col],
                cmap=cmap,
                vmin=col_min,
                vmax=col_max,
            )
    if 'analyzed' in df.columns:
        def analyzed_style(val: Any) -> str:
            return ANALYZED_COLORS.get(val, '')

        styled = styled.map(analyzed_style, subset=['analyzed'])
    return styled


def print_color_legend() -> None:
    header = "=" * 80
    lines = [header, "COLUMN COLOR GRADIENTS", header]
    for column, cmap in COLUMN_COLORMAPS.items():
        lines.append(f"  {column:<22} -> {cmap} (min -> max)")
    lines.append("  analyzed             -> Green (YES) / Red (NO)")
    lines.append(header)
    print("\n".join(lines))
