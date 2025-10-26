import pandas as pd
from typing import Any

# Column → colormap mapping consistent with the notebook legend
COLUMN_COLORMAPS = {
    'n_estimators': 'Reds',
    'eu_complexity': 'Oranges',
    'series_length': 'YlOrBr',
    'n_features': 'YlGn',
    'mean_eu': 'Greens',
    'avg_depth': 'GnBu',
    'avg_leaves': 'Blues',
    'avg_nodes': 'BuPu',
    'train_size': 'Purples',
    'test_size': 'RdPu',
}

ANALYZED_COLORS = {'YES': 'background-color: #b6e192', 'NO': 'background-color: #f6583e', 'N/A': ''}

def style_summary_table(df: pd.DataFrame) -> Any:
    styled = df.style
    # Apply gradients to numeric columns according to the legend
    for col, cmap in COLUMN_COLORMAPS.items():
        if col in df.columns:
            styled = styled.background_gradient(subset=[col], cmap=cmap)
    # Custom coloring for the 'analyzed' column
    if 'analyzed' in df.columns:
        def analyzed_style(val):
            return ANALYZED_COLORS.get(val, '')
        styled = styled.applymap(analyzed_style, subset=['analyzed'])
    return styled

def print_color_legend():
    print("""================================================================================
COLUMN COLOR GRADIENTS
================================================================================
  n_estimators         → Reds (white → red)
  eu_complexity        → Oranges (white → orange)
  series_length        → YlOrBr (yellow → orange → brown)
  n_features           → YlGn (yellow → green)
  mean_eu              → Greens (white → green)
  avg_depth            → GnBu (green → blue)
  avg_leaves           → Blues (white → blue)
  avg_nodes            → BuPu (blue → purple)
  train_size           → Purples (white → purple)
  test_size            → RdPu (red → purple)
  analyzed             → Green (YES) / Red (NO)
================================================================================
""")
