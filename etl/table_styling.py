
"""
Styling functions for dataset_complexity_analysis notebook
Provides color gradients for all numeric columns
"""

import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.cm as cm


def highlight_column(s, cmap_name='RdYlGn_r'):
    """
    Apply color gradient to a numeric column, handling NaN values
    
    Parameters:
    -----------
    s : pd.Series
        Column to apply gradient to
    cmap_name : str
        Matplotlib colormap name (default: 'RdYlGn_r')
    
    Returns:
    --------
    list : List of CSS background-color strings
    """
    # Extract valid values
    valid_mask = pd.notna(s)
    if not valid_mask.any():
        return [''] * len(s)
    
    valid_values = s[valid_mask]
    vmin, vmax = valid_values.min(), valid_values.max()
    
    # Normalize values
    if vmax > vmin:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = lambda x: 0.5
    
    # Use specified colormap
    cmap = cm.get_cmap(cmap_name)
    
    # Apply colors only to valid values
    colors = []
    for val in s:
        if pd.notna(val):
            rgba = cmap(norm(val))
            colors.append(f'background-color: {mcolors.rgb2hex(rgba[:3])}')
        else:
            colors.append('')
    
    return colors


def style_summary_table(summary_df):
    """
    Apply complete styling to summary DataFrame with all column gradients
    
    Parameters:
    -----------
    summary_df : pd.DataFrame
        DataFrame with dataset summary
    
    Returns:
    --------
    pd.Styler : Styled DataFrame
    """
    styled = summary_df.style.format({
        'n_estimators': lambda x: f'{int(x):,}' if pd.notna(x) else 'N/A',
        'series_length': lambda x: f'{int(x):,}' if pd.notna(x) else 'N/A',
        'n_features': lambda x: f'{int(x):,}' if pd.notna(x) else 'N/A',
        'mean_eu': lambda x: f'{x:.2f}' if pd.notna(x) else 'N/A',
        'eu_complexity': lambda x: f'{x:,.0f}' if pd.notna(x) else 'N/A',
        'analyzed': lambda x: x,
        'avg_depth': lambda x: f'{x:.2f}' if pd.notna(x) else 'N/A',
        'avg_leaves': lambda x: f'{x:,.2f}' if pd.notna(x) else 'N/A',
        'avg_nodes': lambda x: f'{x:,.2f}' if pd.notna(x) else 'N/A',
        'train_size': lambda x: f'{int(x):,}' if pd.notna(x) else 'N/A',
        'test_size': lambda x: f'{int(x):,}' if pd.notna(x) else 'N/A',
    })
    
    # Apply gradients to all numeric columns with progressive chromatic transition
    # Reds → Oranges → Yellows → Greens → Blues → Purples
    styled = styled.apply(lambda s: highlight_column(s, 'Reds'), subset=['n_estimators'])
    styled = styled.apply(lambda s: highlight_column(s, 'Oranges'), subset=['eu_complexity'])
    styled = styled.apply(lambda s: highlight_column(s, 'YlOrBr'), subset=['series_length'])
    styled = styled.apply(lambda s: highlight_column(s, 'YlGn'), subset=['n_features'])
    styled = styled.apply(lambda s: highlight_column(s, 'Greens'), subset=['mean_eu'])
    styled = styled.apply(lambda s: highlight_column(s, 'GnBu'), subset=['avg_depth'])
    styled = styled.apply(lambda s: highlight_column(s, 'Blues'), subset=['avg_leaves'])
    styled = styled.apply(lambda s: highlight_column(s, 'BuPu'), subset=['avg_nodes'])
    styled = styled.apply(lambda s: highlight_column(s, 'Purples'), subset=['train_size'])
    styled = styled.apply(lambda s: highlight_column(s, 'RdPu'), subset=['test_size'])
    
    # Bold EU columns
    styled = styled.set_properties(
        subset=['n_features', 'mean_eu', 'eu_complexity'],
        **{'font-weight': 'bold'}
    )
    
    # Color analyzed column
    styled = styled.apply(
        lambda x: ['background-color: #d4edda; font-weight: bold' if v == 'YES' else 'background-color: #f8d7da' if v == 'NO' else '' for v in x],
        subset=['analyzed']
    )
    
    return styled


# Color mapping documentation
COLUMN_COLORS = {
    'n_estimators': 'Reds (white → red)',
    'eu_complexity': 'Oranges (white → orange)',
    'series_length': 'YlOrBr (yellow → orange → brown)',
    'n_features': 'YlGn (yellow → green)',
    'mean_eu': 'Greens (white → green)',
    'avg_depth': 'GnBu (green → blue)',
    'avg_leaves': 'Blues (white → blue)',
    'avg_nodes': 'BuPu (blue → purple)',
    'train_size': 'Purples (white → purple)',
    'test_size': 'RdPu (red → purple)',
    'analyzed': 'Green (YES) / Red (NO)',
}


def print_color_legend():
    """Print the color legend for all columns"""
    print("=" * 80)
    print("COLUMN COLOR GRADIENTS")
    print("=" * 80)
    for col, gradient in COLUMN_COLORS.items():
        print(f"  {col:20s} → {gradient}")
    print("=" * 80)
