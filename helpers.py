"""
General helper utilities used across the project.
"""
from typing import Dict, Any, Optional


def increment_info_counter(info: Optional[Dict[str, Any]], key: str, amount: int = 1, prefix: str = ""):
    """
    Helper to increment info counters with optional prefix.
    
    Args:
        info: Optional dictionary to store statistics (None is safe, does nothing)
        key: Counter key
        amount: Amount to increment (default: 1)
        prefix: Optional prefix for the key (default: "")
        
    Example:
        >>> info = {}
        >>> increment_info_counter(info, 'count', 1)
        >>> increment_info_counter(info, 'hits', 1, prefix='cache_')
        >>> info
        {'count': 1, 'cache_hits': 1}
    """
    if info is not None:
        full_key = f"{prefix}{key}" if prefix else key
        info[full_key] = info.get(full_key, 0) + amount