"""
Redis helper functions for Preferred Reasons (PR) database.
"""
import redis
import json
import datetime
from typing import Optional, Set, List, Tuple


def get_pr_candidate(pr_connection: redis.Redis, scan_count: int = 100) -> Optional[Tuple[str, Set[str]]]:
    """
    Get the best candidate from PR database based on selection criteria:
    1. Minimum number of timestamps in set
    2. If tie, minimum maximum timestamp (earliest last access)
    
    Uses SCAN to iterate through keys without blocking.
    
    Args:
        pr_connection: Redis connection to PR database
        scan_count: Number of keys to fetch per SCAN iteration
        
    Returns:
        Tuple of (key, set of timestamps) or None if PR is empty
    """
    try:
        # Build list of (key, timestamps_set, max_timestamp) using SCAN
        candidates = []
        cursor = 0
        
        while True:
            cursor, keys = pr_connection.scan(cursor, match='*', count=scan_count)
            
            for key in keys:
                value = pr_connection.get(key)
                if value:
                    timestamps = json.loads(value)
                    if isinstance(timestamps, list):
                        timestamps_set = set(timestamps)
                    else:
                        timestamps_set = set([timestamps]) if timestamps else set()
                    
                    if len(timestamps_set) > 0:
                        max_timestamp = max(timestamps_set)
                        candidates.append((key, timestamps_set, len(timestamps_set), max_timestamp))
            
            if cursor == 0:
                # Completed full scan
                break
        
        if not candidates:
            return None
        
        # Sort by: 1) number of timestamps (ascending), 2) max timestamp (ascending)
        candidates.sort(key=lambda x: (x[2], x[3]))
        
        # Return the best candidate
        best_key, best_timestamps, _, _ = candidates[0]
        return (best_key, best_timestamps)
        
    except Exception as e:
        print(f"Error getting PR candidate: {e}")
        return None


def add_timestamp_to_pr(pr_connection: redis.Redis, key: str, timestamp: str, icf_metadata: dict = {}) -> bool:
    """
    Add a timestamp to a PR key's set of timestamps.
    
    Args:
        pr_connection: Redis connection to PR database
        key: The bitmap key
        timestamp: ISO format timestamp string
        
    Returns:
        bool: True if successful
    """
    try:
        value = pr_connection.get(key)
        if value:
            timestamps = json.loads(value)
            if isinstance(timestamps, list):
                timestamps_set = set(timestamps)
            else:
                timestamps_set = set([timestamps]) if timestamps else set()
        else:
            timestamps_set = set()
        
        timestamps_set.add(timestamp)
        icf_metadata['timestamp'] = (list(timestamps_set))
        pr_connection.set(key, json.dumps(icf_metadata))
        return True
        
    except Exception as e:
        print(f"Error adding timestamp to PR: {e}")
        return False


def remove_from_pr(pr_connection: redis.Redis, key: str) -> bool:
    """
    Remove a key from PR database.
    
    Args:
        pr_connection: Redis connection to PR database
        key: The bitmap key to remove
        
    Returns:
        bool: True if successful
    """
    try:
        result = pr_connection.delete(key)
        return result > 0
    except Exception as e:
        print(f"Error removing from PR: {e}")
        return False


def insert_to_pr(pr_connection: redis.Redis, key: str, timestamp: Optional[str] = None, icf_metadata: dict = None) -> bool:
    """
    Insert a new key into PR database with initial timestamp.
    
    Args:
        pr_connection: Redis connection to PR database
        key: The bitmap key
        timestamp: ISO format timestamp (default: current time)
        
    Returns:
        bool: True if successful
    """
    try:
        if timestamp is None:
            timestamp = datetime.datetime.now().isoformat()
            icf_metadata['timestamp'] = timestamp 
        
        pr_connection.set(key, json.dumps(icf_metadata))
        return True
        
    except Exception as e:
        print(f"Error inserting to PR: {e}")
        return False


def count_pr_keys(pr_connection: redis.Redis, scan_count: int = 100) -> int:
    """
    Count the number of keys in PR database using SCAN.
    
    Args:
        pr_connection: Redis connection to PR database
        scan_count: Number of keys to fetch per SCAN iteration
        
    Returns:
        int: Number of keys in PR database
    """
    try:
        count = 0
        cursor = 0
        
        while True:
            cursor, keys = pr_connection.scan(cursor, match='*', count=scan_count)
            count += len(keys)
            
            if cursor == 0:
                break
        
        return count
        
    except Exception as e:
        print(f"Error counting PR keys: {e}")
        return 0