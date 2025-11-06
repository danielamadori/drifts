#!/usr/bin/env python3
"""
Live monitoring of dataset tests
Run this in a separate terminal while tests are running
"""
import time
import os
import json
from pathlib import Path
from datetime import datetime

def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def read_log_tail(log_file, lines=30):
    """Read last N lines from log file"""
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.readlines()
            return content[-lines:] if len(content) > lines else content
    except FileNotFoundError:
        return ["[Log file not found yet]"]
    except Exception as e:
        return [f"[Error reading log: {e}]"]

def read_json_results(json_file):
    """Read JSON results file"""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return {"error": "Invalid JSON"}
    except Exception as e:
        return {"error": str(e)}

def format_time_elapsed(start_time):
    """Format elapsed time"""
    if not start_time:
        return "Unknown"
    
    elapsed = datetime.now() - start_time
    hours = elapsed.seconds // 3600
    minutes = (elapsed.seconds % 3600) // 60
    seconds = elapsed.seconds % 60
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"

def monitor():
    """Main monitoring loop"""
    log_file = Path("test_datasets_workers.log")
    json_file = Path("test_datasets_workers.json")
    
    print("="*80)
    print(" DATASET TEST MONITOR - Live View")
    print("="*80)
    print()
    print("Starting monitor... Press Ctrl+C to exit")
    print()
    
    start_time = None
    
    try:
        while True:
            clear_screen()
            
            # Header
            print("="*80)
            print(f" DATASET TEST MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*80)
            print()
            
            # Get start time from log if available
            if not start_time and log_file.exists():
                try:
                    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                        first_line = f.readline()
                        if first_line.startswith('['):
                            # Try to parse timestamp from first line
                            start_time = datetime.now()  # Approximation
                except:
                    pass
            
            # Show elapsed time
            if start_time:
                print(f"Elapsed Time: {format_time_elapsed(start_time)}")
                print()
            
            # Read JSON results
            results = read_json_results(json_file)
            
            if results:
                print("-"*80)
                print(" PROGRESS SUMMARY")
                print("-"*80)
                
                if "error" in results:
                    print(f"Error reading results: {results['error']}")
                else:
                    success_count = len(results.get('success', []))
                    failed_count = len(results.get('failed', {}))
                    total = results.get('config', {}).get('total_datasets', '?')
                    
                    print(f"Total Datasets: {total}")
                    print(f"Completed Successfully: {success_count}")
                    print(f"Failed: {failed_count}")
                    print(f"Progress: {success_count + failed_count}/{total}")
                    
                    if failed_count > 0:
                        print()
                        print("Failed Datasets:")
                        for dataset, info in results.get('failed', {}).items():
                            print(f"  - {dataset}: {info.get('status', 'unknown')}")
                
                print()
            else:
                print("-"*80)
                print(" PROGRESS SUMMARY")
                print("-"*80)
                print("Waiting for test to start...")
                print()
            
            # Show log tail
            print("-"*80)
            print(" LAST 25 LINES OF LOG")
            print("-"*80)
            
            log_lines = read_log_tail(log_file, 25)
            for line in log_lines:
                print(line.rstrip())
            
            print()
            print("-"*80)
            print("Refreshing in 5 seconds... (Press Ctrl+C to exit)")
            print("-"*80)
            
            # Wait before refresh
            time.sleep(5)
            
    except KeyboardInterrupt:
        print()
        print()
        print("="*80)
        print(" Monitor stopped by user")
        print("="*80)
        
        # Show final summary if available
        results = read_json_results(json_file)
        if results and "error" not in results:
            print()
            print("FINAL SUMMARY:")
            print(f"  Success: {len(results.get('success', []))}")
            print(f"  Failed: {len(results.get('failed', {}))}")
            print()
            
            if results.get('failed'):
                print("Failed datasets:")
                for dataset, info in results['failed'].items():
                    error = info.get('error', 'Unknown error')[:100]
                    print(f"  - {dataset}: {error}")
        
        print()

if __name__ == "__main__":
    monitor()

