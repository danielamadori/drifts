#!/usr/bin/env python3
"""
Random Path Worker with Raw Info Logging to Redis

This worker logs raw iteration data to Redis LOGS database as JSON:
- Raw info dicts from rcheck_cache and ar_check_cache calls (no aggregation)
- Outcomes observed in the worker
- Cache sizes at each iteration
- Timestamps
- Full bitmaps without truncation for complete analysis

Log key format: WORKERIP:WORKERNAME:ITERATION
Log value: JSON string (parse with json.loads())

Example retrieval:
    import redis, json
    logs_db = redis.Redis(host='localhost', port=6379, db=10, decode_responses=True)
    log_json = logs_db.get("hostname:worker_12345:150")
    log_data = json.loads(log_json)
    print(log_data['car_processing']['candidate_bitmap'])  # Full bitmap string
"""

from cost_function import cost_function
import redis
import time
import datetime
import json
import socket
import os
from redis_helpers.forest import retrieve_forest
from redis_helpers.endpoints import retrieve_monotonic_dict
from redis_helpers.icf import random_key_from_can, bitmap_to_icf, delete_from_can, cache_dominated_icf, cache_dominated_bitmap
from redis_helpers.preferred import get_pr_candidate, add_timestamp_to_pr, remove_from_pr, count_pr_keys
from rcheck_cache import rcheck_cache, saturate
from icf_eu_encoding import icf_to_bitmap_mask, bitmap_mask_to_string
import numpy as np
import random
import argparse


def get_worker_id():
    """Get unique worker identifier based on hostname and PID"""
    hostname = socket.gethostname()
    pid = os.getpid()
    return f"{hostname}:worker_{pid}"


def get_cache_sizes(caches):
    """Get current cache sizes"""
    return {name: len(cache_set) for name, cache_set in caches.items()}


def get_db_sizes(connections):
    """Get current database sizes"""
    return {
        'PR': count_pr_keys(connections['PR']),
        'CAN': connections['CAN'].dbsize(),
        'R': connections['R'].dbsize(),
        'NR': connections['NR'].dbsize(),
        'CAR': connections['CAR'].dbsize(),
        'AR': connections['AR'].dbsize(),
        'GP': connections['GP'].dbsize(),
        'BP': connections['BP'].dbsize(),
        'AP': connections['AP'].dbsize()
    }


def log_iteration_to_redis(logs_db, worker_id, iteration, log_data):
    """
    Log iteration data to Redis LOGS database as JSON
    
    Stores the log entry as a JSON string that can be parsed back to a dict.
    Full bitmaps are stored without truncation for complete analysis.
    
    Args:
        logs_db: Redis connection to LOGS database
        worker_id: Unique worker identifier
        iteration: Iteration number
        log_data: Dictionary containing all log information
    """
    log_key = f"{worker_id}:{iteration}"
    try:
        # Store as JSON string - can be parsed back with json.loads()
        logs_db.set(log_key, json.dumps(log_data))
    except Exception as e:
        print(f"Warning: Failed to log iteration {iteration} to Redis: {e}")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Random Path Worker with Raw Info Redis Logging')
    parser.add_argument('--redis-host', default='localhost', 
                       help='Redis server host (default: localhost)')
    parser.add_argument('--redis-port', type=int, default=6379,
                       help='Redis server port (default: 6379)')
    parser.add_argument('--worker-name', default=None,
                       help='Worker name (default: auto-generated from hostname:pid)')
    parser.add_argument('--verbose-stdout', action='store_true',
                       help='Also print summary to stdout')
    parser.add_argument('--log-cache-sizes', action='store_true',
                       help='Log cache sizes at each iteration (may be expensive)')
    parser.add_argument('--log-db-sizes', action='store_true',
                       help='Log database sizes at each iteration (expensive)')
    
    args = parser.parse_args()
    
    # Get worker identifier
    if args.worker_name:
        worker_id = args.worker_name
    else:
        worker_id = get_worker_id()
    
    print(f"Worker ID: {worker_id}")
    
    # Establish Redis connections
    connections = {}
    db_mapping = {
        'DATA': 0,
        'CAN': 1,
        'R': 2,
        'NR': 3,
        'CAR': 4,
        'AR': 5,
        'GP': 6,
        'BP': 7,
        'PR': 8,
        'AP': 9,
        'LOGS': 10  # NEW: Logs database
    }

    print(f"Connecting to Redis at {args.redis_host}:{args.redis_port}")
    
    for name, db_id in db_mapping.items():
        try:
            conn = redis.Redis(host=args.redis_host, port=args.redis_port, db=db_id, decode_responses=True)
            conn.ping()
            connections[name] = conn
            print(f"Connected to Redis DB {db_id} ({name})")
        except redis.ConnectionError:
            print(f"Failed to connect to Redis DB {db_id} ({name})")
            return False

    print(f"Established {len(connections)} Redis connections")

    # Download RF, EU and Label from DATA
    print("\n=== Worker Initialization ===")

    print("Loading Random Forest from DATA['RF']...")
    rf_data = retrieve_forest(connections['DATA'], 'RF')
    if rf_data is None:
        print("Failed to load Random Forest")
        return False
    print(f"Loaded Random Forest with {len(rf_data.trees)} trees")

    print("Loading Endpoints Universe from DATA['EU']...")
    eu_data = retrieve_monotonic_dict(connections['DATA'], 'EU')
    if eu_data is None:
        print("Failed to load Endpoints Universe")
        return False
    print(f"Loaded EU with {len(eu_data)} features")

    print("Loading target label from DATA['label']...")
    label = connections['DATA'].get('label')
    if label is None:
        print("Failed to load target label")
        return False
    print(f"Target label: {label}")

    # Get the nodes from the forest
    nodes = []
    for tree in rf_data.trees:
        nodes.append(tree.root)

    # Initialize caches
    caches = {
        'R': set(),
        'NR': set(),
        'GP': set(),
        'BP': set(),
        'AR': set(),
        'AP': set()
    }
    print(f"Initialized in-memory caches for 6 databases")

    # Main Worker Loop
    print("\n=== Starting Main Worker Loop ===")
    print(f"Logging to Redis LOGS database with key prefix: {worker_id}")

    iteration = 0
    good = 0
    bad = 0

    start_time = time.time()

    def check_and_select_pr():
        """Check PR and select a candidate if available"""
        pr_result = get_pr_candidate(connections['PR'])
        if pr_result is None:
            return None
        pr_key, pr_timestamps = pr_result
        return pr_key
    
    def check_and_select_car():
        """Check CAR and select a candidate AR if available"""
        car_key = random_key_from_can(connections['CAR'])
        return car_key

    try:
        # Initialize queue
        pr_candidate = check_and_select_pr()
        if pr_candidate:
            queue = [pr_candidate]
        else:
            candidate_bitmap = random_key_from_can(connections['CAN'])
            if candidate_bitmap is None:
                # Check if CAR also empty - stop if both are empty
                car_size = connections['CAR'].dbsize()
                if car_size == 0:
                    print("Both CAN and CAR are empty - stopping worker")
                    return True
                else:
                    print(f"CAN empty but CAR has {car_size} entries - continuing with empty queue")
            queue = [candidate_bitmap] if candidate_bitmap else []
        
        # Initialize CAR queue
        car_queue = []

        while True:
            # Refill queues if empty at the start of each iteration
            
            # Refill CAN queue if empty
            if len(queue) == 0:
                # Try PR first
                pr_candidate = check_and_select_pr()
                if pr_candidate:
                    queue = [pr_candidate]
                else:
                    # Try CAN
                    candidate_bitmap = random_key_from_can(connections['CAN'])
                    if candidate_bitmap is not None:
                        queue = [candidate_bitmap]
            
            # Refill CAR queue if empty
            if len(car_queue) == 0:
                car_candidate = random_key_from_can(connections['CAR'])
                if car_candidate is not None:
                    car_queue.append(car_candidate)
            
            # If both queues are still empty after refill attempts, stop
            if len(queue) == 0 and len(car_queue) == 0:
                print("Both CAN and CAR queues and databases are empty - stopping worker")
                break

            iteration += 1
            
            # Initialize log data for this iteration
            iteration_log = {
                'worker_id': worker_id,
                'iteration': iteration,
                'timestamp_start': datetime.datetime.now().isoformat(),
                'queue_size': len(queue),
                'car_queue_size': len(car_queue),
                'car_processing': None,
                'can_processing': None,
                'outcomes': {},
                'timings': {}
            }
            
            iteration_start_time = time.time()

            # === CAR PROCESSING ===
            car_start_time = time.time()
            car_candidate = None
            if len(car_queue) > 0:
                car_candidate = car_queue.pop()  # Pop from end (stack behavior)
            
            if car_candidate:
                car_icf = bitmap_to_icf(car_candidate, eu_data)
                
                from ar_check_cache import ar_check_cache
                
                # Get raw info from ar_check_cache
                car_info = {}
                ar_result = ar_check_cache(
                    connections=connections,
                    icf=car_icf,
                    label=label,
                    nodes=saturate(car_icf, nodes),
                    eu_data=eu_data,
                    forest=rf_data,
                    caches=caches,
                    info=car_info
                )
                
                car_time = time.time() - car_start_time
                
                # Log CAR processing with raw info dict
                iteration_log['car_processing'] = {
                    'candidate_bitmap': car_candidate,  # Full bitmap, no truncation
                    'icf_size': len(car_icf),
                    'result': 'CONFIRMED_AR' if ar_result else 'NOT_AR',
                    'time_seconds': car_time,
                    'raw_info': car_info  # Raw info dict from ar_check_cache
                }
                
                # Delete from CAR after processing
                delete_from_can(connections['CAR'], car_candidate)
                
                if ar_result:
                    iteration_log['outcomes']['car_confirmed_ar'] = True
                    
                    # Generate extensions for confirmed anti-reason
                    car_extension_icfs = [extension for extension in [rf_data.inflate_interval(car_icf, eu_data, feature, direction) 
                                                                       for feature in eu_data.keys() for direction in ["low", "high"]] 
                                          if extension is not None]
                    
                    # Filter extensions for anti-reasons
                    car_extension_bitmaps = []
                    car_extensions_filtered_ar = 0
                    car_extensions_filtered_r = 0
                    car_extensions_filtered_ap = 0
                    
                    for ext_icf in car_extension_icfs:
                        # Check if dominated by AR (anti-reasons)
                        if cache_dominated_icf(ext_icf, eu_data, connections['AR'], caches['AR'], 
                                              reverse=False, scan=10, use_db=True):
                            car_extensions_filtered_ar += 1
                            continue
                        
                        # Check if shares sample with R (reasons)
                        from redis_helpers.icf import cache_shares_sample_with_r
                        if cache_shares_sample_with_r(ext_icf, eu_data, connections['R'], caches.get('R', set()),
                                                     scan=10, use_db=True):
                            car_extensions_filtered_r += 1
                            continue
                        
                        # Check forest profile against AP (anti-reason profiles)
                        ext_forest_bitmap = bitmap_mask_to_string(rf_data.icf_profile_to_bitmap(ext_icf))
                        if cache_dominated_bitmap(ext_forest_bitmap, connections['AP'], caches['AP'], 
                                                 reverse=False, scan=10, use_db=True):
                            car_extensions_filtered_ap += 1
                            continue
                        
                        # Extension passed all filters
                        ext_bitmap = bitmap_mask_to_string(icf_to_bitmap_mask(ext_icf, eu_data))
                        car_extension_bitmaps.append(ext_bitmap)
                    
                    # Log extension stats
                    iteration_log['car_processing']['extensions'] = {
                        'total': len(car_extension_icfs),
                        'added': len(car_extension_bitmaps),
                        'filtered': len(car_extension_icfs) - len(car_extension_bitmaps),
                        'filtered_by_ar': car_extensions_filtered_ar,
                        'filtered_by_r_sharing': car_extensions_filtered_r,
                        'filtered_by_ap': car_extensions_filtered_ap
                    }
                    
                    # Add extensions to CAR queue (stack)
                    random.shuffle(car_extension_bitmaps)
                    car_queue.extend(car_extension_bitmaps)
                    
                else:
                    iteration_log['outcomes']['car_not_ar'] = True

            # === CAN PROCESSING ===
            if len(queue) == 0:
                # Queue is empty, skip CAN processing this iteration
                # Still log the iteration with only CAR processing
                iteration_log['can_processing'] = {
                    'skipped': True,
                    'reason': 'queue_empty'
                }
                
                # Finalize log entry
                iteration_time = time.time() - iteration_start_time
                iteration_log['timings'] = {
                    'total_seconds': iteration_time,
                    'car_seconds': car_time if car_candidate else 0.0,
                    'can_seconds': 0.0
                }
                iteration_log['timestamp_end'] = datetime.datetime.now().isoformat()
                
                # Add cache sizes if requested
                if args.log_cache_sizes:
                    iteration_log['cache_sizes'] = get_cache_sizes(caches)
                
                # Add database sizes if requested
                if args.log_db_sizes:
                    iteration_log['db_sizes'] = get_db_sizes(connections)
                
                # Log to Redis
                log_iteration_to_redis(connections['LOGS'], worker_id, iteration, iteration_log)
                
                # Continue to next iteration
                continue
            
            can_start_time = time.time()
            current_bitmap = queue.pop()
            current_icf = bitmap_to_icf(current_bitmap, eu_data)

            # Get raw info from rcheck_cache
            iteration_info = {}
            result = rcheck_cache(
                connections=connections,
                icf=current_icf, 
                label=label, 
                nodes=saturate(current_icf, nodes),
                eu_data=eu_data,
                forest=rf_data,
                caches=caches,
                info=iteration_info
            )

            can_time = time.time() - can_start_time
            
            # Log CAN processing with raw info dict
            iteration_log['can_processing'] = {
                'candidate_bitmap': current_bitmap,  # Full bitmap, no truncation
                'icf_size': len(current_icf),
                'result': 'GOOD' if result else 'BAD',
                'time_seconds': can_time,
                'raw_info': iteration_info  # Raw info dict from rcheck_cache
            }

            if result:
                good += 1
                iteration_log['outcomes']['good'] = True
                
                # Generate and filter extensions
                extension_icfs = [extension for extension in [rf_data.inflate_interval(current_icf, eu_data, feature, direction) 
                                                             for feature in eu_data.keys() for direction in ["low", "high"]] 
                                 if extension is not None]
                
                # Filter extensions
                extension_bitmaps = []
                for ext_icf in extension_icfs:
                    if cache_dominated_icf(ext_icf, eu_data, connections['R'], caches['R'], 
                                          reverse=False, scan=10, use_db=True):
                        continue
                    if cache_dominated_icf(ext_icf, eu_data, connections['NR'], caches['NR'], 
                                          reverse=True, scan=10, use_db=True):
                        continue
                    ext_forest_bitmap = bitmap_mask_to_string(rf_data.icf_profile_to_bitmap(ext_icf))
                    if cache_dominated_bitmap(ext_forest_bitmap, connections['GP'], caches['GP'], 
                                             reverse=False, scan=10, use_db=True):
                        continue
                    if cache_dominated_bitmap(ext_forest_bitmap, connections['BP'], caches['BP'], 
                                             reverse=True, scan=10, use_db=True):
                        continue
                    from redis_helpers.icf import cache_shares_sample_with_r
                    if cache_shares_sample_with_r(ext_icf, eu_data, connections['AR'], caches.get('AR', set()),
                                                 scan=10, use_db=True):
                        continue
                    ext_bitmap = bitmap_mask_to_string(icf_to_bitmap_mask(ext_icf, eu_data))
                    extension_bitmaps.append(ext_bitmap)
                
                iteration_log['can_processing']['extensions'] = {
                    'total': len(extension_icfs),
                    'added': len(extension_bitmaps),
                    'filtered': len(extension_icfs) - len(extension_bitmaps)
                }
                
                # Handle extensions
                if len(extension_bitmaps) == 0:
                    pr_check = connections['PR'].get(current_bitmap)
                    if pr_check:
                        remove_from_pr(connections['PR'], current_bitmap)
                        iteration_log['outcomes']['pr_removed_all_filtered'] = True
                    pr_candidate = check_and_select_pr()
                    if pr_candidate:
                        queue.append(pr_candidate)
                else:
                    current_time = datetime.datetime.now().isoformat()
                    for ext_bitmap in extension_bitmaps:
                        # Calculate ICF for extension
                        icf = bitmap_to_icf(ext_bitmap, eu_data)
                        sample_data = json.loads(connections['DATA'].get(icf['sample_key'] + "_meta"))
                        print(">>>>>>>>>>>>>>>>> Sample Data:", sample_data)
                        cost = cost_function(
                            sample=sample_data['sample_dict'],
                            icf=icf, sigmas=sample_data["sigmas"]
                        )
                        # Store ICF bitmap in R with metadata
                        icf_metadata = {
                            # 'sample_key': icf['sample_key'],
                            # 'dataset_name': sample_data['dataset_name'],
                            # 'class_label': sample_data['actual_label'],
                            # 'test_index': sample_data['test_index'],
                            # 'prediction_correct': sample_data['prediction_correct'],
                            'timestamp': current_time,
                            'cost': cost
                        }
                        connections['CAN'].set(ext_bitmap, json.dumps(icf_metadata))
                    random.shuffle(extension_bitmaps)
                    queue.extend(extension_bitmaps)
                    
                    pr_check = connections['PR'].get(current_bitmap)
                    if pr_check:
                        current_timestamp = datetime.datetime.now().isoformat()
                        add_timestamp_to_pr(connections['PR'], current_bitmap, current_timestamp)
                        iteration_log['outcomes']['pr_timestamp_added'] = True
                
                delete_from_can(connections['CAN'], current_bitmap)

            else:
                bad += 1
                iteration_log['outcomes']['bad'] = True
                
                pr_check = connections['PR'].get(current_bitmap)
                if pr_check:
                    current_timestamp = datetime.datetime.now().isoformat()
                    add_timestamp_to_pr(connections['PR'], current_bitmap, current_timestamp)
                    iteration_log['outcomes']['pr_timestamp_added'] = True
                
                pr_candidate = check_and_select_pr()
                if pr_candidate:
                    queue.append(pr_candidate)
                
                delete_from_can(connections['CAN'], current_bitmap)

            # === FINALIZE LOG ENTRY ===
            iteration_time = time.time() - iteration_start_time
            iteration_log['timings'] = {
                'total_seconds': iteration_time,
                'car_seconds': car_time if car_candidate else 0.0,
                'can_seconds': can_time
            }
            iteration_log['timestamp_end'] = datetime.datetime.now().isoformat()
            
            # Add cache sizes if requested
            if args.log_cache_sizes:
                iteration_log['cache_sizes'] = get_cache_sizes(caches)
            
            # Add database sizes if requested
            if args.log_db_sizes:
                iteration_log['db_sizes'] = get_db_sizes(connections)
            
            # Log to Redis
            log_iteration_to_redis(connections['LOGS'], worker_id, iteration, iteration_log)
            
            # Optional stdout summary
            if args.verbose_stdout:
                print(f"Iteration {iteration}: "
                      f"{'CAR ' + iteration_log['car_processing']['result'] + ', ' if car_candidate else ''}"
                      f"CAN {iteration_log['can_processing']['result']}, "
                      f"Time: {iteration_time:.4f}s")
            
            # Periodic progress report
            if iteration % 100 == 0:
                elapsed = time.time() - start_time
                print(f"Iteration {iteration}: {good + bad} processed ({good} good, {bad} bad), "
                      f"Speed: {iteration/elapsed:.2f} iter/s, Queue: {len(queue)} (CAN) / {len(car_queue)} (CAR)")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()

    print("\n=== Worker Loop Completed ===")
    
    # Final summary
    final_time = time.time()
    total_elapsed = final_time - start_time

    print(f"\n{'='*80}")
    print(f"=== Final Statistics ===")
    print(f"Worker ID: {worker_id}")
    print(f"Total runtime: {total_elapsed:.2f}s")
    print(f"Total iterations: {iteration}")
    if total_elapsed > 0:
        print(f"Average speed: {iteration/total_elapsed:.2f} iterations/second")
    print(f"Processed: {good + bad} (good={good}, bad={bad})")
    if (good + bad) > 0:
        print(f"Good rate: {100*good/(good+bad):.1f}%")
    print(f"\nAll iteration logs saved to Redis LOGS database")
    print(f"Log keys: {worker_id}:1 through {worker_id}:{iteration}")
    print("=" * 80)

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)