#!/usr/bin/env python3
"""
Random Path Worker with CAR Processing and AR Sample-Sharing Filter

This worker processes candidates from CAN, PR, and CAR databases using rcheck_cache,
automatically checks if non-reasons are anti-reasons using ar_check_cache, and filters
ICFs that share samples with anti-reasons.

Key features:
- CAR processing at each iteration (validates candidate anti-reasons)
- AR sample-sharing filter (rejects ICFs that overlap with ARs)
- Comprehensive statistics tracking with enhanced verbose logging
- Integration of rcheck and ar_check systems
"""

import redis
import time
import datetime
from redis_helpers.forest import retrieve_forest
from redis_helpers.endpoints import retrieve_monotonic_dict
from redis_helpers.icf import random_key_from_can, bitmap_to_icf, delete_from_can, cache_dominated_icf, cache_dominated_bitmap
from redis_helpers.preferred import get_pr_candidate, add_timestamp_to_pr, remove_from_pr, count_pr_keys
from rcheck_cache import rcheck_cache, saturate
from icf_eu_encoding import icf_to_bitmap_mask, bitmap_mask_to_string
import numpy as np
import random
import argparse


def hamming_distance_comprehension(s1, s2):
    """Using list comprehension for a more Pythonic approach"""
    if len(s1) != len(s2):
        raise ValueError("Strings must have the same length")
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Random Path Worker - rcheck + AR check + CAR processing')
    parser.add_argument('--redis-host', default='localhost', 
                       help='Redis server host (default: localhost)')
    parser.add_argument('--redis-port', type=int, default=6379,
                       help='Redis server port (default: 6379)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Establish Redis connections
    connections = {}
    db_mapping = {
        'DATA': 0,
        'CAN': 1,
        'R': 2,
        'NR': 3,
        'CAR': 4,      # Candidate Anti-Reasons
        'AR': 5,       # Anti-Reasons
        'GP': 6,       # Good Profiles
        'BP': 7,       # Bad Profiles
        'PR': 8,       # Preferred Reasons
        'AP': 9        # Anti-Reason Profiles
    }

    print(f"Connecting to Redis at {args.redis_host}:{args.redis_port}")
    
    for name, db_id in db_mapping.items():
        try:
            conn = redis.Redis(host=args.redis_host, port=args.redis_port, db=db_id, decode_responses=True)
            conn.ping()
            connections[name] = conn
            print(f"Connected to Redis DB {db_id} ({name})")
        except redis.ConnectionError:
            print(f"Failed to connect to Redis DB {db_id} ({name}) at {args.redis_host}:{args.redis_port}")
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

    # Initialize caches for R, NR, GP, BP, AR, AP
    caches = {
        'R': set(),
        'NR': set(),
        'GP': set(),
        'BP': set(),
        'AR': set(),  # For AR sample-sharing checks
        'AP': set()
    }
    print(f"Initialized in-memory caches for 6 databases")

    # Main Worker Loop
    print("\n=== Starting Main Worker Loop ===")

    # Configuration flags
    VERBOSE_ITERATION = True  # Set to True to enable iteration-wise printing
    REPORT_INTERVAL = 100     # Report every N iterations

    iteration = 0

    # Global statistics
    good = 0
    bad = 0

    # Comprehensive info aggregation (rcheck_cache + ar_check_cache + CAR)
    total_info = {
        # rcheck_cache statistics
        'iterations': 0,
        'dominated_by_R': 0,
        'dominated_by_NR': 0, 
        'forest_dominated_in_GP': 0,
        'forest_dominated_in_BP': 0,
        'shares_sample_with_AR': 0,  # NEW: AR sample-sharing in domination checks
        'early_stop_good': 0,
        'early_stop_bad': 0,
        'tie_case': 0,
        'deleted_from_R': 0,
        'deleted_from_NR': 0,
        'deleted_from_GP': 0,
        'deleted_from_BP': 0,
        'deleted_from_CAN': 0,
        'extensions_total': 0,
        'extensions_filtered_R': 0,
        'extensions_filtered_NR': 0,
        'extensions_filtered_GP': 0,
        'extensions_filtered_BP': 0,
        'extensions_filtered_AR_sharing': 0,  # NEW: AR sample-sharing in extension filtering
        'extensions_added': 0,
        'prev_filtered_R': 0,
        'prev_filtered_NR': 0,
        'prev_filtered_GP': 0,
        'prev_filtered_BP': 0,
        
        # AR check statistics
        'ar_checks_from_rcheck': 0,
        'ar_found_from_rcheck': 0,
        'ar_iterations': 0,
        'ar_dominated_by_AR': 0,
        'ar_profile_dominated_by_AP': 0,
        'ar_share_by_reason_in_R': 0,
        'ar_profile_dominated_by_GP': 0,
        'ar_early_stop_true': 0,
        'ar_early_stop_false': 0,
        'ar_tie_case': 0,
        'deleted_from_CAR': 0,
        'ar_extensions_total': 0,
        'ar_extensions_filtered_by_AR': 0,
        'ar_extensions_filtered_by_R': 0,
        'ar_extensions_added': 0,
        'ar_extensions_added_to_CAR': 0,
        
        # CAR processing statistics  # NEW SECTION
        'car_checks': 0,
        'car_processed': 0,
        'car_confirmed_ar': 0,
        'car_not_ar': 0,
        
        # PR statistics
        'pr_checks': 0,
        'pr_selected': 0,
        'pr_removed_all_filtered': 0,
        'pr_timestamp_added': 0
    }

    # Timing variables
    start_time = time.time()
    last_report_time = start_time

    # Helper function to check and select from PR
    def check_and_select_pr():
        """Check PR and select a candidate if available"""
        total_info['pr_checks'] += 1
        
        pr_result = get_pr_candidate(connections['PR'])
        if pr_result is None:
            if VERBOSE_ITERATION:
                print("PR: No candidates in PR database")
            return None
        
        pr_key, pr_timestamps = pr_result
        total_info['pr_selected'] += 1
        
        if VERBOSE_ITERATION:
            print(f"PR: Selected key with {len(pr_timestamps)} accesses")
            print(f"PR: Last access at {max(pr_timestamps)}")
            print(f"PR: Key = {pr_key[:50]}...")
        
        return pr_key
    
    # NEW: Helper function to check and select from CAR
    def check_and_select_car():
        """Check CAR and select a candidate AR if available"""
        total_info['car_checks'] += 1
        
        car_key = random_key_from_can(connections['CAR'])
        if car_key is None:
            if VERBOSE_ITERATION:
                print("CAR: No candidates in CAR database")
            return None
        
        total_info['car_processed'] += 1
        
        if VERBOSE_ITERATION:
            print(f"CAR: Selected candidate AR: {car_key[:50]}...")
        
        return car_key

    try:
        # Check PR at the beginning
        if VERBOSE_ITERATION:
            print("\n=== Initial PR Check ===")
        
        pr_candidate = check_and_select_pr()
        if pr_candidate:
            queue = [pr_candidate]
            if VERBOSE_ITERATION:
                print("Starting with PR candidate")
        else:
            # Get random candidate ICF from CAN
            candidate_bitmap = random_key_from_can(connections['CAN'])
            if candidate_bitmap is None:
                print("No candidate ICF found in CAN - stopping worker")
                return True
            
            if VERBOSE_ITERATION:
                print(f"Starting with CAN candidate: {candidate_bitmap[:50]}...")
            queue = [candidate_bitmap]

        while True:
            # Check PR when queue is empty
            if len(queue) == 0:
                if VERBOSE_ITERATION:
                    print("\n=== Queue Empty - Checking PR ===")
                
                pr_candidate = check_and_select_pr()
                if pr_candidate:
                    queue = [pr_candidate]
                else:
                    # Get random candidate ICF from CAN
                    candidate_bitmap = random_key_from_can(connections['CAN'])
                    if candidate_bitmap is None:
                        print("No candidate ICF found in CAN - stopping worker")
                        break
                    
                    if VERBOSE_ITERATION:
                        print(f"No PR candidates, using CAN candidate: {candidate_bitmap[:50]}...")
                    queue = [candidate_bitmap]

            if iteration % REPORT_INTERVAL == 0:
                # Calculate timing
                current_time = time.time()
                total_elapsed = current_time - start_time
                round_elapsed = current_time - last_report_time
                last_report_time = current_time

                print(f"\n{'='*80}")
                print(f"=== Report at iteration {iteration} ===")
                print(f"Time - Total: {total_elapsed:.2f}s, Last {REPORT_INTERVAL}: {round_elapsed:.2f}s")
                if round_elapsed > 0:
                    print(f"Speed: {REPORT_INTERVAL/round_elapsed:.2f} iterations/second")
                print(f"Queue size: {len(queue)}")
                print(f"Processed: {good + bad} (good={good}, bad={bad})")
                
                print(f"\n--- rcheck_cache statistics ---")
                rcheck_stats = [
                    'iterations', 'dominated_by_R', 'dominated_by_NR', 
                    'forest_dominated_in_GP', 'forest_dominated_in_BP',
                    'shares_sample_with_AR',  # NEW
                    'early_stop_good', 'early_stop_bad', 'tie_case',
                    'deleted_from_R', 'deleted_from_NR', 'deleted_from_GP', 
                    'deleted_from_BP', 'deleted_from_CAN',
                    'extensions_total', 'extensions_filtered_R', 'extensions_filtered_NR',
                    'extensions_filtered_GP', 'extensions_filtered_BP', 
                    'extensions_filtered_AR_sharing',  # NEW
                    'extensions_added'
                ]
                for key in rcheck_stats:
                    if key in total_info:
                        print(f"  {key}: {total_info[key]}")
                
                print(f"\n--- AR check statistics ---")
                ar_stats = [
                    'ar_checks_from_rcheck', 'ar_found_from_rcheck',
                    'ar_iterations', 'ar_dominated_by_AR', 'ar_profile_dominated_by_AP',
                    'ar_share_by_reason_in_R', 'ar_profile_dominated_by_GP',
                    'ar_early_stop_true', 'ar_early_stop_false', 'ar_tie_case',
                    'deleted_from_CAR', 'ar_extensions_total', 
                    'ar_extensions_filtered_by_AR', 'ar_extensions_filtered_by_R',
                    'ar_extensions_added', 'ar_extensions_added_to_CAR'
                ]
                for key in ar_stats:
                    if key in total_info:
                        print(f"  {key}: {total_info[key]}")
                
                print(f"\n--- CAR processing statistics ---")  # NEW SECTION
                car_stats = ['car_checks', 'car_processed', 'car_confirmed_ar', 'car_not_ar']
                for key in car_stats:
                    if key in total_info:
                        print(f"  {key}: {total_info[key]}")
                
                print(f"\n--- PR statistics ---")
                pr_stats = ['pr_checks', 'pr_selected', 'pr_removed_all_filtered', 'pr_timestamp_added']
                for key in pr_stats:
                    if key in total_info:
                        print(f"  {key}: {total_info[key]}")
                
                print(f"\n--- Cache sizes ---")
                for cache_name, cache_set in caches.items():
                    print(f"  {cache_name}: {len(cache_set)} entries")
                
                print(f"\n--- Database sizes ---")
                pr_size = count_pr_keys(connections['PR'])
                print(f"  PR: {pr_size} entries")
                car_size = connections['CAR'].dbsize()
                print(f"  CAR: {car_size} entries")
                ar_size = connections['AR'].dbsize()
                print(f"  AR: {ar_size} entries")
                
                print("=" * 80)

            iteration += 1
            if VERBOSE_ITERATION:
                print(f"\n{'='*80}")
                print(f"--- Iteration {iteration} ---")

            # NEW: FIRST - Check CAR for candidate anti-reasons
            car_start_time = time.time()
            car_candidate = check_and_select_car()
            if car_candidate:
                if VERBOSE_ITERATION:
                    print(f"\n=== Processing CAR Candidate ===")
                
                # Convert bitmap to ICF
                car_icf = bitmap_to_icf(car_candidate, eu_data)
                
                # Import ar_check_cache
                from ar_check_cache import ar_check_cache
                
                # Check if it's an anti-reason
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
                
                # Aggregate CAR check stats
                for key, value in car_info.items():
                    if key in total_info and key not in ["path", "ar_path"]:
                        total_info[key] += value
                
                car_time = time.time() - car_start_time
                
                if ar_result:
                    total_info['car_confirmed_ar'] += 1
                    if VERBOSE_ITERATION:
                        print(f"CAR: Result: ✓ CONFIRMED as anti-reason")
                else:
                    total_info['car_not_ar'] += 1
                    if VERBOSE_ITERATION:
                        print(f"CAR: Result: ✗ NOT an anti-reason")
                
                # Delete from CAR after processing
                delete_from_can(connections['CAR'], car_candidate)
                
                if VERBOSE_ITERATION:
                    print(f"CAR: ICF size: {len(car_icf)} features")
                    print(f"CAR: Time: {car_time:.4f}s")
                    
                    # Show key internal stats from ar_check_cache
                    print(f"CAR: Internal stats:")
                    ar_check_stats = [
                        ('ar_iterations', 'iterations'),
                        ('ar_dominated_by_AR', 'dominated by AR'),
                        ('ar_profile_dominated_by_AP', 'profile dominated by AP'),
                        ('ar_share_by_reason_in_R', 'shares sample with R'),
                        ('ar_profile_dominated_by_GP', 'profile dominated by GP'),
                        ('ar_early_stop_true', 'early stop (true)'),
                        ('ar_early_stop_false', 'early stop (false)'),
                    ]
                    for key, label_text in ar_check_stats:
                        if key in car_info and car_info[key] > 0:
                            print(f"  - {label_text}: {car_info[key]}")
                    
                    # Show extension stats only if confirmed as anti-reason
                    if ar_result:
                        car_extensions_total = car_info.get('ar_extensions_total', 0)
                        car_extensions_added = car_info.get('ar_extensions_added', 0)
                        car_extensions_to_car = car_info.get('ar_extensions_added_to_CAR', 0)
                        car_filtered = car_extensions_total - car_extensions_added
                        
                        if car_extensions_total > 0:
                            print(f"CAR: Extensions: {car_extensions_total} total → {car_extensions_added} added ({car_filtered} filtered)")
                            if car_extensions_to_car > 0:
                                print(f"  - Added back to CAR database: {car_extensions_to_car}")
            
            # SECOND: Pop from queue (CAN/PR candidates)
            can_start_time = time.time()
            current_bitmap = queue.pop()
            
            # Convert bitmap to ICF
            current_icf = bitmap_to_icf(current_bitmap, eu_data)

            if VERBOSE_ITERATION:
                print(f"\n=== Processing CAN Candidate ===")
                print(f"CAN: Bitmap: {current_bitmap[:50]}...")
                print(f"CAN: ICF size: {len(current_icf)} features")

            # Use rcheck_cache - this will also check AR automatically if result is False
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

            # Aggregate info statistics
            for key, value in iteration_info.items():
                if key in total_info:
                    if key != "path" and key != "ar_path":
                        total_info[key] += value 
                else:
                    total_info[key] = total_info.get(key, 0) + value if key not in ["path", "ar_path"] else None

            can_time = time.time() - can_start_time

            if result:
                if VERBOSE_ITERATION:
                    print(f"CAN: Result: ✓ GOOD (Reason)")
                    print(f"CAN: Time: {can_time:.4f}s")
                    
                    # Show key internal stats from rcheck_cache
                    print(f"CAN: Internal stats:")
                    rcheck_stats = [
                        ('iterations', 'iterations'),
                        ('dominated_by_R', 'dominated by R'),
                        ('dominated_by_NR', 'dominated by NR'),
                        ('forest_dominated_in_GP', 'forest dominated in GP'),
                        ('forest_dominated_in_BP', 'forest dominated in BP'),
                        ('shares_sample_with_AR', 'shares sample with AR'),
                        ('early_stop_good', 'early stop (good)'),
                        ('early_stop_bad', 'early stop (bad)'),
                    ]
                    for key, label_text in rcheck_stats:
                        if key in iteration_info and iteration_info[key] > 0:
                            print(f"  - {label_text}: {iteration_info[key]}")
                    
                    # Show AR check stats if triggered
                    if iteration_info.get('ar_checks_from_rcheck', 0) > 0:
                        print(f"CAN: AR checks triggered: {iteration_info['ar_checks_from_rcheck']}")
                        print(f"  - ARs found: {iteration_info.get('ar_found_from_rcheck', 0)}")
                
                good += 1
                
                # Generate and filter feature extensions using cached domination checks
                extension_icfs = [extension for extension in [rf_data.inflate_interval(current_icf, eu_data, feature, direction) 
                                                             for feature in eu_data.keys() for direction in ["low", "high"]] 
                                 if extension is not None]
                
                total_info['extensions_total'] = total_info.get('extensions_total', 0) + len(extension_icfs)
                
                # Filter extensions using caches
                extension_bitmaps = []
                for ext_icf in extension_icfs:
                    # Check if dominated by R (good samples)
                    if cache_dominated_icf(ext_icf, eu_data, connections['R'], caches['R'], 
                                          reverse=False, scan=10, use_db=True):
                        total_info['extensions_filtered_R'] = total_info.get('extensions_filtered_R', 0) + 1
                        continue
                    
                    # Check if dominated by NR (bad samples) 
                    if cache_dominated_icf(ext_icf, eu_data, connections['NR'], caches['NR'], 
                                          reverse=True, scan=10, use_db=True):
                        total_info['extensions_filtered_NR'] = total_info.get('extensions_filtered_NR', 0) + 1
                        continue
                    
                    # Check forest profile against GP
                    ext_forest_bitmap = bitmap_mask_to_string(rf_data.icf_profile_to_bitmap(ext_icf))
                    if cache_dominated_bitmap(ext_forest_bitmap, connections['GP'], caches['GP'], 
                                             reverse=False, scan=10, use_db=True):
                        total_info['extensions_filtered_GP'] = total_info.get('extensions_filtered_GP', 0) + 1
                        continue
                    
                    # Check forest profile against BP
                    if cache_dominated_bitmap(ext_forest_bitmap, connections['BP'], caches['BP'], 
                                             reverse=True, scan=10, use_db=True):
                        total_info['extensions_filtered_BP'] = total_info.get('extensions_filtered_BP', 0) + 1
                        continue
                    
                    # NEW: Check if extension shares a sample with any AR
                    from redis_helpers.icf import cache_shares_sample_with_r
                    if cache_shares_sample_with_r(ext_icf, eu_data, connections['AR'], caches.get('AR', set()),
                                                 scan=10, use_db=True):
                        total_info['extensions_filtered_AR_sharing'] = total_info.get('extensions_filtered_AR_sharing', 0) + 1
                        continue
                    
                    # Extension passed all filters
                    ext_bitmap = bitmap_mask_to_string(icf_to_bitmap_mask(ext_icf, eu_data))
                    extension_bitmaps.append(ext_bitmap)
                
                total_info['extensions_added'] = total_info.get('extensions_added', 0) + len(extension_bitmaps)
                
                if VERBOSE_ITERATION:
                    extensions_total = iteration_info.get('extensions_total', 0)
                    extensions_added = iteration_info.get('extensions_added', 0)
                    extensions_filtered = extensions_total - extensions_added
                    
                    if extensions_total > 0:
                        print(f"CAN: Extensions: {extensions_total} total → {extensions_added} added ({extensions_filtered} filtered)")
                        
                        # Show filtering breakdown
                        filter_breakdown = []
                        if iteration_info.get('extensions_filtered_R', 0) > 0:
                            filter_breakdown.append(f"R: {iteration_info['extensions_filtered_R']}")
                        if iteration_info.get('extensions_filtered_NR', 0) > 0:
                            filter_breakdown.append(f"NR: {iteration_info['extensions_filtered_NR']}")
                        if iteration_info.get('extensions_filtered_GP', 0) > 0:
                            filter_breakdown.append(f"GP: {iteration_info['extensions_filtered_GP']}")
                        if iteration_info.get('extensions_filtered_BP', 0) > 0:
                            filter_breakdown.append(f"BP: {iteration_info['extensions_filtered_BP']}")
                        if iteration_info.get('extensions_filtered_AR_sharing', 0) > 0:
                            filter_breakdown.append(f"AR-sharing: {iteration_info['extensions_filtered_AR_sharing']}")
                        
                        if filter_breakdown:
                            print(f"  - Filtered by: {', '.join(filter_breakdown)}")
                
                # Check if all extensions filtered
                if len(extension_bitmaps) == 0:
                    if VERBOSE_ITERATION:
                        print("CAN: All extensions filtered - checking PR")
                    
                    pr_check = connections['PR'].get(current_bitmap)
                    if pr_check:
                        remove_from_pr(connections['PR'], current_bitmap)
                        total_info['pr_removed_all_filtered'] += 1
                        if VERBOSE_ITERATION:
                            print(f"CAN: Removed from PR (all extensions filtered)")
                    
                    # Check PR for next candidate
                    pr_candidate = check_and_select_pr()
                    if pr_candidate:
                        queue.append(pr_candidate)
                else:
                    # Add extensions to CAN and queue
                    current_time = datetime.datetime.now().isoformat()
                    for ext_bitmap in extension_bitmaps:
                        connections['CAN'].set(ext_bitmap, current_time)

                    if VERBOSE_ITERATION:
                        print(f"CAN: Added {len(extension_bitmaps)} extensions to CAN and queue")

                    # Shuffle and add to queue
                    random.shuffle(extension_bitmaps)
                    queue.extend(extension_bitmaps)
                    
                    # Add timestamp to PR if it was from PR
                    pr_check = connections['PR'].get(current_bitmap)
                    if pr_check:
                        current_timestamp = datetime.datetime.now().isoformat()
                        add_timestamp_to_pr(connections['PR'], current_bitmap, current_timestamp)
                        total_info['pr_timestamp_added'] += 1
                
                # Delete from CAN
                delete_from_can(connections['CAN'], current_bitmap)
                if VERBOSE_ITERATION:
                    print(f"CAN: Deleted from CAN")

            else:
                if VERBOSE_ITERATION:
                    print(f"CAN: Result: ✗ BAD (Non-Reason)")
                    print(f"CAN: Time: {can_time:.4f}s")
                    
                    # Show key internal stats from rcheck_cache
                    print(f"CAN: Internal stats:")
                    rcheck_stats = [
                        ('iterations', 'iterations'),
                        ('dominated_by_R', 'dominated by R'),
                        ('dominated_by_NR', 'dominated by NR'),
                        ('forest_dominated_in_GP', 'forest dominated in GP'),
                        ('forest_dominated_in_BP', 'forest dominated in BP'),
                        ('shares_sample_with_AR', 'shares sample with AR'),
                        ('early_stop_good', 'early stop (good)'),
                        ('early_stop_bad', 'early stop (bad)'),
                    ]
                    for key, label_text in rcheck_stats:
                        if key in iteration_info and iteration_info[key] > 0:
                            print(f"  - {label_text}: {iteration_info[key]}")
                
                bad += 1
                
                # Add timestamp to PR even for bad results
                pr_check = connections['PR'].get(current_bitmap)
                if pr_check:
                    current_timestamp = datetime.datetime.now().isoformat()
                    add_timestamp_to_pr(connections['PR'], current_bitmap, current_timestamp)
                    total_info['pr_timestamp_added'] += 1
                    if VERBOSE_ITERATION:
                        print(f"CAN: Added timestamp to PR (bad result)")
                
                # Check PR after bad result
                if VERBOSE_ITERATION:
                    print("CAN: Checking PR after bad result")
                
                pr_candidate = check_and_select_pr()
                if pr_candidate:
                    queue.append(pr_candidate)
                
                # Delete from CAN
                delete_from_can(connections['CAN'], current_bitmap)
                if VERBOSE_ITERATION:
                    print(f"CAN: Deleted from CAN")

            # Show iteration summary
            if VERBOSE_ITERATION:
                print(f"\n{'─'*80}")
                if car_candidate:
                    iteration_time = car_time + can_time
                    print(f"Iteration {iteration} complete: {iteration_time:.4f}s")
                    print(f"  Breakdown: CAR {car_time:.4f}s + CAN {can_time:.4f}s")
                else:
                    iteration_time = can_time
                    print(f"Iteration {iteration} complete: {iteration_time:.4f}s (CAN only)")
                print(f"{'='*80}")

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
    print(f"Total runtime: {total_elapsed:.2f}s")
    print(f"Total iterations: {iteration}")
    if total_elapsed > 0:
        print(f"Average speed: {iteration/total_elapsed:.2f} iterations/second")
    print(f"Processed: {good + bad} (good={good}, bad={bad})")
    if (good + bad) > 0:
        print(f"Good rate: {100*good/(good+bad):.1f}%")

    print(f"\n--- rcheck_cache Final Statistics ---")
    rcheck_stats = [
        'iterations', 'dominated_by_R', 'dominated_by_NR', 
        'forest_dominated_in_GP', 'forest_dominated_in_BP',
        'shares_sample_with_AR',  # NEW
        'early_stop_good', 'early_stop_bad', 'tie_case',
        'deleted_from_R', 'deleted_from_NR', 'deleted_from_GP', 
        'deleted_from_BP', 'deleted_from_CAN',
        'extensions_total', 'extensions_filtered_R', 'extensions_filtered_NR',
        'extensions_filtered_GP', 'extensions_filtered_BP',
        'extensions_filtered_AR_sharing',  # NEW
        'extensions_added'
    ]
    for key in rcheck_stats:
        if key in total_info and not key.startswith('prev_'):
            print(f"  {key}: {total_info[key]}")
    
    print(f"\n--- AR Check Final Statistics ---")
    ar_stats = [
        'ar_checks_from_rcheck', 'ar_found_from_rcheck',
        'ar_iterations', 'ar_dominated_by_AR', 'ar_profile_dominated_by_AP',
        'ar_share_by_reason_in_R', 'ar_profile_dominated_by_GP',
        'ar_early_stop_true', 'ar_early_stop_false', 'ar_tie_case',
        'deleted_from_CAR', 'ar_extensions_total', 
        'ar_extensions_filtered_by_AR', 'ar_extensions_filtered_by_R',
        'ar_extensions_added', 'ar_extensions_added_to_CAR'
    ]
    for key in ar_stats:
        if key in total_info:
            print(f"  {key}: {total_info[key]}")
    
    if total_info.get('ar_checks_from_rcheck', 0) > 0:
        ar_hit_rate = 100 * total_info.get('ar_found_from_rcheck', 0) / total_info['ar_checks_from_rcheck']
        print(f"\n  AR Discovery Rate: {ar_hit_rate:.1f}% ({total_info.get('ar_found_from_rcheck', 0)}/{total_info['ar_checks_from_rcheck']} non-reasons were ARs)")
    
    print(f"\n--- CAR Processing Final Statistics ---")  # NEW SECTION
    car_stats = ['car_checks', 'car_processed', 'car_confirmed_ar', 'car_not_ar']
    for key in car_stats:
        if key in total_info:
            print(f"  {key}: {total_info[key]}")
    
    if total_info.get('car_processed', 0) > 0:
        car_confirm_rate = 100 * total_info.get('car_confirmed_ar', 0) / total_info['car_processed']
        print(f"\n  CAR Confirmation Rate: {car_confirm_rate:.1f}% ({total_info.get('car_confirmed_ar', 0)}/{total_info['car_processed']} candidates confirmed as ARs)")
    
    print(f"\n--- PR Statistics ---")
    pr_stats = ['pr_checks', 'pr_selected', 'pr_removed_all_filtered', 'pr_timestamp_added']
    for key in pr_stats:
        if key in total_info:
            print(f"  {key}: {total_info[key]}")
    
    print(f"\n--- Final Cache Sizes ---")
    for cache_name, cache_set in caches.items():
        print(f"  {cache_name}: {len(cache_set)} entries")

    # Final database sizes
    print(f"\n--- Final Database Sizes ---")
    final_pr_size = count_pr_keys(connections['PR'])
    print(f"  PR: {final_pr_size} entries")
    final_can_size = connections['CAN'].dbsize()
    print(f"  CAN: {final_can_size} entries")
    final_r_size = connections['R'].dbsize()
    print(f"  R (Reasons): {final_r_size} entries")
    final_nr_size = connections['NR'].dbsize()
    print(f"  NR (Non-Reasons): {final_nr_size} entries")
    final_car_size = connections['CAR'].dbsize()
    print(f"  CAR (Candidate ARs): {final_car_size} entries")
    final_ar_size = connections['AR'].dbsize()
    print(f"  AR (Anti-Reasons): {final_ar_size} entries")
    
    print("=" * 80)

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)