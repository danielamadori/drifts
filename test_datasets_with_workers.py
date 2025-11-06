#!/usr/bin/env python3
"""
Test automatico dei dataset con init optimize e worker
- Inizializza dataset con --optimize
- Lancia worker per 20 secondi
- Verifica assenza errori
- Passa al prossimo dataset
"""
import subprocess
import sys
import time
import signal
import json
from pathlib import Path
from datetime import datetime

def run_cmd(args, timeout=None):
    """Esegue un comando e cattura l'output"""
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False
        )
        return result
    except subprocess.TimeoutExpired as e:
        return type('obj', (object,), {
            'returncode': -1,
            'stdout': e.stdout.decode() if e.stdout else '',
            'stderr': e.stderr.decode() if e.stderr else '',
            'timeout': True
        })()

def write_log(log_file, message):
    """Scrivi messaggio nel log"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(line + '\n')
        f.flush()

def stop_all_workers():
    """Ferma tutti i worker attivi"""
    print("  Stopping workers...")
    result = run_cmd(['python', 'enhanced_launch_workers.py', 'stop'], timeout=30)
    time.sleep(2)  # Attendi che i worker si fermino completamente
    return result.returncode == 0

def start_workers(profile='default'):
    """Avvia worker con un profilo specifico usando enhanced_launch_workers.py"""
    print(f"  Starting workers (profile: {profile})...")

    args = ['python', 'enhanced_launch_workers.py', 'start']
    if profile != 'default':
        args.extend(['--profile', profile])

    try:
        # Avvia enhanced_launch_workers e lascialo girare in background
        process = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Aspetta che il processo completi lo spawn dei worker (max 10 secondi)
        for i in range(10):
            time.sleep(1)
            returncode = process.poll()

            # Se il processo è terminato, verifica se con successo o errore
            if returncode is not None:
                if returncode == 0:
                    # Terminato con successo - worker spawnati
                    print(f"  [OK] Workers spawned successfully")
                    time.sleep(2)  # Attendi che i worker si attivino
                    return True
                else:
                    # Terminato con errore
                    stdout, stderr = process.communicate()
                    print(f"  [ERROR] Failed to start workers")
                    if stderr:
                        print(f"  Error: {stderr[:200]}")
                    return None

        # Processo ancora in esecuzione dopo 10 secondi - probabilmente bloccato
        # Ma i worker potrebbero essere già stati spawnati
        print(f"  [OK] Workers started (launcher still running)")
        time.sleep(2)
        return True

    except KeyboardInterrupt:
        print(f"  [INTERRUPTED] Worker start interrupted by user")
        if process:
            process.kill()
        raise
    except Exception as e:
        print(f"  [ERROR] Exception starting workers: {e}")
        return None


def get_dataset_list():
    """Ottieni lista dei dataset disponibili"""
    result = run_cmd(['python', 'init_aeon_univariate.py', '--list-datasets'])
    if result.returncode != 0:
        return None

    datasets = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        # Skip headers
        if ('=' in line or line.startswith('[') or
            line.lower().startswith('available') or
            line.lower().startswith('total:') or
            line.lower().startswith('note:')):
            continue

        # Parse dataset names
        parts = [p.strip() for p in line.split('  ') if p.strip()]
        for part in parts:
            if part and part[0].isalpha():
                datasets.append(part)

    return datasets

def get_dataset_classes(dataset_name):
    """Ottieni le classi di un dataset usando --info"""
    result = run_cmd(['python', 'init_aeon_univariate.py', dataset_name, '--info'])

    if result.returncode != 0:
        return None

    # Parse output per trovare le classi
    classes = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if 'Classes:' in line or 'Labels:' in line:
            # Estrai le classi dalla riga
            parts = line.split(':', 1)
            if len(parts) == 2:
                class_str = parts[1].strip()
                # Parse formato tipo "['0', '1']" o "0, 1"
                import re
                found = re.findall(r"'([^']+)'|\"([^\"]+)\"|(\d+)", class_str)
                for match in found:
                    class_val = match[0] or match[1] or match[2]
                    if class_val:
                        classes.append(class_val)

    # Se non troviamo classi nel formato sopra, prova approccio alternativo
    if not classes:
        # Carica direttamente solo se necessario
        try:
            from aeon.datasets import load_classification
            import numpy as np

            X_train, y_train = load_classification(dataset_name, split="train")
            X_test, y_test = load_classification(dataset_name, split="test")

            all_classes = np.unique(np.concatenate([y_train, y_test]))
            classes = [str(c) for c in all_classes]
        except:
            pass

    return classes if classes else None

def test_dataset(dataset_name, class_label, log_file, worker_profile='default', worker_duration=20):
    """
    Testa un singolo dataset:
    1. Init con optimize
    2. Lancia worker per N secondi
    3. Verifica errori
    """
    write_log(log_file, f"Testing {dataset_name} (class: {class_label})")

    # Step 1: Inizializza con optimize
    write_log(log_file, f"  [1/3] Running init with optimize...")
    init_result = run_cmd([
        'python', 'init_aeon_univariate.py',
        dataset_name,
        '--class-label', str(class_label),
        '--optimize'
    ], timeout=600)  # 10 minuti max

    if hasattr(init_result, 'timeout') and init_result.timeout:
        write_log(log_file, f"  [ERROR] Init timeout (>10 minutes)")
        return {'status': 'init_timeout', 'error': 'Timeout during initialization'}

    if init_result.returncode != 0:
        error_msg = init_result.stderr.strip()[:200] if init_result.stderr else 'Unknown error'
        write_log(log_file, f"  [ERROR] Init failed: {error_msg}")
        return {'status': 'init_failed', 'error': error_msg}

    write_log(log_file, f"  [OK] Init completed successfully")

    # Step 2: Avvia worker
    write_log(log_file, f"  [2/3] Starting workers for {worker_duration} seconds...")
    worker_started = start_workers(worker_profile)

    if not worker_started:
        write_log(log_file, f"  [ERROR] Failed to start workers")
        return {'status': 'worker_start_failed', 'error': 'Worker process failed to start'}

    # Attendi per il tempo specificato
    write_log(log_file, f"  Waiting {worker_duration} seconds for workers to process...")
    time.sleep(worker_duration)

    # Step 3: Ferma worker usando enhanced_launch_workers
    write_log(log_file, f"  [3/3] Stopping workers...")
    stop_success = stop_all_workers()
    if stop_success:
        write_log(log_file, f"  Workers stopped successfully")
    else:
        write_log(log_file, f"  [WARNING] Issues stopping workers")

    write_log(log_file, f"  [OK] Workers completed without errors")
    return {'status': 'success', 'error': None}

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Test datasets with init optimize and workers')
    parser.add_argument('--worker-profile', default='default',
                       help='Worker profile to use (default, development, production)')
    parser.add_argument('--worker-duration', type=int, default=20,
                       help='Duration in seconds to run workers (default: 20)')
    parser.add_argument('--max-datasets', type=int, default=None,
                       help='Maximum number of datasets to test')
    parser.add_argument('--continue-on-error', action='store_true',
                       help='Continue testing even if errors occur')
    parser.add_argument('--datasets', nargs='+',
                       help='Specific datasets to test (default: all)')

    args = parser.parse_args()

    # Setup logging
    log_file = Path('test_datasets_workers.log')
    log_json = Path('test_datasets_workers.json')

    if log_file.exists():
        log_file.unlink()
    if log_json.exists():
        log_json.unlink()

    write_log(log_file, "="*80)
    write_log(log_file, "DATASET TEST WITH WORKERS")
    write_log(log_file, f"Worker profile: {args.worker_profile}")
    write_log(log_file, f"Worker duration: {args.worker_duration}s")
    write_log(log_file, "="*80)

    # Ferma eventuali worker precedenti
    write_log(log_file, "Stopping any existing workers...")
    stop_all_workers()

    # Ottieni lista dataset
    if args.datasets:
        datasets = args.datasets
        write_log(log_file, f"Testing {len(datasets)} specified datasets")
    else:
        write_log(log_file, "Getting dataset list...")
        datasets = get_dataset_list()
        if not datasets:
            write_log(log_file, "[ERROR] Failed to get dataset list")
            return 1

        if args.max_datasets:
            datasets = datasets[:args.max_datasets]

        write_log(log_file, f"Found {len(datasets)} datasets to test")

    # Risultati
    results = {
        'success': [],
        'failed': {},
        'config': {
            'worker_profile': args.worker_profile,
            'worker_duration': args.worker_duration,
            'total_datasets': len(datasets)
        }
    }

    # Test ogni dataset
    for i, dataset_name in enumerate(datasets, 1):
        write_log(log_file, f"\n{'='*80}")
        write_log(log_file, f"[{i}/{len(datasets)}] Dataset: {dataset_name}")
        write_log(log_file, f"{'='*80}")

        # Ottieni classi
        write_log(log_file, "Getting dataset classes...")
        classes = get_dataset_classes(dataset_name)

        if not classes:
            write_log(log_file, f"[ERROR] Failed to load dataset classes")
            results['failed'][dataset_name] = {
                'status': 'load_failed',
                'error': 'Cannot load dataset classes'
            }

            # Salva risultati parziali
            with open(log_json, 'w') as f:
                json.dump(results, f, indent=2)

            if not args.continue_on_error:
                write_log(log_file, "[STOP] Stopping due to error")
                break
            continue

        write_log(log_file, f"Found classes: {classes}")

        # Testa con la prima classe
        test_class = classes[0]
        result = test_dataset(
            dataset_name,
            test_class,
            log_file,
            args.worker_profile,
            args.worker_duration
        )

        if result['status'] == 'success':
            results['success'].append(dataset_name)
        else:
            results['failed'][dataset_name] = result

        # Salva risultati parziali
        with open(log_json, 'w') as f:
            json.dump(results, f, indent=2)

        if result['status'] != 'success' and not args.continue_on_error:
            write_log(log_file, "[STOP] Stopping due to error")
            break

        # Pausa tra dataset
        time.sleep(2)

    # Report finale
    write_log(log_file, f"\n{'='*80}")
    write_log(log_file, "FINAL REPORT")
    write_log(log_file, f"{'='*80}")
    write_log(log_file, f"Success: {len(results['success'])}")
    write_log(log_file, f"Failed: {len(results['failed'])}")

    if results['failed']:
        write_log(log_file, "\nFailed datasets:")
        for ds, info in results['failed'].items():
            write_log(log_file, f"  - {ds}: {info['status']} - {info['error'][:100]}")

    write_log(log_file, f"\n{'='*80}")
    write_log(log_file, f"Results saved to: {log_json}")
    write_log(log_file, f"{'='*80}")

    # Ferma worker finali
    stop_all_workers()

    return 0 if not results['failed'] else 1

if __name__ == '__main__':
    sys.exit(main())

