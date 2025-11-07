"""
Test dei dataset presenti nella cartella results/
- Testa con e senza Docker
- Si ferma al primo errore
"""
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

def get_datasets_from_results():
    """
    Estrae i nomi dei dataset dai file .zip in results/
    Formato file: DatasetName_classLabel_optimize_seed.zip
    """
    results_dir = Path('results')
    if not results_dir.exists():
        print(f"ERROR: Directory {results_dir} non trovata")
        return []

    datasets = set()
    for zip_file in results_dir.glob('*.zip'):
        # Estrai il nome del dataset (prima parte prima di '_')
        dataset_name = zip_file.stem.split('_')[0]
        datasets.add(dataset_name)

    return sorted(list(datasets))

def write_log(log_file, message):
    """Scrivi messaggio nel log"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(line + '\n')
        f.flush()

def run_test_without_docker(datasets, log_file):
    """Esegui test senza Docker"""
    write_log(log_file, "\n" + "="*80)
    write_log(log_file, "TESTING WITHOUT DOCKER")
    write_log(log_file, "="*80)

    # Costruisci il comando
    cmd = [
        sys.executable,
        'test_datasets_with_workers.py',
        '--worker-duration', '20',
        '--datasets'
    ] + datasets

    write_log(log_file, f"Command: {' '.join(cmd)}")
    write_log(log_file, f"Testing {len(datasets)} datasets")
    write_log(log_file, "")

    # Esegui
    try:
        result = subprocess.run(cmd, check=True)
        write_log(log_file, "[OK] Test senza Docker completato con successo")
        return True
    except subprocess.CalledProcessError as e:
        write_log(log_file, f"[ERROR] Test senza Docker fallito con exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        write_log(log_file, "[INTERRUPTED] Test interrotto dall'utente")
        raise

def run_test_with_docker(datasets, log_file):
    """Esegui test con Docker"""
    write_log(log_file, "\n" + "="*80)
    write_log(log_file, "TESTING WITH DOCKER")
    write_log(log_file, "="*80)

    # Costruisci il comando per Docker
    datasets_str = ' '.join(datasets)
    docker_cmd = [
        'docker', 'exec', '-i', 'drifts-container',
        'python', 'test_datasets_with_workers.py',
        '--worker-duration', '20',
        '--datasets'
    ] + datasets

    write_log(log_file, f"Command: {' '.join(docker_cmd)}")
    write_log(log_file, f"Testing {len(datasets)} datasets in Docker")
    write_log(log_file, "")

    # Verifica che il container sia attivo
    try:
        check_result = subprocess.run(
            ['docker', 'ps', '--filter', 'name=drifts-container', '--format', '{{.Names}}'],
            capture_output=True,
            text=True
        )
        if 'drifts-container' not in check_result.stdout:
            write_log(log_file, "[WARNING] Container drifts-container non in esecuzione")
            write_log(log_file, "[INFO] Tentativo di avvio del container...")

            start_result = subprocess.run(
                ['docker', 'start', 'drifts-container'],
                capture_output=True,
                text=True
            )

            if start_result.returncode != 0:
                write_log(log_file, "[ERROR] Impossibile avviare il container")
                write_log(log_file, "Suggerimento: esegui 'docker_tests.bat rebuild' per ricreare il container")
                return False

            write_log(log_file, "[OK] Container avviato")
    except Exception as e:
        write_log(log_file, f"[ERROR] Errore nel verificare il container: {e}")
        return False

    # Esegui il test
    try:
        result = subprocess.run(docker_cmd, check=True)
        write_log(log_file, "[OK] Test con Docker completato con successo")
        return True
    except subprocess.CalledProcessError as e:
        write_log(log_file, f"[ERROR] Test con Docker fallito con exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        write_log(log_file, "[INTERRUPTED] Test interrotto dall'utente")
        raise

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Test dei dataset presenti in results/ con e senza Docker'
    )
    parser.add_argument('--skip-local', action='store_true',
                       help='Salta i test locali (senza Docker)')
    parser.add_argument('--skip-docker', action='store_true',
                       help='Salta i test con Docker')
    parser.add_argument('--docker-only', action='store_true',
                       help='Esegui solo test con Docker (equivalente a --skip-local)')
    parser.add_argument('--local-only', action='store_true',
                       help='Esegui solo test locali (equivalente a --skip-docker)')

    args = parser.parse_args()

    # Gestisci alias
    if args.docker_only:
        args.skip_local = True
    if args.local_only:
        args.skip_docker = True

    # Setup log
    log_file = Path('test_results_datasets.log')
    if log_file.exists():
        log_file.unlink()

    write_log(log_file, "="*80)
    write_log(log_file, "TEST DATASET DA RESULTS/")
    write_log(log_file, "="*80)
    write_log(log_file, "")

    # Ottieni dataset
    write_log(log_file, "Scansione directory results/...")
    datasets = get_datasets_from_results()

    if not datasets:
        write_log(log_file, "[ERROR] Nessun dataset trovato in results/")
        return 1

    write_log(log_file, f"Trovati {len(datasets)} dataset:")
    for ds in datasets:
        write_log(log_file, f"  - {ds}")
    write_log(log_file, "")

    # Risultati
    results = {
        'datasets': datasets,
        'local': {'status': 'skipped', 'success': False},
        'docker': {'status': 'skipped', 'success': False}
    }

    # Test senza Docker
    if not args.skip_local:
        success = run_test_without_docker(datasets, log_file)
        results['local'] = {'status': 'completed', 'success': success}

        if not success:
            write_log(log_file, "")
            write_log(log_file, "[STOP] Test fallito senza Docker, interruzione")

            # Salva risultati
            with open('test_results_datasets.json', 'w') as f:
                json.dump(results, f, indent=2)

            return 1

    # Test con Docker
    if not args.skip_docker:
        success = run_test_with_docker(datasets, log_file)
        results['docker'] = {'status': 'completed', 'success': success}

        if not success:
            write_log(log_file, "")
            write_log(log_file, "[STOP] Test fallito con Docker, interruzione")

            # Salva risultati
            with open('test_results_datasets.json', 'w') as f:
                json.dump(results, f, indent=2)

            return 1

    # Report finale
    write_log(log_file, "")
    write_log(log_file, "="*80)
    write_log(log_file, "REPORT FINALE")
    write_log(log_file, "="*80)

    if not args.skip_local:
        status = "✓ SUCCESSO" if results['local']['success'] else "✗ FALLITO"
        write_log(log_file, f"Test locali (senza Docker): {status}")

    if not args.skip_docker:
        status = "✓ SUCCESSO" if results['docker']['success'] else "✗ FALLITO"
        write_log(log_file, f"Test con Docker: {status}")

    write_log(log_file, "="*80)
    write_log(log_file, "")

    # Salva risultati
    with open('test_results_datasets.json', 'w') as f:
        json.dump(results, f, indent=2)

    write_log(log_file, f"Risultati salvati in test_results_datasets.json")

    # Exit code
    all_success = (
        (args.skip_local or results['local']['success']) and
        (args.skip_docker or results['docker']['success'])
    )

    return 0 if all_success else 1

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Test interrotto dall'utente")
        sys.exit(130)

