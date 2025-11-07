"""Script di test per verificare il sistema"""
from pathlib import Path

print("=" * 60)
print("VERIFICA SISTEMA DI TEST")
print("=" * 60)

# 1. Verifica directory results
results_dir = Path('results')
print(f"\n1. Directory results/ esiste: {results_dir.exists()}")

if results_dir.exists():
    zip_files = list(results_dir.glob('*.zip'))
    print(f"   File .zip trovati: {len(zip_files)}")

    # Estrai dataset
    datasets = set()
    for zip_file in zip_files:
        dataset_name = zip_file.stem.split('_')[0]
        datasets.add(dataset_name)

    datasets = sorted(list(datasets))
    print(f"   Dataset unici: {len(datasets)}")
    print("\n   Lista dataset:")
    for ds in datasets:
        print(f"     - {ds}")

# 2. Verifica script esistenti
print("\n2. Script di test:")
scripts = [
    'test_datasets_with_workers.py',
    'test_results_datasets.py',
    'enhanced_launch_workers.py',
    'init_aeon_univariate.py'
]

for script in scripts:
    exists = Path(script).exists()
    status = "✓" if exists else "✗"
    print(f"   {status} {script}")

# 3. Verifica batch files
print("\n3. Script batch:")
batches = [
    'test_results_datasets.bat',
    'tests.bat',
    'docker_tests.bat'
]

for batch in batches:
    exists = Path(batch).exists()
    status = "✓" if exists else "✗"
    print(f"   {status} {batch}")

print("\n" + "=" * 60)
print("VERIFICA COMPLETATA")
print("=" * 60)

