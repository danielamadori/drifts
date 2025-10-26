"""scripts/analyze_results.py

Script autonomo che unisce le funzioni di conteggio e diagnostica viste nel notebook
in un unico file eseguibile.

Usi principali:
  - Conta le voci `redis_backup_db*.json` dentro `results/` (zip o cartelle)
  - Stampa un rapporto sui file zip/cartelle senza manifest o corrotti
  - Esporta un CSV con i conteggi per dataset

Esempi:
  python3 scripts/analyze_results.py --results results --out analyzed_counts_results_only.csv --verbose
  python3 scripts/analyze_results.py --results results --list-missing

Nota: questo script non cancella file automaticamente; qualsiasi rimozione va
fatta manualmente dopo che confermi quali elementi eliminare.
"""
#!/usr/bin/env python3
from pathlib import Path
import json
import re
import zipfile
import argparse
import sys
from typing import Dict, List, Optional

try:
    import pandas as pd
except Exception:
    print("pandas non trovato: installalo con `pip install pandas`", file=sys.stderr)
    raise

DB_TO_CAT = {1: "CAN", 2: "R", 3: "NR", 4: "CAR", 5: "AR", 6: "GP", 7: "BP", 8: "PR", 9: "AP"}
CAT_LIST = [DB_TO_CAT[i] for i in sorted(DB_TO_CAT)]
RE_DB = re.compile(r"redis_backup_db(\d+)\.json$")


def _detect_db_index(filename: str) -> Optional[int]:
    m = RE_DB.search(filename)
    return int(m.group(1)) if m else None


def _count_in_dir(ds_dir: Path, verbose: bool = False) -> Optional[Dict[str, int]]:
    agg = {cat: 0 for cat in DB_TO_CAT.values()}
    any_found = False
    for p in ds_dir.rglob("redis_backup_db*.json"):
        dbi = _detect_db_index(p.name)
        if dbi is None:
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            if verbose:
                print(f"⚠️ Errore leggendo JSON: {p}")
            continue
        if isinstance(data, dict) and isinstance(data.get("entries"), list):
            cat = DB_TO_CAT.get(dbi)
            if cat:
                agg[cat] += len(data["entries"])
                any_found = True
    if not any_found and verbose:
        print(f"Manifest not found in directory: {ds_dir}")
    return agg if any_found else None


def _count_in_zip(zip_path: Path, verbose: bool = False) -> Optional[Dict[str, int]]:
    agg = {cat: 0 for cat in DB_TO_CAT.values()}
    any_found = False
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            for name in z.namelist():
                if RE_DB.search(name):
                    dbi = _detect_db_index(name)
                    if dbi is None:
                        continue
                    try:
                        raw = z.read(name)
                        try:
                            text = raw.decode('utf-8')
                        except Exception:
                            text = raw.decode('latin-1', errors='ignore')
                        data = json.loads(text)
                    except Exception:
                        if verbose:
                            print(f"⚠️ Errore leggendo {name} in {zip_path}")
                        continue
                    if isinstance(data, dict) and isinstance(data.get('entries'), list):
                        cat = DB_TO_CAT.get(dbi)
                        if cat:
                            agg[cat] += len(data['entries'])
                            any_found = True
    except zipfile.BadZipFile:
        if verbose:
            print(f"⚠️ Corrupted zip: {zip_path}")
        return None
    except Exception as e:
        if verbose:
            print(f"⚠️ Errore leggendo zip: {zip_path} -> {e}")
        return None
    if not any_found and verbose:
        print(f"Manifest not found in {zip_path}")
    return agg if any_found else None


def compute_counts_from_results(results_dir: Path, verbose: bool = False) -> pd.DataFrame:
    """Restituisce DataFrame con colonne: dataset + CAT_LIST
    Opzionale: verbose=True stamperà avvisi su zip/dir senza manifest.
    """
    rows = []
    if not results_dir.exists():
        if verbose:
            print(f"Results directory not found: {results_dir}")
        return pd.DataFrame(columns=["dataset", *CAT_LIST])
    for entry in sorted(results_dir.iterdir()):
        if entry.is_dir():
            ds_name = entry.name.split("_")[0] if "_" in entry.name else entry.name
            agg = _count_in_dir(entry, verbose=verbose)
            if agg:
                rows.append({"dataset": ds_name, **agg})
        elif entry.is_file() and entry.suffix.lower() == '.zip':
            ds_name = entry.stem.split("_")[0] if "_" in entry.stem else entry.stem
            agg = _count_in_zip(entry, verbose=verbose)
            if agg:
                rows.append({"dataset": ds_name, **agg})
        else:
            continue
    if not rows:
        return pd.DataFrame(columns=["dataset", *CAT_LIST])
    return pd.DataFrame(rows)


def list_missing_manifests(results_dir: Path) -> List[str]:
    """Ritorna lista di path (str) per cui non è stato trovato manifest o sono corrotti."""
    missing = []
    if not results_dir.exists():
        return missing
    for entry in sorted(results_dir.iterdir()):
        if entry.is_dir():
            found = False
            for p in entry.rglob("redis_backup_db*.json"):
                if RE_DB.search(p.name):
                    found = True
                    break
            if not found:
                missing.append(str(entry))
        elif entry.is_file() and entry.suffix.lower() == '.zip':
            # aperto e controllato se contiene almeno un redis_backup_db*.json valido
            ok = False
            try:
                with zipfile.ZipFile(entry, 'r') as z:
                    for name in z.namelist():
                        if RE_DB.search(name):
                            try:
                                raw = z.read(name)
                                try:
                                    text = raw.decode('utf-8')
                                except Exception:
                                    text = raw.decode('latin-1', errors='ignore')
                                data = json.loads(text)
                                if isinstance(data, dict) and isinstance(data.get('entries'), list):
                                    ok = True
                                    break
                            except Exception:
                                continue
            except zipfile.BadZipFile:
                missing.append(str(entry))
                continue
            except Exception:
                missing.append(str(entry))
                continue
            if not ok:
                missing.append(str(entry))
    return missing


# --- NEW: notebook fixer and patcher functions (moved here from separate scripts) ---

def fix_notebook_calls(nb_path: Path = Path('models_analysis_enriched.ipynb')) -> int:
    """Fix notebook code cells replacing underscored helper calls.

    Replacements applied:
      - _compute_counts_from_results( -> compute_counts_from_results(
      - _load_analyzed_df( -> load_analyzed_df(
      - _cast_dataset_str( -> cast_dataset_str(

    Returns 0 on success, 1 on error.
    """
    if not nb_path.exists():
        print('Notebook not found:', nb_path)
        return 1
    try:
        nb = json.loads(nb_path.read_text(encoding='utf-8'))
    except Exception as e:
        print('Failed to read notebook:', e)
        return 1
    changed = False
    for cell in nb.get('cells', []):
        if cell.get('cell_type') != 'code':
            continue
        src = '\n'.join(cell.get('source', []))
        src_new = src.replace('_compute_counts_from_results(', 'compute_counts_from_results(')
        src_new = src_new.replace('_load_analyzed_df(', 'load_analyzed_df(')
        src_new = src_new.replace('_cast_dataset_str(', 'cast_dataset_str(')
        if src_new != src:
            cell['source'] = [line + '\n' for line in src_new.splitlines()]
            changed = True
    if changed:
        nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
        print('Fixed notebook calls in', nb_path)
    else:
        print('No changes needed in', nb_path)
    return 0


def patch_models_analysis_enriched(nb_path: Path = Path('models_analysis_enriched.ipynb')) -> int:
    """Patch the notebook to move support functions to drifts_results.

    Behaviour (matches previous script):
      - replace the cell containing '# === Funzioni di supporto ===' with an import cell
      - optionally find a block that starts with 'Robust handling when there are NO records' and remove it and following cells

    Returns 0 on success, 1 on error.
    """
    if not nb_path.exists():
        print('Notebook not found:', nb_path)
        return 1
    try:
        nb = json.loads(nb_path.read_text(encoding='utf-8'))
    except Exception as e:
        print('Failed to read notebook:', e)
        return 1
    cells = nb.get('cells', [])

    # Prepare new import cell source
    import_cell_src = [
        "from pathlib import Path\n",
        "from drifts_results import compute_counts_from_results, load_analyzed_df, cast_dataset_str, CAT_LIST, DB_TO_CAT\n",
        "# Detect notebook base directory as robustly as possible\n",
        "try:\n",
        "    from IPython import get_ipython\n",
        "    ip = get_ipython()\n",
        "    BASE_DIR = Path(ip.run_line_magic('pwd', '')).resolve()\n",
        "except Exception:\n",
        "    BASE_DIR = Path.cwd().resolve()\n",
        "RESULTS_DIR = BASE_DIR / 'results'\n",
        "FR_CSV = BASE_DIR / 'forest_report.csv'\n",
    ]

    # Find indices
    support_idx = None
    robust_idx = None
    for i, cell in enumerate(cells):
        src = '\n'.join(cell.get('source', []))
        if '# === Funzioni di supporto ===' in src and support_idx is None:
            support_idx = i
        if 'Robust handling when there are NO records' in src and robust_idx is None:
            robust_idx = i

    # If the canonical marker isn't found, insert the import cell near the top
    # (before the first code cell) so the notebook will import the shared helpers.
    if support_idx is None:
        insert_at = 0
        for i, cell in enumerate(cells):
            if cell.get('cell_type') == 'code':
                insert_at = i
                break
        new_cell = {
            'cell_type': 'code',
            'metadata': {},
            'source': import_cell_src,
            'outputs': [],
            'execution_count': None,
        }
        cells.insert(insert_at, new_cell)
        support_idx = insert_at
        print('Support functions marker not found; inserted import cell at index', insert_at)

    # Replace support cell with import cell
    cells[support_idx]['cell_type'] = 'code'
    cells[support_idx]['metadata'] = {}
    cells[support_idx]['source'] = import_cell_src
    cells[support_idx]['outputs'] = []
    cells[support_idx]['execution_count'] = None
    print('Replaced support functions cell at index', support_idx)

    # If robust block found, remove it and all following cells
    if robust_idx is not None:
        print('Found robust block at', robust_idx, 'removing to end')
        del cells[robust_idx:]
    else:
        print('No robust duplicate block found; nothing else removed')

    nb['cells'] = cells
    nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
    print('Notebook patched successfully')
    return 0


# --- end inserted functions ---

def main(argv=None):
    p = argparse.ArgumentParser(description="Analizza la cartella results e conta i manifest redis backup")
    p.add_argument('--results', '-r', default='results', help='Percorso alla cartella results')
    p.add_argument('--out', '-o', default='analyzed_counts_results_only.csv', help='File CSV di output')
    p.add_argument('--verbose', '-v', action='store_true', help='Stampa diagnostica dettagliata')
    p.add_argument('--list-missing', action='store_true', help='Stampa la lista degli zip/cartelle senza manifest')
    p.add_argument('--fix-notebook-calls', action='store_true', help='Correggi chiamate con underscore nel notebook (models_analysis_enriched.ipynb)')
    p.add_argument('--patch-notebook', action='store_true', help='Patch notebook per importare le funzioni da drifts_results')
    p.add_argument('--notebook', default='models_analysis_enriched.ipynb', help='Percorso al notebook da modificare')
    args = p.parse_args(argv)

    results_dir = Path(args.results)
    nb_path = Path(args.notebook)

    # Handle notebook utilities first (they exit after operation)
    if args.fix_notebook_calls:
        return fix_notebook_calls(nb_path)
    if args.patch_notebook:
        return patch_models_analysis_enriched(nb_path)

    if args.list_missing:
        missing = list_missing_manifests(results_dir)
        if missing:
            print("Missing or corrupted (no manifest):")
            for m in missing:
                print(" -", m)
        else:
            print("Nessun file mancante o corrotto rilevato in", results_dir)
        return 0

    df = compute_counts_from_results(results_dir, verbose=args.verbose)
    out_path = Path(args.out)
    df.to_csv(out_path, index=False)
    print(f"Wrote: {out_path.resolve()} (rows: {len(df)})")
    if args.verbose:
        print(df.head().to_string())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
