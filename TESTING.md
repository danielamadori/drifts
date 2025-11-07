# Testing dei Dataset

## Panoramica

Questo progetto include diversi strumenti per testare i dataset con worker e ottimizzazione.

## File Principali

### Script di Test

- **`test_datasets_with_workers.py`** - Test automatico dei dataset con init optimize e worker
- **`test_results_datasets.py`** - Test specifico per i dataset presenti in `results/`
- **`test_redis_backup.py`** - Test del sistema di backup Redis

### Script Batch (Windows)

- **`tests.bat`** - Test di tutti i dataset disponibili (senza Docker)
- **`docker_tests.bat`** - Test di tutti i dataset dentro Docker
- **`test_results_datasets.bat`** - Test dei dataset in `results/` (con/senza Docker)
- **`monitor_tests.bat`** - Monitoraggio dei test in esecuzione

## Guida Rapida

### Test dei Dataset in Results/

Per testare **solo i dataset già presenti nella cartella `results/`**:

#### Modalità Interattiva (Consigliata)
```bat
test_results_datasets.bat
```
Mostra un prompt prima di iniziare. Testa sia locale che Docker.

#### Esecuzione Immediata
```bat
test_results_datasets.bat now
```
Avvia subito i test (locale + Docker).

#### Solo Test Locali (senza Docker)
```bat
test_results_datasets.bat local
```
O in Python:
```bash
python test_results_datasets.py --local-only
```

#### Solo Test con Docker
```bat
test_results_datasets.bat docker
```
O in Python:
```bash
python test_results_datasets.py --docker-only
```

### Test di Tutti i Dataset

#### Senza Docker
```bat
tests.bat
```
Testa tutti i dataset disponibili in modalità locale.

Opzioni:
- `tests.bat interactive` - Modalità interattiva (default)
- `tests.bat now` - Esecuzione immediata
- `tests.bat check` - Controllo rapido del sistema

#### Con Docker
```bat
docker_tests.bat run
```
Esegue i test dentro il container Docker.

Opzioni:
- `docker_tests.bat run` - Esegue i test (default)
- `docker_tests.bat rebuild` - Ricostruisce il container prima di testare

### Test Manuale con Python

Per testare dataset specifici:
```bash
python test_datasets_with_workers.py --datasets Coffee ECG200 Wine
```

Parametri disponibili:
- `--worker-profile` - Profilo worker da usare (default, development, production)
- `--worker-duration` - Durata in secondi dei worker (default: 20)
- `--max-datasets` - Numero massimo di dataset da testare
- `--continue-on-error` - Continua anche se ci sono errori
- `--datasets` - Lista specifica di dataset da testare

## Come Funzionano i Test

### Flusso di Test Standard

Per ogni dataset:

1. **Inizializzazione** - Esegue `init_aeon_univariate.py` con `--optimize`
2. **Avvio Worker** - Lancia i worker usando `enhanced_launch_workers.py`
3. **Elaborazione** - Attende per la durata specificata (default: 20s)
4. **Stop Worker** - Ferma i worker
5. **Verifica** - Controlla se ci sono stati errori

### Comportamento in Caso di Errore

**Comportamento di default**: Il test **si ferma al primo errore**.

Per continuare anche in caso di errore, usa:
```bash
python test_datasets_with_workers.py --continue-on-error
```

### Output dei Test

I test generano diversi file di output:

#### Test Generici
- `test_datasets_workers.log` - Log dettagliato dell'esecuzione
- `test_datasets_workers.json` - Riepilogo JSON dei risultati

#### Test da Results/
- `test_results_datasets.log` - Log dettagliato
- `test_results_datasets.json` - Riepilogo JSON con risultati locale/Docker

### Formato del JSON di Output

```json
{
  "success": ["Dataset1", "Dataset2"],
  "failed": {
    "Dataset3": {
      "status": "init_failed",
      "error": "Messaggio di errore"
    }
  },
  "config": {
    "worker_profile": "default",
    "worker_duration": 20,
    "total_datasets": 3
  }
}
```

## Test con Docker

### Prerequisiti Docker

1. Docker deve essere installato e in esecuzione
2. L'immagine `drifts:latest` deve essere disponibile
3. Il container `drifts-container` deve essere creato

### Costruzione Container

Per creare/ricostruire il container:
```bat
docker_maintenance.bat clean-rebuild
```

### Verifica Container

Per verificare che il container sia attivo:
```bash
docker ps --filter "name=drifts-container"
```

### Avvio/Stop Container

Avvio:
```bash
docker start drifts-container
```

Stop:
```bash
docker stop drifts-container
```

## Troubleshooting

### Errore: "Container not found"

Se il container Docker non viene trovato:
```bat
docker_tests.bat rebuild
```

### Errore: "Worker start failed"

Verifica che:
1. Redis sia in esecuzione
2. `worker_config.yaml` sia presente e corretto
3. I moduli Python necessari siano installati

Per controllare il sistema:
```bat
tests.bat check
```

### I Worker Non Si Fermano

Per fermare manualmente tutti i worker:
```bash
python enhanced_launch_workers.py stop
```

### Test Molto Lenti

Il tempo di esecuzione dipende da:
- Numero di dataset
- Complessità dei dataset
- Durata dei worker (`--worker-duration`)

Per test più rapidi:
```bash
python test_datasets_with_workers.py --max-datasets 2 --worker-duration 10
```

## Best Practices

### Prima di Iniziare Test Lunghi

1. Controlla il sistema:
   ```bat
   tests.bat check
   ```

2. Testa con pochi dataset:
   ```bash
   python test_datasets_with_workers.py --max-datasets 2
   ```

3. Verifica i log:
   ```
   test_datasets_workers.log
   ```

### Per Test di Produzione

1. Usa il profilo production:
   ```bash
   python test_datasets_with_workers.py --worker-profile production
   ```

2. Aumenta la durata dei worker:
   ```bash
   python test_datasets_with_workers.py --worker-duration 60
   ```

3. Salva i log in modo persistente

### Per Test Rapidi/Debug

1. Limita i dataset:
   ```bash
   python test_datasets_with_workers.py --max-datasets 1
   ```

2. Riduci la durata:
   ```bash
   python test_datasets_with_workers.py --worker-duration 5
   ```

3. Continua su errori per vedere tutti i problemi:
   ```bash
   python test_datasets_with_workers.py --continue-on-error
   ```

## Confronto tra gli Script di Test

| Script | Cosa Testa | Docker | Stop su Errore | Quando Usarlo |
|--------|------------|--------|----------------|---------------|
| `test_results_datasets.py` | Solo dataset in `results/` | Opzionale | Sì | Test veloci su dataset già elaborati |
| `test_datasets_with_workers.py` | Tutti o specifici dataset | No | Configurable | Test completi, sviluppo |
| `tests.bat` | Tutti i dataset | No | Sì | Test locali completi |
| `docker_tests.bat` | Tutti i dataset | Sì | Sì | Test in ambiente isolato |

## Esempi Pratici

### Scenario 1: Verificare i Dataset già Elaborati

```bat
# Test rapido locale
test_results_datasets.bat local

# Test completo (locale + Docker)
test_results_datasets.bat
```

### Scenario 2: Test di Sviluppo

```bash
# Testa 3 dataset specifici
python test_datasets_with_workers.py --datasets Coffee Wine ECG200 --worker-duration 15

# Testa i primi 5 dataset
python test_datasets_with_workers.py --max-datasets 5
```

### Scenario 3: Test Completo Pre-Release

```bat
# 1. Verifica sistema
tests.bat check

# 2. Test locale completo
tests.bat now

# 3. Test Docker completo
docker_tests.bat run
```

### Scenario 4: Debug di un Dataset Specifico

```bash
# Test singolo dataset con log dettagliato
python test_datasets_with_workers.py --datasets ProblematicDataset --worker-duration 30
```

## File di Configurazione

### worker_config.yaml

Contiene i profili di configurazione per i worker:
- `default` - Configurazione standard
- `development` - Per sviluppo/debug
- `production` - Per esecuzione in produzione

### Modifica Profili

Edita `worker_config.yaml` per personalizzare:
- Numero di worker
- Risorse allocate
- Parametri di logging
- Timeout

## Riferimenti

- Vedi `enhanced_launch_workers.py` per la gestione dei worker
- Vedi `init_aeon_univariate.py` per l'inizializzazione dei dataset
- Vedi `DOCKER_GUIDE.md` per maggiori dettagli su Docker (se disponibile)

