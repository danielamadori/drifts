# Guida Docker per Drifts

## Panoramica

Questa guida descrive come usare Docker per eseguire i test dei dataset in un ambiente isolato e riproducibile.

## Componenti Docker

### Immagine
- **Nome**: `drifts:latest`
- **Base**: Python con dipendenze scientifiche
- **Include**: Redis, Python, librerie ML, worker

### Container
- **Nome**: `drifts-container`
- **Porte esposte**:
  - `6379` - Redis
  - `8888` - Jupyter (se abilitato)
- **Volumi montati**:
  - `logs/` - Log dei worker
  - `workers/` - Script worker
  - `results/` - Risultati dei test
  - `fig/` - Figure e grafici

## File Docker

- **`Dockerfile`** - Definizione dell'immagine
- **`docker_tests.bat`** - Script per eseguire test in Docker
- **`docker_maintenance.bat`** - Manutenzione container/immagine
- **`docker/supervisord.conf`** - Configurazione supervisor (se usato)

## Comandi Rapidi

### Build e Avvio

```bat
# Build iniziale
docker build -t drifts:latest .

# Crea e avvia container
docker run -d --name drifts-container -p 6379:6379 -p 8888:8888 ^
  -v "%cd%\logs:/app/logs" ^
  -v "%cd%\workers:/app/workers" ^
  -v "%cd%\results:/app/results" ^
  -v "%cd%\fig:/app/fig" ^
  drifts:latest
```

### Gestione Container

```bash
# Avvia container esistente
docker start drifts-container

# Ferma container
docker stop drifts-container

# Riavvia container
docker restart drifts-container

# Rimuovi container
docker rm drifts-container

# Mostra container attivi
docker ps

# Mostra tutti i container (anche fermi)
docker ps -a
```

### Gestione Immagine

```bash
# Rimuovi immagine
docker rmi drifts:latest

# Rebuild completo (senza cache)
docker build --no-cache -t drifts:latest .

# Mostra immagini
docker images
```

### Accesso al Container

```bash
# Shell interattiva
docker exec -it drifts-container /bin/bash

# Esegui comando singolo
docker exec drifts-container python --version

# Esegui comando interattivo
docker exec -it drifts-container python
```

### Log e Debug

```bash
# Mostra log del container
docker logs drifts-container

# Segui log in tempo reale
docker logs -f drifts-container

# Ultime 100 righe
docker logs --tail 100 drifts-container

# Ispeziona container
docker inspect drifts-container
```

## Script di Manutenzione

### docker_maintenance.bat

Script Windows per la manutenzione del container.

#### Comandi Disponibili

```bat
# Ferma container
docker_maintenance.bat stop

# Avvia container
docker_maintenance.bat start

# Riavvia container
docker_maintenance.bat restart

# Rebuild immagine
docker_maintenance.bat rebuild

# Pulizia completa e rebuild
docker_maintenance.bat clean-rebuild

# Rimuovi container
docker_maintenance.bat remove

# Mostra stato
docker_maintenance.bat status

# Mostra aiuto
docker_maintenance.bat help
```

#### Esempi d'Uso

```bat
# Ricostruire dopo modifiche al Dockerfile
docker_maintenance.bat rebuild

# Ripartire da zero
docker_maintenance.bat clean-rebuild

# Verificare stato
docker_maintenance.bat status
```

## Esecuzione Test in Docker

### Metodo 1: Script Batch (Consigliato)

```bat
# Esegui test
docker_tests.bat run

# Rebuild e poi testa
docker_tests.bat rebuild
```

### Metodo 2: Comando Diretto

```bash
# Test di tutti i dataset
docker exec -it drifts-container python test_datasets_with_workers.py --worker-duration 20

# Test di dataset specifici
docker exec -it drifts-container python test_datasets_with_workers.py --datasets Coffee Wine

# Test dei dataset in results/
docker exec -it drifts-container python test_results_datasets.py
```

### Metodo 3: Shell Interattiva

```bash
# Entra nel container
docker exec -it drifts-container /bin/bash

# Dentro il container
python test_datasets_with_workers.py --max-datasets 2
python test_results_datasets.py --local-only
exit
```

## Volumi e Persistenza

### Directory Condivise

I volumi permettono di condividere dati tra host e container:

| Directory Host | Directory Container | Scopo |
|----------------|---------------------|-------|
| `.\logs` | `/app/logs` | Log dei worker |
| `.\workers` | `/app/workers` | Script worker personalizzati |
| `.\results` | `/app/results` | Risultati elaborati |
| `.\fig` | `/app/fig` | Figure e grafici |

### Accesso ai File

I file sono accessibili sia dall'host che dal container:

**Dall'host (Windows):**
```bat
type results\test_results.json
notepad logs\worker.log
```

**Dal container:**
```bash
docker exec drifts-container cat /app/results/test_results.json
docker exec drifts-container tail /app/logs/worker.log
```

### Backup dei Dati

Per fare backup dei dati:
```bat
# Copia da container a host
docker cp drifts-container:/app/data ./backup_data

# Copia da host a container
docker cp ./backup_data drifts-container:/app/data
```

## Porte e Servizi

### Redis (porta 6379)

Redis viene eseguito nel container ed è accessibile dall'host:

```bash
# Da host Windows (se redis-cli installato)
redis-cli -h localhost -p 6379 ping

# Dal container
docker exec drifts-container redis-cli ping
```

### Jupyter (porta 8888)

Se Jupyter è configurato:

```bash
# Avvia Jupyter nel container
docker exec -d drifts-container jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser

# Accedi da browser su host
http://localhost:8888
```

## Troubleshooting

### Container Non Si Avvia

**Problema**: `docker start drifts-container` fallisce

**Soluzioni**:
1. Verifica se esiste:
   ```bash
   docker ps -a | findstr drifts-container
   ```

2. Rimuovi e ricrea:
   ```bat
   docker_maintenance.bat clean-rebuild
   ```

3. Controlla i log:
   ```bash
   docker logs drifts-container
   ```

### Porta Già in Uso

**Problema**: Errore "port is already allocated"

**Soluzioni**:
1. Trova processo che usa la porta:
   ```bat
   netstat -ano | findstr :6379
   ```

2. Ferma il servizio/processo in conflitto

3. O usa porte diverse:
   ```bash
   docker run -p 6380:6379 -p 8889:8888 ...
   ```

### Immagine Obsoleta

**Problema**: Modifiche al codice non hanno effetto

**Soluzione**: Rebuild dell'immagine
```bat
docker_maintenance.bat rebuild
```

### Spazio Disco Insufficiente

**Problema**: Docker occupa troppo spazio

**Soluzioni**:
1. Rimuovi container non usati:
   ```bash
   docker container prune
   ```

2. Rimuovi immagini non usate:
   ```bash
   docker image prune
   ```

3. Pulizia completa (ATTENZIONE: rimuove tutto):
   ```bash
   docker system prune -a
   ```

### Volumi Non Sincronizzati

**Problema**: File modificati su host non visibili nel container

**Soluzioni**:
1. Verifica mount volumi:
   ```bash
   docker inspect drifts-container | findstr Mounts -A 20
   ```

2. Riavvia container:
   ```bat
   docker_maintenance.bat restart
   ```

3. Ricrea container con volumi corretti:
   ```bat
   docker_maintenance.bat clean-rebuild
   ```

### Permission Denied

**Problema**: Errori di permessi nel container

**Soluzione**: Assicurati che i volumi abbiano i permessi corretti su Windows

## Best Practices

### Sviluppo

1. **Usa volumi per codice in sviluppo**
   - Monta directory del progetto per modifiche in tempo reale

2. **Rebuild solo quando necessario**
   - Rebuild dopo modifiche a `Dockerfile` o `requirements.txt`
   - Usa `docker_maintenance.bat rebuild`

3. **Mantieni container leggeri**
   - Non installare pacchetti non necessari
   - Pulisci cache dopo install

### Testing

1. **Isola i test**
   - Usa Docker per test riproducibili
   - Evita dipendenze dall'ambiente locale

2. **Monitora le risorse**
   ```bash
   docker stats drifts-container
   ```

3. **Salva sempre i risultati**
   - Usa volumi per persistere risultati
   - Backup regolari di `results/`

### Produzione

1. **Usa tag versionate**
   ```bash
   docker build -t drifts:v1.0.0 .
   docker tag drifts:v1.0.0 drifts:latest
   ```

2. **Configura restart policy**
   ```bash
   docker run --restart=unless-stopped ...
   ```

3. **Monitora log**
   - Configura log rotation
   - Usa strumenti di monitoring esterni

## Configurazione Avanzata

### Dockerfile Personalizzato

Per modificare l'immagine:

1. Edita `Dockerfile`
2. Rebuild:
   ```bat
   docker_maintenance.bat rebuild
   ```

### Variabili d'Ambiente

Passa variabili al container:
```bash
docker run -e REDIS_HOST=localhost -e WORKERS=4 drifts:latest
```

O usa file `.env`:
```bash
docker run --env-file .env drifts:latest
```

### Networking

Per connettere più container:
```bash
# Crea network
docker network create drifts-network

# Avvia container nella network
docker run --network drifts-network --name drifts-container drifts:latest
docker run --network drifts-network --name redis redis:latest
```

### Resource Limits

Limita risorse del container:
```bash
docker run --memory="2g" --cpus="2.0" drifts:latest
```

## Comandi Utili

### Pulizia

```bash
# Rimuovi container fermi
docker container prune

# Rimuovi immagini non usate
docker image prune

# Rimuovi volumi non usati
docker volume prune

# Pulizia totale
docker system prune -a --volumes
```

### Monitoring

```bash
# Statistiche in tempo reale
docker stats drifts-container

# Spazio occupato
docker system df

# Processi nel container
docker top drifts-container
```

### Copia File

```bash
# Da container a host
docker cp drifts-container:/app/results/data.json ./data.json

# Da host a container
docker cp ./config.yaml drifts-container:/app/config.yaml
```

## Riferimenti

- [Documentazione Docker](https://docs.docker.com/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Dockerfile Reference](https://docs.docker.com/engine/reference/builder/)

## Supporto

Per problemi con Docker:
1. Controlla i log: `docker logs drifts-container`
2. Ispeziona il container: `docker inspect drifts-container`
3. Verifica risorse: `docker stats drifts-container`
4. Rebuild da zero: `docker_maintenance.bat clean-rebuild`

