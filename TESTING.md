# Testing Guide

This document collects the available workflows for running and monitoring the
dataset test suite on Windows. For Linux/macOS equivalents, refer to the inline
comments inside the Python scripts.

## 1. Local host testing

Use `tests.bat` to orchestrate the full workflow of
`test_datasets_with_workers.py --worker-duration 20`.

```powershell
# Guided flow (pauses before starting)
tests.bat

# Start immediately after a short countdown
tests.bat now

# Run pre-flight checks (Python version, syntax, required files)
tests.bat check
```

Both the guided and immediate flows stop on the first failure and print the JSON
summary (`test_datasets_workers.json`) when available.

## 2. Docker test runs

Launch the tests inside the Docker container when you want isolation or Redis
pre-configured.

```powershell
# Run tests using an existing container/image
docker_tests.bat

# Clean rebuild the image and container before running tests
docker_tests.bat --rebuild
```

If you need a rebuild or verification without executing tests, rely on
`docker_maintenance.bat`:

```powershell
# Validate Docker install, container status, and key services
docker_maintenance.bat verify

# Rebuild the Docker image and container, verifying notebooks are copied
docker_maintenance.bat clean-rebuild
```

## 3. Monitoring progress

Use `monitor_tests.bat` to keep an eye on long-running jobs.

```powershell
# Launch python monitor_live.py (host)
monitor_tests.bat monitor

# Tail the host-side log with auto-refresh
monitor_tests.bat tail

# Show quick status (Python processes, log tail, JSON summary)
monitor_tests.bat status

# Watch tests inside the Docker container with live refresh
monitor_tests.bat docker
```

## 4. Expected outputs

Every execution of the test suite writes two files in the project root:

- `test_datasets_workers.log` – detailed log of the latest run
- `test_datasets_workers.json` – structured summary of successes and failures

The monitoring commands above read those files directly, so no manual parsing is
required unless you need deeper troubleshooting.

---

For additional orchestration details, consult `README.md` or the inline help for
each batch script (`tests.bat help`, etc.).
