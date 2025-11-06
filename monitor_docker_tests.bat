@echo off
REM Monitor tests running inside Docker container

echo ========================================
echo DOCKER TEST MONITOR
echo ========================================
echo.

:check
cls
echo ========================================
echo DOCKER TEST MONITOR - %date% %time%
echo ========================================
echo.

REM Check if container is running
docker ps --filter "name=drifts-container" --format "{{.Status}}" | findstr "Up" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Container is not running
    echo.
    pause
    exit /b 1
)

echo [OK] Container Status: Running
echo.

REM Check if test log exists
docker exec drifts-container test -f test_datasets_workers.log >nul 2>&1
if errorlevel 1 (
    echo [INFO] Test not started yet or log file not created
    echo Waiting for test to start...
) else (
    echo === Last 20 lines of test log ===
    docker exec drifts-container tail -20 test_datasets_workers.log
    echo.
    echo === JSON Results (if available) ===
    docker exec drifts-container test -f test_datasets_workers.json >nul 2>&1
    if not errorlevel 1 (
        docker exec drifts-container cat test_datasets_workers.json
    ) else (
        echo Results file not created yet
    )
)

echo.
echo ----------------------------------------
echo Refreshing in 10 seconds... (Ctrl+C to exit)
echo ----------------------------------------
timeout /t 10 /nobreak >nul

goto check

