@echo off
setlocal enabledelayedexpansion

REM Monitoring utilities for dataset tests

set "PROJECT_ROOT=%~dp0"
cd /d "%PROJECT_ROOT%"

if "%~1"=="" goto monitor

set "COMMAND=%~1"
if /I "%COMMAND%"=="monitor" goto monitor
if /I "%COMMAND%"=="tail" goto tail
if /I "%COMMAND%"=="status" goto status
if /I "%COMMAND%"=="docker" goto docker
if /I "%COMMAND%"=="help" goto help

echo Unknown option: %COMMAND%
echo.

:help
echo Usage: monitor_tests.bat [monitor^|tail^|status^|docker^|help]
echo.
echo   monitor   Launch python monitor_live.py (host)
echo   tail      Follow test_datasets_workers.log on host
echo   status    Show quick status summary (processes, logs, JSON)
echo   docker    Monitor tests running inside Docker container
echo   help      Show this help
echo.
exit /b 1

:monitor
echo.
echo ========================================
echo  DATASET TEST MONITOR
echo ========================================
echo.
echo This will show live progress of tests.
echo Run this in a separate window while
echo tests are running in another terminal.
echo.
echo Press any key to start monitoring...
pause >nul

python monitor_live.py

pause
exit /b %ERRORLEVEL%

:tail
echo.
echo ========================================
echo  LIVE LOG MONITORING
echo ========================================
echo.
echo Monitoring: test_datasets_workers.log
echo Press Ctrl+C to stop
echo.
echo Waiting for log file to be created...
echo.

:waitlog
if not exist test_datasets_workers.log (
    timeout /t 2 /nobreak >nul
    goto waitlog
)

echo Log file found! Showing live updates...
echo ========================================
echo.

powershell -Command "Get-Content test_datasets_workers.log -Wait -Tail 20"

pause
exit /b %ERRORLEVEL%

:status
echo ========================================
echo TEST STATUS CHECK
echo ========================================
echo.

echo === Python Processes ===
tasklist /FI "IMAGENAME eq python.exe" 2>NUL | find "python.exe"
if errorlevel 1 echo No Python processes running

echo.
echo === Last 10 lines of log ===
if exist test_datasets_workers.log (
    powershell -Command "Get-Content test_datasets_workers.log -Tail 10"
) else (
    echo Log file not found
)

echo.
echo === Results JSON ===
if exist test_datasets_workers.json (
    type test_datasets_workers.json
) else (
    echo Results file not found yet
)

echo.
pause
exit /b 0

:docker
echo ========================================
echo DOCKER TEST MONITOR
echo ========================================
echo.

:docker_loop
cls
echo ========================================
echo DOCKER TEST MONITOR - %date% %time%
echo ========================================
echo.

docker ps --filter "name=drifts-container" --format "{{.Status}}" | findstr "Up" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Container is not running
    echo.
    pause
    exit /b 1
)

echo [OK] Container Status: Running
echo.

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

goto docker_loop
