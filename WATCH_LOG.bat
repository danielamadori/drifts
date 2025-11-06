@echo off
REM Simple log tail monitoring
REM Run this in a separate terminal

cd /d C:\Users\danie\Projects\drifts

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
    timeout /t 2 /nobreak > nul
    goto waitlog
)

echo Log file found! Showing live updates...
echo ========================================
echo.

powershell -Command "Get-Content test_datasets_workers.log -Wait -Tail 20"

pause

