@echo off
REM Quick status check
cd /d C:\Users\danie\Projects\drifts

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

