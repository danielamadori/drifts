@echo off
echo ========================================
echo LAUNCHING DATASET TESTS
echo ========================================
echo.
echo Starting tests in 3 seconds...
timeout /t 3 /nobreak

cd /d C:\Users\danie\Projects\drifts

echo Running: python test_datasets_with_workers.py --worker-duration 20
echo.

REM Execute with output visible
python -u test_datasets_with_workers.py --worker-duration 20

echo.
echo ========================================
echo Test execution completed
echo Exit code: %ERRORLEVEL%
echo ========================================
echo.

if exist test_datasets_workers.json (
    echo Results available in:
    echo - test_datasets_workers.log
    echo - test_datasets_workers.json
    echo.
    type test_datasets_workers.json
)

echo.
pause

