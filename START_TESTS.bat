@echo off
setlocal enabledelayedexpansion

REM ====================================
REM DATASET TESTING - ALL DATASETS
REM ====================================

cd /d "%~dp0"

echo.
echo ========================================
echo  DATASET TESTING WITH WORKERS
echo ========================================
echo.
echo This will test ALL datasets with:
echo - Init + Optimize for each dataset
echo - Workers running for 20 seconds
echo - STOPS on first error
echo.
echo Estimated time: 4-13 hours for all datasets
echo.
echo Press Ctrl+C to cancel, or
pause

echo.
echo ========================================
echo Starting tests...
echo ========================================
echo.

REM Execute the test
python test_datasets_with_workers.py --worker-duration 20

REM Capture exit code
set EXIT_CODE=%ERRORLEVEL%

echo.
echo ========================================
if %EXIT_CODE%==0 (
    echo ALL TESTS COMPLETED SUCCESSFULLY!
    echo Check test_datasets_workers.json for results
) else (
    echo TESTS STOPPED DUE TO ERROR
    echo Check test_datasets_workers.log for details
)
echo ========================================
echo.

REM Show summary if JSON exists
if exist test_datasets_workers.json (
    echo.
    echo === RESULTS SUMMARY ===
    type test_datasets_workers.json
    echo.
)

echo.
echo Exit code: %EXIT_CODE%
echo.
pause

exit /b %EXIT_CODE%

