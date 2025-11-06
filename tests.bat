@echo off
setlocal enabledelayedexpansion

REM Unified test runner for Windows hosts

set "PROJECT_ROOT=%~dp0"
cd /d "%PROJECT_ROOT%"

if "%~1"=="" goto interactive

set "COMMAND=%~1"
if /I "%COMMAND%"=="interactive" goto interactive
if /I "%COMMAND%"=="now" goto run_now
if /I "%COMMAND%"=="check" goto check
if /I "%COMMAND%"=="help" goto help

echo Unknown option: %COMMAND%
echo.

:help
echo Usage: tests.bat [interactive^|now^|check^|help]
echo.
echo   interactive   Guided run that pauses before starting tests (default)
echo   now           Start tests immediately after a short countdown
echo   check         Run quick system checks for Python and worker scripts
echo   help          Show this help
echo.
exit /b 1

:interactive
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

call :execute_tests
set "EXIT_CODE=%ERRORLEVEL%"
goto finalize

:run_now
echo ========================================
echo LAUNCHING DATASET TESTS
echo ========================================
echo.
echo Starting tests in 3 seconds...
timeout /t 3 /nobreak >nul
echo Running: python test_datasets_with_workers.py --worker-duration 20
echo.

call :execute_tests
set "EXIT_CODE=%ERRORLEVEL%"
goto finalize

:execute_tests
python -u test_datasets_with_workers.py --worker-duration 20
set "EXIT_CODE=%ERRORLEVEL%"

echo.
echo ========================================
if %EXIT_CODE%==0 (
    echo Test execution completed successfully
) else (
    echo Test execution stopped due to error
)
echo Exit code: %EXIT_CODE%
echo ========================================
echo.

if exist test_datasets_workers.json (
    echo Results available in:
    echo - test_datasets_workers.log
    echo - test_datasets_workers.json
    echo.
    type test_datasets_workers.json
    echo.
)

exit /b %EXIT_CODE%

:check
echo ====================================
echo QUICK SYSTEM CHECK
echo ====================================
echo.

echo [1/5] Checking Python version...
python --version
if errorlevel 1 (
    echo ERROR: Python not found
    exit /b 1
)
echo.

echo [2/5] Testing enhanced_launch_workers.py syntax...
python -m py_compile enhanced_launch_workers.py
if errorlevel 1 (
    echo ERROR: Syntax error in enhanced_launch_workers.py
    exit /b 1
)
echo OK: enhanced_launch_workers.py syntax valid
echo.

echo [3/5] Testing test_datasets_with_workers.py syntax...
python -m py_compile test_datasets_with_workers.py
if errorlevel 1 (
    echo ERROR: Syntax error in test_datasets_with_workers.py
    exit /b 1
)
echo OK: test_datasets_with_workers.py syntax valid
echo.

echo [4/5] Checking worker_config.yaml...
if not exist worker_config.yaml (
    echo ERROR: worker_config.yaml not found
    exit /b 1
)
echo OK: worker_config.yaml found
echo.

echo [5/5] Checking worker scripts...
if not exist worker_cache.py (
    echo ERROR: worker_cache.py not found
    exit /b 1
)
echo OK: worker_cache.py found
if not exist worker_cache_logged.py (
    echo WARNING: worker_cache_logged.py not found
) else (
    echo OK: worker_cache_logged.py found
)
echo.

echo ====================================
echo ALL CHECKS PASSED
echo ====================================
echo.
echo System is ready to use:
echo   - python enhanced_launch_workers.py start
echo   - python test_datasets_with_workers.py --max-datasets 2
echo.
exit /b 0

:finalize
echo.
pause
exit /b %EXIT_CODE%
