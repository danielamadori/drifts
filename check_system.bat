@echo off
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

