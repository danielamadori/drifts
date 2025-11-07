@echo off
echo ========================================
echo ESECUZIONE TEST DATASET DA RESULTS
echo ========================================
echo.

echo [1] Verifico Redis...
docker ps | findstr redis >nul 2>&1
if errorlevel 1 (
    echo Redis non trovato, avvio container...
    docker run -d -p 6379:6379 --name test-redis redis:latest
    timeout /t 5 /nobreak >nul
    echo Redis avviato!
) else (
    echo Redis gia attivo!
)

echo.
echo [2] Eseguo test dei dataset...
echo.

python test_results_datasets.py --local-only

echo.
echo ========================================
echo FINE TEST
echo ========================================
pause

