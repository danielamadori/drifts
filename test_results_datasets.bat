@echo off
setlocal enabledelayedexpansion

REM Test dei dataset presenti in results/ con e senza Docker

set "PROJECT_ROOT=%~dp0"
cd /d "%PROJECT_ROOT%"

set "ARG=%~1"
if "%ARG%"=="" goto interactive
if /I "%ARG%"=="interactive" goto interactive
if /I "%ARG%"=="now" goto run_now
if /I "%ARG%"=="local" goto run_local
if /I "%ARG%"=="docker" goto run_docker
if /I "%ARG%"=="help" goto help

echo Opzione non riconosciuta: %ARG%
echo.
goto help

:help
echo Uso: test_results_datasets.bat [interactive^|now^|local^|docker^|help]
echo.
echo   interactive   Modalita guidata con pausa (default)
echo   now           Esegui test immediatamente (locale + Docker)
echo   local         Esegui solo test locali (senza Docker)
echo   docker        Esegui solo test con Docker
echo   help          Mostra questo aiuto
echo.
echo Esempi:
echo   test_results_datasets.bat              ^(modalita interattiva^)
echo   test_results_datasets.bat now          ^(esegui subito entrambi^)
echo   test_results_datasets.bat local        ^(solo test locali^)
echo   test_results_datasets.bat docker       ^(solo test Docker^)
echo.
exit /b 1

:interactive
echo.
echo ========================================
echo  TEST DATASET DA RESULTS/
echo ========================================
echo.
echo Verranno testati tutti i dataset presenti in results/
echo con entrambe le modalita:
echo   1. Test locali (senza Docker)
echo   2. Test con Docker
echo.
echo Il test si fermera al primo errore.
echo.
echo Premi Ctrl+C per annullare, oppure
pause
echo.

call :execute_tests
set "EXIT_CODE=%ERRORLEVEL%"
goto finalize

:run_now
echo.
echo ========================================
echo  ESECUZIONE IMMEDIATA TEST
echo ========================================
echo.
echo Avvio test in 3 secondi...
timeout /t 3 /nobreak >nul
echo.

call :execute_tests
set "EXIT_CODE=%ERRORLEVEL%"
goto finalize

:run_local
echo.
echo ========================================
echo  TEST SOLO LOCALI
echo ========================================
echo.
echo Esecuzione test senza Docker...
echo.

python -u test_results_datasets.py --local-only
set "EXIT_CODE=%ERRORLEVEL%"
goto finalize

:run_docker
echo.
echo ========================================
echo  TEST SOLO DOCKER
echo ========================================
echo.
echo Esecuzione test con Docker...
echo.

python -u test_results_datasets.py --docker-only
set "EXIT_CODE=%ERRORLEVEL%"
goto finalize

:execute_tests
echo Esecuzione: python test_results_datasets.py
echo.

python -u test_results_datasets.py
set "EXIT_CODE=%ERRORLEVEL%"

echo.
echo ========================================
if %EXIT_CODE%==0 (
    echo Test completati con successo
) else (
    echo Test falliti o interrotti
)
echo Exit code: %EXIT_CODE%
echo ========================================
echo.

if exist test_results_datasets.json (
    echo Risultati disponibili in:
    echo   - test_results_datasets.log
    echo   - test_results_datasets.json
    echo.
    echo Riepilogo risultati:
    type test_results_datasets.json
    echo.
)

exit /b %EXIT_CODE%

:finalize
echo.
if exist test_results_datasets.json (
    echo ========================================
    echo RIEPILOGO FINALE
    echo ========================================
    type test_results_datasets.json
    echo.
)

echo.
echo Log completo disponibile in: test_results_datasets.log
echo.
pause
exit /b %EXIT_CODE%

