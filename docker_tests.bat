@echo off
setlocal enabledelayedexpansion

REM Run dataset tests inside the Docker container

set "PROJECT_ROOT=%~dp0"
cd /d "%PROJECT_ROOT%"

set "IMAGE_NAME=drifts:latest"
set "CONTAINER_NAME=drifts-container"
set "RUN_COMMAND=python test_datasets_with_workers.py --worker-duration 20"
set "REQUEST_REBUILD="

if "%~1"=="" goto run

set "ARG=%~1"
if /I "%ARG%"=="run" goto run
if /I "%ARG%"=="rebuild" goto rebuild
if /I "%ARG%"=="--rebuild" goto rebuild
if /I "%ARG%"=="help" goto help

echo Unknown option: %ARG%
echo.

:help
echo Usage: docker_tests.bat [run^|rebuild^|--rebuild^|help]
echo.
echo   run        Execute dataset tests inside the Docker container (default)
echo   rebuild    Clean rebuild of the container before running tests
echo   --rebuild  Alias for ^"rebuild^"
echo   help       Show this help
echo.
exit /b 1

:rebuild
echo ========================================
echo CLEAN REBUILD REQUESTED
echo ========================================
echo.
call docker_maintenance.bat clean-rebuild
if errorlevel 1 (
    echo [ERROR] Clean rebuild failed. Aborting.
    exit /b 1
)
goto ensure_running

:run
echo ========================================
echo DOCKER TEST EXECUTION
echo ========================================
echo.

:ensure_running
REM Ensure the container is running before executing tests
docker ps --filter "name=%CONTAINER_NAME%" --format "{{.Names}}" | findstr "%CONTAINER_NAME%" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Container not running. Attempting to start...
    docker start %CONTAINER_NAME% >nul 2>&1
    if errorlevel 1 (
        echo [INFO] Container not found or failed to start. Launching a new container...
        docker run -d ^
            --name %CONTAINER_NAME% ^
            -p 6379:6379 ^
            -p 8888:8888 ^
            -v "%cd%\logs:/app/logs" ^
            -v "%cd%\workers:/app/workers" ^
            -v "%cd%\results:/app/results" ^
            -v "%cd%\fig:/app/fig" ^
            %IMAGE_NAME%
        if errorlevel 1 (
            echo [ERROR] Unable to start container. Consider running: docker_tests.bat --rebuild
            exit /b 1
        )
    )
)

echo [OK] Container ready: %CONTAINER_NAME%
echo.
echo ========================================
echo Running tests inside container...
echo Command: %RUN_COMMAND%
echo ========================================
echo.

docker exec -it %CONTAINER_NAME% %RUN_COMMAND%
set "EXIT_CODE=%ERRORLEVEL%"

echo.
echo ========================================
if %EXIT_CODE%==0 (
    echo Docker tests completed successfully
) else (
    echo Docker tests stopped due to error
)
echo Exit code: %EXIT_CODE%
echo ========================================
echo.

echo Fetching results summary (if available)...
docker exec %CONTAINER_NAME% test -f test_datasets_workers.json >nul 2>&1
if errorlevel 1 (
    echo No JSON summary created yet.
) else (
    docker exec %CONTAINER_NAME% cat test_datasets_workers.json
)
echo.

pause
exit /b %EXIT_CODE%
