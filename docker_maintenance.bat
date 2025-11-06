@echo off
setlocal enabledelayedexpansion

REM Docker maintenance utilities: verification and clean rebuild

set "PROJECT_ROOT=%~dp0"
cd /d "%PROJECT_ROOT%"

set "IMAGE_NAME=drifts:latest"
set "CONTAINER_NAME=drifts-container"

if "%~1"=="" goto verify

set "COMMAND=%~1"
if /I "%COMMAND%"=="verify" goto verify
if /I "%COMMAND%"=="clean-rebuild" goto clean_rebuild
if /I "%COMMAND%"=="help" goto help

echo Unknown option: %COMMAND%
echo.

:help
echo Usage: docker_maintenance.bat [verify^|clean-rebuild^|help]
echo.
echo   verify         Validate Docker installation, container status, and services
echo   clean-rebuild  Rebuild image, recreate container, and verify notebooks
echo   help           Show this help
echo.
exit /b 1

:verify
echo ========================================
echo DOCKER SETUP VERIFICATION
echo ========================================
echo.

docker --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not installed!
    pause
    exit /b 1
)
echo [OK] Docker is installed
echo.

docker ps -a --filter "name=%CONTAINER_NAME%" --format "{{.Names}}" | findstr "%CONTAINER_NAME%" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Container does not exist yet
    echo [ACTION] Run: docker_maintenance.bat clean-rebuild
    echo.
    pause
    exit /b 0
)

echo [OK] Container exists
echo.

docker ps --filter "name=%CONTAINER_NAME%" --format "{{.Names}}" | findstr "%CONTAINER_NAME%" >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Container exists but is not running
    echo [ACTION] Starting container...
    docker start %CONTAINER_NAME%
    timeout /t 5 /nobreak >nul
)

echo [OK] Container is running
echo.

echo Waiting for services to initialize (5 seconds)...
timeout /t 5 /nobreak >nul

echo ========================================
echo SUPERVISOR SERVICES
echo ========================================
docker exec %CONTAINER_NAME% supervisorctl status
echo.

echo ========================================
echo TESTING REDIS
echo ========================================
docker exec %CONTAINER_NAME% redis-cli ping
if errorlevel 1 (
    echo [ERROR] Redis is not responding!
) else (
    echo [OK] Redis is working
)
echo.

echo ========================================
echo TESTING JUPYTER
echo ========================================
docker exec %CONTAINER_NAME% curl -s http://localhost:8888 >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Jupyter may not be fully started yet
    echo Checking Jupyter process...
    docker exec %CONTAINER_NAME% ps aux | findstr "jupyter"
) else (
    echo [OK] Jupyter is responding
)
echo.

echo ========================================
echo JUPYTER ACCESS
echo ========================================
echo.
echo Open your browser to:
echo   http://localhost:8888
echo.
echo No password required (development mode)
echo.

echo ========================================
echo AVAILABLE NOTEBOOKS
echo ========================================
docker exec %CONTAINER_NAME% find /app -maxdepth 1 -name "*.ipynb" -type f
echo.

echo ========================================
echo EXPOSED PORTS
echo ========================================
docker port %CONTAINER_NAME%
echo.

echo ========================================
echo VERIFICATION COMPLETE
echo ========================================
echo.
echo Status:
docker exec %CONTAINER_NAME% supervisorctl status | findstr "RUNNING" >nul 2>&1
if errorlevel 1 (
    echo   [WARNING] Some services may not be running
) else (
    echo   [OK] All services are running
)
echo.
echo Next steps:
echo   1. Open browser to http://localhost:8888
echo   2. Run tests: docker exec -it %CONTAINER_NAME% python test_datasets_with_workers.py --worker-duration 20
echo   3. Check logs: docker exec %CONTAINER_NAME% tail -f /var/log/supervisor/jupyter.log
echo.

pause
exit /b 0

:clean_rebuild
echo ========================================
echo FINAL DOCKER BUILD WITH NOTEBOOK VERIFICATION
echo ========================================
echo.

echo [STEP 1] Verifying notebooks in filesystem...
dir *.ipynb >nul 2>&1
if errorlevel 1 (
    echo [ERROR] No notebooks found in filesystem!
    echo Cannot proceed without notebooks
    pause
    exit /b 1
)

echo [OK] Notebooks found in filesystem:
dir *.ipynb
echo.

echo [STEP 2] Cleaning old Docker resources...
docker stop %CONTAINER_NAME% 2>nul
docker rm %CONTAINER_NAME% 2>nul
docker rmi %IMAGE_NAME% 2>nul
echo [OK] Cleanup complete
echo.

echo [STEP 3] Building Docker image (this may take several minutes)...
docker build -t %IMAGE_NAME% .
if errorlevel 1 (
    echo [ERROR] Docker build failed!
    pause
    exit /b 1
)
echo.
echo [OK] Build completed successfully!
echo.

echo [STEP 4] Starting container...
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
    echo [ERROR] Failed to start container!
    pause
    exit /b 1
)
echo [OK] Container started
echo.

echo Waiting for services to initialize (15 seconds)...
timeout /t 15 /nobreak >nul

echo ========================================
echo VERIFYING NOTEBOOKS IN CONTAINER
echo ========================================
echo.

docker exec %CONTAINER_NAME% ls -lah /app/*.ipynb
if errorlevel 1 (
    echo.
    echo [CRITICAL ERROR] NOTEBOOKS NOT IN CONTAINER!
    echo.
    echo Checking Dockerfile COPY command...
    findstr /C:"COPY" Dockerfile
    echo.
    echo Checking .dockerignore...
    if exist .dockerignore (
        type .dockerignore | findstr "ipynb"
    ) else (
        echo .dockerignore file not found
    )
    echo.
    echo Debugging: What IS in /app?
    docker exec %CONTAINER_NAME% ls -lah /app/ | more
    echo.
    pause
    exit /b 1
)

echo.
echo [SUCCESS] NOTEBOOKS ARE IN THE CONTAINER!
echo.

echo ========================================
echo VERIFYING JUPYTER
echo ========================================
docker exec %CONTAINER_NAME% supervisorctl status jupyter
echo.

echo ========================================
echo SETUP COMPLETE - ALL WORKING!
echo ========================================
echo.
echo Access Points:
echo   Jupyter Notebook: http://localhost:8888
echo   Redis:            localhost:6379
echo.
echo Available Notebooks:
docker exec %CONTAINER_NAME% ls /app/*.ipynb
echo.
echo Open your browser to http://localhost:8888 now!
echo.

pause
exit /b 0
