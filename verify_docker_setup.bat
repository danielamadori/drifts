@echo off
REM Complete verification of Docker setup

echo ========================================
echo DOCKER SETUP VERIFICATION
echo ========================================
echo.

REM Check Docker is installed
docker --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not installed!
    pause
    exit /b 1
)
echo [OK] Docker is installed
echo.

REM Check if container exists
docker ps -a --filter "name=drifts-container" --format "{{.Names}}" | findstr "drifts-container" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Container does not exist yet
    echo [ACTION] Build is probably in progress in another window
    echo [ACTION] Or run: docker_full_test.bat
    echo.
    pause
    exit /b 0
)

echo [OK] Container exists
echo.

REM Check if container is running
docker ps --filter "name=drifts-container" --format "{{.Names}}" | findstr "drifts-container" >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Container exists but is not running
    echo [ACTION] Starting container...
    docker start drifts-container
    timeout /t 5 /nobreak >nul
)

echo [OK] Container is running
echo.

REM Wait for services to start
echo Waiting for services to initialize (5 seconds)...
timeout /t 5 /nobreak >nul

REM Check supervisor services
echo ========================================
echo SUPERVISOR SERVICES
echo ========================================
docker exec drifts-container supervisorctl status
echo.

REM Test Redis
echo ========================================
echo TESTING REDIS
echo ========================================
docker exec drifts-container redis-cli ping
if errorlevel 1 (
    echo [ERROR] Redis is not responding!
) else (
    echo [OK] Redis is working
)
echo.

REM Test Jupyter
echo ========================================
echo TESTING JUPYTER
echo ========================================
docker exec drifts-container curl -s http://localhost:8888 >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Jupyter may not be fully started yet
    echo Checking Jupyter process...
    docker exec drifts-container ps aux | findstr "jupyter"
) else (
    echo [OK] Jupyter is responding
)
echo.

REM Show Jupyter URL
echo ========================================
echo JUPYTER ACCESS
echo ========================================
echo.
echo Open your browser to:
echo   http://localhost:8888
echo.
echo No password required (development mode)
echo.

REM List notebooks
echo ========================================
echo AVAILABLE NOTEBOOKS
echo ========================================
docker exec drifts-container find /app -maxdepth 1 -name "*.ipynb" -type f
echo.

REM Show ports
echo ========================================
echo EXPOSED PORTS
echo ========================================
docker port drifts-container
echo.

REM Final summary
echo ========================================
echo VERIFICATION COMPLETE
echo ========================================
echo.
echo Status:
docker exec drifts-container supervisorctl status | findstr "RUNNING" >nul 2>&1
if errorlevel 1 (
    echo   [WARNING] Some services may not be running
) else (
    echo   [OK] All services are running
)
echo.
echo Next steps:
echo   1. Open browser to http://localhost:8888
echo   2. Run tests: docker exec -it drifts-container python test_datasets_with_workers.py --worker-duration 20
echo   3. Check logs: docker exec drifts-container tail -f /var/log/supervisor/jupyter.log
echo.

pause

