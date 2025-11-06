@echo off
REM Complete Docker setup and test execution

echo ========================================
echo DOCKER COMPLETE SETUP AND TEST
echo ========================================
echo.

REM Step 1: Build Docker image
echo [STEP 1/4] Building Docker image...
echo This may take 5-10 minutes on first build.
echo.
docker build -t drifts:latest .
if errorlevel 1 (
    echo [ERROR] Docker build failed!
    pause
    exit /b 1
)
echo [OK] Docker image built successfully
echo.

REM Step 2: Stop and remove any existing container
echo [STEP 2/4] Cleaning up old containers...
docker stop drifts-container 2>nul
docker rm drifts-container 2>nul
echo [OK] Cleanup completed
echo.

REM Step 3: Start container
echo [STEP 3/4] Starting Docker container...
docker run -d ^
    --name drifts-container ^
    -p 6379:6379 ^
    -p 8888:8888 ^
    -v "%cd%\logs:/app/logs" ^
    -v "%cd%\workers:/app/workers" ^
    -v "%cd%\results:/app/results" ^
    drifts:latest

if errorlevel 1 (
    echo [ERROR] Failed to start container!
    pause
    exit /b 1
)
echo [OK] Container started
echo.

REM Wait for Redis to be ready
echo Waiting for Redis to start (10 seconds)...
timeout /t 10 /nobreak >nul

REM Verify Redis is running
echo Checking Redis...
docker exec drifts-container redis-cli ping
if errorlevel 1 (
    echo [WARNING] Redis not responding, waiting more...
    timeout /t 5 /nobreak >nul
)
echo [OK] Redis is running
echo.

REM Step 4: Run tests inside container
echo [STEP 4/4] Starting tests inside Docker container...
echo ========================================
echo.
echo Running tests with configuration from worker_config.yaml
echo   Profile: default
echo   Workers: 1 (worker_cache.py)
echo   Duration: 20 seconds per dataset
echo   Total datasets: 88
echo.
echo This will take approximately 4-8 hours.
echo.
echo Press Ctrl+C to stop, then run: docker stop drifts-container
echo.
echo ========================================
echo.

docker exec -it drifts-container python test_datasets_with_workers.py --worker-duration 20

echo.
echo ========================================
echo Tests completed
echo ========================================
echo.

REM Show results
echo Fetching results...
docker exec drifts-container cat test_datasets_workers.json

echo.
pause

