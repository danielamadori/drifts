@echo off
REM Final verification and rebuild script

echo ========================================
echo FINAL DOCKER BUILD WITH NOTEBOOK VERIFICATION
echo ========================================
echo.

cd /d C:\Users\danie\Projects\drifts

REM Step 1: Verify notebooks exist
echo [STEP 1] Verifying notebooks in filesystem...
dir *.ipynb >nul 2>&1
if errorlevel 1 (
    echo [ERROR] No notebooks found in filesystem!
    echo Creating minimal notebooks...
    echo Cannot proceed without notebooks
    pause
    exit /b 1
)

echo [OK] Notebooks found in filesystem:
dir *.ipynb
echo.

REM Step 2: Stop and clean
echo [STEP 2] Cleaning old Docker resources...
docker stop drifts-container 2>nul
docker rm drifts-container 2>nul
docker rmi drifts:latest 2>nul
echo [OK] Cleanup complete
echo.

REM Step 3: Build
echo [STEP 3] Building Docker image (this takes 5-10 minutes)...
echo.
docker build -t drifts:latest .
if errorlevel 1 (
    echo [ERROR] Docker build failed!
    pause
    exit /b 1
)
echo.
echo [OK] Build completed successfully!
echo.

REM Step 4: Start container
echo [STEP 4] Starting container...
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

REM Wait for services
echo Waiting for services to initialize (15 seconds)...
timeout /t 15 /nobreak >nul

REM Step 5: Verify notebooks IN CONTAINER
echo ========================================
echo [STEP 5] VERIFYING NOTEBOOKS IN CONTAINER
echo ========================================
echo.

docker exec drifts-container ls -lah /app/*.ipynb
if errorlevel 1 (
    echo.
    echo [CRITICAL ERROR] NOTEBOOKS NOT IN CONTAINER!
    echo.
    echo Checking Dockerfile COPY command...
    findstr /C:"COPY" Dockerfile
    echo.
    echo Checking .dockerignore...
    type .dockerignore | findstr "ipynb"
    echo.
    echo Debugging: What IS in /app?
    docker exec drifts-container ls -lah /app/ | more
    echo.
    pause
    exit /b 1
)

echo.
echo [SUCCESS] NOTEBOOKS ARE IN THE CONTAINER!
echo.

REM Verify Jupyter
echo ========================================
echo VERIFYING JUPYTER
echo ========================================
docker exec drifts-container supervisorctl status jupyter
echo.

REM Show access info
echo ========================================
echo SETUP COMPLETE - ALL WORKING!
echo ========================================
echo.
echo Access Points:
echo   Jupyter Notebook: http://localhost:8888
echo   Redis:            localhost:6379
echo.
echo Available Notebooks:
docker exec drifts-container ls /app/*.ipynb
echo.
echo Open your browser to http://localhost:8888 now!
echo.

pause

