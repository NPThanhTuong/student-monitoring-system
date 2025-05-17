@echo off
REM Setup script for facial recognition with HDFS on Windows

echo Creating necessary directories...
if not exist data\known_faces mkdir data\known_faces
if not exist saved_images mkdir saved_images

echo Checking for Docker...
docker --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Docker is not installed or not in PATH. Please install Docker Desktop and try again.
    pause
    exit /b 1
)

echo Starting all docker containers...
docker-compose up -d
if %ERRORLEVEL% neq 0 (
    echo Failed to start Docker containers. See error message above.
    pause
    exit /b 1
)

echo Waiting for all services to be ready...
timeout /t 30 /nobreak

echo.
echo Setup complete!
echo.

echo.

echo classification server is running
if exist app/server.py (
    start "" cmd /k python -m app.server
) else (
    echo File server.py is not exist!
    pause
)

echo Waiting for application setup to be ready...
timeout /t 5 /nobreak

echo.

pause