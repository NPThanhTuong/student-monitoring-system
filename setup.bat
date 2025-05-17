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

echo kafka_stream_service.py is running...
if exist kafka_stream_service.py (
    start "" cmd /k python kafka_stream_service.py
) else (
    echo File kafka_stream_service.py is not exist!
)

echo spark_stream_service.py is running...
if exist spark_stream_service.py (
    start "" cmd /k python spark_stream_service.py
) else (
    echo File spark_stream_service.py is not exist!
)

echo Waiting for application setup to be ready...
timeout /t 20 /nobreak

echo.

pause