@echo off
REM LFS Analyzer Setup Script for Windows
REM This script helps you set up Ollama and run the LFS analyzer

echo ========================================
echo LFS Analyzer Setup and Runner
echo ========================================
echo.

REM Check if Ollama is installed
where ollama >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Ollama is not installed!
    echo.
    echo Please install Ollama first:
    echo 1. Go to: https://ollama.ai
    echo 2. Download and install Ollama for Windows
    echo 3. Run this script again
    echo.
    pause
    exit /b 1
)

echo [OK] Ollama is installed
echo.

REM Check if llama3.2 model is downloaded
echo Checking for llama3.2 model...
ollama list | findstr "llama3.2" >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [SETUP] llama3.2 model not found. Downloading...
    echo This will download about 2GB. Please wait...
    echo.
    ollama pull llama3.2
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to download model
        pause
        exit /b 1
    )
    echo.
    echo [OK] Model downloaded successfully!
) else (
    echo [OK] llama3.2 model is ready
)

echo.
echo Checking Python dependencies...
pip show ollama >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [SETUP] Installing Python dependencies...
    pip install -r requirement.txt
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to install dependencies
        pause
        exit /b 1
    )
) else (
    echo [OK] Dependencies are installed
)

echo.
echo ========================================
echo Setup Complete! Starting LFS Analyzer...
echo ========================================
echo.

REM Run the analyzer
python lfs_analyzer.py

pause
