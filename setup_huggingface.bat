@echo off
echo ========================================
echo Hugging Face API Setup
echo ========================================
echo.
echo This will set your Hugging Face API token as an environment variable.
echo.
echo Get your token from: https://huggingface.co/settings/tokens
echo.
set /p token="Enter your Hugging Face API token (or press Enter to use default): "

if "%token%"=="" (
    echo Using default token from code...
) else (
    setx HUGGINGFACE_API_KEY "%token%"
    echo.
    echo ✓ Token saved as environment variable: HUGGINGFACE_API_KEY
    echo ✓ Please restart your terminal for changes to take effect
)

echo.
echo Installing required packages...
pip install llama-index-llms-huggingface-api
echo.
echo ========================================
echo Setup complete!
echo ========================================
echo.
echo Next steps:
echo 1. Restart your terminal (if you set a new token)
echo 2. Run your NLP script: python NLP/NLP.py
echo.
pause
