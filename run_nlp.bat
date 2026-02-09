@echo off
REM NLP Query Engine Launcher
REM 
REM IMPORTANT: Set your Hugging Face token before running!
REM Get token from: https://huggingface.co/settings/tokens
REM
REM Option 1: Set token in environment (RECOMMENDED)
REM   Run this command first in your terminal:
REM   set HUGGINGFACE_TOKEN=your_actual_token_here
REM
REM Option 2: Edit this file and uncomment the line below:
REM set HUGGINGFACE_TOKEN=hf_YOUR_TOKEN_HERE
REM Note: No spaces allowed after the token!

echo Starting NLP Query Engine with Falcon-7B (FREE model)...
C:\Users\harik\AppData\Local\Microsoft\WindowsApps\python3.10.exe NLP/NLP.py --model model/skilldev_model.pkl
pause
