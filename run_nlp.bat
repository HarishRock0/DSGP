@echo off
REM NLP Query Engine Launcher
REM 
REM IMPORTANT: Set your Groq API key before running!
REM Get API key from: https://console.groq.com/keys
REM
REM Option 1: Set token in environment (RECOMMENDED)
REM   Run this command first in your terminal:
REM   set GROQ_API_KEY=your_actual_key_here
REM
REM Option 2: Edit this file and uncomment the line below:
REM set GROQ_API_KEY=gsk_YOUR_KEY_HERE
REM Note: No spaces allowed after the key!

echo Starting NLP Query Engine with Groq API (Llama 3.3 70B)...
C:\Users\harik\AppData\Local\Microsoft\WindowsApps\python3.10.exe NLP/NLP.py --model model/skilldev_model.pkl
pause
