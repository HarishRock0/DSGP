# 🚀 Setup Guide - Fix Query Engine Initialization

## Problem
The error "Query engine not initialized" occurs because the Hugging Face API configuration was failing.

## ✅ Solution: Use Local Ollama (Free, Fast, No API Keys!)

### Step 1: Install Required Packages
```bash
pip install llama-index llama-index-llms-ollama llama-index-experimental
```

Or use the updated requirements file:
```bash
pip install -r requirement.txt
```

### Step 2: Install Ollama
Download and install Ollama from: https://ollama.ai

**Windows:**
- Download the installer from https://ollama.ai/download/windows
- Run the installer
- Ollama will start automatically

### Step 3: Download the Llama Model
Open a terminal and run:
```bash
ollama pull llama3.1:8b
```

This downloads the Llama 3.1 8B model (about 4.7GB). Alternatives:
- `ollama pull llama2` (smaller, faster)
- `ollama pull mistral` (good alternative)
- `ollama pull phi3` (smallest option)

### Step 4: Start Ollama Server
```bash
ollama serve
```

Keep this terminal open while running your application.

### Step 5: Test Your Setup
Run your NLP script:
```bash
python NLP/NLP.py
```

You should see:
```
✅ LlamaIndex PandasQueryEngine ready with local Ollama!
📌 Using model: llama3.1:8b
```

## 🎯 Benefits of Ollama vs Hugging Face API:
- ✅ **FREE** - No API costs
- ✅ **FAST** - Runs locally on your machine
- ✅ **PRIVATE** - Your data stays on your computer
- ✅ **NO INTERNET** - Works offline after model download
- ✅ **NO API KEYS** - No token management needed

## 🔧 Troubleshooting

### "Ollama connection error"
- Make sure Ollama is running: `ollama serve`
- Check if the model is downloaded: `ollama list`

### "Model not found"
- Download the model: `ollama pull llama3.1:8b`

### "Out of memory"
- Use a smaller model: `ollama pull phi3`
- Or close other applications

## 📝 Alternative: Use Hugging Face (If you prefer)
If you still want to use Hugging Face API:

1. Get a valid token from https://huggingface.co/settings/tokens
2. Request access to meta-llama/Llama-3.1-8B-Instruct
3. Update the token in NLP.py (line 366)
4. Install: `pip install llama-index-llms-huggingface`

## ✨ What Changed?
- Removed dependency on Hugging Face API
- Switched to local Ollama LLM
- Updated imports in NLP.py
- Added required packages to requirement.txt
- Better error messages with setup instructions
