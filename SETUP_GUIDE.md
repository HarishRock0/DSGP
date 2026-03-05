# 🚀 Quick Setup Guide - NLP Query Engine

## 📋 Setup Steps

### Step 1: Get Your Groq Cloud API Key

1. Go to: https://console.groq.com/keys
2. Sign in or create a free account
3. Click **"Create API Key"**
4. Name it (e.g., "DSGP-NLP")
5. **Copy the API key** (starts with `gsk_`)

### Step 2: Set Environment Variable

**Option A: For Single Session (Quick Test)**

Open PowerShell in this folder and run:
```powershell
$env:GROQ_API_KEY = "gsk_YOUR_KEY_HERE"
python3.10 NLP/NLP.py --model model/skilldev_model.pkl
```

**Option B: Using Batch File (Recommended)**

1. Open `run_nlp.bat` in a text editor
2. Find this line:
   ```bat
   REM set GROQ_API_KEY=gsk_YOUR_KEY_HERE
   ```
3. Remove `REM` and replace with your key:
   ```bat
   set GROQ_API_KEY=gsk_YOUR_KEY_HERE
   ```
4. Save and double-click `run_nlp.bat` to run

**Option C: Permanent Environment Variable (Best for Development)**

1. Open Windows Settings → System → About → Advanced system settings
2. Click "Environment Variables"
3. Under "User variables", click "New"
4. Variable name: `GROQ_API_KEY`
5. Variable value: `gsk_YOUR_KEY_HERE`
6. Click OK
7. **Restart your terminal/PowerShell**
8. Run: `python3.10 NLP/NLP.py --model model/skilldev_model.pkl`

### Step 3: Verify Installation

Run in PowerShell:
```powershell
pip list | findstr llama-index
```

You should see:
- `llama-index`
- `llama-index-llms-groq`
- `llama-index-experimental`

If not, install:
```powershell
pip install -r requirement.txt
```

### Step 4: Test the Query Engine

Run the application:
```powershell
python3.10 NLP/NLP.py --model model/skilldev_model.pkl
```

You should see:
```
✅ PandasQueryEngine ready (general queries)
✅ Direct LLM ready (resource allocation)
```

Try a test query:
```
📝 Your query: How many people have vision difficulties?
```

## 🔧 Troubleshooting

### Error: "Missing GROQ_API_KEY environment variable"

**Cause:** Environment variable not set

**Fix:** Follow Step 2 above to set `GROQ_API_KEY`

### Error: "Query engine not initialized"

**Possible causes:**
1. API key not set → Set environment variable
2. Invalid API key → Regenerate key at https://console.groq.com/keys
3. No internet connection → Check connectivity

### Error: "No module named 'llama_index'"

**Fix:**
```powershell
pip install -r requirement.txt
```

### Error: "Model file not found"

**Fix:** Make sure `model/skilldev_model.pkl` exists. If not, you can use CSV directly:
```powershell
python3.10 NLP/NLP.py --csv data/LFS-2023.csv
```

## 📁 Tech Stack

- **LLM Provider:** Groq Cloud API (Llama 3.3 70B Versatile)
- **LLM Framework:** LlamaIndex with PandasQueryEngine
- **Cluster Analysis:** Local transformers (BART-large-MNLI, MiniLM)
- **No HuggingFace API token required** — all LLM inference runs through Groq

## 📞 Need Help?

If you still encounter issues:

1. Check your API key is valid: https://console.groq.com/keys
2. Verify Groq service status: https://status.groq.com
3. Test internet connection
4. Review error messages carefully

## 🎯 Quick Reference

**Run with model:**
```powershell
$env:GROQ_API_KEY = "gsk_YOUR_KEY"
python3.10 NLP/NLP.py --model model/skilldev_model.pkl
```

**Run with CSV:**
```powershell
$env:GROQ_API_KEY = "gsk_YOUR_KEY"
python3.10 NLP/NLP.py --csv data/LFS-2023.csv
```

**Using batch file:**
```cmd
run_nlp.bat
```

---

**Security Reminder:** Never commit API keys to Git! They are now in `.gitignore`.
