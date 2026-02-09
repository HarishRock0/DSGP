# 🚀 Quick Setup Guide - NLP Query Engine

## ⚠️ IMPORTANT: Token Security Issue

Your previous Hugging Face token was **exposed on GitHub** and has been removed from the code for security.

**You MUST regenerate a new token before using this application!**

## 📋 Setup Steps

### Step 1: Regenerate Your Hugging Face Token

1. Go to: https://huggingface.co/settings/tokens
2. Click **"New token"**
3. Name it (e.g., "DSGP-NLP")
4. Select **"Read"** permission (sufficient for inference)
5. Click **"Generate token"**
6. **Copy the token** (you won't see it again!)

### Step 2: Set Environment Variable

**Option A: For Single Session (Quick Test)**

Open PowerShell in this folder and run:
```powershell
$env:HUGGINGFACE_TOKEN = "hf_YOUR_NEW_TOKEN_HERE"
python3.10 NLP/NLP.py --model model/skilldev_model.pkl
```

**Option B: Using Batch File (Recommended)**

1. Open `run_nlp.bat` in a text editor
2. Find this line:
   ```bat
   REM set HUGGINGFACE_TOKEN=your_actual_token_here
   ```
3. Remove `REM` and replace with your token:
   ```bat
   set HUGGINGFACE_TOKEN=hf_YOUR_NEW_TOKEN_HERE
   ```
4. Save and double-click `run_nlp.bat` to run

**Option C: Permanent Environment Variable (Best for Development)**

1. Open Windows Settings → System → About → Advanced system settings
2. Click "Environment Variables"
3. Under "User variables", click "New"
4. Variable name: `HUGGINGFACE_TOKEN`
5. Variable value: `hf_YOUR_NEW_TOKEN_HERE`
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
- `llama-index-llms-huggingface-api`
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
✅ LlamaIndex PandasQueryEngine ready with Hugging Face API!
📌 Using model: deepseek-ai/DeepSeek-R1 (ONLINE - Inference API)
🌐 All queries will be processed via Hugging Face API
```

Try a test query:
```
📝 Your query: How many people have vision difficulties?
```

## 🔧 Troubleshooting

### Error: "❌ Hugging Face token not found!"

**Cause:** Environment variable not set

**Fix:** Follow Step 2 above to set `HUGGINGFACE_TOKEN`

### Error: "Query engine not initialized"

**Possible causes:**
1. Token not set → Set environment variable
2. Invalid/expired token → Regenerate token
3. No internet connection → Check connectivity
4. Model access denied → Ensure token has "Read" permission

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

## 📁 What Was Cleaned

During code cleanup, the following were removed:
- ✅ Hardcoded API tokens (security risk)
- ✅ Ollama offline setup files (not needed)
- ✅ Test files (test_*.py)
- ✅ Duplicate setup scripts
- ✅ Unused imports (torch, transformers for local GPU)
- ✅ NLPClusterQueryEngine class (unused)

Everything now uses **Hugging Face Inference API** only.

## 📞 Need Help?

If you still encounter issues:

1. Check your token is valid: https://huggingface.co/settings/tokens
2. Verify model access: https://huggingface.co/deepseek-ai/DeepSeek-R1
3. Test internet connection
4. Review error messages carefully

## 🎯 Quick Reference

**Run with model:**
```powershell
$env:HUGGINGFACE_TOKEN = "hf_YOUR_TOKEN"
python3.10 NLP/NLP.py --model model/skilldev_model.pkl
```

**Run with CSV:**
```powershell
$env:HUGGINGFACE_TOKEN = "hf_YOUR_TOKEN"
python3.10 NLP/NLP.py --csv data/LFS-2023.csv
```

**Using batch file:**
```cmd
run_nlp.bat
```

---

**Security Reminder:** Never commit tokens to Git! They are now in `.gitignore`.
