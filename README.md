# LFS Analyzer - Quick Start

## 🚀 Setup (3 Steps):

### 1. Install Ollama
Download from: https://ollama.ai

### 2. Download AI Model
```powershell
ollama pull llama3.2
```

### 3. Install Dependencies
```powershell
pip install -r requirement.txt
```

## 💻 Run

```powershell
# Easy way - use batch file
.\run_analyzer.bat

# Or run directly
python lfs_analyzer.py
```

## 📖 Usage

```
📝 Your question: What is the average age?
📝 Your question: /insights employment
📝 Your question: /overview
📝 Your question: quit
```

## ⚠️ Troubleshooting

**Error: "ollama: command not found"**
→ Install Ollama from https://ollama.ai

**Error: "model not found"**
→ Run: `ollama pull llama3.2`

**Error: Python packages**
→ Run: `pip install -r requirement.txt`

That's it! 🎉
