# DSGP - Data Science Group Project
## LFS-2023 Labor Force Survey Analysis with AI

AI-powered analysis of Sri Lanka's Labor Force Survey 2023 data using DeepSeek-R1 model via Hugging Face API.

---

## ⚠️ IMPORTANT: Security Notice

**Your Hugging Face token was exposed on GitHub and needs to be regenerated!**

📖 **See [SETUP_GUIDE.md](SETUP_GUIDE.md) for complete step-by-step instructions**

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirement.txt
```

### 2. Regenerate Your Hugging Face Token (REQUIRED)
Your old token was compromised. Follow these steps:

1. Go to: https://huggingface.co/settings/tokens
2. Create a **new token** with "Read" permission
3. Copy the new token

### 3. Set Up Environment Variable

**Option A - Quick Test (PowerShell):**
```powershell
$env:HUGGINGFACE_TOKEN = "hf_YOUR_TOKEN_HERE"
python3.10 NLP/NLP.py --model model/skilldev_model.pkl
```

**Option B - Permanent (Recommended):**

Edit `run_nlp.bat` and uncomment this line with your new token:
```bat
set HUGGINGFACE_TOKEN=hf_YOUR_NEW_TOKEN_HERE
```

Then run:
```cmd
run_nlp.bat
```

See **[SETUP_GUIDE.md](SETUP_GUIDE.md)** for detailed instructions.

### 4. Run the NLP Query Engine
```bash
python NLP/NLP.py
```

Or with a trained model:
```bash
python NLP/NLP.py --model model/skilldev_model.pkl
```

---

## 💬 Usage

Ask questions in natural language:

```
📝 Your query: How many people have vision difficulties?
📝 Your query: What is the average income by district?
📝 Your query: Compare employment rates by gender
📝 Your query: Show disability distribution across sectors
📝 Your query: quit
```

### Example Questions

**Demographics:**
- "How many people are in each ethnic group?"
- "What's the gender distribution?"

**Employment & Income:**
- "What's the employment rate?"
- "Show average income by district"
- "How many people work informally?"

**Disabilities:**
- "How many people have severe mobility difficulties?"
- "Compare disability rates by sector"

**Clusters** (if using trained model):
- "What are the characteristics of each cluster?"
- "Compare clusters by income levels"

---

## 📂 Project Structure

```
DSGP/
├── NLP/                    # NLP Query Engine
│   ├── NLP.py             # Main LLM query engine
│   └── README.md          # NLP documentation
├── data/                  # LFS-2023 dataset
│   └── LFS-2023.csv
├── model/                 # Trained clustering models
│   └── skilldev_model.pkl
├── examples/              # Example scripts
│   ├── init_database.py   # Database initialization example
│   └── query_walking_difficulties.py
├── agents/                # Multi-agent system
├── dataloader/            # Data loading utilities
├── service/               # Business logic services
├── ui/                    # User interface (Streamlit)
└── requirement.txt        # Python dependencies
```

---

## 🔧 Configuration

### Environment Variables
- `HUGGINGFACE_TOKEN` (required): Your Hugging Face API token

### Data Sources
- **CSV**: `data/LFS-2023.csv` (default)
- **Model**: `model/skilldev_model.pkl` (optional, includes clusters)

---

## 📊 Features

✅ **AI-Powered Analysis** - DeepSeek-R1 via Hugging Face API  
✅ **Natural Language Queries** - Ask questions in plain English  
✅ **LFS-2023 Expert** - Pre-configured with survey metadata  
✅ **Cluster Analysis** - Automatic cluster detection & analysis  
✅ **Secure** - Environment variable-based token management  

---

## 🛠️ Development

### Install Development Dependencies
```bash
pip install -r requirement.txt
```

### Project Components

- **NLP Engine**: `NLP/NLP.py` - LlamaIndex + Hugging Face integration
- **Data Loaders**: `dataloader/` - CSV and database loading
- **Services**: `service/` - Business logic layer
- **UI**: `ui/` - Streamlit web interface
- **Agents**: `agents/` - Multi-agent orchestration

---

## ⚠️ Troubleshooting

**"Query engine not initialized"**
- Ensure `HUGGINGFACE_TOKEN` is set
- Restart terminal after setting token
- Check internet connection

**"No data loaded"**
- Verify `data/LFS-2023.csv` exists
- Or specify correct model path with `--model`

**Import errors**
```bash
pip install -r requirement.txt
```

**Token errors**
- Get new token: https://huggingface.co/settings/tokens
- Ensure token has 'inference' permission
- Check for typos in environment variable

---

## 📝 License

Data Science Group Project - Academic Research

---

## 🤝 Contributing

This is an academic project. For questions or collaboration, please contact the project team.

---

**Documentation:**
- [NLP Engine Documentation](NLP/README.md)
- [Requirements](requirement.txt)

**Quick Links:**
- Hugging Face: https://huggingface.co
- DeepSeek-R1: https://huggingface.co/deepseek-ai/DeepSeek-R1
