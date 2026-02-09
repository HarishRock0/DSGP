# DSGP - Data Science Group Project
## LFS-2023 Labor Force Survey Analysis with AI

AI-powered analysis of Sri Lanka's Labor Force Survey 2023 data using DeepSeek-R1 model via Hugging Face API.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirement.txt
```

### 2. Set Up Hugging Face API Token
Get your token from: https://huggingface.co/settings/tokens

**Windows (PowerShell):**
```powershell
setx HUGGINGFACE_TOKEN "your_token_here"
```

Then **restart your terminal**.

### 3. Run the NLP Query Engine
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
