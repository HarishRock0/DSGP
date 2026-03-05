# NLP Query Engine - LFS-2023 Data Analysis

Clean, production-ready LLM query engine for analyzing Sri Lanka's Labor Force Survey 2023 data using Llama 3.3 70B via Groq Cloud API.

## Features

✅ **AI-Powered Analysis** - Uses Llama 3.3 70B model via Groq Cloud API  
✅ **LFS-2023 Expert** - Pre-configured with comprehensive column descriptions  
✅ **Cluster Analysis** - Automatically detects and analyzes cluster data  
✅ **Natural Language Queries** - Ask questions in plain English  
✅ **Secure** - Uses environment variables for API keys  

## Setup

### 1. Install Dependencies
```bash
pip install -r requirement.txt
```

### 2. Set Groq API Key
```bash
# Windows PowerShell
$env:GROQ_API_KEY = "gsk_YOUR_KEY_HERE"

# Or permanent (Windows)
setx GROQ_API_KEY "gsk_YOUR_KEY_HERE"
# Then restart terminal
```

Get your API key from: https://console.groq.com/keys

### 3. Run
```bash
python NLP/NLP.py
```

Or with a specific model file:
```bash
python NLP/NLP.py --model path/to/skilldev_model.pkl
```

## Usage

### Interactive Mode
```
💬 LLM Query Engine - Llama 3.3 70B via Groq Cloud API
Ask questions about your LFS-2023 data!

📝 Your query: How many people have vision difficulties?
📝 Your query: What is the average income by district?
📝 Your query: Compare employment rates by gender
📝 Your query: quit
```

### Example Questions

**Demographics:**
- "How many people are in each ethnic group?"
- "What's the gender distribution?"
- "Show marital status breakdown"

**Disabilities:**
- "How many people have vision difficulties?"
- "What percentage have severe mobility issues?"
- "Compare disability rates by sector (urban/rural)"

**Employment:**
- "What's the employment rate?"
- "How many people work informally?"
- "Compare average hours worked by gender"

**Income & Poverty:**
- "What's the average monthly income?"
- "Show income distribution by district"
- "How many people earn below 30000?"

**Clusters (if model with clusters):**
- "What are the characteristics of each cluster?"
- "How many people in cluster 0?"
- "Compare clusters by employment status"

## Architecture

```
LLMQueryEngine
├── Data Loading (model pickle or CSV)
├── Groq Cloud API Configuration
│   ├── Model: Llama 3.3 70B Versatile
│   ├── API key from environment variable
│   └── Online inference via Groq Cloud
├── LlamaIndex PandasQueryEngine
│   ├── Direct DataFrame access
│   ├── Column descriptions
│   └── Analysis guidelines
└── Interactive Query Loop
```

## Code Structure

- **COLUMN_DESCRIPTIONS**: Comprehensive LFS-2023 column mappings
- **COLUMN_VALUE_SCALE**: Disability/difficulty scale values
- **EMPLOYMENT_STATUS**: Employment type mappings
- **SECTOR_MAP**: Urban/Rural/Estate mappings
- **LLMQueryEngine**: Main query engine class
  - `__init__()`: Load data and initialize API
  - `analyze_data()`: Process natural language queries

## Configuration

### Environment Variables
- `GROQ_API_KEY` (required): Your Groq Cloud API key

### Command Line Arguments
- `--model`: Path to model pickle file (optional)
- `--csv`: Path to CSV file (default: data/LFS-2023.csv)

## Dependencies

Core packages:
- `llama-index` - LLM framework
- `llama-index-llms-groq` - Groq Cloud API integration
- `llama-index-experimental` - PandasQueryEngine
- `groq` - Groq API client
- `pandas`, `numpy` - Data processing
- `streamlit` - Web interface (optional)

See [requirement.txt](../requirement.txt) for full list.

## Security

🔒 **Never commit API tokens**
- Tokens are read from environment variables
- No hardcoded credentials in code
- `.gitignore` prevents `.env` files from being committed

## Performance

- **Model**: Llama 3.3 70B Versatile (via Groq Cloud)
- **No local download**: Runs entirely via Groq API
- **Internet required**: Yes (for API calls)
- **Response time**: 2-10 seconds depending on query complexity

## Troubleshooting

### "Query engine not initialized"
- Check `GROQ_API_KEY` is set
- Verify internet connection
- Ensure Groq API key is valid

### "No data loaded"
- Verify model file exists at specified path
- Or ensure CSV file exists at data/LFS-2023.csv

### Import errors
```bash
pip install -r requirement.txt
```

## License

Part of DSGP - Data Science Group Project
