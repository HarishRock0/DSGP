# NLP Query Engine - LFS-2023 Data Analysis

Clean, production-ready LLM query engine for analyzing Sri Lanka's Labor Force Survey 2023 data using DeepSeek-R1 via Hugging Face API.

## Features

✅ **AI-Powered Analysis** - Uses DeepSeek-R1 model via Hugging Face Inference API  
✅ **LFS-2023 Expert** - Pre-configured with comprehensive column descriptions  
✅ **Cluster Analysis** - Automatically detects and analyzes cluster data  
✅ **Natural Language Queries** - Ask questions in plain English  
✅ **Secure** - Uses environment variables for API tokens  

## Setup

### 1. Install Dependencies
```bash
pip install -r requirement.txt
```

### 2. Set Hugging Face Token
```bash
# Windows PowerShell
setx HUGGINGFACE_TOKEN "your_token_here"

# Then restart terminal
```

Get your token from: https://huggingface.co/settings/tokens

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
💬 LLM Query Engine - DeepSeek-R1 via Hugging Face API
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
├── Hugging Face API Configuration
│   ├── Model: deepseek-ai/DeepSeek-R1
│   ├── Token from environment variable
│   └── Online inference (no local download)
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
- `HUGGINGFACE_TOKEN` (required): Your Hugging Face API token

### Command Line Arguments
- `--model`: Path to model pickle file (optional)
- `--csv`: Path to CSV file (default: data/LFS-2023.csv)

## Dependencies

Core packages:
- `llama-index` - LLM framework
- `llama-index-llms-huggingface-api` - Hugging Face integration
- `llama-index-experimental` - PandasQueryEngine
- `pandas`, `numpy` - Data processing
- `streamlit` - Web interface (optional)

See [requirement.txt](../requirement.txt) for full list.

## Security

🔒 **Never commit API tokens**
- Tokens are read from environment variables
- No hardcoded credentials in code
- `.gitignore` prevents `.env` files from being committed

## Performance

- **Model**: DeepSeek-R1 (online API)
- **No local download**: Runs entirely via API
- **Internet required**: Yes (for API calls)
- **Response time**: 2-10 seconds depending on query complexity

## Troubleshooting

### "Query engine not initialized"
- Check `HUGGINGFACE_TOKEN` is set
- Verify internet connection
- Ensure token has 'inference' permission

### "No data loaded"
- Verify model file exists at specified path
- Or ensure CSV file exists at data/LFS-2023.csv

### Import errors
```bash
pip install -r requirement.txt
```

## License

Part of DSGP - Data Science Group Project
