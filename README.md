# DSGP — Data Science Group Project

## Sri Lanka District Decision-Support & Governance Platform

An AI-powered social-welfare analytics platform for Sri Lanka, covering **poverty analysis**, **child protection**, and **mental health** — built with a multi-agent LangChain architecture, Groq Cloud LLM inference, machine-learning clustering, and an interactive Streamlit dashboard.

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Multi-Agent Architecture** | LangChain-based coordinator, NLP recommendation, and insight-generation agents |
| **LLM-Powered Queries** | Groq Cloud API with Llama 3.3 70B for natural-language data analysis |
| **Smart Resource Allocation** | Multi-factor need scoring (income, education, disability, informality, sector) with outlier guardrails |
| **NLP Clustering Engine** | Zero-shot classification (BART-large-MNLI) + semantic similarity (MiniLM) for cluster analysis |
| **SkillDev ML Model** | KMeans clustering (k=4) identifying intervention-based workforce segments |
| **Interactive Dashboard** | Streamlit UI with Plotly charts, KPI metrics, and trend visualizations |
| **NL-to-SQL** | Natural language to SQL query generation for the LFS-2023 database |
| **Typed Signal Contracts** | Pydantic models for inter-agent communication |

---

##  Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit UI (ui/)                    │
│         home.py → poverty / childcase / mentalhealth    │
├─────────────────────────────────────────────────────────┤
│                  Service Layer (service/)                │
│               RecommendationService (facade)            │
├───────────────────┬─────────────────────────────────────┤
│   CoordinatorAgent│       InsightGeneratorAgent          │
│   (LangChain)     │       (trend & demographic data)     │
├───────────────────┤                                      │
│ NLPRecommendation │  Signals (Pydantic I/O contracts)    │
│ Agent (embeddings)│                                      │
├───────────────────┴─────────────────────────────────────┤
│            Data Layer                                    │
│  PovertyDataLoader · PovertyInsightsDataLoader           │
│  DatabaseManager (SQLite) · SQLQueryGenerator            │
├─────────────────────────────────────────────────────────┤
│            NLP / ML Engines                              │
│  LLMQueryEngine (Groq Cloud API + LlamaIndex)           │
│  NLPClusterQueryEngine (local transformers models)      │
│  SkillDev KMeans Model (scikit-learn)                   │
└─────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Step 1: Get Your Groq Cloud API Key

1. Go to: https://console.groq.com/keys
2. Sign in or create a free account
3. Click **"Create API Key"**
4. Name it (e.g., `DSGP-NLP`)
5. **Copy the API key** — it starts with `gsk_`

### Step 2: Set the Environment Variable

**Option A — Single Session (Quick Test)**

Open PowerShell in the project folder and run:
```powershell
$env:GROQ_API_KEY = "gsk_YOUR_KEY_HERE"
python3.10 NLP/NLP.py --model model/skilldev_model.pkl
```

**Option B — Batch File (Recommended)**

1. Open `run_nlp.bat` in a text editor
2. Find the line:
   ```bat
   REM set GROQ_API_KEY=gsk_YOUR_KEY_HERE
   ```
3. Remove `REM` and replace with your actual key:
   ```bat
   set GROQ_API_KEY=gsk_YOUR_KEY_HERE
   ```
4. Save and double-click `run_nlp.bat` to run

**Option C — Permanent Environment Variable (Best for Development)**

1. Open *Windows Settings → System → About → Advanced system settings*
2. Click **Environment Variables**
3. Under *User variables*, click **New**
4. Variable name: `GROQ_API_KEY`
5. Variable value: `gsk_YOUR_KEY_HERE`
6. Click OK, then **restart your terminal/PowerShell**

### Step 3: Install Dependencies

```bash
pip install -r requirement.txt
```

> **Note:** Some packages (`torch`, `sentence-transformers`, `transformers`, `langchain-core`, `scipy`) may not be listed in `requirement.txt`. Install them separately if needed.

Verify key packages are installed:
```powershell
pip list | findstr llama-index
```

You should see `llama-index`, `llama-index-llms-groq`, and `llama-index-experimental`. If not, re-run `pip install -r requirement.txt`.

### Step 4: Launch the Dashboard

```bash
streamlit run ui/home.py
```

### Step 5: Run the NLP CLI Engine

```bash
python NLP/NLP.py --model model/skilldev_model.pkl
```

Or via the batch launcher:
```cmd
run_nlp.bat
```

On success you should see:
```
 PandasQueryEngine ready (general queries)
 Direct LLM ready (resource allocation)
```

---


---

##  Project Structure

```
DSGP/
├── agents/                        # Multi-agent system (LangChain)
│   ├── base_agent.py              # Abstract agent contract
│   ├── coordinator_agent.py       # Central orchestration agent
│   ├── orchestrator.py            # Lightweight alternative orchestrator
│   ├── nlp_recommendation_agent.py # Sentence-transformer recommendations
│   └── insight_generator_agent.py # Poverty trend & demographic insights
│
├── signals/                       # Pydantic signal contracts
│   ├── nlp_signals.py             # NLPQuerySignal, RecommendationSignal
│   ├── child_nlp_signals.py       # Child protection signals
│   ├── mental_health_nlp_signals.py
│   └── insight_signals/
│       └── poverty_insight_signals.py
│
├── NLP/                           # NLP & LLM query engines
│   ├── NLP.py                     # LLMQueryEngine (Groq Cloud API) + NLPClusterQueryEngine
│   └── README.md                  # NLP module documentation
│
├── ML/                            # Machine learning
│   └── skilldev.ipynb             # SkillDev KMeans clustering notebook
│
├── database/                      # Database layer
│   ├── db_setup.py                # SQLite database manager
│   └── sql_generator.py           # NL-to-SQL query generator
│
├── dataloader/                    # Data loading utilities
│   ├── poverty_data_loader.py     # Poverty + demographic data loader
│   ├── child_case_data_loader.py  # Child protection (stub)
│   ├── mental_helalth_data_loader.py # Mental health (stub)
│   └── insight/
│       ├── poverty_insights.py    # Poverty insight data loader
│       ├── child_cases_insights.py
│       └── mental_health_insights.py
│
├── service/                       # Business logic layer
│   ├── recommendation_service.py  # Facade over CoordinatorAgent
│   ├── child_protection_service.py
│   └── mental_health_service.py
│
├── ui/                            # Streamlit web interface
│   ├── home.py                    # Main entry point (3-card dashboard)
│   └── pages/
│       ├── poverty.py             # Poverty analysis dashboard
│       ├── childcase.py           # Child protection (in progress)
│       └── mentalhealth.py        # Mental health (in progress)
│
├── api/controller/                # REST API (currently inactive)
│   └── poverty_controller.py      # FastAPI endpoint (commented out)
│
├── data/                          # Datasets
│   └── LFS-2023.csv              # Sri Lanka Labor Force Survey 2023
│
├── model/                         # Trained models
│   ├── poverty_model.pkl          # Sentence-transformer embeddings
│   └── skilldev_model.pkl         # SkillDev KMeans cluster model
│
├── examples/                      # Example scripts
│   ├── init_database.py
│   └── query_walking_difficulties.py
│
├── requirement.txt                # Python dependencies
├── run_nlp.bat                    # Windows batch launcher
└── README.md
```

---

## 🔧 Configuration

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `GROQ_API_KEY` | Groq Cloud API key for Llama 3.3 70B LLM inference |

### Data Sources

| File | Description |
|------|-------------|
| `data/LFS-2023.csv` | Sri Lanka Labor Force Survey 2023 |
| `data/Povertylines.xlsx` | District-level poverty line data |
| `data/demographic_district_wise.xlsx` | District demographics (population, area) |

### Models
### Models
 
| File | Description |
|------|-------------|
| `model/skilldev_model.pkl` | KMeans clustering model (4 workforce segments) |
| `model/poverty_model.pkl` | Sentence-transformer for district embeddings |
| `model/poverty_risk_model.pkl` | Risk scoring model for poverty classification |
| `model/child_case_nlp.pkl` | NLP model for child protection case analysis |
| `model/child_case_risk_model.pkl` | Risk scoring model for child welfare cases |
| `model/resource allocation models/poverty_risk_model.pkl` | Poverty risk model used in resource allocation pipeline |
| `model/resource allocation models/child_welfare_pipeline.pkl` | End-to-end child welfare resource allocation pipeline |


---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **LLM / NLP** | Groq Cloud API (Llama 3.3 70B), LlamaIndex, PyTorch Transformers (local), sentence-transformers |
| **ML** | scikit-learn, PyTorch, SciPy |
| **Agents** | LangChain Core (Runnable interface) |
| **Data** | Pandas, NumPy, SQLite, openpyxl |
| **Visualization** | Plotly, Streamlit |
| **Contracts** | Pydantic |

---

## 📋 Implementation Status

| Module | Status                                              |
|--------|-----------------------------------------------------|
| Poverty analysis pipeline | Complete — full NLP → recommendation → insight → UI flow |
| NLP / LLM query engine | Complete — Groq LLM + cluster analysis modes        |
| SkillDev ML clustering | Complete — 4-cluster workforce segmentation         |
| Database & NL-to-SQL | Complete — SQLite + pattern-based SQL generation    |
| Streamlit dashboard | Complete — poverty dashboard with Plotly charts     |
| Child protection pipeline | In Progress — UI scaffold exists, data pipeline stubbed |
| FastAPI REST endpoint | Complete — teh backend is working using fast api    | |

---

## ⚠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| `GROQ_API_KEY not found` | Set the environment variable — see Step 2 above |
| `Query engine not initialized` | Ensure `GROQ_API_KEY` is set, restart terminal, check internet |
| Invalid API key | Regenerate at https://console.groq.com/keys |
| `No data loaded` | Verify `data/LFS-2023.csv` exists |
| `No module named 'llama_index'` | Run `pip install -r requirement.txt` |
| Import errors | Install missing packages: `torch`, `transformers`, `sentence-transformers`, `langchain-core` |
| Model file not found | Ensure `model/skilldev_model.pkl` and `model/poverty_model.pkl` exist; or run with `--csv data/LFS-2023.csv` |

---

## 📝 License

Data Science Group Project — Academic Research

---

## 🤝 Contributing

This is an academic project. For questions or collaboration, please contact the project team.

---

**External Links:**
- [Groq Cloud Console](https://console.groq.com)
- [Groq Service Status](https://status.groq.com)
- [LlamaIndex Docs](https://docs.llamaindex.ai)
- [LangChain Docs](https://python.langchain.com)

> **Security Reminder:** Never commit API keys to Git! The `.gitignore` is already configured to exclude them.