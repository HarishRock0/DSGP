# DSGP — Data Science Group Project

## Sri Lanka District Decision-Support & Governance Platform

An AI-powered social-welfare analytics platform for Sri Lanka, covering **poverty analysis**, **child protection**, and **mental health** — built with a multi-agent LangChain architecture, Groq Cloud LLM inference, machine-learning clustering, and an interactive Streamlit dashboard.

---

## 📊 Key Features

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

## 🏗️ Architecture

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

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirement.txt
```

> **Note:** Some packages used in the project (`torch`, `sentence-transformers`, `transformers`, `langchain-core`, `scipy`) may not be in `requirement.txt` — install them separately if needed.

### 2. Set Up Environment Variables

| Variable | Required For | How to Get |
|----------|-------------|------------|
| `GROQ_API_KEY` | LLM queries (Llama 3.3 70B) | https://console.groq.com/keys |

**PowerShell (quick test):**
```powershell
$env:GROQ_API_KEY = "gsk_YOUR_KEY_HERE"
```

**Permanent (Windows):**  
Set them in *System → Advanced system settings → Environment Variables*.

📖 See [SETUP_GUIDE.md](SETUP_GUIDE.md) for detailed setup instructions.

### 3. Launch the Dashboard

```bash
streamlit run ui/home.py
```

### 4. Run the NLP CLI Engine

```bash
python NLP/NLP.py --model model/skilldev_model.pkl
```

Or via the batch launcher:
```cmd
run_nlp.bat
```

---

## 💬 NLP Query Engine Usage

The CLI supports two engines that are selected automatically based on query type:

### LLM Engine (Groq Cloud API — Llama 3.3 70B)

General data analysis and resource allocation queries:

```
📝 Your query: How many people have vision difficulties?
📝 Your query: What is the average income by district?
📝 Your query: I have 100 taxis, whom should I give them to?
📝 Your query: Compare employment rates by gender
```

### Cluster Engine (Local Transformers — zero-shot + semantic)

Cluster analysis using the trained SkillDev model:

```
📝 Your query: /clusters
📝 Your query: /compare
📝 Your query: /insights
📝 Your query: What are the characteristics of each cluster?
```

### SkillDev Cluster Segments

| Cluster | Label | Description |
|---------|-------|-------------|
| 0 | High Skill Gap | Needs Job Matching |
| 1 | Digitally Excluded | Needs Tech Training |
| 2 | Economically Vulnerable | Needs Social Safety Net |
| 3 | Stable Workforce | Needs Leadership / Advanced Skills |

---

## 📂 Project Structure

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
├── SETUP_GUIDE.md                 # Detailed setup instructions
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

| File | Description |
|------|-------------|
| `model/skilldev_model.pkl` | KMeans clustering model (4 workforce segments) |
| `model/poverty_model.pkl` | Sentence-transformer for district embeddings |

---

## 🧪 Testing

```bash
# Verify imports and Groq API connectivity
python test_groq.py

# Quick import check
python quick_test.py

# End-to-end resource allocation test
python test_query.py
```

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

## ⚠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| `GROQ_API_KEY not found` | Set the `GROQ_API_KEY` environment variable — see [Setup Guide](SETUP_GUIDE.md) |
| `Query engine not initialized` | Ensure `GROQ_API_KEY` is set, restart terminal, check internet |
| `No data loaded` | Verify `data/LFS-2023.csv` exists |
| Import errors | Run `pip install -r requirement.txt` and install missing packages (`torch`, `transformers`, `sentence-transformers`, `langchain-core`) |
| Model file not found | Ensure `model/skilldev_model.pkl` and `model/poverty_model.pkl` exist |

---

## 📋 Implementation Status

| Module | Status |
|--------|--------|
| Poverty analysis pipeline | ✅ Complete — full NLP → recommendation → insight → UI flow |
| NLP / LLM query engine | ✅ Complete — Groq LLM + cluster analysis modes |
| SkillDev ML clustering | ✅ Complete — 4-cluster workforce segmentation |
| Database & NL-to-SQL | ✅ Complete — SQLite + pattern-based SQL generation |
| Streamlit dashboard | ✅ Complete — poverty dashboard with Plotly charts |
| Child protection pipeline | 🚧 In Progress — UI scaffold exists, data pipeline stubbed |
| Mental health pipeline | 🚧 In Progress — UI scaffold exists, data pipeline stubbed |
| FastAPI REST endpoint | ⏸️ Paused — code exists but is commented out |

---

## 📝 License

Data Science Group Project — Academic Research

---

## 🤝 Contributing

This is an academic project. For questions or collaboration, please contact the project team.

---

**Documentation:**
- [NLP Engine Documentation](NLP/README.md)
- [Setup Guide](SETUP_GUIDE.md)
- [Requirements](requirement.txt)

**External Links:**
- [Groq Cloud Console](https://console.groq.com)
- [LlamaIndex](https://docs.llamaindex.ai)
- [LangChain](https://python.langchain.com)
