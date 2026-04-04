# 🤖 LFS Agentic AI — Sri Lanka Labour Force Survey 2023

An **Agentic AI system** for natural language analysis and resource allocation over Sri Lanka's Labour Force Survey 2023 (LFS-2023) dataset — 18,937 respondents × 134 features across all 9 provinces.

Designed for **welfare societies and policy makers** to identify vulnerable populations, allocate resources, and generate labour market insights — all through plain English queries.

---

## 🏗️ Architecture

```
NLP.py  (Entry Point)
│
├── GPU Detection (CUDA auto-detected, falls back to CPU)
├── Loads skilldev_model.pkl  ← Pre-trained K-Means clustering model
│
└── LFSAgent  (Engines/agent.py)
      │
      ├── [STEP 1] Conversational Guard
      │     └── Handles greetings / help / meta queries directly
      │
      ├── [STEP 2] LLAMA GATE  ← Engines/NLPC.py
      │     ├── Llama 3.2 3B (Ollama, LOCAL — offline capable)
      │     ├── Classifies query into 7 intent categories
      │     ├── Calculates tool alignment score (0–1)
      │     └── Keyword fallback when Ollama is unavailable
      │
      ├── [STEP 3a] Fast Path — Tool-Aligned (score > 0.4)
      │     └── Routes directly to the best-fit @tool function
      │
      ├── [STEP 3b] Fallback — Out-of-Scope (score ≤ 0.4)
      │     └── Groq / ReActAgent (cloud LLM, autonomous tool selection)
      │
      ├── LLMQueryEngine  (Engines/LLMQ.py)
      │     ├── Groq Cloud API — Llama 3.3 70B Versatile
      │     ├── PandasQueryEngine — general statistical queries
      │     ├── handle_allocation() — CORE resource allocation pipeline
      │     │     ├── LLM selects best vulnerability cluster
      │     │     ├── Need score per person (6 weighted factors)
      │     │     ├── Rank + select Top-N beneficiaries
      │     │     ├── Outlier guardrail (centroid distance trust level)
      │     │     └── LLM formats & explains final output
      │     ├── compare_clusters()
      │     ├── get_insights()
      │     └── analyze_data()
      │
      └── Tool Registry  (Engines/tools.py)
            └── 8 @tool functions → LlamaIndex FunctionTools
```

### Two-Tier LLM Strategy

| Tier | Engine | Role | Connectivity |
|---|---|---|---|
| **Tier 1** | Llama 3.2 3B via Ollama (local) | Intent classification, tool routing | ✅ Offline |
| **Tier 2** | Groq Cloud — Llama 3.3 70B | Complex reasoning, output formatting | 🌐 Internet needed |

---

## 🛠️ Available Tools (8 Registered)

| Tool | Trigger Example |
|---|---|
| `allocate_resources` | *"Give 50 laptops to the most vulnerable workers"* |
| `compare_clusters` | *"How do the population groups differ?"* |
| `query_cluster` | *"Tell me about cluster 2"* |
| `get_insights` | *"What are the key employment trends?"* |
| `analyze_demographics` | *"Show age and gender distribution"* |
| `find_outliers` | *"Find unusual or anomalous records"* |
| `get_cluster_stats` | *"How is the data split across clusters?"* |
| `get_data_schema` | *"What columns are in the dataset?"* |

---

## 👥 Population Clusters (K-Means, pre-trained)

| Cluster | Label | Primary Welfare Focus |
|---|---|---|
| 0 | **Economically Vulnerable** — Needs Social Safety Net | Income support, food rations |
| 1 | **High Skill Gap** — Needs Job Matching | Vocational training, job placement |
| 2 | **Digitally Excluded** — Needs Tech Training | Devices, internet, digital literacy |
| 3 | **Stable Workforce** — Needs Leadership/Advanced Skills | Advanced upskilling |

---

## 📊 Need Score Formula (Resource Allocation)

Every person is scored 0–100 across 6 weighted factors:

| Factor | Weight | Signal Used |
|---|---|---|
| Income | 40% | Q45_A_1 (monthly salary in LKR) |
| Education | 15% | EDU (grade level) |
| Disability | 15% | P15–P20 (6 disability dimensions) |
| Informality | 10% | Q47 (workplace formality) + Q46 (EPF/ETF) |
| Sector deprivation | 10% | SECTOR (Estate > Rural > Urban) |
| Item-specific | 10% | e.g. computer literacy for laptops, mobility for transport |

Higher score = greater need = higher allocation priority.

---

## ⚙️ Setup

### Prerequisites
- Python 3.9+
- [Ollama](https://ollama.com) installed and running locally
- Groq API key (free at https://console.groq.com/keys)
- `skilldev_model.pkl` + `scaler.pkl` in the `model/` directory

### 1. Install Dependencies
```bash
pip install -r requirement.txt
```

### 2. Pull Llama 3.2 3B (for local intent detection)
```bash
ollama pull llama3.2:3b
ollama serve
```

### 3. Set Groq API Key
```bash
# Windows PowerShell
$env:GROQ_API_KEY = "gsk_YOUR_KEY_HERE"

# Or create a .env file in the project root:
# GROQ_API_KEY=gsk_YOUR_KEY_HERE
```

### 4. Run
```bash
# From project root
python NLP/NLP.py
```

---

## 💬 Usage Examples

### Resource Allocation (Welfare Societies)
```
📝 Your query: Give 50 sewing machines to the most vulnerable women in estate sector
📝 Your query: Allocate 100 food rations to the neediest households
📝 Your query: Who should receive the 30 laptops for digital upskilling?
📝 Your query: Identify 20 beneficiaries for vocational training scholarships
```

### Cluster & Demographics
```
📝 Your query: Compare all population clusters
📝 Your query: Tell me about the digitally excluded group
📝 Your query: Show employment statistics for the estate sector
📝 Your query: What is the gender distribution in cluster 1?
```

### Insights & Analysis
```
📝 Your query: What are the main skill gaps in the workforce?
📝 Your query: Insights on disability and employment
📝 Your query: Which districts have the highest poverty concentration?
📝 Your query: Find outliers or anomalous cases in the data
```

### Session Control
```
reset    → Clear conversation memory for a fresh session
quit     → Exit the application
```

---

## 📁 Code Structure

```
NLP/
├── NLP.py                  ← Entry point: GPU detection, model loading, chat loop
├── README.md
└── Engines/
    ├── agent.py            ← LFSAgent: 2-tier routing logic (LLAMA gate + ReActAgent)
    ├── LLMQ.py             ← LLMQueryEngine: Groq LLM + resource allocation pipeline
    ├── NLPC.py             ← NLPClusterQueryEngine: Llama 3.2 intent + cluster operations
    ├── tools.py            ← 8 @tool functions registered as LlamaIndex FunctionTools
    └── constants.py        ← Shared lookup tables (columns, districts, sectors, etc.)
```

---

## 🗄️ Dataset Reference (LFS-2023 Key Columns)

| Column | Description |
|---|---|
| `SECTOR` | 1=Urban, 2=Rural, 3=Estate |
| `DISTRICT` | 25 districts (11=Colombo … 92=Kegalle) |
| `AGE`, `SEX` | Age in years; 1=Male, 2=Female |
| `EDU` | 0=No schooling → 16=Postgraduate |
| `Q45_A_1` | Monthly income/salary in LKR |
| `Q16` | Employment: 1=Employee, 2=Employer, 3=Self-employed, 4=Family worker |
| `Q47` | 1=Formal workplace, 2=Informal |
| `Q60A`, `Q60B` | Computer literacy; Smartphone literacy |
| `Q61` | Internet use in last 12 months |
| `P15–P20` | Disability: Vision, Hearing, Mobility, Cognition, Self-care, Communication |
| `cluster_id` | K-Means cluster assignment (0–3) |
| `cluster_label` | Human-readable cluster name |
| `distance_to_center` | Euclidean distance to cluster centroid (used for outlier guardrail) |

---

## ⚠️ Known Limitations

- **Income data (~86% missing)**: `Q45_A_1` is only populated for employed respondents (~2,600 of 18,937). Records with no income are scored using employment status as a proxy.
- **No skill assessments**: LFS does not include direct skill test scores — skill gap clusters are inferred from education + digital literacy proxies.
- **Llama 3.2 accuracy**: Intent classification uses a 3B parameter model — works very well for clear queries, but ReActAgent remains as a fallback.
- **Offline mode**: Without Ollama running, automatic keyword fallback activates but with reduced routing accuracy.

---

## 🔧 Troubleshooting

| Problem | Fix |
|---|---|
| `GROQ_API_KEY not set` | Set env variable or add to `.env` file in project root |
| `Ollama not reachable` | Run `ollama serve` in a separate terminal |
| `Model 'llama3.2:3b' not found` | Run `ollama pull llama3.2:3b` |
| `skilldev_model.pkl not found` | Place model file in `model/skilldev_model.pkl` |
| `scaler.pkl not found` | Outlier detection will use approximate scaling — accuracy reduced |
| `Import errors` | Run `pip install -r requirement.txt` |

---

## 🔒 Security

- API keys are read from environment variables — **never hardcoded**
- `.env` files excluded from version control via `.gitignore`
- Individual record data is exposed in CLI — add authentication before any web deployment

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `llama-index` | LLM agent framework + PandasQueryEngine |
| `llama-index-llms-groq` | Groq Cloud API integration |
| `llama-index-experimental` | PandasQueryEngine |
| `torch` | GPU detection + tensor operations |
| `requests` | Ollama REST API communication |
| `pandas`, `numpy` | Data processing |
| `scikit-learn` | K-Means, StandardScaler |
| `python-dotenv` | `.env` file loading |

See [requirement.txt](../requirement.txt) for full pinned versions.

---

## 📄 License

Part of DSGP — Data Science Group Project
