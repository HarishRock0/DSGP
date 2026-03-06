import os
import pickle
import pandas as pd
import numpy as np
import warnings
import torch
import sys
import argparse
import re
import json

# LlamaIndex imports for pandas query engine
from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.core import Settings
try:
    from llama_index.llms.groq import Groq
    _LLAMA_INDEX_IMPORT_ERROR = None
except ImportError as import_error:
    Groq = None
    _LLAMA_INDEX_IMPORT_ERROR = import_error

# Transformers imports for clustering engine (optional)
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
except ImportError:
    pipeline = None
    AutoTokenizer = None
    AutoModelForSequenceClassification = None

warnings.filterwarnings('ignore')

# Comprehensive column descriptions for LFS-2023 dataset matching actual CSV column names
COLUMN_DESCRIPTIONS = {
    # Identification & Location
    "YEAR": "Survey year (2023)",
    "MONTH": "Survey month (1-12)",
    "SECTOR": "Residential Sector (1=Urban, 2=Rural, 3=Estate)",
    "DISTRICT": "Administrative District code (11=Colombo, 12=Gampaha, 13=Kalutara, 21=Kandy, 22=Matale, 23=Nuwara Eliya, 31=Galle, 32=Matara, 33=Hambantota, 41=Jaffna, 42=Kilinochchi, 43=Mannar, 44=Vavuniya, 45=Mullaitivu, 51=Batticaloa, 52=Ampara, 53=Trincomalee, 61=Kurunegala, 62=Puttalam, 71=Anuradhapura, 72=Polonnaruwa, 81=Badulla, 82=Monaragala, 91=Ratnapura, 92=Kegalle)",
    "PSU": "Primary Sampling Unit",
    "HUNIT": "Housing Unit number",
    "HHOLD": "Household number",
    "SERNO": "Serial number of person within household",

    # Demographics
    "RSHIP": "Relationship to head of household (1=Head, 2=Spouse, 3=Child, 4=Parent, 5=Other relative, 6=Non-relative)",
    "SEX": "Gender (1=Male, 2=Female)",
    "BYEAR": "Birth year",
    "BMONTH": "Birth month",
    "AGE": "Age in years (numeric)",
    "ETH": "Ethnic Group (1=Sinhala, 2=SL Tamil, 3=Indian Tamil, 4=Moor, 5=Malay, 6=Burgher, 9=Other)",
    "REL": "Religion (1=Buddhist, 2=Hindu, 3=Islam, 4=Roman Catholic, 5=Other Christian, 9=Other)",
    "MARITAL": "Marital Status (1=Never Married, 2=Married, 3=Widowed, 4=Divorced, 5=Separated)",
    "EDU": "Highest Education Level (0/19=No schooling, 1-10=Grade 1-10, 11=O/L, 12=Passed O/L, 13=A/L, 14=Passed A/L, 15=Degree, 16=Postgraduate)",
    "DEGREE": "Degree field of study (if applicable)",
    "CUEDU": "Currently in Education (1=Yes, 2=No)",

    # Literacy
    "SIN": "Sinhala Literacy (1=Can read/write, 2=Cannot read/write)",
    "TAMIL": "Tamil Literacy (1=Can read/write, 2=Cannot read/write)",
    "ENG": "English Literacy (1=Can read/write, 2=Cannot read/write)",

    # Disability/Difficulty Questions (P15-P20 all use same scale: 1=None, 2=Some, 3=A lot, 4=Cannot do)
    "P15": "Vision Difficulty - Even with glasses (1=None, 2=Some, 3=A lot, 4=Cannot do)",
    "P16": "Hearing Difficulty - Even with hearing aid (1=None, 2=Some, 3=A lot, 4=Cannot do)",
    "P17": "Mobility/Walking Difficulty - Walking or climbing steps (1=None, 2=Some, 3=A lot, 4=Cannot do)",
    "P18": "Cognitive Difficulty - Remembering or concentrating (1=None, 2=Some, 3=A lot, 4=Cannot do)",
    "P19": "Self-care Difficulty - Washing or dressing (1=None, 2=Some, 3=A lot, 4=Cannot do)",
    "P20": "Communication Difficulty - Using usual language (1=None, 2=Some, 3=A lot, 4=Cannot do)",
    "P21": "Education/Training Participation - Last 12 months",

    # Employment
    "Q2": "Work Activity - Did person work for pay/profit in last 7 days? (1=Yes, 2=No)",
    "Q8": "Occupation code - Main job/task performed (ISCO-08 coded, stored as string)",
    "Q16": "Employment Status (1=Employee public/private, 2=Employer, 3=Own account worker/self-employed, 4=Contributing family worker)",
    "Q20": "Hours Worked - Total actual hours per week at main job (underemployment if < 40)",
    "Q36": "Job Search - Looked for work in reference period (1=Yes, 2=No)",
    "Q43": "Availability - Available to start work (1=Yes, 2=No)",

    # Income & Poverty
    "Q45_A_1": "Monthly Income/Salary in LKR - stored as STRING with spaces for missing values. Must convert to numeric with pd.to_numeric(errors='coerce'). Only ~2600 of ~18937 rows have actual numeric values.",

    # Formality & Benefits
    "Q46": "EPF/ETF Benefits (1=Yes/formal, 2=No/informal)",
    "Q47": "Workplace Formality (1=Formal/Registered, 2=Informal/Not registered)",

    # Digital Skills
    "Q60A": "Computer Literacy (1=Can use, 2=Cannot use)",
    "Q60B": "Smartphone/Tablet (1=Can use, 2=Cannot use)",
    "Q61": "Internet Use - Used internet in last 12 months (1=Yes, 2=No)",
    "Q64": "Internet Use frequency",

    # Weighting
    "Annual_Factor": "Survey weight / Annual expansion factor for population estimates"
}

# Value scales for disability/difficulty questions
COLUMN_VALUE_SCALE = {
    1: "No difficulty/None",
    2: "Some difficulty/Minor",
    3: "A lot of difficulty/Major",
    4: "Cannot do at all/Severe"
}

# Employment status mapping
EMPLOYMENT_STATUS = {
    1: "Employee (public/private)",
    2: "Employer",
    3: "Own Account Worker (Self-employed)",
    4: "Contributing Family Worker"
}

# Sector mapping
SECTOR_MAP = {
    1: "Urban",
    2: "Rural",
    3: "Estate"
}

# District mapping
DISTRICT_MAP = {
    11: "Colombo", 12: "Gampaha", 13: "Kalutara",
    21: "Kandy", 22: "Matale", 23: "Nuwara Eliya",
    31: "Galle", 32: "Matara", 33: "Hambantota",
    41: "Jaffna", 42: "Kilinochchi", 43: "Mannar", 44: "Vavuniya", 45: "Mullaitivu",
    51: "Batticaloa", 52: "Ampara", 53: "Trincomalee",
    61: "Kurunegala", 62: "Puttalam",
    71: "Anuradhapura", 72: "Polonnaruwa",
    81: "Badulla", 82: "Monaragala",
    91: "Ratnapura", 92: "Kegalle"
}

# Ethnicity mapping
ETHNICITY_MAP = {
    1: "Sinhala", 2: "SL Tamil", 3: "Indian Tamil",
    4: "Moor", 5: "Malay", 6: "Burgher", 9: "Other"
}

# Religion mapping
RELIGION_MAP = {
    1: "Buddhist", 2: "Hindu", 3: "Islam",
    4: "Roman Catholic", 5: "Other Christian", 9: "Other"
}

# Marital status mapping
MARITAL_MAP = {
    1: "Never married", 2: "Married", 3: "Widowed",
    4: "Divorced", 5: "Separated"
}

# Province to districts mapping
PROVINCE_DISTRICTS = {
    "Western": [11, 12, 13],
    "Central": [21, 22, 23],
    "Southern": [31, 32, 33],
    "Northern": [41, 42, 43, 44, 45],
    "Eastern": [51, 52, 53],
    "North Western": [61, 62],
    "North Central": [71, 72],
    "Uva": [81, 82],
    "Sabaragamuwa": [91, 92]
}

# GPU Detection with detailed diagnostics
print("\n" + "="*60)
print("🔍 GPU Detection & Diagnostics")
print("="*60)
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Built: {torch.version.cuda if torch.version.cuda else 'NO (CPU-only PyTorch)'}")

if torch.cuda.is_available():
    print(f"🎮 GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"📊 CUDA Version: {torch.version.cuda}")
    print(f"🔢 GPU Count: {torch.cuda.device_count()}")
    DEVICE = "cuda"
    print("✅ Using GPU for inference")
else:
    print("💻 Running on CPU")
    print("\n⚠️ GPU NOT detected. Common fixes:")
    print("   1. Install PyTorch with CUDA support:")
    print("      pip uninstall torch")
    print("      pip install torch --index-url https://download.pytorch.org/whl/cu121")
    print("   2. Update NVIDIA drivers: https://www.nvidia.com/Download/index.aspx")
    print("   3. Verify CUDA installation: nvidia-smi")
    DEVICE = "cpu"
print("="*60 + "\n")


class SkillDev:
    """Minimal stub to support unpickling SkillDev instances."""
    pass

class NLPClusterQueryEngine:
    """
    NLP-based query engine that uses pretrained models to understand requests
    and access cluster data from the trained SkillDev model
    """
    
    def __init__(self, model_path):
        """Load the trained SkillDev model"""
        print("🔄 Loading trained clustering model...")
        with open(model_path, 'rb') as f:
            self.skilldev_model = pickle.load(f)
        
        self.df = self.skilldev_model.df
        self.kmeans = self.skilldev_model.kmeans
        self.features = self.skilldev_model.features
        
        print("✅ Model loaded successfully!")

        # Initialize pretrained NLP models
        print(f"\n🤖 Loading pretrained NLP models on {DEVICE.upper()}...")
        try:
            self.classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if torch.cuda.is_available() else -1
            )
            print(f"✅ Zero-shot classifier loaded on {DEVICE.upper()}")
        except Exception as e:
            print(f"⚠️ Could not load zero-shot classifier: {e}")
            self.classifier = None

        try:
            self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            self.model = AutoModelForSequenceClassification.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(DEVICE)
            print(f"✅ Semantic model loaded on {DEVICE.upper()}")
        except Exception as e:
            print(f"⚠️ Could not load semantic model: {e}")
            self.tokenizer = None
            self.model = None
    
    def understand_query(self, query):
        """Use NLP to understand user query and extract intent"""
        print(f"\n🔍 Analyzing query: '{query}'")
        
        if not self.classifier:
            print("⚠️ Classifier not available, using keyword matching")
            return self._keyword_intent(query)
        
        # Possible intents
        intents = [
            "find records in a specific cluster",
            "compare clusters",
            "analyze demographic patterns",
            "identify outliers",
            "get cluster statistics"
        ]
        
        try:
            result = self.classifier(query, intents, multi_class=False)
            top_intent = result['labels'][0]
            confidence = result['scores'][0]
            
            print(f"📌 Detected Intent: {top_intent}")
            print(f"💯 Confidence: {confidence:.2%}")
            
            return {
                'intent': top_intent,
                'confidence': confidence,
                'query': query
            }
        except Exception as e:
            print(f"⚠️ Intent detection failed: {e}")
            return self._keyword_intent(query)
    
    def _keyword_intent(self, query):
        """Fallback keyword-based intent detection"""
        query_lower = query.lower()
        
        if any(kw in query_lower for kw in ['cluster', 'group', 'segment']):
            intent = "find records in a specific cluster"
        elif any(kw in query_lower for kw in ['compare', 'difference', 'vs']):
            intent = "compare clusters"
        elif any(kw in query_lower for kw in ['pattern', 'analyze', 'demographic']):
            intent = "analyze demographic patterns"
        elif any(kw in query_lower for kw in ['outlier', 'extreme', 'unusual']):
            intent = "identify outliers"
        else:
            intent = "get cluster statistics"
        
        return {
            'intent': intent,
            'confidence': 0.5,
            'query': query
        }
    
    def query_clusters(self, query):
        """Execute query against the cluster data"""
        intent_result = self.understand_query(query)
        intent = intent_result['intent']
        
        print(f"\n⚙️ Executing query...")
        
        if "specific cluster" in intent:
            return self._get_cluster_records(query)
        elif "compare" in intent:
            return self._compare_clusters()
        elif "pattern" in intent:
            return self._analyze_patterns(query)
        elif "outlier" in intent:
            return self._find_outliers()
        else:
            return self._get_cluster_stats()
    
    def _get_cluster_records(self, query):
        """Get records from a specific cluster"""
        print("\n📋 Cluster Records:")
        
        # Extract cluster number from query if possible
        import re
        cluster_nums = re.findall(r'\d+', query)
        
        if cluster_nums:
            cluster_id = int(cluster_nums[0]) % self.skilldev_model.n_clusters
        else:
            cluster_id = 0
        
        cluster_data = self.df[self.df['cluster_id'] == cluster_id]
        
        print(f"Cluster {cluster_id}: {len(cluster_data)} records")
        print(cluster_data[self.features[:5]].head(10))
        
        return cluster_data
    
    def _compare_clusters(self):
        """Compare statistics across clusters"""
        print("\n📊 Cluster Comparison:")
        
        for cluster_id in range(self.skilldev_model.n_clusters):
            cluster_data = self.df[self.df['cluster_id'] == cluster_id]
            print(f"\nCluster {cluster_id}:")
            print(f"  Records: {len(cluster_data)}")
            print(f"  Mean values: {cluster_data[self.features[:3]].mean().round(2).to_dict()}")
    
    def _analyze_patterns(self, query):
        """Analyze demographic patterns in clusters"""
        print("\n🔬 Pattern Analysis:")
        
        # Show variance across clusters for each feature
        for feature in self.features[:5]:
            cluster_means = self.df.groupby('cluster_id')[feature].mean()
            print(f"\n{feature}:")
            print(cluster_means.round(2))
        
        return self.df.groupby('cluster_id')[self.features[:5]].mean()
    
    def _find_outliers(self):
        """Identify outlier records"""
        print("\n⚠️ Outlier Detection:")  
        
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.df[self.features])
        
        # Records with extreme values (|z-score| > 3)
        outliers = np.where((np.abs(X_scaled) > 3).any(axis=1))[0]
        
        print(f"Found {len(outliers)} outlier records")
        if len(outliers) > 0:
            print(self.df.iloc[outliers[:10]][self.features[:5]])
        
        return self.df.iloc[outliers]
    
    def _get_cluster_stats(self):
        """Get comprehensive cluster statistics"""
        print("\n📈 Cluster Statistics:")
        
        stats = {
            'Total Records': len(self.df),
            'Clusters': self.skilldev_model.n_clusters,
            'Distribution': self.df['cluster_id'].value_counts().to_dict()
        }
        
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        return stats
    
    def interactive_query(self):
        """Interactive query loop"""
        print("\n" + "="*60)
        print("💬 NLP Cluster Query Engine - Interactive Mode")
        print("="*60)
        print("Ask questions about your cluster data!")
        print("Examples:")
        print("  - 'Show records in cluster 0'")
        print("  - 'Compare all clusters'")
        print("  - 'What patterns exist in the data?'")
        print("  - 'Find outlier records'")
        print("  - 'Cluster statistics'")
        print("Type 'quit' to exit\n")
        
        while True:
            query = input("📝 Your query: ").strip()
            
            if query.lower() == 'quit':
                print("✅ Goodbye!")
                break
            
            if query:
                self.query_clusters(query)


class LLMQueryEngine:
    """
    LLM-powered query engine for Sri Lanka LFS-2023 data.

    PRIMARY USE CASE: Efficient resource allocation (identify beneficiaries, prioritize
    by need, allocate items).  Resource-allocation queries bypass PandasQueryEngine
    entirely — everything is pre-computed in Python and sent to the LLM for
    formatting only, guaranteeing 100 % accurate numbers.

    SECONDARY: General demographic / statistical queries routed through
    PandasQueryEngine.
    """

    # Education level labels used in scoring & display
    EDU_MAP = {
        0: 'No schooling', 1: 'Grade 1', 2: 'Grade 2', 3: 'Grade 3',
        4: 'Grade 4', 5: 'Grade 5', 6: 'Grade 6', 7: 'Grade 7',
        8: 'Grade 8', 9: 'Grade 9', 10: 'Grade 10', 11: 'O/L',
        12: 'Passed O/L', 13: 'A/L', 14: 'Passed A/L', 15: 'Degree',
        16: 'Postgraduate', 19: 'No schooling',
    }

    def __init__(self, model_path=None, df=None, csv_path="lfs_clustered_data.csv"):
        print("🤖 Initializing LLM Query Engine …")

        # ---- Load data ----
        if df is not None:
            self.df = df.copy()
            self.has_clusters = 'cluster_id' in df.columns
        elif model_path and os.path.exists(model_path):
            print(f"📂 Loading model from {model_path}")
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            self.df = model.df.copy() if hasattr(model, 'df') and model.df is not None else None
            self.has_clusters = self.df is not None and 'cluster_id' in self.df.columns
        elif os.path.exists(csv_path):
            print(f"📂 Loading data from {csv_path}")
            self.df = pd.read_csv(csv_path)
            self.has_clusters = 'cluster_id' in self.df.columns
            print(f"✅ Loaded {len(self.df)} records × {len(self.df.columns)} columns")
        else:
            self.df = None
            self.has_clusters = False
            print("⚠️ No data loaded")

        # ---- LLM setup ----
        self.llm = None
        self.query_engine = None
        try:
            if Groq is None:
                raise ImportError(
                    "Missing llama-index Groq LLM package. "
                    "Install 'llama-index-llms-groq'."
                ) from _LLAMA_INDEX_IMPORT_ERROR

            groq_api_key = os.getenv("GROQ_API_KEY")
            if not groq_api_key:
                raise ValueError("Missing GROQ_API_KEY environment variable.")

            self.llm = Groq(
                api_key=groq_api_key,
                model="llama-3.3-70b-versatile",
                temperature=0.0,
                timeout=120.0,
            )
            Settings.llm = self.llm

            if self.df is not None:
                self.query_engine = PandasQueryEngine(
                    df=self.df,
                    instruction_str=self._build_instruction_str(),
                    verbose=False,
                    synthesize_response=True,
                )
                print("✅ PandasQueryEngine ready (general queries)")
            print("✅ Direct LLM ready (resource allocation)")

        except Exception as e:
            print(f"⚠️ LLM not available: {e}")
            print("Set GROQ_API_KEY env var.  Get key → https://console.groq.com/keys")

        self.last_question = None
        self.last_answer = None

    # ==================================================================
    #  QUERY-TYPE DETECTION
    # ==================================================================

    def _detect_analysis_type(self, question_lower: str):
        """Return (analysis_type, params) tuple."""

        # ---- Resource allocation ----
        alloc_match = re.search(
            r'(?:give|distribut|allocat|provide|deliver|hand\s*out|send|assign|target)\s+'
            r'(\d+)\s+(\w+)',
            question_lower,
        )
        if alloc_match:
            return "resource_allocation", {
                "num_items": int(alloc_match.group(1)),
                "item_type": alloc_match.group(2),
            }

        # Phrases that imply allocation even without a verb+number
        if any(kw in question_lower for kw in [
            'who should', 'identify beneficiar', 'target population',
            'most in need', 'most vulnerable', 'priority list',
            'neediest', 'poorest', 'allocate resource',
        ]):
            num_match = re.search(r'(\d+)', question_lower)
            return "resource_allocation", {
                "num_items": int(num_match.group(1)) if num_match else 10,
                "item_type": "items",
            }

        return "general", {}

    # ==================================================================
    #  NEED-SCORE COMPUTATION  (core of resource allocation)
    # ==================================================================

    def _compute_need_score(self, row, item_type: str, question_lower: str) -> float:
        """
        Compute a composite need score for one person (higher = greater need).

        Factors (weighted):
          1. Income (40 %)  — lower income → higher score
          2. Education (15 %) — lower education → higher score
          3. Disability burden (15 %) — more difficulty → higher score
          4. Informality (10 %) — informal workers → higher score
          5. Sector (10 %) — Estate > Rural > Urban
          6. Item-specific (10 %) — e.g. computer literacy for laptops
        """
        score = 0.0
        weights = {
            'income': 0.40,
            'education': 0.15,
            'disability': 0.15,
            'informality': 0.10,
            'sector': 0.10,
            'item_specific': 0.10,
        }

        # 1. Income component (0-100, lower income → higher score)
        income = row.get('Q45_A_1')
        if pd.notna(income):
            # Scale: 0 LKR → 100, 100k+ → 0
            score += weights['income'] * max(0.0, 100.0 - (float(income) / 1000.0))
        else:
            score += weights['income'] * 50  # Unknown income → moderate need

        # 2. Education (0-100, lower edu → higher score)
        edu = row.get('EDU')
        if pd.notna(edu):
            edu = float(edu)
            if edu <= 5:
                score += weights['education'] * 100
            elif edu <= 10:
                score += weights['education'] * 70
            elif edu <= 12:
                score += weights['education'] * 40
            elif edu <= 14:
                score += weights['education'] * 20
            else:
                score += weights['education'] * 5
        else:
            score += weights['education'] * 50

        # 3. Disability burden (sum of P15-P20; each 1-4)
        disability_cols = ['P15', 'P16', 'P17', 'P18', 'P19', 'P20']
        d_total = 0
        d_count = 0
        for dc in disability_cols:
            v = row.get(dc)
            if pd.notna(v):
                d_total += (float(v) - 1)  # 0 = no difficulty
                d_count += 1
        if d_count > 0:
            # max possible = 3*6 = 18
            score += weights['disability'] * min(100.0, (d_total / 18.0) * 100.0)

        # 4. Informality
        q47 = row.get('Q47')
        if pd.notna(q47) and float(q47) == 2:
            score += weights['informality'] * 100
        q46_val = row.get('Q46')
        if pd.notna(q46_val) and float(q46_val) == 2:
            score += weights['informality'] * 50  # no EPF/ETF

        # 5. Sector
        sector = row.get('SECTOR')
        if pd.notna(sector):
            sector_scores = {3: 100, 2: 60, 1: 20}
            score += weights['sector'] * sector_scores.get(int(sector), 40)

        # 6. Item-specific bonus
        is_tech = any(kw in question_lower for kw in ['laptop', 'computer', 'tablet', 'phone', 'device'])
        is_transport = any(kw in question_lower for kw in ['taxi', 'vehicle', 'transport', 'bus', 'wheel'])
        is_food = any(kw in question_lower for kw in ['food', 'meal', 'ration', 'nutrition'])

        if is_tech:
            q60a = row.get('Q60A')
            # Prefer people who CAN use computers but can't afford them
            if pd.notna(q60a) and float(q60a) == 1:
                score += weights['item_specific'] * 80
            age = row.get('AGE')
            if pd.notna(age) and 16 <= float(age) <= 45:
                score += weights['item_specific'] * 20
        elif is_transport:
            p17 = row.get('P17')
            if pd.notna(p17) and float(p17) >= 2:
                score += weights['item_specific'] * 100  # mobility difficulty
            age = row.get('AGE')
            if pd.notna(age) and 18 <= float(age) <= 65:
                score += weights['item_specific'] * 20
        elif is_food:
            age = row.get('AGE')
            if pd.notna(age):
                if float(age) < 18 or float(age) > 60:
                    score += weights['item_specific'] * 80  # children / elderly
            rship = row.get('RSHIP')
            if pd.notna(rship) and float(rship) == 1:
                score += weights['item_specific'] * 40  # head of household
        else:
            # Generic: prefer working-age unemployed
            q2 = row.get('Q2')
            if pd.notna(q2) and float(q2) == 2:
                score += weights['item_specific'] * 60  # not working

        return round(score, 2)

    # ==================================================================
    #  RESOURCE ALLOCATION  — fully pre-computed, direct LLM call
    # ==================================================================

    def _handle_resource_allocation(self, question: str, num_items: int, item_type: str) -> str:
        """
        End-to-end resource allocation pipeline:
        1. Score every person with income data
        2. Rank by need score (descending)
        3. Build beneficiary table + summary statistics
        4. Send self-contained prompt to LLM (NO code generation)
        5. Return LLM-formatted answer
        """
        df = self.df
        question_lower = question.lower()

        # ---- 1-A. Determine target cluster via LLM + safe keyword matching ----
        # Maps simple lowercase keywords → exact cluster_label strings in the CSV.
        keyword_map = {
            'skill gap':          'High Skill Gap - Needs Job Matching',
            'digitally excluded': 'Digitally Excluded - Needs Tech Training',
            'vulnerable':         'Economically Vulnerable - Needs Social Safety Net',
            'stable':             'Stable Workforce - Needs Leadership/Advanced Skills',
        }

        # Safest default if nothing matches (most likely to benefit from resources)
        target_cluster_exact = 'Economically Vulnerable - Needs Social Safety Net'

        if self.llm is not None and 'cluster_label' in df.columns:
            try:
                available_labels = df['cluster_label'].dropna().unique().tolist()
                cluster_prompt = (
                    f"You are helping allocate resources to Sri Lankan workers.\n"
                    f"User question: {question}\n\n"
                    f"Available clusters:\n" +
                    "\n".join(f"  - {lbl}" for lbl in available_labels) +
                    "\n\nWhich single cluster should be prioritised for this "
                    "resource allocation? Reply with ONLY the cluster name, "
                    "with no extra text or punctuation."
                )
                llm_response = str(self.llm.complete(cluster_prompt))
                llm_text = llm_response.lower()
                for keyword, cluster_name in keyword_map.items():
                    if keyword in llm_text:
                        target_cluster_exact = cluster_name
                        break
                print(f"🎯 Target cluster (LLM): {target_cluster_exact}")
            except Exception as e:
                print(f"⚠️ Cluster detection LLM call failed ({e}), "
                      f"falling back to keyword scan of question")
                for keyword, cluster_name in keyword_map.items():
                    if keyword in question_lower:
                        target_cluster_exact = cluster_name
                        break
                print(f"🎯 Target cluster (keyword fallback): {target_cluster_exact}")
        else:
            # No LLM available or no cluster_label column — match against question text
            for keyword, cluster_name in keyword_map.items():
                if keyword in question_lower:
                    target_cluster_exact = cluster_name
                    break
            print(f"🎯 Target cluster (question keyword): {target_cluster_exact}")

        # ---- 1-B. Filter to target cluster, sort closest-to-centroid first ----
        if 'cluster_label' in df.columns:
            filtered_df = df[df['cluster_label'] == target_cluster_exact].copy()
            if len(filtered_df) == 0:
                print(f"⚠️ Cluster '{target_cluster_exact}' not found in data — "
                      "using full dataset as fallback")
                filtered_df = df.copy()

            # Sort by proximity to centroid so the most representative individuals
            # are selected first (ascending = closest first)
            if 'distance_to_center' in filtered_df.columns:
                pool = filtered_df.sort_values(by='distance_to_center', ascending=True)
            else:
                pool = filtered_df
        else:
            # CSV predates cluster columns — fall back to full dataset
            pool = df.copy()

        if len(pool) == 0:
            return "⚠️ No data available for need-based resource allocation."

        # ---- 2. Score & rank ----
        print(f"📊 Scoring {len(pool)} candidates …")
        pool['_need_score'] = pool.apply(
            lambda r: self._compute_need_score(r, item_type, question_lower), axis=1
        )
        pool = pool.sort_values('_need_score', ascending=False)

        n = min(num_items, len(pool))
        beneficiaries = pool.head(n).reset_index(drop=False)

        # ---- 2-B. Outlier Guardrail — Confidence Score ----
        # Measures how representative the selected candidates are by their
        # average distance to the cluster centroid.
        if 'distance_to_center' in beneficiaries.columns:
            avg_dist = round(float(beneficiaries['distance_to_center'].mean()), 2)
        else:
            avg_dist = None

        if avg_dist is None:
            trust_level = 'Unknown'
            trust_guidance = ('The distance_to_center column is unavailable, so '
                              'representativeness cannot be mathematically assessed. '
                              'Apply standard manual verification.')
        elif avg_dist < 0.5:
            trust_level = 'High'
            trust_guidance = ('These individuals are PERFECT ARCHETYPES for this '
                              'intervention — they sit extremely close to the cluster '
                              'centroid and are highly representative of the lifestyle '
                              'and needs profile. You may allocate resources with high '
                              'confidence.')
        elif avg_dist < 1.5:
            trust_level = 'Moderate'
            trust_guidance = ('These are GOOD MATCHES but show some individual variance '
                              'from the cluster centre. Most will benefit strongly from '
                              'the intervention; a small minority may have edge-case '
                              'circumstances worth reviewing.')
        else:
            trust_level = 'Low'
            trust_guidance = ('These individuals are PERIPHERAL OUTLIERS who only '
                              'partially match the cluster profile. Be transparent that '
                              'they fall at the boundary of the target group and may '
                              'require manual case-by-case verification before resources '
                              'are committed.')

        print(f'🔍 Outlier Guardrail — avg distance: {avg_dist}, trust: {trust_level}')

        # ---- 3. Build translated beneficiary table ----
        rows = []
        for i, (_, row) in enumerate(beneficiaries.iterrows(), 1):
            r = {'No': i, 'Index': int(row['index'])}
            r['Income (LKR)'] = f"Rs. {row['Q45_A_1']:,.0f}" if pd.notna(row['Q45_A_1']) else 'N/A'
            r['Need Score'] = row['_need_score']
            if pd.notna(row.get('AGE')):
                r['Age'] = int(row['AGE'])
            if pd.notna(row.get('SEX')):
                r['Sex'] = 'Male' if int(row['SEX']) == 1 else 'Female'
            if pd.notna(row.get('EDU')):
                r['Education'] = self.EDU_MAP.get(int(row['EDU']), f"Code {int(row['EDU'])}")
            if pd.notna(row.get('SECTOR')):
                r['Sector'] = SECTOR_MAP.get(int(row['SECTOR']), str(int(row['SECTOR'])))
            if pd.notna(row.get('DISTRICT')):
                r['District'] = DISTRICT_MAP.get(int(row['DISTRICT']), str(int(row['DISTRICT'])))
            if pd.notna(row.get('Q16')):
                r['Employment'] = EMPLOYMENT_STATUS.get(int(row['Q16']), str(int(row['Q16'])))
            if pd.notna(row.get('Q47')):
                r['Formality'] = 'Formal' if int(row['Q47']) == 1 else 'Informal'
            # Disability flags
            disability_labels = []
            for dc, dl in [('P15','Vision'), ('P16','Hearing'), ('P17','Mobility'),
                           ('P18','Cognition'), ('P19','Self-care'), ('P20','Communication')]:
                dv = row.get(dc)
                if pd.notna(dv) and int(dv) >= 2:
                    sev = {2: 'Some', 3: 'A lot', 4: 'Cannot do'}.get(int(dv), '')
                    disability_labels.append(f"{dl}:{sev}")
            if disability_labels:
                r['Disabilities'] = '; '.join(disability_labels)
            rows.append(r)

        table_df = pd.DataFrame(rows)
        table_str = table_df.to_string(index=False) if len(table_df) <= 200 else table_df.head(200).to_string(index=False)

        # ---- 4. Summary statistics ----
        incomes = beneficiaries['Q45_A_1'].dropna()
        sector_dist = {}
        if 'SECTOR' in beneficiaries.columns:
            for code, label in SECTOR_MAP.items():
                cnt = int((beneficiaries['SECTOR'] == code).sum())
                if cnt > 0:
                    sector_dist[label] = cnt

        district_dist = {}
        if 'DISTRICT' in beneficiaries.columns:
            for code, cnt in beneficiaries['DISTRICT'].value_counts().head(10).items():
                district_dist[DISTRICT_MAP.get(int(code), str(int(code)))] = int(cnt)

        gender_dist = {}
        if 'SEX' in beneficiaries.columns:
            gender_dist['Male'] = int((beneficiaries['SEX'] == 1).sum())
            gender_dist['Female'] = int((beneficiaries['SEX'] == 2).sum())

        summary = (
            f"\n=== SUMMARY STATISTICS (PRE-COMPUTED — EXACT) ===\n"
            f"Total beneficiaries selected: {n}\n"
            f"Items to distribute: {num_items} {item_type}\n"
            f"Average income: Rs. {incomes.mean():,.0f}\n"
            f"Median income: Rs. {incomes.median():,.0f}\n"
            f"Income range: Rs. {incomes.min():,.0f} – Rs. {incomes.max():,.0f}\n"
            f"Average need score: {beneficiaries['_need_score'].mean():.1f} / 100\n"
            f"Gender: {gender_dist}\n"
            f"Sector distribution: {sector_dist}\n"
            f"Top districts: {district_dist}\n"
        )

        # ---- 5. Overall dataset context ----
        overall_income = df['Q45_A_1'].dropna()
        dataset_context = (
            f"\n=== DATASET CONTEXT ===\n"
            f"Total records: {len(df)}\n"
            f"Records with income data: {len(overall_income)}\n"
            f"Overall income — Mean: Rs. {overall_income.mean():,.0f}, "
            f"Median: Rs. {overall_income.median():,.0f}\n"
            f"Sector: Urban={int((df['SECTOR']==1).sum())}, "
            f"Rural={int((df['SECTOR']==2).sum())}, "
            f"Estate={int((df['SECTOR']==3).sum())}\n"
        )

        # ---- 6. Build final prompt for LLM  (NO code generation) ----
        final_prompt = f"""You are a resource-allocation analyst for Sri Lanka's Labour Force Survey (LFS-2023).

USER QUESTION: {question}

I have ALREADY identified the top {n} beneficiaries using a multi-factor need scoring system.
The scoring considers: income (40%), education level (15%), disability burden (15%),
employment informality (10%), sector deprivation (10%), and item-specific factors (10%).

All data below is PRE-COMPUTED by Python and is 100% accurate.
DO NOT recalculate. DO NOT return Python code. Only format and present the results.

COLUMN MEANINGS (for reference):
"""
        # Add relevant column descriptions
        relevant_cols = ['Q45_A_1', 'SEX', 'AGE', 'EDU', 'SECTOR', 'DISTRICT',
                         'Q16', 'Q47', 'P15', 'P16', 'P17', 'P18', 'P19', 'P20']
        for col in relevant_cols:
            if col in COLUMN_DESCRIPTIONS:
                final_prompt += f"- {col.upper()}: {COLUMN_DESCRIPTIONS[col]}\n"

        final_prompt += """
VALUE MAPPINGS (use these to translate codes in your responses):

=== ADMINISTRATIVE & IDENTIFICATION ===
SECTOR: 1=Urban, 2=Rural, 3=Estate

DISTRICT:
  11=Colombo, 12=Gampaha, 13=Kalutara, 21=Kandy, 22=Matale, 23=Nuwara Eliya,
  31=Galle, 32=Matara, 33=Hambantota, 41=Jaffna, 42=Kilinochchi, 43=Mannar,
  44=Vavuniya, 45=Mullaitivu, 51=Batticaloa, 52=Ampara, 53=Trincomalee,
  61=Kurunegala, 62=Puttalam, 71=Anuradhapura, 72=Polonnaruwa,
  81=Badulla, 82=Monaragala, 91=Ratnapura, 92=Kegalle

=== PERSONAL CHARACTERISTICS ===
SEX: 1=Male, 2=Female
MARITAL: 1=Never married, 2=Married, 3=Widowed, 4=Divorced, 5=Separated
ETH: 1=Sinhala, 2=SL Tamil, 3=Indian Tamil, 4=Moor, 5=Malay, 6=Burgher, 9=Other

=== EMPLOYMENT ===
Q16: 1=Employee (public/private), 2=Employer, 3=Own account worker, 4=Contributing family worker
Q47: 1=Formal, 2=Informal

=== DISABILITY (P15-P20) ===
1=No difficulty, 2=Some difficulty, 3=A lot of difficulty, 4=Cannot do at all

IMPORTANT: All codes have ALREADY been translated in the table below. Present them as-is.
"""

        final_prompt += f"""
=== BENEFICIARY ALLOCATION LIST (ALL {n} beneficiaries) ===

{table_str}

{summary}
{dataset_context}

FOR RESOURCE ALLOCATION OF {num_items} {item_type.upper()}:

CRITICAL: You MUST show ALL {n} beneficiaries. DO NOT abbreviate with "..." or skip rows.

Your task:
1. Present ALL {n} beneficiaries in a clear numbered table
2. For each beneficiary, show: Row number, Dataset Index, Income, Need Score,
   Age, Sex, Education, Sector, District, Employment status, and any disabilities
3. After the complete list, provide:
   - Total beneficiaries: {n}
   - Average income of selected group vs overall population
   - Need score statistics
   - Distribution by sector, district (top 5), gender
   - Brief allocation rationale (why these people were selected)
   - 2-3 policy recommendations

=== MATHEMATICAL CONFIDENCE / OUTLIER GUARDRAIL ===
Average Distance to Cluster Centroid: {avg_dist if avg_dist is not None else 'N/A'}
Trust Level: {trust_level}

You MUST incorporate this confidence level into your response:
{trust_guidance}

REMEMBER: Show EVERY SINGLE ROW. No abbreviations. All numbers are pre-computed and exact.
DO NOT return Python code. Return ONLY the formatted analysis.

Provide your analysis:"""

        # ---- 7. Call LLM directly (bypass PandasQueryEngine) ----
        if self.llm is not None:
            try:
                print(f"🤖 Sending allocation prompt to LLM ({len(final_prompt)} chars) …")
                response = self.llm.complete(final_prompt)
                answer = str(response)
                self.last_question = question
                self.last_answer = answer
                return answer
            except Exception as e:
                print(f"⚠️ LLM call failed: {e}")
                # Fall through to pre-computed fallback

        # ---- Fallback: return pre-computed data directly ----
        fallback = f"📊 Resource Allocation: {num_items} {item_type}\n\n"
        fallback += f"BENEFICIARY LIST ({n} people, ranked by need score):\n\n"
        fallback += table_str + "\n"
        fallback += summary
        fallback += dataset_context
        return fallback

    # ==================================================================
    #  RESOURCE ALLOCATION & REAL PROFILE SEARCH  (df.query based)
    # ==================================================================

    @staticmethod
    def _derive_employment_label(row) -> str:
        """Derive a human-readable employment label from Q2 / Q16 / Q36."""
        q2 = row.get('Q2')
        q16 = row.get('Q16')
        q36 = row.get('Q36')

        # Q2: 1 = worked last 7 days, 2 = did not work
        if pd.notna(q2) and int(q2) == 1:
            if pd.notna(q16):
                return {
                    1: 'Employee',
                    2: 'Employer',
                    3: 'Self-employed',
                    4: 'Family Worker',
                }.get(int(q16), 'Employed')
            return 'Employed'

        # Not working → check if actively looking for work
        if pd.notna(q36) and int(q36) == 1:
            return 'Unemployed'

        return 'Inactive'

    def handle_allocation(self, query: str) -> str:
        """
        Resource Allocation & Real Profile Search.

        The LLM's ONLY role is translating the user's resource request into a
        strictly valid Pandas boolean query string.  All data, counts, and
        allocation numbers are produced exclusively by df.query() on the real
        DataFrame — guaranteeing 100 % accuracy and zero hallucination.
        """
        df = self.df
        if df is None:
            return "⚠️ No data loaded."

        # ---- Extract quantity & item type from query ----
        num_match = re.search(r'(\d+)', query)
        num_items = int(num_match.group(1)) if num_match else 10

        item_match = re.search(
            r'\d+\s+([\w][\w\s]*?)(?:\s+(?:to|for|among|across|in|between|whom)\b|$)',
            query, re.IGNORECASE,
        )
        item_type = item_match.group(1).strip() if item_match else "items"

        # ---- Compute HH_SIZE (family size) if missing ----
        if 'HH_SIZE' not in df.columns:
            hh_keys = ['DISTRICT', 'PSU', 'HUNIT', 'HHOLD']
            if all(c in df.columns for c in hh_keys):
                hh_sizes = (
                    df.groupby(hh_keys)
                    .size()
                    .reset_index(name='HH_SIZE')
                )
                df = df.merge(hh_sizes, on=hh_keys, how='left')
                self.df = df  # persist for subsequent calls

        # ---- Ask LLM for a Pandas query string (ONLY) ----
        col_desc_text = "\n".join(
            f"  {k}: {v}" for k, v in COLUMN_DESCRIPTIONS.items()
        )
        prompt = (
            "You are a data-filtering expert for Sri Lanka's Labour Force Survey "
            "(LFS-2023, ~18 937 rows).  Your ONLY job is to translate a resource-"
            "allocation request into a valid Pandas df.query() boolean expression.\n\n"
            f"AVAILABLE COLUMNS:\n{col_desc_text}\n\n"
            f"USER REQUEST: \"{query}\"\n"
            f"RESOURCE / PRODUCT: \"{item_type}\"\n\n"
            "Think about WHO would benefit most from this resource, then write a "
            "targeted but NOT overly restrictive filter.  Prefer 2-3 conditions "
            "maximum so we get enough matches to distribute across districts.\n\n"
            "EXAMPLES OF VALID OUTPUT (return text exactly like these):\n"
            "  SEX == 2 & AGE >= 18 & AGE <= 60\n"
            "  AGE >= 18 & AGE <= 65\n"
            "  P17 >= 2 & AGE >= 18\n"
            "  AGE >= 16 & AGE <= 45\n\n"
            "RULES:\n"
            "1. Return ONLY the Pandas query string — no explanation, no code "
            "blocks, no surrounding quotes, no backticks.\n"
            "2. Use only column names listed above.\n"
            "3. Use Python operators: ==  !=  >  <  >=  <=\n"
            "4. Use & for AND, | for OR.  Parenthesise OR groups.\n"
            "5. Keep it broad enough to return hundreds of matches.\n"
            "6. Do NOT use columns that have mostly empty values (Q60A, Q61 "
            "have many NaN).  Prefer AGE, SEX, SECTOR, EDU, Q2, Q16.\n\n"
            "Your query string:"
        )

        pandas_query_str = None
        if self.llm is not None:
            try:
                raw = str(self.llm.complete(prompt)).strip()
                # Sanitise: strip markdown fences, stray quotes / backticks
                cleaned = re.sub(r'^```[\w]*\n?', '', raw)
                cleaned = re.sub(r'\n?```$', '', cleaned)
                cleaned = cleaned.strip('`"\' \n')
                pandas_query_str = cleaned
                print(f"🔍 LLM Filter: {pandas_query_str}")
            except Exception as e:
                print(f"⚠️ LLM query generation failed: {e}")

        if not pandas_query_str:
            return "⚠️ Could not generate filter criteria from LLM."

        # ---- Execute df.query() with progressive relaxation ----
        conditions = [c.strip() for c in pandas_query_str.split('&')]
        filtered_df = None
        used_query = pandas_query_str

        # Try full query first, then drop conditions from the end one by one
        for n_conds in range(len(conditions), 0, -1):
            attempt_query = ' & '.join(conditions[:n_conds])
            try:
                result = df.query(attempt_query)
                if len(result) > 0:
                    filtered_df = result.copy()
                    used_query = attempt_query
                    if n_conds < len(conditions):
                        print(f"🔄 Relaxed filter to: {used_query}")
                    break
            except Exception as err:
                print(f"⚠️ Filter error ({attempt_query}): {err}")
                continue

        if filtered_df is None or len(filtered_df) == 0:
            # Ultimate fallback: working-age adults
            fallback_q = 'AGE >= 18 & AGE <= 65'
            try:
                filtered_df = df.query(fallback_q).copy()
                used_query = fallback_q
                print(f"🔄 Using fallback filter: {fallback_q}")
            except Exception:
                return "⚠️ Could not filter data from the dataset."

        total_matches = len(filtered_df)
        print(f"✅ Total Matches Found (filter): {total_matches:,}")

        # ---- Cluster-aware refinement ----
        cluster_name_used = None
        if 'cluster_label' in filtered_df.columns and self.llm is not None:
            available_labels = filtered_df['cluster_label'].dropna().unique().tolist()
            if available_labels:
                # Ask LLM which cluster best fits the resource
                keyword_map = {
                    'skill gap':          'High Skill Gap - Needs Job Matching',
                    'digitally excluded': 'Digitally Excluded - Needs Tech Training',
                    'vulnerable':         'Economically Vulnerable - Needs Social Safety Net',
                    'stable':             'Stable Workforce - Needs Leadership/Advanced Skills',
                }
                try:
                    cluster_prompt = (
                        "You are allocating resources to Sri Lankan workers.\n"
                        f"Resource: {item_type}\n"
                        f"User request: {query}\n\n"
                        "Available workforce clusters:\n" +
                        "\n".join(f"  - {lbl}" for lbl in available_labels) +
                        "\n\nWhich single cluster should be PRIORITISED? "
                        "Reply with ONLY the cluster name, nothing else."
                    )
                    llm_cluster = str(self.llm.complete(cluster_prompt)).strip()
                    # Match via keyword
                    target_cluster = 'Economically Vulnerable - Needs Social Safety Net'
                    for kw, cname in keyword_map.items():
                        if kw in llm_cluster.lower():
                            target_cluster = cname
                            break

                    cluster_subset = filtered_df[
                        filtered_df['cluster_label'] == target_cluster
                    ]
                    if len(cluster_subset) > 0:
                        filtered_df = cluster_subset.copy()
                        cluster_name_used = target_cluster
                        print(f"🎯 Cluster: {cluster_name_used} "
                              f"({len(filtered_df):,} candidates)")
                    else:
                        print(f"⚠️ Cluster '{target_cluster}' empty after "
                              "filter — using all filtered matches")
                except Exception as e:
                    print(f"⚠️ Cluster selection failed: {e}")

        # Sort by distance_to_center if available (closest = most representative)
        if 'distance_to_center' in filtered_df.columns:
            filtered_df = filtered_df.sort_values(
                'distance_to_center', ascending=True
            )

        total_matches = len(filtered_df)
        print(f"✅ Final candidate pool: {total_matches:,}")

        # ---- District-wise proportional allocation ----
        district_counts = (
            filtered_df['DISTRICT']
            .value_counts()
            .sort_values(ascending=False)
        )
        total_in_filter = district_counts.sum()

        district_items = []       # (name, units)
        allocated_so_far = 0
        entries = list(district_counts.items())

        for i, (code, count) in enumerate(entries):
            if i == len(entries) - 1:
                units = max(0, num_items - allocated_so_far)
            else:
                units = round(num_items * (count / total_in_filter))
            allocated_so_far += units
            name = DISTRICT_MAP.get(int(code), f"District {int(code)}")
            if units > 0:
                district_items.append((name, units))

        # ---- Profile table — proportionally sampled across districts ----
        # Pick from each district in proportion to its allocation so profiles
        # reflect real geographic spread (not just the first rows = Colombo).
        n_profiles = min(num_items, total_matches)
        sampled_parts = []
        for code, count in district_counts.items():
            district_share = round(n_profiles * (count / total_in_filter))
            if district_share == 0:
                continue
            district_rows = filtered_df[filtered_df['DISTRICT'] == code]
            sampled_parts.append(
                district_rows.head(district_share)
            )
        if sampled_parts:
            sampled_df = pd.concat(sampled_parts, ignore_index=False)
        else:
            sampled_df = filtered_df.head(n_profiles)
        # Trim / pad to exact count
        if len(sampled_df) > n_profiles:
            sampled_df = sampled_df.head(n_profiles)
        elif len(sampled_df) < n_profiles:
            remaining = filtered_df[~filtered_df.index.isin(sampled_df.index)]
            sampled_df = pd.concat([
                sampled_df,
                remaining.head(n_profiles - len(sampled_df))
            ])

        # ---- Employment mapping (Q2: 1=Employed, 2=Unemployed, 3=Inactive) ----
        _Q2_MAP = {1: "Employed", 2: "Unemployed", 3: "Inactive"}

        profile_rows = []
        for idx, row in sampled_df.iterrows():
            district_name = (
                DISTRICT_MAP.get(int(row['DISTRICT']),
                                 str(int(row['DISTRICT'])))
                if pd.notna(row.get('DISTRICT')) else 'N/A'
            )
            sector_label = (
                SECTOR_MAP.get(int(row['SECTOR']), str(int(row['SECTOR'])))
                if pd.notna(row.get('SECTOR')) else 'N/A'
            )
            emp_label = (
                _Q2_MAP.get(int(row['Q2']), str(int(row['Q2'])))
                if pd.notna(row.get('Q2')) else 'N/A'
            )
            hh_size = (
                int(row['HH_SIZE'])
                if 'HH_SIZE' in row.index and pd.notna(row.get('HH_SIZE'))
                else 'N/A'
            )
            # Income: Q45_A_1 may be string with spaces — coerce safely
            try:
                inc_val = pd.to_numeric(row.get('Q45_A_1'), errors='coerce')
                income_str = f"Rs. {inc_val:,.0f}" if pd.notna(inc_val) else 'N/A'
            except Exception:
                income_str = 'N/A'

            cluster_val = (
                str(row['cluster_label'])
                if 'cluster_label' in row.index and pd.notna(row.get('cluster_label'))
                else 'N/A'
            )

            profile_rows.append({
                'Line #': idx,
                'District': district_name,
                'Sector': sector_label,
                'Employment': emp_label,
                'Family': hh_size,
                'Income Rs.': income_str,
                'Cluster': cluster_val,
            })

        # ---- Assemble formatted output ----
        out = []
        out.append(f"\n✅ DATA-DRIVEN ANALYSIS FOR: {query}")
        if cluster_name_used:
            out.append(f"🎯 Target Cluster: {cluster_name_used}")
        out.append(f"\n📊 Total Matches Found: {total_matches:,}")
        out.append(f"\n📍 District-wise Allocation:")
        out.append(f"{'─' * 45}")
        for name, units in district_items:
            out.append(f"  {name:<20}: {units} units")
        out.append(f"\n👤 Targeted User Profiles ({n_profiles} shown):")
        hdr = (f"{'Line #':<8}| {'District':<15}| {'Sector':<8}| "
               f"{'Employment':<12}| {'Family':<7}| {'Income Rs.':<14}| {'Cluster'}")
        out.append(f"{'─' * len(hdr)}")
        out.append(hdr)
        out.append(f"{'─' * len(hdr)}")
        for pr in profile_rows:
            out.append(
                f"{str(pr['Line #']):<8}| {pr['District']:<15}| "
                f"{pr['Sector']:<8}| {pr['Employment']:<12}| "
                f"{str(pr['Family']):<7}| {pr['Income Rs.']:<14}| "
                f"{pr['Cluster']}"
            )

        return "\n".join(out)

    # ==================================================================
    #  PRE-COMPUTED STATISTICS  (for general queries)
    # ==================================================================

    def _compute_statistics(self, question: str) -> str:
        """Pre-compute relevant statistics so the LLM uses exact numbers."""
        df = self.df
        parts = []
        ql = question.lower()

        parts.append(f"DATASET: {len(df)} rows × {len(df.columns)} columns")
        parts.append(f"\nAge: min={df['AGE'].min()}, max={df['AGE'].max()}, "
                     f"mean={df['AGE'].mean():.1f}, median={df['AGE'].median():.0f}")
        parts.append(f"Gender: Male(1)={int((df['SEX']==1).sum())}, "
                     f"Female(2)={int((df['SEX']==2).sum())}")
        parts.append(f"Sector: Urban(1)={int((df['SECTOR']==1).sum())}, "
                     f"Rural(2)={int((df['SECTOR']==2).sum())}, "
                     f"Estate(3)={int((df['SECTOR']==3).sum())}")

        # Income
        if any(kw in ql for kw in ['income','salary','poverty','poor','rich','wage',
                                    'earning','resource','allocat','beneficiar',
                                    'give','distribut','q45']):
            inc = df['Q45_A_1'].dropna()
            if len(inc):
                parts.append(f"\nINCOME (Q45_A_1) — {len(inc)} with data / {len(df)} total:")
                parts.append(f"  Mean: Rs. {inc.mean():,.0f}  | Median: Rs. {inc.median():,.0f}")
                parts.append(f"  Min: Rs. {inc.min():,.0f}  | Max: Rs. {inc.max():,.0f}")
                parts.append(f"  25th: Rs. {inc.quantile(.25):,.0f} | 75th: Rs. {inc.quantile(.75):,.0f}")
                parts.append(f"  <20k: {int((inc<20000).sum())} | <10k: {int((inc<10000).sum())} | Zero: {int((inc==0).sum())}")

        # Employment
        if any(kw in ql for kw in ['employ','work','job','occupation','formal',
                                    'informal','labor','labour','q16','q2','q47']):
            parts.append("\nEMPLOYMENT:")
            q2 = df['Q2'].dropna()
            parts.append(f"  Worked last 7d (Q2): Yes={int((q2==1).sum())}, No={int((q2==2).sum())}")
            for c, l in EMPLOYMENT_STATUS.items():
                cnt = int((df['Q16']==c).sum())
                if cnt: parts.append(f"  Q16={c} ({l}): {cnt}")
            parts.append(f"  Formal(Q47=1): {int((df['Q47']==1).sum())}, "
                         f"Informal(Q47=2): {int((df['Q47']==2).sum())}")

        # Education
        if any(kw in ql for kw in ['education','school','degree','literacy',
                                    'edu','literat','computer','digital']):
            parts.append("\nEDUCATION:")
            edu = df['EDU'].dropna()
            for c in sorted(edu.unique()):
                parts.append(f"  EDU={int(c)} ({self.EDU_MAP.get(int(c), f'Grade {int(c)}')}): "
                             f"{int((edu==c).sum())}")
            if 'Q60A' in df.columns:
                parts.append(f"  Computer literate (Q60A=1): {int((df['Q60A']==1).sum())}")
                parts.append(f"  Not computer literate (Q60A=2): {int((df['Q60A']==2).sum())}")

        # Disability
        if any(kw in ql for kw in ['disab','difficult','vision','hear','walk',
                                    'mobil','self-care','communicat','p15','p16',
                                    'p17','p18','p19','p20']):
            parts.append("\nDISABILITY:")
            for col, lbl in [('P15','Vision'),('P16','Hearing'),('P17','Mobility'),
                              ('P18','Cognition'),('P19','Self-care'),('P20','Communication')]:
                if col in df.columns:
                    p = df[col].dropna()
                    parts.append(f"  {col} ({lbl}): Some+={int((p>=2).sum())}, "
                                 f"A lot+={int((p>=3).sum())}, Cannot={int((p==4).sum())}")

        # District
        if any(kw in ql for kw in ['district','region','province','colombo','kandy','galle','jaffna']):
            parts.append("\nDISTRICT:")
            for c in sorted(df['DISTRICT'].dropna().unique()):
                parts.append(f"  {int(c)} ({DISTRICT_MAP.get(int(c),'?')}): "
                             f"{int((df['DISTRICT']==c).sum())}")

        # Ethnicity
        if any(kw in ql for kw in ['ethnic','sinhala','tamil','moor','malay','burgher']):
            parts.append("\nETHNICITY:")
            for c, l in ETHNICITY_MAP.items():
                cnt = int((df['ETH']==c).sum())
                if cnt: parts.append(f"  {c} ({l}): {cnt}")

        # Marital
        if any(kw in ql for kw in ['marital','married','widow','divorced','single']):
            parts.append("\nMARITAL:")
            for c, l in MARITAL_MAP.items():
                cnt = int((df['MARITAL']==c).sum())
                if cnt: parts.append(f"  {c} ({l}): {cnt}")

        return "\n".join(parts)

    def _detect_relevant_columns(self, question_lower: str) -> list:
        """Return columns relevant to the question (used for context)."""
        kw_map = {
            'income':['Q45_A_1','Q47','Q2','Q16'], 'salary':['Q45_A_1','Q16'],
            'poverty':['Q45_A_1','SECTOR','DISTRICT','EDU'], 'employ':['Q2','Q16','Q20','Q47','Q8'],
            'education':['EDU','CUEDU','DEGREE'], 'disab':['P15','P16','P17','P18','P19','P20'],
            'difficult':['P15','P16','P17','P18','P19','P20'], 'vision':['P15'],
            'hear':['P16'], 'walk':['P17'], 'mobil':['P17'],
            'computer':['Q60A','Q60B'], 'digital':['Q60A','Q60B','Q61','Q64'],
            'internet':['Q61','Q64'], 'gender':['SEX'], 'male':['SEX'], 'female':['SEX'],
            'ethnic':['ETH'], 'religion':['REL'], 'district':['DISTRICT'],
            'sector':['SECTOR'], 'age':['AGE'], 'literacy':['SIN','TAMIL','ENG'],
            'marital':['MARITAL'],
        }
        cols = set()
        for kw, cl in kw_map.items():
            if kw in question_lower:
                cols.update(cl)
        return list(cols) or ['AGE','SEX','SECTOR','DISTRICT','EDU','Q45_A_1','Q2','Q16']

    # ==================================================================
    #  INSTRUCTION STRING  (for PandasQueryEngine — general queries only)
    # ==================================================================

    def _build_instruction_str(self) -> str:
        s = ("You are a data analyst for Sri Lanka's Labour Force Survey (LFS-2023).\n"
             "Answer by writing Python code against the DataFrame 'df'.\n\n"
             "DATASET: ~18,937 rows × 128 columns.\n\n"
             "COLUMN MEANINGS:\n")
        for col, desc in COLUMN_DESCRIPTIONS.items():
            s += f"- {col}: {desc}\n"

        s += "\n=== VALUE MAPPINGS ===\n"
        s += "SECTOR: " + ", ".join(f"{k}={v}" for k, v in SECTOR_MAP.items()) + "\n"
        s += "DISTRICT: " + ", ".join(f"{k}={v}" for k, v in DISTRICT_MAP.items()) + "\n"
        s += "Q16: " + ", ".join(f"{k}={v}" for k, v in EMPLOYMENT_STATUS.items()) + "\n"
        s += "ETH: " + ", ".join(f"{k}={v}" for k, v in ETHNICITY_MAP.items()) + "\n"
        s += "REL: " + ", ".join(f"{k}={v}" for k, v in RELIGION_MAP.items()) + "\n"
        s += "MARITAL: " + ", ".join(f"{k}={v}" for k, v in MARITAL_MAP.items()) + "\n"
        s += "P15-P20: 1=None, 2=Some, 3=A lot, 4=Cannot do\n"

        s += """
RULES:
- Output ONLY raw Python — no markdown, no ``` blocks.
- Use only df, pd, np (already imported). UPPERCASE column names only.
- Q45_A_1 already numeric; use .dropna() when filtering income.
- Translate codes to labels in print(): e.g. print("District: 11 (Colombo)").
- Show counts AND percentages. Use INCLUSIVE filters.
"""
        return s

    # ==================================================================
    #  MAIN ENTRY POINT
    # ==================================================================

    def analyze_data(self, question: str):
        """Route question to resource-allocation or general analysis."""
        print(f"\n🔍 Processing: {question}")

        if self.df is None:
            return "⚠️ No data loaded."

        ql = question.lower()

        # Quick answers
        if any(kw in ql for kw in ['columns', 'column names', 'fields', 'structure', 'schema']):
            info = [f"• {c}: {COLUMN_DESCRIPTIONS.get(c, 'Survey variable')}"
                    for c in self.df.columns]
            return f"📊 {len(self.df.columns)} columns:\n" + "\n".join(info)

        if 'shape' in ql or ('how many' in ql and 'row' in ql):
            return f"📊 {self.df.shape[0]:,} rows × {self.df.shape[1]} columns"

        # Detect type
        analysis_type, params = self._detect_analysis_type(ql)

        # ---- Resource allocation (primary path) ----
        if analysis_type == "resource_allocation":
            return self._handle_resource_allocation(
                question,
                params.get("num_items", 10),
                params.get("item_type", "items"),
            )

        # ---- General query (secondary path via PandasQueryEngine) ----
        if self.query_engine is None and self.llm is None:
            return "⚠️ Query engine not initialized."

        stats = self._compute_statistics(question)
        enhanced = f"""{question}

=== PRE-COMPUTED STATISTICS (EXACT — calculated by Python) ===
{stats}

INSTRUCTIONS:
1. Use the pre-computed statistics above — they are EXACT.
2. You may generate Python code for additional breakdowns not in the stats.
3. Translate ALL codes to meanings. Be specific, cite actual numbers.
4. DO NOT fabricate numbers."""

        # Try PandasQueryEngine first
        if self.query_engine is not None:
            try:
                response = self.query_engine.query(enhanced)
                answer = self._translate_codes_in_response(str(response))
                self.last_question = question
                self.last_answer = answer
                return answer
            except Exception as e:
                print(f"⚠️ PandasQueryEngine error: {e}")

        # Fallback: direct LLM
        if self.llm is not None:
            try:
                response = self.llm.complete(enhanced)
                answer = self._translate_codes_in_response(str(response))
                self.last_question = question
                self.last_answer = answer
                return answer
            except Exception as e:
                print(f"⚠️ Direct LLM error: {e}")

        return f"Here are the relevant statistics:\n\n{stats}"

    # ==================================================================
    #  POST-PROCESSING
    # ==================================================================

    def _translate_codes_in_response(self, text: str) -> str:
        """Translate remaining numeric codes in LLM output."""
        text = re.sub(r'\bSector:\s*1\b', 'Sector: 1 (Urban)', text)
        text = re.sub(r'\bSector:\s*2\b', 'Sector: 2 (Rural)', text)
        text = re.sub(r'\bSector:\s*3\b', 'Sector: 3 (Estate)', text)
        for code, name in DISTRICT_MAP.items():
            text = re.sub(rf'\bDistrict:\s*{code}\b', f'District: {code} ({name})', text)
        text = re.sub(r'\bSex:\s*1\b', 'Sex: 1 (Male)', text)
        text = re.sub(r'\bSex:\s*2\b', 'Sex: 2 (Female)', text)
        return text

    # ==================================================================
    #  CLUSTER & INSIGHT HELPERS  (interactive mode)
    # ==================================================================

    def ask_about_clusters(self, question: str) -> str:
        if not self.has_clusters:
            return "⚠️ No cluster information in the dataset."
        return self.analyze_data(f"Regarding the cluster_id column: {question}")

    def compare_clusters(self) -> str:
        if not self.has_clusters:
            return "⚠️ No cluster information in the dataset."
        return self.analyze_data(
            "Compare all clusters: count, mean age, mean income, dominant sector per cluster_id."
        )

    def get_insights(self, topic: str = None) -> str:
        if topic:
            return self.analyze_data(f"Key insights about {topic} in the dataset.")
        return self.analyze_data(
            "5 key insights covering income, employment, education, disability, regions."
        )


# Main execution
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='NLP Query Engine for Cluster Analysis')
    parser.add_argument('--mode', choices=['classic', 'llm'], default='llm',
                        help='Query mode: classic (rule-based) or llm (AI-powered)')
    parser.add_argument('--model', type=str, help='Path to model pickle file')

    args = parser.parse_args()

    # Load model
    if args.model:
        model_path = args.model
    else:
        model_path = os.path.join('..', 'model', 'skilldev_model.pkl')

    candidate_paths = [
        model_path,
        os.path.join('..', 'model', 'skilldev_model.pkl'),
        os.path.join('model', 'skilldev_model.pkl'),
    ]

    valid_path = None
    for path in candidate_paths:
        if os.path.exists(path):
            valid_path = path
            break

    if args.mode == 'classic':
        if not valid_path:
            print("❌ Model not found!")
            exit(1)
        print(f"✅ Using model from: {valid_path}\n")
        engine = NLPClusterQueryEngine(valid_path)
        engine.interactive_query()
    else:
        # LLM mode can work with CSV directly (no model needed)
        if valid_path:
            print(f"✅ Using model from: {valid_path}\n")
            engine = LLMQueryEngine(model_path=valid_path)
        else:
            print("📂 No model found, loading CSV directly...\n")
            engine = LLMQueryEngine()

        print("\n" + "=" * 70)
        print("💬 LLM Query Interface")
        print("=" * 70)
        print("Commands: /clusters, /compare, /insights [topic], quit\n")

        while True:
            query = input("📝 Your query: ").strip()

            if query.lower() in ['quit', 'exit']:
                print("✅ Goodbye!")
                break

            if not query:
                continue

            if query.startswith('/'):
                cmd_parts = query.split(' ', 1)
                cmd = cmd_parts[0].lower()
                cmd_arg = cmd_parts[1] if len(cmd_parts) > 1 else None

                if cmd == '/clusters':
                    question = cmd_arg or "What can you tell me about the clusters?"
                    print(engine.ask_about_clusters(question))
                elif cmd == '/compare':
                    print(engine.compare_clusters())
                elif cmd == '/insights':
                    print(engine.get_insights(cmd_arg))
                else:
                    print("⚠️ Unknown command. Use /clusters, /compare, /insights, or quit.")
            else:
                # --- Allocation detection ---
                # Catches:  "i have 100 cars", "give 50 sewing machines",
                #           "distribute 200 laptops to …", etc.
                _ql = query.lower()
                _has_number = bool(re.search(r'\d+', _ql))

                # Pattern 1: "i have <N> <things>" — always allocation
                _i_have_pattern = bool(re.search(
                    r'i\s+have\s+\d+', _ql
                ))

                # Pattern 2: number + action/resource keyword
                _ALLOC_KEYWORDS = [
                    'give', 'distribut', 'allocat', 'provide', 'deliver',
                    'send', 'assign', 'hand out', 'target', 'whom',
                    'sewing', 'laptop', 'computer', 'taxi', 'food',
                    'wheel', 'machine', 'book', 'phone', 'tablet',
                    'device', 'package', 'kit', 'tool', 'vehicle',
                    'bus', 'bicycle', 'ration', 'meal', 'uniform',
                    'scholarship', 'medicine', 'aid', 'supply',
                    'equipment', 'furniture', 'seed', 'fertilizer',
                    'car', 'truck', 'tractor', 'motorbike', 'motor',
                    'wheelchair', 'blanket', 'tent', 'house', 'loan',
                    'grant', 'voucher', 'coupon', 'subsidy',
                ]
                _kw_match = _has_number and any(
                    kw in _ql for kw in _ALLOC_KEYWORDS
                )

                if _i_have_pattern or _kw_match:
                    print(engine.handle_allocation(query))
                else:
                    print(engine.analyze_data(query))

            print()
