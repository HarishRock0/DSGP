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
    # Demographics
    "RSHIP": "Relationship to head of household (1=Head, 2=Spouse, 3=Child)",
    "SEX": "Gender (1=Male, 2=Female)",
    "ETH": "Ethnic Group (1=Sinhala, 2=SL Tamil, 3=Indian Tamil, 4=Moor, 5=Malay, 6=Burgher, 9=Other)",
    "MARITAL": "Marital Status (1=Never Married, 2=Married, 3=Widowed, 4=Divorced, 5=Separated)",
    "AGE": "Age in years (numeric, range typically 15-65+)",
    "EDU": "Highest Education Level (00=Grade 1, 05=Grade 5, 11=O/L, 13=A/L, 15=Degree, 16=PostGrad, 19=No Schooling)",

    # Literacy
    "SIN": "Sinhala Literacy (1=Can read/write, 2=Cannot read/write)",
    "TAMIL": "Tamil Literacy (1=Can read/write, 2=Cannot read/write)",
    "ENG": "English Literacy (1=Can read/write, 2=Cannot read/write)",

    # Disability/Difficulty Questions
    "P15": "Vision Difficulty - Even with glasses (1=None, 2=Some, 3=A lot, 4=Cannot do)",
    "P16": "Hearing Difficulty - Even with hearing aid (1=None, 2=Some, 3=A lot, 4=Cannot do)",
    "P17": "Mobility Difficulty - Walking or climbing steps (1=None, 2=Some, 3=A lot, 4=Cannot do)",
    "P18": "Cognitive Difficulty - Remembering or concentrating (1=None, 2=Some, 3=A lot, 4=Cannot do)",
    "P19": "Self-care Difficulty - Washing or dressing (1=None, 2=Some, 3=A lot, 4=Cannot do)",
    "P20": "Communicating Difficulty - Using usual language (1=None, 2=Some, 3=A lot, 4=Cannot do)",
    "P21": "Education/Training Participation - Last 12 months",

    # Employment
    "Q2": "Work Activity - Did person work for pay/profit in last week? (1=Yes, 2=No)",
    "Q8": "Occupation - Main job/task performed (ISCO-08 coded)",
    "Q16": "Employment Status (1=Public Employee, 2=Private Employee, 3=Employer, 4=Own account worker, 5=Contributing family worker)",
    "Q20": "Hours Worked - Total actual hours per week at main job (Identifies underemployment)",

    # Income & Poverty
    "Q45_A_1": "Monthly Income - Total gross salary or profit in last month (PRIMARY POVERTY INDICATOR)",

    # Formality
    "Q47": "Informal Flag - Workplace formality (1=Formal/Registered/Accounts, 2=Informal/Not registered)",

    # Digital Skills
    "Q60A": "Computer Literacy (1=Can use computer, 2=Cannot use)",
    "Q61": "Internet Use - Used internet in last 12 months",

    # Location
    "SECTOR": "Residential Sector (1=Urban, 2=Rural, 3=Estate)",
    "DISTRICT": "Administrative District"
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
    1: "Public Employee",
    2: "Private Employee",
    3: "Employer",
    4: "Own Account Worker (Self-employed)",
    5: "Contributing Family Worker"
}

# Sector mapping
SECTOR_MAP = {
    1: "Urban",
    2: "Rural",
    3: "Estate"
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
    LLM-powered pandas query engine using LlamaIndex & Ollama (free & local)
    """
    
    def __init__(self, model_path=None, df=None, csv_path="data/LFS-2023.csv"):
        """
        Initialize the LLM Query Engine with LlamaIndex PandasQueryEngine
        
        Args:
            model_path: Path to a trained clustering model (optional)
            df: DataFrame to analyze (optional)
            csv_path: Path to CSV file to load
        """
        print("🤖 Initializing LLM Query Engine with LlamaIndex + Pandas...")
        
        # Load data
        if df is not None:
            self.df = df
            self.has_clusters = 'cluster_id' in df.columns
        elif model_path and os.path.exists(model_path):
            print(f"📂 Loading model from {model_path}")
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            self.df = model.df if hasattr(model, 'df') else None
            self.has_clusters = self.df is not None and 'cluster_id' in self.df.columns
        elif os.path.exists(csv_path):
            print(f"📂 Loading data from {csv_path}")
            self.df = pd.read_csv(csv_path)
            self.has_clusters = 'cluster_id' in self.df.columns
            print(f"✅ Loaded {len(self.df)} records with {len(self.df.columns)} columns")
        else:
            self.df = None
            self.has_clusters = False
            print("⚠️ No data loaded")
        
        # Initialize LlamaIndex with Groq API
        try:
            if Groq is None:
                raise ImportError(
                    "Missing llama-index Groq LLM package. Install 'llama-index-llms-groq'."
                ) from _LLAMA_INDEX_IMPORT_ERROR

            groq_api_key = os.getenv("GROQ_API_KEY")
            if not groq_api_key:
                raise ValueError(
                    "Missing Groq API key. Set GROQ_API_KEY environment variable."
                )

            Settings.llm = Groq(
                api_key=groq_api_key,
                model="llama-3.3-70b-versatile",
                temperature=0.1,
                timeout=120.0
            )
            
            # Create instruction prefix with comprehensive dataset descriptions
            instruction_str = """You are a data analyst specializing in labor force surveys (LFS-2023 Sri Lanka).
Answer questions about the dataset using data analysis.

DATASET CONTEXT:
This dataset includes demographics, literacy, disability/difficulty indicators, employment, income,
informality, digital literacy, and location information for Sri Lanka.

COLUMN MEANINGS:
"""
            for col, desc in COLUMN_DESCRIPTIONS.items():
                instruction_str += f"- {col}: {desc}\n"

            instruction_str += "\nDISABILITY/DIFFICULTY SCALE (P15-P20):\n"
            for val, label in COLUMN_VALUE_SCALE.items():
                instruction_str += f"  {val} = {label}\n"

            instruction_str += "\nEMPLOYMENT STATUS (Q16):\n"
            for val, label in EMPLOYMENT_STATUS.items():
                instruction_str += f"  {val} = {label}\n"

            instruction_str += "\nSECTOR (SECTOR):\n"
            for val, label in SECTOR_MAP.items():
                instruction_str += f"  {val} = {label}\n"

            instruction_str += """
CRITICAL - CODE GENERATION RULES:
- Use UPPERCASE column names ONLY: SEX, EDU, AGE, Q2, Q16, Q60A, SECTOR, DISTRICT, P15-P21, Q45_A_1, Q47, Q61, etc.
- NEVER use lowercase column names - they won't exist in the dataframe!
- Never use markdown code blocks (``` ```). Output ONLY raw Python code.
- Do NOT include ```python, ```, or any markdown formatting.
- Code must use only: df, pd, np (pandas, numpy already imported)
- Always print actual data, not just summaries
- Filter the DataFrame using actual integer/string values that match the dataset

LAPTOP/TAXI DISTRIBUTION STRATEGY:
When asked "who should I give X items to", prioritize by:
1. Education level (EDU): Higher education = better resource use (11=O/L, 13=A/L, 15+=Degree)
2. Computer literacy (Q60A): 1=Can use, 2=Cannot use - STRONGLY PREFER 1 for laptops
3. Employment (Q2, Q16): Employed people benefit more (Q2==1 for worked last week, Q16 in [1-4])
4. Age (AGE): For taxis/vehicles prefer working age (18-65)
5. Location (SECTOR): Can filter by 1=Urban, 2=Rural, 3=Estate as needed
6. Income (Q45_A_1): Lower income may indicate greater need
7. Add reset_index() to get row numbers for identification

Code template for INCLUSIVE selection (to get 100+ recipients):
# IMPORTANT: Use INCLUSIVE criteria - don't filter too aggressively!
# For broad distribution (100+ recipients), use minimal filters
result = df[(df['AGE'] >= 16) & (df['AGE'] <= 75)].copy()  # Wide age range
# OPTIONAL: Sort by education to prioritize better-educated
result = result.sort_values(by=['EDU'], ascending=False, na_position='last')
# Reset index and take first X
result = result.reset_index(drop=False).head(100)
# Display results
print(f"Selected {len(result)} recipients:")
print(result[['index', 'SEX', 'EDU', 'Q16', 'AGE', 'SECTOR', 'Q45_A_1']].to_string())
print(f"\\nTotal eligible (age 16-75): {len(df[(df['AGE'] >= 16) & (df['AGE'] <= 75)])}")

ANALYSIS GUIDELINES:
1. Disability questions (P15-P20): 1=No difficulty, 2=Some, 3=A lot, 4=Cannot do
   - P15=Vision, P16=Hearing, P17=Mobility, P18=Cognition, P19=Self-care, P20=Communication

2. Poverty indicator: Q45_A_1 (Monthly Income) is PRIMARY poverty measure
   - Also consider Q47 (1=Formal/Registered, 2=Informal/Not registered) for job security
   - Q2 shows if person worked last week (employment activity)

3. Employment analysis:
   - Q16 = Employment status (1=Public, 2=Private, 3=Employer, 4=Self-employed, 5=Unpaid family)
   - Q20 = Hours worked per week (underemployment indicator)
   - Q8 = Occupation type (ISCO-08 coded)

4. Skills & literacy:
   - EDU = Education level (00-06=primary, 11=O/L, 13=A/L, 15+=Degree)
   - SIN, TAMIL, ENG = Language literacy (1=can read/write, 2=cannot)
   - Q60A = Computer literacy (1=yes, 2=no)
   - Q61 = Internet use

5. Demographics & location:
   - SEX = Gender (1=Male, 2=Female)
   - AGE = Age in years (numeric)
   - ETH = Ethnicity
   - MARITAL = Marital status
   - SECTOR = Urban/Rural/Estate (1/2/3)
   - DISTRICT = Administrative district

THE INDEX COLUMN IS IMPORTANT - IT IDENTIFIES EACH PERSON UNIQUELY!

ALWAYS provide:
1. The actual filtered dataframe with index numbers
2. Show key columns: index, SEX, EDU, Q60A, Q2, Q16, AGE, SECTOR
3. Print total count of eligible people
4. Use INCLUSIVE criteria (not too narrow) to get meaningful results.

Return actual data and clear insights. Be specific and concise."""


            # Initialize PandasQueryEngine if we have data
            if self.df is not None:
                self.query_engine = PandasQueryEngine(
                    df=self.df,
                    instruction_str=instruction_str,
                    verbose=True,
                    synthesize_response=True
                )
                print("✅ LlamaIndex PandasQueryEngine ready!")
            else:
                self.query_engine = None
                print("⚠️ No query engine created - no data available")
                
        except Exception as e:
            print(f"⚠️ LlamaIndex/Groq not available: {e}")
            print("Please set your Groq API key in GROQ_API_KEY environment variable.")
            print("Get your key at: https://console.groq.com/keys")
            self.query_engine = None
                
        except Exception as e:
            print(f"⚠️ LlamaIndex/Groq not available: {e}")
            print("Please set your Groq API key in GROQ_API_KEY environment variable.")
            print("Get your key at: https://console.groq.com/keys")
            self.query_engine = None
        
        # Conversation context for follow-up questions
        self.last_question = None
        self.last_answer = None
    
    
    def analyze_data(self, question: str):
        """Answer questions about the data using LlamaIndex PandasQueryEngine"""
        print(f"\n🔍 Processing question: {question}")
        
        if self.df is None:
            return "⚠️ No data loaded."
        
        if self.query_engine is None:
            return "⚠️ Query engine not initialized."
        
        try:
            response = self.query_engine.query(question)
            self.last_question = question
            self.last_answer = str(response)
            return str(response)
        except Exception as e:
            error_msg = f"⚠️ Error: {str(e)}"
            print(error_msg)
            return error_msg
    



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
    
    if not valid_path:
        print("❌ Model not found!")
        exit(1)
    
    print(f"✅ Using model from: {valid_path}\n")
    
    # Initialize engine
    if args.mode == 'classic':
        engine = NLPClusterQueryEngine(valid_path)
        engine.interactive_query()
    else:
        engine = LLMQueryEngine(model_path=valid_path)
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
                    print("⚠️ Unknown command")
            else:
                print(engine.analyze_data(query))
            
            print()
