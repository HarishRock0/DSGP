import os
import pickle
import pandas as pd
import numpy as np
import warnings
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import sys

# LlamaIndex imports for pandas query engine
from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core import Settings

warnings.filterwarnings('ignore')

# Comprehensive column descriptions for LFS-2023 dataset
COLUMN_DESCRIPTIONS = {
    # Demographics
    "p3": "Relationship to head of household (1=Head, 2=Spouse, 3=Child)",
    "p4": "Gender (1=Male, 2=Female)",
    "p7": "Ethnic Group (1=Sinhala, 2=SL Tamil, 3=Indian Tamil, 4=Moor, 5=Malay, 6=Burgher, 9=Other)",
    "p9": "Marital Status (1=Never Married, 2=Married, 3=Widowed, 4=Divorced, 5=Separated)",
    "p10": "Highest Education Level (00=Grade 1, 05=Grade 5, 11=O/L, 13=A/L, 15=Degree, 16=PostGrad, 19=No Schooling)",

    # Literacy
    "p12": "Sinhala Literacy (1=Can read/write, 2=Cannot read/write)",
    "p13": "Tamil Literacy (1=Can read/write, 2=Cannot read/write)",
    "p14": "English Literacy (1=Can read/write, 2=Cannot read/write)",

    # Disability/Difficulty Questions
    "p15": "Vision Difficulty - Even with glasses (1=None, 2=Some, 3=A lot, 4=Cannot do)",
    "p16": "Hearing Difficulty - Even with hearing aid (1=None, 2=Some, 3=A lot, 4=Cannot do)",
    "p17": "Mobility Difficulty - Walking or climbing steps (1=None, 2=Some, 3=A lot, 4=Cannot do)",
    "p18": "Cognitive Difficulty - Remembering or concentrating (1=None, 2=Some, 3=A lot, 4=Cannot do)",
    "p19": "Self-care Difficulty - Washing or dressing (1=None, 2=Some, 3=A lot, 4=Cannot do)",
    "p20": "Communicating Difficulty - Using usual language (1=None, 2=Some, 3=A lot, 4=Cannot do)",
    "p21": "Education/Training Participation - Last 12 months",

    # Employment
    "q2": "Work Activity - Did person work for pay/profit in last week? (1=Yes, 2=No)",
    "q8": "Occupation - Main job/task performed (ISCO-08 coded)",
    "q16": "Employment Status (1=Public Employee, 2=Private Employee, 3=Employer, 4=Own account worker, 5=Contributing family worker)",
    "q20": "Hours Worked - Total actual hours per week at main job (Identifies underemployment)",

    # Income & Poverty
    "q45_a_1": "Monthly Income - Total gross salary or profit in last month (PRIMARY POVERTY INDICATOR)",

    # Formality
    "q47": "Informal Flag - Workplace formality (1=Formal/Registered/Accounts, 2=Informal/Not registered)",

    # Digital Skills
    "q60a": "Computer Literacy (1=Can use computer, 2=Cannot use)",
    "q61": "Internet Use - Used internet in last 12 months",

    # Location
    "sector": "Residential Sector (1=Urban, 2=Rural, 3=Estate)",
    "district": "Administrative District"
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
        
        # Initialize LlamaIndex with Hugging Face Inference API (ONLINE)
        try:
            # Get Hugging Face token from environment variable (REQUIRED)
            hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")
            if not hf_token:
                raise ValueError(
                    "❌ Hugging Face token not found!\n"
                    "   Set environment variable: HUGGINGFACE_TOKEN\n"
                    "   Get token from: https://huggingface.co/settings/tokens"
                )
            
            # Use Hugging Face Inference API with DeepSeek-R1 model (ONLINE - no local download)
            Settings.llm = HuggingFaceInferenceAPI(
                model_name="deepseek-ai/DeepSeek-R1",
                token=hf_token,
                context_window=8192,
                num_output=1024,
                temperature=0.1
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
                instruction_str += f"- {col.upper()}: {desc}\n"

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
ANALYSIS GUIDELINES:
1. Disability questions (P15-P20): 1=No difficulty, 2=Some, 3=A lot, 4=Cannot do
   - To find people WITH difficulties: df[df['pXX'] > 1]

2. Poverty indicator: Q45_A_1 (Monthly Income) is PRIMARY poverty measure
   - Also consider Q47 (informal vs formal employment)
   - Q2 shows if person worked last week

3. Employment analysis:
   - Q16 = Employment status
   - Q20 = Hours worked (underemployment)
   - Q8 = Occupation/job type

4. Skills & literacy:
   - P10 = Education level
   - P12-P14 = Language literacy (Sinhala, Tamil, English)
   - Q60A = Computer literacy
   - Q61 = Internet use

5. Demographics & location:
   - P4 = Gender (1=Male, 2=Female)
   - P7 = Ethnicity
   - P9 = Marital status
   - SECTOR = Urban/Rural/Estate
   - DISTRICT = Geographic location

Return actual data and clear insights. Be specific and concise."""

            # Initialize PandasQueryEngine if we have data
            if self.df is not None:
                self.query_engine = PandasQueryEngine(
                    df=self.df,
                    instruction_str=instruction_str,
                    verbose=True,
                    synthesize_response=True
                )
                print("✅ LlamaIndex PandasQueryEngine ready with Hugging Face API!")
            print("📌 Using model: deepseek-ai/DeepSeek-R1 (ONLINE - Inference API)")
            print("🌐 All queries will be processed via Hugging Face API")
        except ImportError as ie:
            print(f"⚠️ Hugging Face API integration not installed: {ie}")
            print("📦 Install with: pip install llama-index-llms-huggingface-api")
            self.query_engine = None
        except Exception as e:
            print(f"⚠️ Hugging Face API error: {e}")
            print("Please check:")
            print("  1. Your Hugging Face token is valid (get from https://huggingface.co/settings/tokens)")
            print("  2. You have access to deepseek-ai/DeepSeek-R1 model")
            print("  3. Your internet connection is working")
            print("  4. Token has 'inference' permission enabled")
            print("\n💡 Visit model page: https://huggingface.co/deepseek-ai/DeepSeek-R1")
            self.query_engine = None
        
        # Conversation context for follow-up questions
        self.last_question = None
        self.last_answer = None
    
    
    def analyze_data(self, question: str):
        """
        Answer questions about the data using LlamaIndex PandasQueryEngine
        
        Args:
            question: User's question about the data
        
        Returns:
            LLM's answer as a string
        """
        print(f"\n🔍 Processing question: {question}")
        
        if self.df is None:
            return "⚠️ No data loaded. Please provide a dataset or CSV path."
        
        if self.query_engine is None:
            return "⚠️ Query engine not initialized. Please check your Hugging Face API configuration and internet connection."
        
        try:
            # Query using LlamaIndex PandasQueryEngine
            response = self.query_engine.query(question)
            
            # Store conversation context
            self.last_question = question
            self.last_answer = str(response)
            
            return str(response)
            
        except Exception as e:
            error_msg = f"⚠️ Error processing query: {str(e)}"
            print(error_msg)
            return error_msg
    
    def _prepare_data_context(self, limit: int):
        """Prepare a concise summary of the data for LLM context"""
        context_parts = []
        
        # Basic info
        context_parts.append(f"Dataset size: {len(self.df)} records")
        context_parts.append(f"Columns: {', '.join(self.df.columns.tolist())}")
        
        # P-column descriptions for context
        desc_lines = []
        for col in self.df.columns:
            # Case-insensitive check
            for desc_col, desc_text in COLUMN_DESCRIPTIONS.items():
                if col.lower() == desc_col.lower():
                    desc_lines.append(f"{col}: {desc_text}")
                    break
        
        if desc_lines:
            context_parts.append("\n=== P-COLUMN DESCRIPTIONS ===")
            context_parts.append("\n".join(desc_lines))
            scale_text = ", ".join([f"{k}={v}" for k, v in COLUMN_VALUE_SCALE.items()])
            context_parts.append(f"Answer scale for P-columns: {scale_text}")
            context_parts.append("="*30)
        
        # Statistical summary for all numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            stats = self.df[numeric_cols].describe().round(2)
            context_parts.append(f"\nNumeric column statistics:\n{stats.to_string()}")
        
        # Cluster information if available
        if self.has_clusters:
            cluster_dist = self.df['cluster_id'].value_counts().sort_index()
            context_parts.append(f"\nCluster distribution:\n{cluster_dist.to_string()}")
            
            # Cluster characteristics
            for cluster_id in sorted(self.df['cluster_id'].unique()):
                cluster_data = self.df[self.df['cluster_id'] == cluster_id]
                if numeric_cols:
                    means = cluster_data[numeric_cols[:5]].mean().round(2)  # Top 5 features
                    context_parts.append(f"\nCluster {cluster_id} average values:\n{means.to_string()}")
        
        # Sample data
        sample_size = min(10, limit)
        context_parts.append(f"\nSample data (first {sample_size} rows):\n{self.df.head(sample_size).to_string()}")
        
        return "\n\n".join(context_parts)
    
    def ask_about_clusters(self, question: str):
        """
        Specifically answer questions about clusters
        
        Args:
            question: Question about clusters
        
        Returns:
            Answer with cluster analysis
        """
        if not self.has_clusters:
            return "⚠️ No cluster information available in the dataset."
        
        if self.llm is None:
            return "⚠️ Ollama not running. Please start Ollama."
        
        # Extract cluster-specific context
        cluster_context = self._get_cluster_analysis()
        
        prompt_text = f"""You are a clustering expert with direct access to cluster data. Analyze the clusters below and answer the question.

IMPORTANT: Do NOT generate SQL queries. Provide direct analysis based on the data provided.

Cluster Analysis:
{cluster_context}

Question: {question}

Provide a direct answer with insights:"""
        
        try:
            answer = self.llm.invoke(prompt_text)
            return answer
        except Exception as e:
            return f"⚠️ Error: {str(e)}"
    
    def _get_cluster_analysis(self):
        """Generate detailed cluster analysis"""
        analysis_parts = []
        
        n_clusters = self.df['cluster_id'].nunique()
        analysis_parts.append(f"Total clusters: {n_clusters}")
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if 'cluster_id' in numeric_cols:
            numeric_cols.remove('cluster_id')
        
        for cluster_id in sorted(self.df['cluster_id'].unique()):
            cluster_data = self.df[self.df['cluster_id'] == cluster_id]
            
            parts = [f"\n--- Cluster {cluster_id} ---"]
            parts.append(f"Size: {len(cluster_data)} records ({len(cluster_data)/len(self.df)*100:.1f}%)")
            
            if numeric_cols:
                # Statistics for this cluster
                means = cluster_data[numeric_cols].mean()
                stds = cluster_data[numeric_cols].std()
                
                parts.append("\nAverage values:")
                for col in numeric_cols[:8]:  # Top 8 features
                    parts.append(f"  {col}: {means[col]:.2f} (±{stds[col]:.2f})")
            
            analysis_parts.append("\n".join(parts))
        
        return "\n".join(analysis_parts)
    
    def compare_clusters(self, cluster_ids: list = None):
        """
        Compare specific clusters or all clusters
        
        Args:
            cluster_ids: List of cluster IDs to compare, or None for all clusters
        
        Returns:
            Comparison analysis
        """
        if not self.has_clusters:
            return "⚠️ No cluster information available."
        
        if self.llm is None:
            return "⚠️ Ollama not running."
        
        if cluster_ids is None:
            cluster_ids = sorted(self.df['cluster_id'].unique())
        
        comparison = self._get_cluster_comparison(cluster_ids)
        
        prompt_text = f"""You are a clustering analyst. Compare these data clusters and explain the key differences.

IMPORTANT: Do NOT generate SQL queries. Provide direct comparison based on the statistics shown.

{comparison}

Provide a clear comparison with specific differences:"""
        
        try:
            answer = self.llm.invoke(prompt_text)
            return answer
        except Exception as e:
            return f"⚠️ Error: {str(e)}"
    
    def _get_cluster_comparison(self, cluster_ids):
        """Generate cluster comparison data"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if 'cluster_id' in numeric_cols:
            numeric_cols.remove('cluster_id')
        
        comparison_data = []
        
        for cluster_id in cluster_ids:
            cluster_data = self.df[self.df['cluster_id'] == cluster_id]
            
            if numeric_cols:
                means = cluster_data[numeric_cols].mean().round(2)
                comparison_data.append(f"\nCluster {cluster_id} ({len(cluster_data)} records):")
                comparison_data.append(means.to_string())
        
        return "\n".join(comparison_data)
    
    def get_insights(self, topic: str = None):
        """
        Get general insights about the data
        
        Args:
            topic: Specific topic to focus on (optional)
        
        Returns:
            Data insights
        """
        if self.df is None:
            return "⚠️ No data loaded."
        
        if self.llm is None:
            return "⚠️ Ollama not running."
        
        data_context = self._prepare_data_context(100)
        
        if topic:
            question = f"What insights can you provide about {topic} in this dataset?"
        else:
            question = "What are the key insights and patterns in this dataset?"
        
        prompt_text = f"""You are a data scientist analyzing this dataset. Provide insights based on the data provided.

IMPORTANT: Do NOT generate SQL queries or code. Analyze the data summary and provide direct insights.

Data:
{data_context}

Question: {question}

Key insights based on the data:"""
        
        try:
            answer = self.llm.invoke(prompt_text)
            return answer
        except Exception as e:
            return f"⚠️ Error: {str(e)}"


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
