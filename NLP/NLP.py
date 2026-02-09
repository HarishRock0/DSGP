import os
import pickle
import pandas as pd
import numpy as np
import warnings

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


class SkillDev:
    """Stub class to support unpickling SkillDev model instances"""
    pass


class LLMQueryEngine:
    """
    LLM-powered pandas query engine using LlamaIndex & Hugging Face API
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
            hf_token = (os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HUGGINGFACE_API_KEY") or "").strip()
            if not hf_token:
                raise ValueError(
                    "❌ Hugging Face token not found!\n"
                    "   Set environment variable: HUGGINGFACE_TOKEN\n"
                    "   Get token from: https://huggingface.co/settings/tokens"
                )
            
            # Use Hugging Face Inference API with Falcon-7B model (ONLINE - FREE)
            Settings.llm = HuggingFaceInferenceAPI(
                model_name="tiiuae/falcon-7b-instruct",
                token=hf_token,
                context_window=2048,
                num_output=512,
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
            print("📌 Using model: tiiuae/falcon-7b-instruct (FREE)")
            print("🌐 All queries will be processed via Hugging Face API")
        except ImportError as ie:
            print(f"⚠️ Hugging Face API integration not installed: {ie}")
            print("📦 Install with: pip install llama-index-llms-huggingface-api")
            self.query_engine = None
        except Exception as e:
            print(f"⚠️ Hugging Face API error: {e}")
            print("Please check:")
            print("  1. Your Hugging Face token is valid (get from https://huggingface.co/settings/tokens)")
            print("  2. You have access to tiiuae/falcon-7b-instruct model")
            print("  3. Your internet connection is working")
            print("  4. Token has 'inference' permission enabled")
            print("\n💡 Visit model page: https://huggingface.co/tiiuae/falcon-7b-instruct")
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


# Main execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='LLM Query Engine for LFS-2023 Data Analysis')
    parser.add_argument('--model', type=str, help='Path to model pickle file')
    parser.add_argument('--csv', type=str, default='data/LFS-2023.csv', help='Path to CSV file')
    
    args = parser.parse_args()
    
    # Determine data source
    if args.model:
        model_path = args.model
        csv_path = None
    else:
        # Try to find model file
        candidate_paths = [
            os.path.join('..', 'model', 'skilldev_model.pkl'),
            os.path.join('model', 'skilldev_model.pkl'),
        ]
        
        model_path = None
        for path in candidate_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        csv_path = args.csv
    
    # Initialize engine
    if model_path:
        print(f"✅ Using model from: {model_path}\n")
        engine = LLMQueryEngine(model_path=model_path)
    else:
        print(f"✅ Using CSV from: {csv_path}\n")
        engine = LLMQueryEngine(csv_path=csv_path)
    
    # Interactive query loop
    print("\n" + "=" * 70)
    print("💬 LLM Query Engine - Falcon-7B via Hugging Face API (FREE)")
    print("=" * 70)
    print("Ask questions about your LFS-2023 data!")
    print("Examples:")
    print("  - How many people have vision difficulties?")
    print("  - What is the average income by district?")
    print("  - Compare employment rates by gender")
    if engine.has_clusters:
        print("  - What are the characteristics of each cluster?")
    print("\nType 'quit' or 'exit' to stop\n")
    
    while True:
        query = input("📝 Your query: ").strip()
        
        if query.lower() in ['quit', 'exit']:
            print("✅ Goodbye!")
            break
        
        if not query:
            continue
        
        response = engine.analyze_data(query)
        print(f"\n{response}\n")
