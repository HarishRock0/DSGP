import os
import pickle
import pandas as pd
import numpy as np
import warnings

from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.llms.groq import Groq
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
        
        # Initialize LlamaIndex with Groq (FAST API)
        try:
            # Get API key from environment variable
            groq_api_key = os.getenv('GROQ_API_KEY')
            
            if not groq_api_key:
                print("⚠️ GROQ_API_KEY environment variable not set!")
                print("Please set it using one of these methods:")
                print("  Windows CMD: set GROQ_API_KEY=your_api_key_here")
                print("  Windows PowerShell: $env:GROQ_API_KEY='your_api_key_here'")
                print("  Linux/Mac: export GROQ_API_KEY=your_api_key_here")
                print("\n💡 Get your API key from: https://console.groq.com/")
                self.query_engine = None
                return
            
            # Use Groq with currently supported model
            Settings.llm = Groq(
                model="llama-3.3-70b-versatile",
                api_key=groq_api_key,
                temperature=0.1,
                max_tokens=8000  # Increased to allow full beneficiary lists
            )
            print("📌 Using model: llama-3.3-70b-versatile (Groq API)")
            print("✅ Groq LLM ready for direct data analysis!")
            print("🚀 All queries will be processed via Groq API (FAST)")
        except ImportError as ie:
            print(f"⚠️ Groq integration not installed: {ie}")
            print("📦 Install with: pip install llama-index-llms-groq")
            self.query_engine = None
        except Exception as e:
            print(f"⚠️ Groq API error: {e}")
            print("Please check:")
            print("  1. Your Groq API key is valid")
            print("  2. You have internet connection")
            print("  3. Groq service is available")
            print("\n💡 Get API key from: https://console.groq.com/")
            self.query_engine = None
        
        # Conversation context for follow-up questions
        self.last_question = None
        self.last_answer = None
    
    
    def analyze_data(self, question: str):
        """
        Answer questions about the data using two-step LLM approach:
        1. Identify relevant columns/clusters for the question
        2. Analyze only relevant data subset
        
        Args:
            question: User's question about the data
        
        Returns:
            LLM's answer as a string
        """
        print(f"\n🔍 Processing question: {question}")
        
        if self.df is None:
            return "⚠️ No data loaded. Please provide a dataset or CSV path."
        
        if Settings.llm is None:
            return "⚠️ LLM not initialized. Please check your Groq API configuration and internet connection."
        
        try:
            from llama_index.core.llms import ChatMessage
            
            # STEP 1: Identify relevant columns and analysis approach
            print("📊 Step 1: Identifying relevant columns for analysis...")
            
            step1_prompt = f"""You are a data analyst. Analyze this question and identify which columns are needed.

AVAILABLE COLUMNS:
"""
            for col, desc in COLUMN_DESCRIPTIONS.items():
                step1_prompt += f"- {col.upper()}: {desc}\n"
            
            step1_prompt += f"""
DATASET HAS {len(self.df)} records with columns: {', '.join(self.df.columns.tolist())}
HAS CLUSTERS: {self.has_clusters}

USER QUESTION: {question}

Respond with ONLY a JSON object (no markdown, no code blocks):
{{
  "relevant_columns": ["col1", "col2", ...],
  "needs_clusters": true/false,
  "analysis_type": "resource_allocation" or "statistics" or "comparison" or "cluster_analysis",
  "filter_criteria": "description of what data to filter (if any)"
}}
"""
            
            messages1 = [ChatMessage(role="user", content=step1_prompt)]
            response1 = Settings.llm.chat(messages1)
            
            # Parse the response to get relevant columns
            import json
            import re
            
            response_text = str(response1.message.content).strip()
            # Try to extract JSON if wrapped in markdown
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(0)
            
            column_info = json.loads(response_text)
            relevant_cols = column_info.get('relevant_columns', [])
            needs_clusters = column_info.get('needs_clusters', False)
            analysis_type = column_info.get('analysis_type', 'statistics')
            
            # Extract number of items from question (e.g., "100 cars")
            number_match = re.search(r'(\d+)\s+\w+', question)
            num_items = int(number_match.group(1)) if number_match else 0
            
            print(f"✅ Identified relevant columns: {', '.join(relevant_cols)}")
            print(f"✅ Analysis type: {analysis_type}")
            if num_items > 0:
                print(f"✅ Number of items to distribute: {num_items}")
            
            # STEP 2: Prepare focused data context with only relevant columns
            print("📊 Step 2: Analyzing relevant data...")
            
            # Get only relevant columns that exist in the dataframe
            existing_relevant_cols = [col for col in relevant_cols if col in self.df.columns or col.upper() in self.df.columns or col.lower() in self.df.columns]
            
            # Build column mapping (handle case sensitivity)
            col_mapping = {}
            for col in existing_relevant_cols:
                for df_col in self.df.columns:
                    if df_col.lower() == col.lower():
                        col_mapping[col] = df_col
                        break
            
            actual_cols = list(col_mapping.values())
            
            # Create focused dataframe
            if actual_cols:
                focused_df = self.df[actual_cols].copy()
            else:
                # If no specific columns identified, use a limited set
                focused_df = self.df.iloc[:, :10].copy()  # First 10 columns
            
            # For resource allocation, prepare beneficiary list
            beneficiary_list = None
            if analysis_type == "resource_allocation" and num_items > 0:
                # Sort by need (lower income = higher priority)
                # Find income column
                income_col = None
                for col in focused_df.columns:
                    if 'q45' in col.lower() or 'income' in col.lower():
                        income_col = col
                        break
                
                if income_col:
                    # Get top N beneficiaries sorted by income (ascending - poorest first)
                    sorted_df = focused_df.dropna(subset=[income_col]).sort_values(income_col)
                    top_beneficiaries = sorted_df.head(num_items)
                    
                    beneficiary_list = f"""
TOP {len(top_beneficiaries)} BENEFICIARIES (Sorted by need - lowest income first):
{top_beneficiaries.to_string(index=True)}
"""
            
            # Prepare compact data context
            df_info = f"""
RELEVANT DATA FOR ANALYSIS:
- Total records: {len(focused_df)}
- Analyzing columns: {', '.join(focused_df.columns.tolist())}

SAMPLE DATA (first 10 rows):
{focused_df.head(10).to_string()}

STATISTICS FOR RELEVANT COLUMNS:
{focused_df.describe(include='all').to_string()}
"""
            
            if beneficiary_list:
                df_info += beneficiary_list
            
            # Add cluster analysis if needed
            if needs_clusters and self.has_clusters:
                try:
                    # Build aggregation for relevant columns only
                    agg_dict = {}
                    for col in focused_df.columns:
                        if col != 'cluster_id' and focused_df[col].dtype in ['int64', 'float64']:
                            agg_dict[col] = ['mean', 'count']
                    
                    if agg_dict:
                        cluster_summary = self.df.groupby('cluster_id').agg(agg_dict).head(20).to_string()
                        df_info += f"\n\nCLUSTER SUMMARY:\n{cluster_summary}\n"
                except Exception as cluster_err:
                    print(f"⚠️ Could not generate cluster summary: {cluster_err}")
            
            # STEP 3: Get final answer from LLM
            final_prompt = f"""You are a data analyst for Sri Lankan labor force survey data.

QUESTION: {question}

ANALYSIS TYPE: {analysis_type}

COLUMN MEANINGS (for reference):
"""
            for col in relevant_cols[:10]:  # Limit to top 10 relevant columns
                if col in COLUMN_DESCRIPTIONS:
                    final_prompt += f"- {col.upper()}: {COLUMN_DESCRIPTIONS[col]}\n"
            
            final_prompt += """

CRITICAL INSTRUCTIONS:
1. Analyze the actual data provided below
2. Calculate statistics from the sample and summary statistics
3. Return ONLY the final answer with actual numbers from the data
4. DO NOT return Python code
5. Be specific and data-driven

"""
            if analysis_type == "resource_allocation":
                final_prompt += f"""
FOR RESOURCE ALLOCATION OF {num_items} ITEMS:

I have provided you with a list of the TOP {num_items} BENEFICIARIES sorted by need (lowest income first).

CRITICAL: You MUST show ALL {num_items} beneficiaries. DO NOT abbreviate with "..." or skip rows.

Your task:
1. List EVERY SINGLE beneficiary from the provided list (all {num_items} rows)
2. For each beneficiary, show:
   - Row number (1 to {num_items})
   - Dataset index
   - Income value
   - Occupation code (if available)
   - Sector (if available)
   - District (if available)

Format your response as a COMPLETE LIST:

BENEFICIARY ALLOCATION LIST (ALL {num_items} items):

[Show ALL rows from 1 to {num_items} - DO NOT use "..." or skip any rows]

No. | Index | Income    | Occupation | Sector | District
----|-------|-----------|------------|--------|----------
1   | X     | Rs. X,XXX | XXXX       | X      | XX
2   | X     | Rs. X,XXX | XXXX       | X      | XX
3   | X     | Rs. X,XXX | XXXX       | X      | XX
[Continue for ALL {num_items} rows without abbreviation]

After the complete list, provide summary statistics:
- Total beneficiaries: {num_items}
- Average income: Rs. X
- Income range: Rs. X (lowest) to Rs. Y (highest)
- Distribution by sector: Urban (X), Rural (Y), Estate (Z)
- Top 5 districts with counts

REMEMBER: Show EVERY SINGLE ROW from the beneficiary list. No abbreviations!
"""
            else:
                final_prompt += """

CRITICAL INSTRUCTIONS:
1. Analyze the actual data provided below
2. Calculate statistics from the sample and summary statistics
3. Return ONLY the final answer with actual numbers from the data
4. DO NOT return Python code
5. Be specific and data-driven
"""
            
            final_prompt += df_info
            final_prompt += f"\n\nProvide your analysis:"
            
            messages2 = [ChatMessage(role="user", content=final_prompt)]
            response2 = Settings.llm.chat(messages2)
            
            # Store conversation context
            self.last_question = question
            self.last_answer = str(response2.message.content)
            
            return str(response2.message.content)
            
        except Exception as e:
            error_msg = f"⚠️ Error processing query: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
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
    print("💬 LLM Query Engine - Llama 3.3 70B via Groq API (FAST)")
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
