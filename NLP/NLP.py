import os
import pickle
import pandas as pd
import numpy as np
import warnings
import torch
import sys
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



# Main execution
#import Engines
import NLP.Engines.NLPC as NLPClusterQueryEngine
import NLP.Engines.LLMQ as LLMQueryEngine 
if __name__ == "__main__":
    #able to run from either root or NLP/ subdirectory
    candidate_paths = [
        os.path.join('..', 'model', 'skilldev_model.pkl'),
        os.path.join('model', 'skilldev_model.pkl'),
    ]

    valid_path = None
    for path in candidate_paths:
        if os.path.exists(path):
            valid_path = path
            break

    # LLM mode – **requires** a pretrained model with cluster data
    if valid_path:
        print(f"✅ Using model from: {valid_path}\n")
        engine = LLMQueryEngine(model_path=valid_path)
    else:
        print("❌ No pretrained model found. LLM mode cannot load the raw CSV directly.")
        print("Please supply a model pickle containing cluster information.")
        sys.exit(1)

        print("\n" + "=" * 70)
        print("💬 LLM Query Interface")
        print("=" * 70)
        print("Commands: /clusters, /compare, /insights [topic], quit\n")

        while True:
            query = input("📝 Your query: ").strip()

            if query.lower() in ['quit', 'exit', 'bye']:
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
