#!/usr/bin/env python
import os
import sys

# Load environment variables (API key should be set externally)
# os.environ['GROQ_API_KEY'] should be set before running this script

import pandas as pd
from NLP.NLP import LLMQueryEngine

# Load CSV
print("📂 Loading data...")
df = pd.read_csv('data/LFS-2023.csv')
print(f"✅ Loaded {len(df)} records")
print(f"✅ Columns (first 10): {df.columns[:10].tolist()}")
print(f"✅ Q2 col exists: {'Q2' in df.columns}")
print(f"✅ SECTOR col exists: {'SECTOR' in df.columns}")
print(f"✅ EDU col exists: {'EDU' in df.columns}")

# Initialize engine
print("\n🚀 Initializing LLM Query Engine...")
try:
    engine = LLMQueryEngine(df=df)
    print("✅ Engine initialized")
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

# Test query
print("\n🔄 Running query: 'i have 100 taxis whom should i give'")
try:
    result = engine.analyze_data('i have 100 taxis whom should i give')
    print("\n✅ Query completed!")
    print(f"Result type: {type(result)}")
except Exception as e:
    print(f"❌ Query failed: {e}")
    import traceback
    traceback.print_exc()
