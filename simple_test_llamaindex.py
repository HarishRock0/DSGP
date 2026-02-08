"""
Simple test for LlamaIndex NLP integration
"""
import pandas as pd
from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.llms.ollama import Ollama as LlamaIndexOllama
from llama_index.core import Settings

print("Testing LlamaIndex setup...")

# Load sample data
print("\n1. Loading CSV data...")
df = pd.read_csv("data/LFS-2023.csv")
print(f"✅ Loaded {len(df)} records with {len(df.columns)} columns")

# Configure LlamaIndex
print("\n2. Configuring LlamaIndex with Ollama...")
Settings.llm = LlamaIndexOllama(
    model="llama3.2:1b",  # Smaller 1B model for low RAM systems
    request_timeout=120.0,
    temperature=0.1
)
print("✅ LLM configured")

# Create query engine
print("\n3. Creating PandasQueryEngine...")
instruction_str = """You are a data analyst. Answer questions about this dataset.

Important columns:
- p17: Do you have difficulty walking or climbing steps?
  Answer scale: 1=No difficulties, 2=Minor difficulties, 3=Major difficulties, 4=Cannot do

When asked for records with difficulties, filter where p17 > 1.
Return actual data rows, not code."""

query_engine = PandasQueryEngine(
    df=df,
    instruction_str=instruction_str,
    verbose=True,
    synthesize_response=True
)
print("✅ Query engine created")

# Test query
print("\n4. Testing query...")
question = "list first 50 records who have difficulties in walking or climbing"
print(f"Question: {question}")

response = query_engine.query(question)
print(f"\n📊 Response:\n{response}")
