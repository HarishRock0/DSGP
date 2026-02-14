#!/usr/bin/env python3
import os
import sys

print("Testing Groq Integration...")
print("=" * 70)

# Check API key
groq_key = os.getenv("GROQ_API_KEY")
print(f"1️⃣  Groq API Key: {'✅ SET' if groq_key else '❌ NOT SET'}")

# Test imports
try:
    from llama_index.llms.groq import Groq
    print(f"2️⃣  Groq import: ✅ SUCCESS")
except ImportError as e:
    print(f"2️⃣  Groq import: ❌ FAILED - {e}")
    sys.exit(1)

try:
    from llama_index.experimental.query_engine import PandasQueryEngine
    print(f"3️⃣  PandasQueryEngine import: ✅ SUCCESS")
except ImportError as e:
    print(f"3️⃣  PandasQueryEngine import: ❌ FAILED - {e}")
    sys.exit(1)

print("=" * 70)
print("✅ All imports successful! System is ready.")
