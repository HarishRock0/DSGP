#!/usr/bin/env python3
"""Quick test to verify Groq integration"""
import os
import sys

print("\n" + "="*70)
print("🔍 Groq Integration Test")
print("="*70)

# Check API key
groq_key = os.getenv("GROQ_API_KEY")
if groq_key:
    print(f"✅ GROQ_API_KEY is set: {groq_key[:20]}...")
else:
    print("❌ GROQ_API_KEY not set!")
    print("   Set it with: $env:GROQ_API_KEY = 'gsk_YOUR_KEY'")
    sys.exit(1)

# Test imports
print("\n📦 Testing imports...")
try:
    from llama_index.core import Settings
    print("✅ LlamaIndex core imported")
except ImportError as e:
    print(f"❌ Failed to import LlamaIndex core: {e}")

try:
    from llama_index.llms.groq import Groq
    print("✅ Groq LLM imported")
except ImportError as e:
    print(f"❌ Failed to import Groq: {e}")
    sys.exit(1)

# Initialize Groq
print("\n🚀 Initializing Groq LLM...")
try:
    Settings.llm = Groq(
        api_key=groq_key,
        model="llama-3.3-70b-versatile",
        temperature=0.1,
    )
    print("✅ Groq LLM initialized successfully!")
    print(f"   Model: mixtral-8x7b-32768")
    print(f"   Temperature: 0.1")
except Exception as e:
    print(f"❌ Failed to initialize Groq: {e}")
    sys.exit(1)

# Test a simple query
print("\n💬 Testing simple query...")
try:
    response = Settings.llm.complete("What is 2+2?")
    print(f"✅ Query successful!")
    print(f"   Response: {response.text}")
except Exception as e:
    print(f"❌ Query failed: {e}")
    sys.exit(1)

print("\n" + "="*70)
print("✅ All tests passed! Groq integration is working.")
print("="*70 + "\n")
