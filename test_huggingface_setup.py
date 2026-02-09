"""
Test Hugging Face API Configuration for Query Engine
"""
import os
import sys

print("="*60)
print("🔍 Testing Hugging Face API Configuration")
print("="*60)

# Test 1: Import packages
print("\n1️⃣ Testing imports...")
try:
    from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
    from llama_index.core import Settings
    print("✅ All required packages imported successfully")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Check API token
print("\n2️⃣ Checking API token...")
hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")
if not hf_token:
    print("❌ Error: HUGGINGFACE_TOKEN environment variable not set!")
    print("   Set it with: set HUGGINGFACE_TOKEN=your_token_here")
    print("   Get token from: https://huggingface.co/settings/tokens")
    sys.exit(1)

print(f"✅ Using token from environment: {hf_token[:10]}...")

# Test 3: Initialize Hugging Face API
print("\n3️⃣ Initializing Hugging Face Inference API...")
try:
    llm = HuggingFaceInferenceAPI(
        model_name="HuggingFaceH4/zephyr-7b-beta",
        token=hf_token,
        context_window=4096,
        num_output=512,
        temperature=0.1
    )
    Settings.llm = llm
    print("✅ Hugging Face API initialized successfully")
    print(f"📌 Model: HuggingFaceH4/zephyr-7b-beta")
except Exception as e:
    print(f"❌ Initialization failed: {e}")
    print("\n🔧 Troubleshooting:")
    print("  1. Check your internet connection")
    print("  2. Verify token at: https://huggingface.co/settings/tokens")
    print("  3. Make sure token has 'inference' permission")
    sys.exit(1)

# Test 4: Test simple query
print("\n4️⃣ Testing simple query...")
try:
    response = llm.complete("What is 2+2? Answer briefly.")
    print(f"✅ Query successful!")
    print(f"📝 Response: {response.text[:100]}...")
except Exception as e:
    print(f"❌ Query failed: {e}")
    print("\n🔧 Possible issues:")
    print("  1. Model might be loading (first time takes longer)")
    print("  2. Check API rate limits")
    print("  3. Verify model access permissions")
    sys.exit(1)

print("\n" + "="*60)
print("✅ All tests passed! Query engine is ready to use.")
print("="*60)
print("\n💡 Next steps:")
print("  - Run your NLP script: python NLP/NLP.py")
print("  - The query engine should initialize without errors")
print("  - If you still see errors, check the specific error message above")
