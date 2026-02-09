"""Test DeepSeek-R1 API connection"""
import os
import sys
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core import Settings

print("="*60)
print("🧪 Testing DeepSeek-R1 via Hugging Face API")
print("="*60)

# Get token from environment
hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")
if not hf_token:
    print("❌ Error: HUGGINGFACE_TOKEN environment variable not set!")
    print("   Set it with: set HUGGINGFACE_TOKEN=your_token_here")
    print("   Get token from: https://huggingface.co/settings/tokens")
    sys.exit(1)

print(f"\n🔑 Using token: {hf_token[:15]}...")

# Initialize API
print("\n🔌 Connecting to Hugging Face Inference API...")
try:
    llm = HuggingFaceInferenceAPI(
        model_name="deepseek-ai/DeepSeek-R1",
        token=hf_token,
        context_window=8192,
        num_output=1024,
        temperature=0.1
    )
    Settings.llm = llm
    print("✅ API connection initialized")
    
    # Test query
    print("\n🧪 Testing API call...")
    response = llm.complete("What is 5+3? Answer with just the number.")
    print(f"✅ API Response: {response.text[:200]}")
    
    print("\n" + "="*60)
    print("✅ SUCCESS! DeepSeek-R1 API is working!")
    print("🌐 Using ONLINE inference - no local download needed")
    print("="*60)
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\n🔧 Troubleshooting:")
    print("  1. Check token validity: https://huggingface.co/settings/tokens")
    print("  2. Verify model access: https://huggingface.co/deepseek-ai/DeepSeek-R1")
    print("  3. Ensure internet connection is active")
    print("  4. Token needs 'inference' permission")
