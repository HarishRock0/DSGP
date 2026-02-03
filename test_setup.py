"""Quick test to verify LFS analyzer is working"""
import os
import sys

print("=" * 60)
print("LFS Analyzer - System Check")
print("=" * 60)

# Check 1: Data file
print("\n[1/4] Checking data file...")
data_path = os.path.join('data', 'LFS-2023.csv')
if os.path.exists(data_path):
    print(f"✅ Found: {data_path}")
else:
    print(f"❌ Missing: {data_path}")
    sys.exit(1)

# Check 2: Ollama
print("\n[2/4] Checking Ollama...")
import subprocess
try:
    result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print("✅ Ollama is installed")
        if 'llama3.2' in result.stdout:
            print("✅ llama3.2 model found")
        else:
            print("⚠️  llama3.2 model not found - run: ollama pull llama3.2")
    else:
        print("⚠️  Ollama installed but not responding")
except FileNotFoundError:
    print("❌ Ollama not installed - download from https://ollama.ai")
    sys.exit(1)

# Check 3: Python packages
print("\n[3/4] Checking Python packages...")
try:
    import pandas
    import ollama as ollama_client
    from langchain_community.llms import Ollama
    print("✅ All packages installed")
except ImportError as e:
    print(f"❌ Missing package: {e}")
    print("Run: pip install -r requirement.txt")
    sys.exit(1)

# Check 4: Load data
print("\n[4/4] Testing data load...")
try:
    import pandas as pd
    df = pd.read_csv(data_path)
    print(f"✅ Loaded {len(df):,} records with {len(df.columns)} columns")
except Exception as e:
    print(f"❌ Error loading data: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ ALL CHECKS PASSED!")
print("=" * 60)
print("\nYou're ready to run:")
print("  python lfs_analyzer.py")
print("  OR")
print("  .\\run_analyzer.bat")
print()
