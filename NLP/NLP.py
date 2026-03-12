import os
import pickle
import pandas as pd
import numpy as np
import warnings
import torch
import sys
import re
import json

warnings.filterwarnings('ignore')

# Make the Engines/ sub-package importable regardless of run directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Shared lookup tables — single source of truth in Engines/constants.py
from Engines.constants import (
    COLUMN_DESCRIPTIONS, COLUMN_VALUE_SCALE, EMPLOYMENT_STATUS,
    SECTOR_MAP, DISTRICT_MAP, ETHNICITY_MAP, RELIGION_MAP,
    MARITAL_MAP, PROVINCE_DISTRICTS,
)

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


# Engine imports (sys.path already set above)
from Engines.NLPC import NLPClusterQueryEngine    # BART intent router
from Engines.LLMQ import LLMQueryEngine           # Groq LLM responder


if __name__ == "__main__":
    # Locate the pretrained model pickle (cluster data required)
    candidate_paths = [
        os.path.join('..', 'model', 'skilldev_model.pkl'),
        os.path.join('model', 'skilldev_model.pkl'),
    ]
    valid_path = next((p for p in candidate_paths if os.path.exists(p)), None)

    if not valid_path:
        print("❌ No pretrained model found. Please supply skilldev_model.pkl.")
        sys.exit(1)

    print(f"✅ Using model: {valid_path}\n")

    # ── Initialise both engines ──────────────────────────────────────────────
    # NLPC loads BART for intent detection
    nlpc = NLPClusterQueryEngine(model_path=valid_path)
    # LLMQ loads the same pickle for data + Groq for LLM responses
    llm  = LLMQueryEngine(model_path=valid_path)

    print("\n" + "=" * 70)
    print("💬  LFS-2023 Query Interface")
    print("    Route: User → BART (NLPC) → intent → LLM handler (LLMQ)")
    print("=" * 70)
    print("Slash commands : /clusters [question]  /compare  /insights [topic]")
    print("Type 'quit' to exit.\n")

    while True:
        query = input("📝 Your query: ").strip()

        if not query:
            continue

        if query.lower() in ('quit', 'exit', 'bye'):
            print("✅ Goodbye!")
            break

        # ── Slash commands bypass BART (explicit user intent) ────────────────
        if query.startswith('/'):
            parts = query.split(' ', 1)
            cmd   = parts[0].lower()
            arg   = parts[1] if len(parts) > 1 else None

            if cmd == '/clusters':
                print(llm.ask_about_clusters(arg or "Tell me about the clusters."))
            elif cmd == '/compare':
                print(llm.compare_clusters())
            elif cmd == '/insights':
                print(llm.get_insights(arg))
            else:
                print("⚠️  Unknown command.  Use /clusters, /compare, /insights, or quit.")

        # ── Free-text queries: BART detects intent, route to LLMQ ───────────
        else:
            result = nlpc.understand_query(query)
            route  = result['route']

            if route == "resource_allocation":
                print(llm.handle_allocation(query))

            elif route == "compare_clusters":
                print(llm.compare_clusters())

            elif route == "cluster_query":
                print(llm.ask_about_clusters(query))

            elif route == "insights":
                print(llm.get_insights())

            else:   # general_analysis (demographic / statistical)
                print(llm.analyze_data(query))

        print()

