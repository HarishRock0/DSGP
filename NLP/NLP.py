import os
import pickle
import warnings
import torch
import sys

warnings.filterwarnings('ignore')

# Make the Engines/ sub-package importable regardless of run directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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


# Agentic AI entry point
from Engines.agent import LFSAgent


if __name__ == "__main__":
    # ── Locate pretrained model ──────────────────────────────────────────────
    candidate_paths = [
        os.path.join('..', 'model', 'skilldev_model.pkl'),
        os.path.join('model', 'skilldev_model.pkl'),
    ]
    valid_path = next((p for p in candidate_paths if os.path.exists(p)), None)

    if not valid_path:
        print("❌ No pretrained model found. Please supply skilldev_model.pkl.")
        sys.exit(1)

    print(f"✅ Using model: {valid_path}")

    # ── Create the agentic AI ────────────────────────────────────────────────
    # LFSAgent initialises LLMQueryEngine + NLPClusterQueryEngine, registers
    # all @tool functions, and builds a ReActAgent that autonomously selects
    # the right tool(s) for every user query.
    agent = LFSAgent(model_path=valid_path, verbose=True)

    print("Available tools  : allocate_resources | compare_clusters | query_cluster")
    print("                   get_insights | analyze_demographics | find_outliers")
    print("                   get_cluster_stats | get_data_schema")
    print("Mode             :", agent.mode)
    print("Type 'quit' to exit, 'reset' to clear conversation memory.\n")

    # ── Agentic chat loop ────────────────────────────────────────────────────
    while True:
        try:
            query = input("📝 Your query: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n✅ Goodbye!")
            break

        if not query:
            continue

        if query.lower() in ('quit', 'exit', 'bye'):
            print("✅ Goodbye!")
            break

        if query.lower() == 'reset':
            agent.reset()
            print("🔄 Conversation memory cleared.")
            continue

        print()
        print(agent.chat(query))
        print()

