
"""
NLP.py — Entry point for the LFS-2023 Agentic AI system.

Run:
    python NLP.py

Requirements:
    • GROQ_API_KEY set in .env
    • skilldev_model.pkl in ../model/ or ./model/
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class SkillDev:
    """Minimal stub to support unpickling SkillDev instances."""
    pass


from Engines.agent import LFSAgent


if __name__ == "__main__":
    # Locate pretrained model
    nlp_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_paths = [
        os.path.join(nlp_dir, '..', 'model', 'skilldev_model.pkl'),
        os.path.join(nlp_dir, 'model', 'skilldev_model.pkl'),
    ]
    valid_path = next((p for p in candidate_paths if os.path.exists(p)), None)

    if not valid_path:
        print("❌ No pretrained model found. Run: python train_model.py")
        sys.exit(1)

    print(f"✅ Model: {valid_path}")

    agent = LFSAgent(model_path=valid_path, verbose=True)

    print("\nTools: allocate_resources | compare_clusters | query_cluster")
    print("       get_insights | analyze_demographics | find_outliers")
    print("       get_cluster_stats | get_data_schema")
    print(f"Mode : {agent.mode}")
    print("Type 'quit' to exit, 'reset' to clear conversation memory.\n")

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
            print("🔄 Memory cleared.")
            continue

        print()
        print(agent.chat(query))
        print()