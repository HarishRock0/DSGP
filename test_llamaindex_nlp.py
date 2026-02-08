"""
Test script for LlamaIndex-based NLP Query Engine
"""
import sys
import os

# Add NLP module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'NLP'))
from NLP import LLMQueryEngine

def main():
    print("="*70)
    print("Testing LlamaIndex NLP Query Engine")
    print("="*70)
    
    # Initialize the query engine
    engine = LLMQueryEngine(csv_path="data/LFS-2023.csv")
    
    if engine.query_engine is None:
        print("\n❌ Failed to initialize query engine")
        print("Make sure Ollama is running: ollama serve")
        print("And llama3.2 is installed: ollama pull llama3.2")
        return
    
    print("\n" + "="*70)
    print("Ready to answer questions!")
    print("="*70)
    
    # Test query
    test_question = "list first 50 records who have difficulties in walking or climbing"
    
    print(f"\n📝 Question: {test_question}")
    print("-"*70)
    
    answer = engine.analyze_data(test_question)
    
    print(f"\n💡 Answer:\n{answer}")
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
