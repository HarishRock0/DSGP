#!/usr/bin/env python
"""Test Ollama integration with NLPC intent detection."""

import requests
import json
import sys
import os

# Add NLP to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'NLP'))

OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
OLLAMA_MODEL = "tinyllama"

def test_ollama_connectivity():
    """Test if Ollama is reachable."""
    print("\n" + "="*70)
    print("Testing Ollama Connectivity")
    print("="*70)
    
    try:
        response = requests.get(OLLAMA_TAGS_URL, timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = data.get('models', [])
            model_names = [m.get('name', 'unknown') for m in models]
            print(f"✅ Ollama is running!")
            print(f"   Available models: {', '.join(model_names)}")
            
            if OLLAMA_MODEL in model_names:
                print(f"✅ Model '{OLLAMA_MODEL}' is available")
                return True
            else:
                print(f"⚠️  Model '{OLLAMA_MODEL}' not found!")
                print(f"   Run: ollama pull {OLLAMA_MODEL}")
                return False
        else:
            print(f"❌ Ollama returned status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot reach Ollama at {OLLAMA_API_URL}")
        print(f"   Error: {e}")
        print(f"   Make sure Ollama is running: ollama serve")
        return False

def test_intent_detection():
    """Test intent detection with a sample query."""
    print("\n" + "="*70)
    print("Testing Intent Detection with Ollama")
    print("="*70)
    
    test_queries = [
        "Give 50 laptops to vulnerable workers",
        "What are the key insights from the data?",
        "Compare all clusters",
        "Hi, how are you?",
    ]
    
    candidate_labels = [
        "allocate or distribute resources to people",
        "compare clusters or segments",
        "find records in a specific cluster",
        "get insights or trends in the data",
        "analyze demographic or statistical patterns",
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: '{query}'")
        
        prompt = f"""You are an intent classifier. Given a user query, classify it into ONE of these intents:

Candidate intents:
{chr(10).join([f'- {label}' for label in candidate_labels])}

User query: "{query}"

Respond with ONLY the best matching intent label from the list above, nothing else."""
        
        try:
            response = requests.post(
                OLLAMA_API_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.3,
                },
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            response_text = result.get('response', '').strip()
            
            # Try to find which label was selected
            detected = None
            for label in candidate_labels:
                if label.lower() in response_text.lower():
                    detected = label
                    break
            
            if not detected:
                detected = "UNKNOWN"
            
            print(f"💬 Response: {response_text[:100]}{'...' if len(response_text) > 100 else ''}")
            print(f"🎯 Detected intent: {detected}")
            
        except Exception as e:
            print(f"❌ Error: {e}")

def test_nlpc_engine():
    """Test the actual NLPC engine with Ollama."""
    print("\n" + "="*70)
    print("Testing NLPC Engine with Ollama")
    print("="*70)
    
    try:
        from Engines.NLPC import NLPClusterQueryEngine
        from sklearn.preprocessing import RobustScaler
        import pickle
        
        model_path = os.path.join(os.path.dirname(__file__), 'model', 'skilldev_model.pkl')
        if not os.path.exists(model_path):
            print(f"⚠️  Model not found at {model_path}")
            print("   Cannot test NLPC engine without model")
            return False
        
        print(f"Loading model from {model_path}...")
        engine = NLPClusterQueryEngine(model_path=model_path)
        
        test_query = "Give 50 laptops to the most vulnerable"
        print(f"\n📝 Testing query: '{test_query}'")
        
        result = engine.understand_query(test_query)
        print(f"\n✅ NLPC Result:")
        print(f"   Intent: {result.get('intent', 'N/A')}")
        print(f"   Route: {result.get('route', 'N/A')}")
        print(f"   Confidence: {result.get('confidence', 'N/A')}")
        
        return True
        
    except ModuleNotFoundError as e:
        print(f"⚠️  Missing module: {e}")
        print("   Install dependencies: pip install -r requirement.txt")
        return False
    except Exception as e:
        print(f"❌ Error testing NLPC: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n🧪 Ollama Integration Test Suite\n")
    
    # Test 1: Ollama connectivity
    ollama_ok = test_ollama_connectivity()
    
    if ollama_ok:
        # Test 2: Intent detection
        test_intent_detection()
        
        # Test 3: NLPC engine (requires model)
        test_nlpc_engine()
    
    print("\n" + "="*70)
    print("Test Suite Complete")
    print("="*70 + "\n")
