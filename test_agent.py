#!/usr/bin/env python
"""Quick test of the conversational agent fixes."""

import os
import sys

# Navigate to NLP directory and test
os.chdir(os.path.join(os.path.dirname(__file__), 'NLP'))
sys.path.insert(0, os.getcwd())

from Engines.agent import LFSAgent

# Load model
model_path = os.path.join('..', 'model', 'skilldev_model.pkl')
agent = LFSAgent(model_path=model_path, verbose=False)

# Test conversational queries
test_queries = [
    "hi",
    "hello",
    "how are you?",
    "what can you do?",
    "help",
]

print("\n" + "=" * 70)
print("Testing Conversational Responses")
print("=" * 70)

for query in test_queries:
    print(f"\n📝 Query: {query}")
    response = agent.chat(query)
    print(f"💬 Response: {response}\n")
    print("-" * 70)

print("\n✅ Conversational test complete!")
