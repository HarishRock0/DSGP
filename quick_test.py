"""Quick test of the fixed Hugging Face configuration"""
import os
import pandas as pd

print("="*60)
print("🧪 Testing Fixed Query Engine")  
print("="*60)

# Create sample data
sample_data = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'income': [50000, 60000, 70000]
})

print("\n📊 Sample Data:")
print(sample_data)

# Import and test
print("\n🔧 Initializing query engine...")
try:
    import sys
    sys.path.append('NLP')
    from NLP import LLMQueryEngine
    
    engine = LLMQueryEngine(df=sample_data)
    
    if engine.query_engine is not None:
        print("\n✅ SUCCESS! Query engine initialized properly!")
        print("🎉 The error is FIXED!")
        
        print("\n💡 You can now use it to analyze your data:")
        print("   - engine.analyze_data('What is the average age?')")
        print("   - engine.analyze_data('Who has the highest income?')")
    else:
        print("\n⚠️ Query engine is None - check error messages above")
        
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
