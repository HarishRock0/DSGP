"""
Initialize SQL database and test NLP engine with SQL integration
"""

import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'database'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'NLP'))

from database.db_setup import DatabaseManager

def initialize_database():
    """Initialize the SQL database from CSV"""
    print("\n" + "="*70)
    print("🗄️ Database Initialization")
    print("="*70)
    
    db = DatabaseManager(
        db_path="db/lfs_database.db",
        csv_path="data/LFS-2023.csv"
    )
    
    # Create database
    if db.create_database():
        print("\n✅ Database initialized successfully!")
        
        # Connect and show info
        if db.connect():
            info = db.get_table_info()
            print(f"\n📋 Database Tables:")
            for table, columns in info.items():
                print(f"  Table: {table}")
                print(f"  Columns ({len(columns)}): {', '.join(columns[:10])}...")
            
            count = db.get_record_count()
            print(f"\n📊 Total records: {count:,}")
            
            db.disconnect()
            return True
    else:
        print("\n❌ Failed to initialize database")
        return False


def test_nlp_with_sql():
    """Test the NLP engine with SQL integration"""
    print("\n" + "="*70)
    print("🤖 Testing NLP Engine with SQL")
    print("="*70)
    
    try:
        # Import directly from NLP module
        sys.path.insert(0, 'NLP')
        from NLP.Engines import LLMQ
        
        # Initialize engine with database
        print("\n🔄 Initializing NLP Query Engine...")
        engine = LLMQ(db_path="db/lfs_database.db")
        
        if engine.sql_gen:
            print("\n✅ NLP engine initialized with SQL support!")
            
            # Test some queries
            test_queries = [
                "list first 5 records",
                "how many records",
                "what is p15",
            ]
            
            print("\n" + "="*70)
            print("Running test queries...")
            print("="*70)
            
            for query in test_queries:
                print(f"\n📝 Query: {query}")
                result = engine._execute_direct_query(query)
                if result:
                    # Show first 300 chars of result
                    preview = result[:300] + ("..." if len(result) > 300 else "")
                    print(f"✅ Result:\n{preview}")
                else:
                    print("⚠️ No result")
            
            return True
        else:
            print("\n⚠️ SQL generator not initialized")
            return False
    
    except Exception as e:
        print(f"\n❌ Error testing NLP: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "="*70)
    print("LFS-2023 Dataset - SQL Database & NLP Integration Setup")
    print("="*70)
    
    # Step 1: Initialize database
    db_ok = initialize_database()
    
    if db_ok:
        # Step 2: Test NLP
        nlp_ok = test_nlp_with_sql()
        
        if nlp_ok:
            print("\n" + "="*70)
            print("✅ All systems initialized successfully!")
            print("="*70)
            print("\n You can now use the NLP query engine with SQL support.")
            print(" Start interactive mode with: python -m NLP.NLP")
        else:
            print("\n⚠️ Database initialized but NLP test failed")
    else:
        print("\n❌ Failed to initialize database")
