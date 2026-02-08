"""
Database setup and management for LFS-2023 dataset
Creates SQLite database from CSV and provides query interface
"""

import sqlite3
import pandas as pd
import os
from pathlib import Path

class DatabaseManager:
    """Manage SQLite database for dataset"""
    
    def __init__(self, db_path="db/lfs_database.db", csv_path="data/LFS-2023.csv"):
        """
        Initialize database manager
        
        Args:
            db_path: Path to SQLite database file
            csv_path: Path to CSV data file
        """
        self.db_path = db_path
        self.csv_path = csv_path
        self.connection = None
        
        # Create db directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
    def create_database(self, force_recreate=False):
        """
        Create database from CSV file
        
        Args:
            force_recreate: If True, delete existing database and recreate
        """
        # Check if database exists
        if os.path.exists(self.db_path) and not force_recreate:
            print(f"✅ Database already exists at {self.db_path}")
            return True
        
        if force_recreate and os.path.exists(self.db_path):
            print(f"🔄 Recreating database...")
            os.remove(self.db_path)
        
        try:
            print(f"📊 Loading CSV data from {self.csv_path}...")
            df = pd.read_csv(self.csv_path)
            
            print(f"📝 Creating SQLite database with {len(df)} records and {len(df.columns)} columns...")
            
            # Connect to database
            conn = sqlite3.connect(self.db_path)
            
            # Create table from dataframe
            df.to_sql('lfs_data', conn, index=False, if_exists='replace')
            
            # Create indexes on commonly queried columns
            cursor = conn.cursor()
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_year ON lfs_data(YEAR)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sector ON lfs_data(SECTOR)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_district ON lfs_data(DISTRICT)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_age ON lfs_data(AGE)")
            
            # Create indexes for P-columns
            for i in range(15, 22):
                cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_p{i} ON lfs_data(P{i})")
            
            conn.commit()
            conn.close()
            
            print(f"✅ Database created successfully at {self.db_path}")
            print(f"📈 Table: lfs_data ({len(df)} rows × {len(df.columns)} columns)")
            return True
            
        except Exception as e:
            print(f"❌ Error creating database: {e}")
            return False
    
    def connect(self):
        """Connect to database"""
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.connection.row_factory = sqlite3.Row  # Access columns by name
            return True
        except Exception as e:
            print(f"❌ Error connecting to database: {e}")
            return False
    
    def disconnect(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            self.connection = None
    
    def execute_query(self, query):
        """
        Execute SQL query and return results as list of dicts
        
        Args:
            query: SQL query string
            
        Returns:
            List of result rows as dictionaries
        """
        if not self.connection:
            if not self.connect():
                return None
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(query)
            
            # Fetch results
            columns = [description[0] for description in cursor.description]
            results = []
            
            for row in cursor.fetchall():
                results.append(dict(zip(columns, row)))
            
            return results
        
        except Exception as e:
            print(f"❌ Query execution error: {e}")
            return None
    
    def get_dataframe_from_query(self, query):
        """
        Execute query and return results as pandas DataFrame
        
        Args:
            query: SQL query string
            
        Returns:
            pandas DataFrame with results
        """
        if not self.connection:
            if not self.connect():
                return None
        
        try:
            return pd.read_sql_query(query, self.connection)
        except Exception as e:
            print(f"❌ Query error: {e}")
            return None
    
    def get_table_info(self):
        """Get information about database tables and columns"""
        if not self.connection:
            if not self.connect():
                return None
        
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT * FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            
            info = {}
            for table in tables:
                table_name = table[1]
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()
                info[table_name] = [col[1] for col in columns]
            
            return info
        except Exception as e:
            print(f"❌ Error getting table info: {e}")
            return None
    
    def get_record_count(self):
        """Get total number of records in dataset"""
        results = self.execute_query("SELECT COUNT(*) as count FROM lfs_data")
        if results:
            return results[0]['count']
        return 0
    
    def get_column_unique_values(self, column_name):
        """Get unique values in a column"""
        try:
            results = self.execute_query(f"SELECT DISTINCT {column_name} FROM lfs_data ORDER BY {column_name}")
            if results:
                return [r[column_name] for r in results]
            return []
        except Exception as e:
            print(f"❌ Error: {e}")
            return []


if __name__ == "__main__":
    # Initialize database manager
    db = DatabaseManager()
    
    # Create database
    db.create_database(force_recreate=False)
    
    # Connect and get info
    if db.connect():
        info = db.get_table_info()
        print(f"\n📋 Database Tables:")
        for table, columns in info.items():
            print(f"\n  {table}: {len(columns)} columns")
            print(f"    {', '.join(columns[:10])}...")
        
        # Get record count
        count = db.get_record_count()
        print(f"\n📊 Total records: {count}")
        
        db.disconnect()
