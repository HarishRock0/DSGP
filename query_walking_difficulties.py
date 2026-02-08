import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'database'))
from db_setup import DatabaseManager
import pandas as pd

db = DatabaseManager(db_path='db/lfs_database.db')
if db.connect():
    # p17 = 'Do you have difficulty walking or climbing steps?'
    # Values: 1=No, 2=Minor, 3=Major, 4=Cannot do
    # We want those with difficulty (value > 1)
    query = 'SELECT * FROM lfs_data WHERE p17 > 1 LIMIT 50'
    results = db.execute_query(query)
    
    if results:
        df = pd.DataFrame(results)
        print(f'Found {len(results)} records with walking/climbing difficulties:\n')
        print(df.to_string())
    else:
        print('No records found')
    db.disconnect()
