"""
SQL query generator for natural language questions
Converts user questions to appropriate SQL queries
"""

from typing import Optional, Dict, List, Tuple

class SQLQueryGenerator:
    """Generate SQL queries from natural language questions"""
    
    def __init__(self, columns: List[str]):
        """
        Initialize SQL generator
        
        Args:
            columns: List of available column names in database
        """
        self.columns = columns
        self.column_lower = {col.lower(): col for col in columns}
    
    def get_column_name(self, user_text: str) -> Optional[str]:
        """
        Find actual column name from user text (case-insensitive)
        
        Args:
            user_text: User's text potentially containing column name
            
        Returns:
            Actual column name or None
        """
        user_lower = user_text.lower()
        
        # Check for exact matches
        for lower_col, actual_col in self.column_lower.items():
            if lower_col in user_lower:
                return actual_col
        
        return None
    
    def generate_list_query(self, limit: Optional[int] = None) -> str:
        """
        Generate SELECT query to list records
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            SQL query string
        """
        if limit is None:
            return "SELECT * FROM lfs_data"
        else:
            return f"SELECT * FROM lfs_data LIMIT {limit}"
    
    def generate_count_query(self, column: Optional[str] = None) -> str:
        """
        Generate COUNT query
        
        Args:
            column: Optional column to count distinct values
            
        Returns:
            SQL query string
        """
        if column:
            return f"SELECT {column}, COUNT(*) as count FROM lfs_data GROUP BY {column} ORDER BY count DESC"
        else:
            return "SELECT COUNT(*) as total_records FROM lfs_data"
    
    def generate_average_query(self, column: str) -> str:
        """
        Generate average/mean query
        
        Args:
            column: Column name to average
            
        Returns:
            SQL query string
        """
        return f"SELECT {column}, AVG(CAST({column} AS FLOAT)) as average, COUNT(*) as count FROM lfs_data WHERE {column} IS NOT NULL GROUP BY {column}"
    
    def generate_stats_query(self, column: str) -> str:
        """
        Generate statistics query
        
        Args:
            column: Column name
            
        Returns:
            SQL query string
        """
        return f"""
        SELECT 
            COUNT(*) as total,
            COUNT(DISTINCT {column}) as unique_values,
            MIN(CAST({column} AS FLOAT)) as min_value,
            MAX(CAST({column} AS FLOAT)) as max_value,
            AVG(CAST({column} AS FLOAT)) as avg_value
        FROM lfs_data 
        WHERE {column} IS NOT NULL
        """
    
    def generate_filter_query(self, column: str, value) -> str:
        """
        Generate filtered query
        
        Args:
            column: Column to filter on
            value: Value to match
            
        Returns:
            SQL query string
        """
        if isinstance(value, str):
            return f"SELECT * FROM lfs_data WHERE {column} = '{value}' LIMIT 100"
        else:
            return f"SELECT * FROM lfs_data WHERE {column} = {value} LIMIT 100"
    
    def generate_where_query(self, conditions: List[Tuple[str, str, any]]) -> str:
        """
        Generate query with WHERE conditions
        
        Args:
            conditions: List of (column, operator, value) tuples
                       operator can be '=', '>', '<', '>=', '<=', 'IN', 'LIKE'
            
        Returns:
            SQL query string
        """
        where_clauses = []
        for col, op, val in conditions:
            if op.upper() == 'LIKE':
                where_clauses.append(f"{col} LIKE '%{val}%'")
            elif op.upper() == 'IN':
                if isinstance(val, list):
                    val_str = ','.join([f"'{v}'" if isinstance(v, str) else str(v) for v in val])
                    where_clauses.append(f"{col} IN ({val_str})")
            else:
                if isinstance(val, str):
                    where_clauses.append(f"{col} {op} '{val}'")
                else:
                    where_clauses.append(f"{col} {op} {val}")
        
        where_string = " AND ".join(where_clauses)
        return f"SELECT * FROM lfs_data WHERE {where_string} LIMIT 100"
    
    def generate_group_by_query(self, group_columns: List[str], aggregate_col: Optional[str] = None) -> str:
        """
        Generate GROUP BY query
        
        Args:
            group_columns: Columns to group by
            aggregate_col: Column to aggregate (count/sum/avg)
            
        Returns:
            SQL query string
        """
        group_str = ", ".join(group_columns)
        
        if aggregate_col:
            return f"SELECT {group_str}, COUNT(*) as count FROM lfs_data GROUP BY {group_str} ORDER BY count DESC"
        else:
            return f"SELECT {group_str}, COUNT(*) as count FROM lfs_data GROUP BY {group_str} ORDER BY {group_str}"
    
    def parse_and_generate(self, question: str) -> Optional[str]:
        """
        Parse natural language question and generate SQL query
        
        Args:
            question: User's natural language question
            
        Returns:
            SQL query string or None if pattern not recognized
        """
        q_lower = question.lower()
        
        # List all records
        if any(pattern in q_lower for pattern in 
               ['list all', 'show all', 'get all', 'display all', 'all records']):
            import re
            match = re.search(r'(\d+)', question)
            limit = int(match.group(1)) if match else None
            return self.generate_list_query(limit)
        
        # Count records
        if any(pattern in q_lower for pattern in ['how many', 'count']):
            col = self.get_column_name(question)
            return self.generate_count_query(col)
        
        # Average/mean
        if any(pattern in q_lower for pattern in ['average', 'mean', 'avg']):
            col = self.get_column_name(question)
            if col:
                return self.generate_average_query(col)
        
        # Statistics
        if any(pattern in q_lower for pattern in ['statistics', 'stats', 'summary']):
            col = self.get_column_name(question)
            if col:
                return self.generate_stats_query(col)
        
        # Filter by value
        if any(pattern in q_lower for pattern in ['where', 'filter', 'find records', 'records with']):
            col = self.get_column_name(question)
            if col:
                import re
                # Try to extract value
                match = re.search(r'(=|is|equals?|where\s+\w+\s*=?\s*)(\w+)', question)
                if match:
                    value = match.group(2)
                    try:
                        value = int(value)
                    except:
                        pass
                    return self.generate_filter_query(col, value)
        
        # Group by queries
        if any(pattern in q_lower for pattern in ['group by', 'grouped by', 'by']):
            col = self.get_column_name(question)
            if col:
                return self.generate_group_by_query([col])
        
        return None


if __name__ == "__main__":
    # Test SQL generator
    sample_columns = ['YEAR', 'MONTH', 'SECTOR', 'DISTRICT', 'AGE', 'P15', 'P16', 'P17']
    
    gen = SQLQueryGenerator(sample_columns)
    
    test_questions = [
        "list all records",
        "show first 50 records",
        "how many records are there",
        "count records by sector",
        "average age",
        "statistics for age",
        "records where age is 30",
        "group by sector"
    ]
    
    print("SQL Query Generator Test\n" + "="*50)
    for q in test_questions:
        sql = gen.parse_and_generate(q)
        print(f"\nQ: {q}")
        print(f"SQL: {sql}")
