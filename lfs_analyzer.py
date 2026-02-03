"""
LFS (Labor Force Survey) Analyzer with LLM-Powered Q&A
Analyzes LFS-2023.csv data and allows users to ask questions about it using AI
"""

import os
import sys
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# Add NLP module to path
NLP_PATH = os.path.join(os.path.dirname(__file__), 'NLP')
if NLP_PATH not in sys.path:
    sys.path.insert(0, NLP_PATH)

from NLP import LLMQueryEngine


class LFSAnalyzer:
    """Analyzer for Labor Force Survey data with LLM capabilities"""
    
    def __init__(self, data_path=None):
        """
        Initialize the LFS Analyzer
        
        Args:
            data_path: Path to LFS-2023.csv file
        """
        print("🔄 Loading LFS-2023 data...")
        
        if data_path is None:
            data_path = os.path.join(os.path.dirname(__file__), 'data', 'LFS-2023.csv')
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"❌ LFS data file not found: {data_path}")
        
        # Load the data
        self.df = pd.read_csv(data_path)
        print(f"✅ Loaded {len(self.df)} records with {len(self.df.columns)} columns")
        
        # Initialize LLM Query Engine (Ollama - free & local)
        print("\n🤖 Initializing AI Query Engine (Ollama)...")
        self.engine = LLMQueryEngine(df=self.df)
        print("✅ Ready to answer questions!\n")
    
    def get_data_overview(self):
        """Get a quick overview of the LFS data"""
        print("📊 LFS-2023 Data Overview")
        print("=" * 60)
        print(f"Total Records: {len(self.df):,}")
        print(f"Total Columns: {len(self.df.columns)}")
        print(f"\nKey Columns: {', '.join(self.df.columns[:15].tolist())}...")
        
        # Show some statistics
        if 'DISTRICT' in self.df.columns:
            print(f"\nUnique Districts: {self.df['DISTRICT'].nunique()}")
            print(f"Top 5 Districts by count:")
            print(self.df['DISTRICT'].value_counts().head().to_string())
        
        if 'AGE' in self.df.columns:
            print(f"\nAge Statistics:")
            print(f"  Mean Age: {self.df['AGE'].mean():.1f}")
            print(f"  Min Age: {self.df['AGE'].min()}")
            print(f"  Max Age: {self.df['AGE'].max()}")
        
        if 'SEX' in self.df.columns:
            print(f"\nGender Distribution:")
            print(self.df['SEX'].value_counts().to_string())
        
        print("\n" + "=" * 60)
    
    def ask(self, question: str):
        """
        Ask a question about the LFS data
        
        Args:
            question: Natural language question
            
        Returns:
            AI-generated answer
        """
        print(f"\n❓ Question: {question}")
        print("🤖 Analyzing...")
        print("-" * 60)
        
        answer = self.engine.analyze_data(question, context_limit=1000)
        
        print(answer)
        print("-" * 60)
        
        return answer
    
    def get_insights(self, topic=None):
        """
        Get insights about a specific topic or general insights
        
        Args:
            topic: Specific topic to analyze (optional)
            
        Returns:
            Insights string
        """
        print(f"\n💡 Getting insights{' about: ' + topic if topic else ''}...")
        print("-" * 60)
        
        answer = self.engine.get_insights(topic)
        
        print(answer)
        print("-" * 60)
        
        return answer
    
    def interactive_mode(self):
        """Start an interactive Q&A session"""
        print("\n" + "=" * 70)
        print("💬 LFS Interactive Q&A Mode")
        print("=" * 70)
        print("\n🎯 What you can do:")
        print("  • Ask any question about the Labor Force Survey data")
        print("  • /overview - Show data overview")
        print("  • /insights [topic] - Get insights about a topic")
        print("  • /sample - Show sample data")
        print("  • /columns - List all column names")
        print("  • quit - Exit\n")
        
        while True:
            try:
                query = input("📝 Your question: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("✅ Goodbye!")
                    break
                
                if not query:
                    continue
                
                # Handle special commands
                if query.startswith('/'):
                    cmd_parts = query.split(' ', 1)
                    cmd = cmd_parts[0].lower()
                    cmd_arg = cmd_parts[1] if len(cmd_parts) > 1 else None
                    
                    if cmd == '/overview':
                        self.get_data_overview()
                    
                    elif cmd == '/insights':
                        self.get_insights(cmd_arg)
                    
                    elif cmd == '/sample':
                        print("\n📋 Sample Data (first 10 rows):")
                        print(self.df.head(10).to_string())
                        print()
                    
                    elif cmd == '/columns':
                        print("\n📋 Column Names:")
                        for i, col in enumerate(self.df.columns, 1):
                            print(f"  {i}. {col}")
                        print()
                    
                    else:
                        print(f"⚠️ Unknown command: {cmd}")
                    
                    continue
                
                # Normal question
                self.ask(query)
                
                print()
                
            except KeyboardInterrupt:
                print("\n✅ Goodbye!")
                break
            except Exception as e:
                print(f"\n⚠️ Error: {e}\n")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='LFS-2023 Data Analyzer with AI-Powered Q&A (Using Ollama - Free & Local)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python lfs_analyzer.py                    # Interactive mode
  python lfs_analyzer.py --overview         # Show overview and exit
  python lfs_analyzer.py --question "..."   # Ask single question

Setup:
  1. Install Ollama: https://ollama.ai
  2. Run: ollama pull llama3.2
  3. Start: ollama serve (if not auto-started)
        """
    )
    
    parser.add_argument('--data', type=str, help='Path to LFS-2023.csv file')
    parser.add_argument('--overview', action='store_true', help='Show data overview and exit')
    parser.add_argument('--question', type=str, help='Ask a single question and exit')
    
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        analyzer = LFSAnalyzer(data_path=args.data)
        
        if args.overview:
            analyzer.get_data_overview()
        elif args.question:
            analyzer.ask(args.question)
        else:
            # Interactive mode
            analyzer.get_data_overview()
            analyzer.interactive_mode()
    
    except FileNotFoundError as e:
        print(f"\n{e}")
        print("Please ensure LFS-2023.csv exists in the data/ folder")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
