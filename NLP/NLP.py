import os
import pickle
import pandas as pd
import numpy as np
from langchain_community.llms import Ollama
import warnings

warnings.filterwarnings('ignore')


class SkillDev:
    """Minimal stub to support unpickling SkillDev instances."""
    pass

class NLPClusterQueryEngine:
    """
    NLP-based query engine that uses pretrained models to understand requests
    and access cluster data from the trained SkillDev model
    """
    
    def __init__(self, model_path):
        """Load the trained SkillDev model"""
        print("🔄 Loading trained clustering model...")
        with open(model_path, 'rb') as f:
            self.skilldev_model = pickle.load(f)
        
        self.df = self.skilldev_model.df
        self.kmeans = self.skilldev_model.kmeans
        self.features = self.skilldev_model.features
        
        print("✅ Model loaded successfully!")
        print(f"📊 Dataset: {len(self.df)} records, {len(self.features)} features")
        print(f"🎯 Clusters: {self.skilldev_model.n_clusters}")
        
        # Initialize pretrained NLP models
        print("\n🤖 Loading pretrained NLP models...")
        try:
            # Zero-shot classification for intent detection
            self.classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli"
            )
            print("✅ Zero-shot classifier loaded")
        except Exception as e:
            print(f"⚠️ Could not load zero-shot classifier: {e}")
            self.classifier = None
        
        try:
            # Semantic similarity model
            self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            self.model = AutoModelForSequenceClassification.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            print("✅ Semantic model loaded")
        except Exception as e:
            print(f"⚠️ Could not load semantic model: {e}")
            self.tokenizer = None
            self.model = None
    
    def understand_query(self, query):
        """Use NLP to understand user query and extract intent"""
        print(f"\n🔍 Analyzing query: '{query}'")
        
        if not self.classifier:
            print("⚠️ Classifier not available, using keyword matching")
            return self._keyword_intent(query)
        
        # Possible intents
        intents = [
            "find records in a specific cluster",
            "compare clusters",
            "analyze demographic patterns",
            "identify outliers",
            "get cluster statistics"
        ]
        
        try:
            result = self.classifier(query, intents, multi_class=False)
            top_intent = result['labels'][0]
            confidence = result['scores'][0]
            
            print(f"📌 Detected Intent: {top_intent}")
            print(f"💯 Confidence: {confidence:.2%}")
            
            return {
                'intent': top_intent,
                'confidence': confidence,
                'query': query
            }
        except Exception as e:
            print(f"⚠️ Intent detection failed: {e}")
            return self._keyword_intent(query)
    
    def _keyword_intent(self, query):
        """Fallback keyword-based intent detection"""
        query_lower = query.lower()
        
        if any(kw in query_lower for kw in ['cluster', 'group', 'segment']):
            intent = "find records in a specific cluster"
        elif any(kw in query_lower for kw in ['compare', 'difference', 'vs']):
            intent = "compare clusters"
        elif any(kw in query_lower for kw in ['pattern', 'analyze', 'demographic']):
            intent = "analyze demographic patterns"
        elif any(kw in query_lower for kw in ['outlier', 'extreme', 'unusual']):
            intent = "identify outliers"
        else:
            intent = "get cluster statistics"
        
        return {
            'intent': intent,
            'confidence': 0.5,
            'query': query
        }
    
    def query_clusters(self, query):
        """Execute query against the cluster data"""
        intent_result = self.understand_query(query)
        intent = intent_result['intent']
        
        print(f"\n⚙️ Executing query...")
        
        if "specific cluster" in intent:
            return self._get_cluster_records(query)
        elif "compare" in intent:
            return self._compare_clusters()
        elif "pattern" in intent:
            return self._analyze_patterns(query)
        elif "outlier" in intent:
            return self._find_outliers()
        else:
            return self._get_cluster_stats()
    
    def _get_cluster_records(self, query):
        """Get records from a specific cluster"""
        print("\n📋 Cluster Records:")
        
        # Extract cluster number from query if possible
        import re
        cluster_nums = re.findall(r'\d+', query)
        
        if cluster_nums:
            cluster_id = int(cluster_nums[0]) % self.skilldev_model.n_clusters
        else:
            cluster_id = 0
        
        cluster_data = self.df[self.df['cluster_id'] == cluster_id]
        
        print(f"Cluster {cluster_id}: {len(cluster_data)} records")
        print(cluster_data[self.features[:5]].head(10))
        
        return cluster_data
    
    def _compare_clusters(self):
        """Compare statistics across clusters"""
        print("\n📊 Cluster Comparison:")
        
        for cluster_id in range(self.skilldev_model.n_clusters):
            cluster_data = self.df[self.df['cluster_id'] == cluster_id]
            print(f"\nCluster {cluster_id}:")
            print(f"  Records: {len(cluster_data)}")
            print(f"  Mean values: {cluster_data[self.features[:3]].mean().round(2).to_dict()}")
    
    def _analyze_patterns(self, query):
        """Analyze demographic patterns in clusters"""
        print("\n🔬 Pattern Analysis:")
        
        # Show variance across clusters for each feature
        for feature in self.features[:5]:
            cluster_means = self.df.groupby('cluster_id')[feature].mean()
            print(f"\n{feature}:")
            print(cluster_means.round(2))
        
        return self.df.groupby('cluster_id')[self.features[:5]].mean()
    
    def _find_outliers(self):
        """Identify outlier records"""
        print("\n⚠️ Outlier Detection:")  
        
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.df[self.features])
        
        # Records with extreme values (|z-score| > 3)
        outliers = np.where((np.abs(X_scaled) > 3).any(axis=1))[0]
        
        print(f"Found {len(outliers)} outlier records")
        if len(outliers) > 0:
            print(self.df.iloc[outliers[:10]][self.features[:5]])
        
        return self.df.iloc[outliers]
    
    def _get_cluster_stats(self):
        """Get comprehensive cluster statistics"""
        print("\n📈 Cluster Statistics:")
        
        stats = {
            'Total Records': len(self.df),
            'Clusters': self.skilldev_model.n_clusters,
            'Distribution': self.df['cluster_id'].value_counts().to_dict()
        }
        
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        return stats
    
    def interactive_query(self):
        """Interactive query loop"""
        print("\n" + "="*60)
        print("💬 NLP Cluster Query Engine - Interactive Mode")
        print("="*60)
        print("Ask questions about your cluster data!")
        print("Examples:")
        print("  - 'Show records in cluster 0'")
        print("  - 'Compare all clusters'")
        print("  - 'What patterns exist in the data?'")
        print("  - 'Find outlier records'")
        print("  - 'Cluster statistics'")
        print("Type 'quit' to exit\n")
        
        while True:
            query = input("📝 Your query: ").strip()
            
            if query.lower() == 'quit':
                print("✅ Goodbye!")
                break
            
            if query:
                self.query_clusters(query)


class LLMQueryEngine:
    """
    LLM-powered database and cluster query engine using Ollama (free & local)
    """
    
    def __init__(self, model_path=None, df=None):
        """
        Initialize the LLM Query Engine
        
        Args:
            model_path: Path to a trained clustering model (optional)
            df: DataFrame to analyze (optional)
        """
        print("🤖 Initializing LLM Query Engine...")
        
        # Load data
        if df is not None:
            self.df = df
            self.has_clusters = 'cluster_id' in df.columns
        elif model_path and os.path.exists(model_path):
            print(f"📂 Loading model from {model_path}")
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            self.df = model.df if hasattr(model, 'df') else df
            self.has_clusters = self.df is not None and 'cluster_id' in self.df.columns
        else:
            self.df = None
            self.has_clusters = False
            print("⚠️ No data loaded - using general query mode")
        
        # Initialize Ollama LLM (local, free)
        try:
            self.llm = Ollama(
                model="llama3.2",
                temperature=0.7
            )
            # Test if Ollama is running
            test_response = self.llm.invoke("test")
            print("✅ LLM Query Engine ready! (Using Ollama - llama3.2)")
        except Exception as e:
            print(f"⚠️ Ollama not available: {e}")
            print("Please install and start Ollama first!")
            print("Download from: https://ollama.ai")
            print("Then run: ollama pull llama3.2")
            self.llm = None
        if self.df is not None:
            print(f"📊 Loaded {len(self.df)} records")
            if self.has_clusters:
                n_clusters = self.df['cluster_id'].nunique()
                print(f"🎯 Found {n_clusters} clusters")
    
    def analyze_data(self, question: str, context_limit: int = 500):
        """
        Answer questions about the data using LLM with data context
        
        Args:
            question: User's question about the data
            context_limit: Maximum number of rows to include in context
        
        Returns:
            LLM's answer as a string
        """
        print(f"\n🔍 Processing question: {question}")
        
        if self.df is None:
            return "⚠️ No data loaded. Please provide a dataset or model path."
        
        if self.llm is None:
            return "⚠️ Ollama not running. Please install and start Ollama, then run: ollama pull llama3.2"
        
        # Prepare data context (smaller for Ollama)
        data_summary = self._prepare_data_context(min(context_limit, 200))
        
        # Create simplified prompt for Ollama
        prompt_text = f"""You are a data analyst. Using this data summary, answer the question.

Data Summary:
{data_summary}

Question: {question}

Provide a clear, concise answer with specific numbers when possible:"""
        
        try:
            # Get answer from Ollama
            answer = self.llm.invoke(prompt_text)
            return answer
        except Exception as e:
            return f"⚠️ Error: {str(e)}\nMake sure Ollama is running: ollama serve"
    
    def _prepare_data_context(self, limit: int):
        """Prepare a concise summary of the data for LLM context"""
        context_parts = []
        
        # Basic info
        context_parts.append(f"Dataset size: {len(self.df)} records")
        context_parts.append(f"Columns: {', '.join(self.df.columns.tolist())}")
        
        # Statistical summary
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            stats = self.df[numeric_cols].describe().round(2)
            context_parts.append(f"\nNumeric column statistics:\n{stats.to_string()}")
        
        # Cluster information if available
        if self.has_clusters:
            cluster_dist = self.df['cluster_id'].value_counts().sort_index()
            context_parts.append(f"\nCluster distribution:\n{cluster_dist.to_string()}")
            
            # Cluster characteristics
            for cluster_id in sorted(self.df['cluster_id'].unique()):
                cluster_data = self.df[self.df['cluster_id'] == cluster_id]
                if numeric_cols:
                    means = cluster_data[numeric_cols[:5]].mean().round(2)  # Top 5 features
                    context_parts.append(f"\nCluster {cluster_id} average values:\n{means.to_string()}")
        
        # Sample data
        sample_size = min(10, limit)
        context_parts.append(f"\nSample data (first {sample_size} rows):\n{self.df.head(sample_size).to_string()}")
        
        return "\n\n".join(context_parts)
    
    def ask_about_clusters(self, question: str):
        """
        Specifically answer questions about clusters
        
        Args:
            question: Question about clusters
        
        Returns:
            Answer with cluster analysis
        """
        if not self.has_clusters:
            return "⚠️ No cluster information available in the dataset."
        
        if self.llm is None:
            return "⚠️ Ollama not running. Please start Ollama."
        
        # Extract cluster-specific context
        cluster_context = self._get_cluster_analysis()
        
        prompt_text = f"""You are a clustering expert. Analyze these clusters and answer the question.

Cluster Analysis:
{cluster_context}

Question: {question}

Answer:"""
        
        try:
            answer = self.llm.invoke(prompt_text)
            return answer
        except Exception as e:
            return f"⚠️ Error: {str(e)}"
    
    def _get_cluster_analysis(self):
        """Generate detailed cluster analysis"""
        analysis_parts = []
        
        n_clusters = self.df['cluster_id'].nunique()
        analysis_parts.append(f"Total clusters: {n_clusters}")
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if 'cluster_id' in numeric_cols:
            numeric_cols.remove('cluster_id')
        
        for cluster_id in sorted(self.df['cluster_id'].unique()):
            cluster_data = self.df[self.df['cluster_id'] == cluster_id]
            
            parts = [f"\n--- Cluster {cluster_id} ---"]
            parts.append(f"Size: {len(cluster_data)} records ({len(cluster_data)/len(self.df)*100:.1f}%)")
            
            if numeric_cols:
                # Statistics for this cluster
                means = cluster_data[numeric_cols].mean()
                stds = cluster_data[numeric_cols].std()
                
                parts.append("\nAverage values:")
                for col in numeric_cols[:8]:  # Top 8 features
                    parts.append(f"  {col}: {means[col]:.2f} (±{stds[col]:.2f})")
            
            analysis_parts.append("\n".join(parts))
        
        return "\n".join(analysis_parts)
    
    def compare_clusters(self, cluster_ids: list = None):
        """
        Compare specific clusters or all clusters
        
        Args:
            cluster_ids: List of cluster IDs to compare, or None for all clusters
        
        Returns:
            Comparison analysis
        """
        if not self.has_clusters:
            return "⚠️ No cluster information available."
        
        if self.llm is None:
            return "⚠️ Ollama not running."
        
        if cluster_ids is None:
            cluster_ids = sorted(self.df['cluster_id'].unique())
        
        comparison = self._get_cluster_comparison(cluster_ids)
        
        prompt_text = f"""Compare these data clusters and explain the key differences:

{comparison}

Provide a clear comparison:"""
        
        try:
            answer = self.llm.invoke(prompt_text)
            return answer
        except Exception as e:
            return f"⚠️ Error: {str(e)}"
    
    def _get_cluster_comparison(self, cluster_ids):
        """Generate cluster comparison data"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if 'cluster_id' in numeric_cols:
            numeric_cols.remove('cluster_id')
        
        comparison_data = []
        
        for cluster_id in cluster_ids:
            cluster_data = self.df[self.df['cluster_id'] == cluster_id]
            
            if numeric_cols:
                means = cluster_data[numeric_cols].mean().round(2)
                comparison_data.append(f"\nCluster {cluster_id} ({len(cluster_data)} records):")
                comparison_data.append(means.to_string())
        
        return "\n".join(comparison_data)
    
    def get_insights(self, topic: str = None):
        """
        Get general insights about the data
        
        Args:
            topic: Specific topic to focus on (optional)
        
        Returns:
            Data insights
        """
        if self.df is None:
            return "⚠️ No data loaded."
        
        if self.llm is None:
            return "⚠️ Ollama not running."
        
        data_context = self._prepare_data_context(100)
        
        if topic:
            question = f"What insights can you provide about {topic} in this dataset?"
        else:
            question = "What are the key insights and patterns in this dataset?"
        
        prompt_text = f"""As a data scientist, analyze this data and provide insights.

Data:
{data_context}

Question: {question}

Key insights:"""
        
        try:
            answer = self.llm.invoke(prompt_text)
            return answer
        except Exception as e:
            return f"⚠️ Error: {str(e)}"


# Main execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='NLP Query Engine for Cluster Analysis')
    parser.add_argument('--mode', choices=['classic', 'llm'], default='llm',
                        help='Query mode: classic (rule-based) or llm (AI-powered)')
    parser.add_argument('--model', type=str, help='Path to model pickle file')
    
    args = parser.parse_args()
    
    # Load model
    if args.model:
        model_path = args.model
    else:
        model_path = os.path.join('..', 'model', 'skilldev_model.pkl')
    
    candidate_paths = [
        model_path,
        os.path.join('..', 'model', 'skilldev_model.pkl'),
        os.path.join('model', 'skilldev_model.pkl'),
    ]
    
    valid_path = None
    for path in candidate_paths:
        if os.path.exists(path):
            valid_path = path
            break
    
    if not valid_path:
        print("❌ Model not found!")
        exit(1)
    
    print(f"✅ Using model from: {valid_path}\n")
    
    # Initialize engine
    if args.mode == 'classic':
        engine = NLPClusterQueryEngine(valid_path)
        engine.interactive_query()
    else:
        engine = LLMQueryEngine(model_path=valid_path)
        print("\n" + "=" * 70)
        print("💬 LLM Query Interface")
        print("=" * 70)
        print("Commands: /clusters, /compare, /insights [topic], quit\n")
        
        while True:
            query = input("📝 Your query: ").strip()
            
            if query.lower() in ['quit', 'exit']:
                print("✅ Goodbye!")
                break
            
            if not query:
                continue
            
            if query.startswith('/'):
                cmd_parts = query.split(' ', 1)
                cmd = cmd_parts[0].lower()
                cmd_arg = cmd_parts[1] if len(cmd_parts) > 1 else None
                
                if cmd == '/clusters':
                    question = cmd_arg or "What can you tell me about the clusters?"
                    print(engine.ask_about_clusters(question))
                elif cmd == '/compare':
                    print(engine.compare_clusters())
                elif cmd == '/insights':
                    print(engine.get_insights(cmd_arg))
                else:
                    print("⚠️ Unknown command")
            else:
                print(engine.analyze_data(query))
            
            print()
