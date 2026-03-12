import os
import re
import pickle
import numpy as np
import torch

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
except ImportError:
    pipeline = None
    AutoTokenizer = None
    AutoModelForSequenceClassification = None

from .constants import COLUMN_DESCRIPTIONS, SECTOR_MAP, DISTRICT_MAP, EMPLOYMENT_STATUS

# Resolve device once at import time
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ──────────────────────────────────────────────────────────────────────────────
# Intent labels used by BART zero-shot classifier.
# The first label that scores highest drives routing in __main__.
# ──────────────────────────────────────────────────────────────────────────────
INTENT_LABELS = [
    "allocate or distribute resources to people",   # → LLM handle_allocation
    "compare clusters or segments",                  # → LLM compare_clusters
    "find records in a specific cluster",            # → LLM ask_about_clusters
    "get insights or trends in the data",            # → LLM get_insights
    "analyze demographic or statistical patterns",   # → LLM analyze_data
    "identify outliers or unusual records",          # → LLM analyze_data
    "get cluster statistics or summary",             # → LLM ask_about_clusters
]

# Map detected intent label → routing key used in __main__
def _map_intent(label: str) -> str:
    label_lower = label.lower()
    if "allocate" in label_lower or "distribut" in label_lower or "resource" in label_lower:
        return "resource_allocation"
    if "compare" in label_lower:
        return "compare_clusters"
    if "specific cluster" in label_lower or "statistics" in label_lower:
        return "cluster_query"
    if "insight" in label_lower or "trend" in label_lower:
        return "insights"
    return "general_analysis"


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

        # Load the saved scaler — must use the same scaler used during K-Means training
        self.scaler = None
        scaler_path = os.path.join(os.path.dirname(model_path), '..', 'models', 'scaler.pkl')
        scaler_path = os.path.normpath(scaler_path)
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as sf:
                self.scaler = pickle.load(sf)
            print(f"✅ Scaler loaded from {scaler_path}")
        else:
            print(f"⚠️  scaler.pkl not found at {scaler_path} — outlier detection will be approximate")

        print("✅ Model loaded successfully!")

        # Initialize pretrained NLP models
        print(f"\n🤖 Loading pretrained NLP models on {DEVICE.upper()}...")
        try:
            self.classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if torch.cuda.is_available() else -1
            )
            print(f"✅ Zero-shot classifier loaded on {DEVICE.upper()}")
        except Exception as e:
            print(f"⚠️ Could not load zero-shot classifier: {e}")
            self.classifier = None

        try:
            self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            self.model = AutoModelForSequenceClassification.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(DEVICE)
            print(f"✅ Semantic model loaded on {DEVICE.upper()}")
        except Exception as e:
            print(f"⚠️ Could not load semantic model: {e}")
            self.tokenizer = None
            self.model = None
    
    def understand_query(self, query):
        """
        Use BART zero-shot classification to detect user intent.
        Returns a dict with keys:
          intent    – human-readable label selected by BART
          route     – routing key for __main__ dispatcher
          confidence – BART confidence score (0-1)
          query     – original query string
        """
        print(f"\n🔍 [NLPC/BART] Analyzing: '{query}'")

        if not self.classifier:
            print("⚠️ BART classifier not available — falling back to keyword matching")
            return self._keyword_intent(query)

        try:
            result = self.classifier(query, INTENT_LABELS, multi_class=False)
            top_label = result['labels'][0]
            confidence = result['scores'][0]
            route = _map_intent(top_label)

            print(f"📌 Intent  : {top_label}")
            print(f"🔀 Route   : {route}")
            print(f"💯 Confidence: {confidence:.2%}")

            return {
                'intent': top_label,
                'route': route,
                'confidence': confidence,
                'query': query,
            }
        except Exception as e:
            print(f"⚠️ BART intent detection failed: {e}")
            return self._keyword_intent(query)
    
    def _keyword_intent(self, query):
        """Fallback keyword-based intent detection (used when BART unavailable)."""
        query_lower = query.lower()

        if any(kw in query_lower for kw in [
            'give', 'distribut', 'allocat', 'provide', 'deliver',
            'sewing', 'laptop', 'computer', 'car', 'truck',
            'scholarship', 'phone', 'tablet', 'device', 'aid',
        ]):
            intent = "allocate or distribute resources to people"
            route = "resource_allocation"
        elif any(kw in query_lower for kw in ['compare', 'difference', 'vs', 'versus']):
            intent = "compare clusters or segments"
            route = "compare_clusters"
        elif any(kw in query_lower for kw in ['insight', 'trend', 'pattern']):
            intent = "get insights or trends in the data"
            route = "insights"
        elif any(kw in query_lower for kw in ['cluster', 'group', 'segment', 'statistic']):
            intent = "find records in a specific cluster"
            route = "cluster_query"
        elif any(kw in query_lower for kw in ['outlier', 'extreme', 'unusual']):
            intent = "identify outliers or unusual records"
            route = "general_analysis"
        else:
            intent = "analyze demographic or statistical patterns"
            route = "general_analysis"

        return {
            'intent': intent,
            'route': route,
            'confidence': 0.5,
            'query': query,
        }
    
    def query_clusters(self, query):
        """Execute query against the cluster data"""
        intent_result = self.understand_query(query)
        route = intent_result['route']

        print(f"\n⚙️ Executing query...")

        if route == "cluster_query":
            return self._get_cluster_records(query)
        elif route == "compare_clusters":
            return self._compare_clusters()
        elif route == "insights":
            return self._analyze_patterns(query)
        elif route == "general_analysis":
            # Check for outlier keywords before falling through to stats
            if any(kw in query.lower() for kw in ['outlier', 'extreme', 'unusual', 'anomal']):
                return self._find_outliers()
            return self._get_cluster_stats()
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
        """Identify outlier records using the saved scaler from training."""
        print("\n⚠️ Outlier Detection:")

        if self.scaler is not None:
            # Use the exact same scaling as K-Means training
            X_scaled = self.scaler.transform(self.df[self.features])
        else:
            # Fallback only if scaler.pkl was not found — log a warning
            print("⚠️  Using fallback scaler — results may differ from cluster assignments")
            from sklearn.preprocessing import StandardScaler
            fallback_scaler = StandardScaler()
            X_scaled = fallback_scaler.fit_transform(self.df[self.features])

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
