import os
import re
import pickle
import numpy as np
import torch
import requests
from dotenv import load_dotenv
import sys

# SkillDev class stub for pickle unpickling
# Must be defined here so pickle can reconstruct the model object
class SkillDev:
    """Minimal stub to support unpickling SkillDev instances."""
    pass


# Custom pickle unpickler to handle pandas compatibility issues
class CompatibleUnpickler(pickle.Unpickler):
    """Custom unpickler that handles pandas StringDtype and SkillDev compatibility."""

    @staticmethod
    def _make_string_dtype_proxy():
        """Return a StringDtype subclass that silently accepts any __init__ args."""
        try:
            from pandas import StringDtype as _SD
            class _StringDtypeCompat(_SD):
                def __init__(self, *args, **kwargs):
                    try:
                        super().__init__()
                    except TypeError:
                        pass
            return _StringDtypeCompat
        except Exception:
            return None

    def find_class(self, module, name):
        if name == 'StringDtype' and 'pandas' in module:
            proxy = self._make_string_dtype_proxy()
            if proxy is not None:
                return proxy
        if name == 'SkillDev':
            return SkillDev
        return super().find_class(module, name)


def load_model_safely(model_path):
    """Load pickled model with compatibility fixes for pandas versions."""
    sys.modules['__main__'].SkillDev = SkillDev

    with open(model_path, 'rb') as f:
        try:
            return CompatibleUnpickler(f).load()
        except Exception as e1:
            print(f"  Custom unpickler failed: {e1}")
            f.seek(0)
            try:
                return pickle.load(f)
            except Exception as e2:
                raise RuntimeError(
                    f"Cannot load model from {model_path}.\n"
                    f"Errors: {e1} | {e2}\n"
                    f"Run: python train_model.py  to regenerate the model."
                ) from e2


# Load environment variables from .env file
load_dotenv()

from .constants import COLUMN_DESCRIPTIONS, SECTOR_MAP, DISTRICT_MAP, EMPLOYMENT_STATUS

# Resolve device once at import time
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Ollama LLM configuration (local inference)
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"  # Match the actual installed model name

# ──────────────────────────────────────────────────────────────────────────────
# Intent labels used by BART zero-shot classifier.
# The first label that scores highest drives routing in __main__.
# ──────────────────────────────────────────────────────────────────────────────
INTENT_LABELS = [
    "allocate or distribute resources to people",   #  LLM handle_allocation
    "compare clusters or segments",                  #  LLM compare_clusters
    "find records in a specific cluster",            #  LLM ask_about_clusters
    "get insights or trends in the data",            #  LLM get_insights
    "analyze demographic or statistical patterns",   #  LLM analyze_data
    "identify outliers or unusual records",          #  LLM analyze_data
    "get cluster statistics or summary",             #  LLM ask_about_clusters
]

# Map detected intent label  routing key used in __main__
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
        self.skilldev_model = load_model_safely(model_path)
        
        self.df = self.skilldev_model.df
        self.kmeans = self.skilldev_model.kmeans
        self.features = self.skilldev_model.features

        # Load the saved scaler — must use the same scaler used during K-Means training
        self.scaler = None
        # Try multiple possible scaler paths (model/scaler.pkl or ../model/scaler.pkl)
        possible_paths = [
            os.path.join(os.path.dirname(model_path), 'scaler.pkl'),
            os.path.join(os.path.dirname(model_path), '..', 'model', 'scaler.pkl'),
        ]
        scaler_path = None
        for path in possible_paths:
            normalized_path = os.path.normpath(path)
            if os.path.exists(normalized_path):
                scaler_path = normalized_path
                break
        
        if scaler_path:
            with open(scaler_path, 'rb') as sf:
                self.scaler = pickle.load(sf)
            print(f" Scaler loaded from {scaler_path}")
        else:
            print(f"  scaler.pkl not found — outlier detection will be approximate")
            print(f"   (searched: {', '.join(possible_paths)})")

        print(" Model loaded successfully!")

        # Initialize Ollama API for intent detection
        print(f"\n Initializing Ollama API for intent detection (model: {OLLAMA_MODEL})...")
        print(f"   Ollama endpoint: {OLLAMA_API_URL}")
        self.classifier = "ollama"  # Flag indicating we use Ollama
        
        # Check if Ollama is running
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                print(f" Ollama is running and reachable")
                available_models = response.json().get('models', [])
                model_names = [m.get('name', 'unknown') for m in available_models]
                print(f"   Available models: {', '.join(model_names)}")
                if OLLAMA_MODEL not in model_names:
                    print(f"  Model '{OLLAMA_MODEL}' not found locally. Available: {model_names}")
                else:
                    print(f" Model '{OLLAMA_MODEL}' is available")
            else:
                print(f"  Ollama returned status {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"  Ollama not reachable at {OLLAMA_API_URL}")
            print(f"   Error: {str(e)[:80]}")
            print(f"   Make sure Ollama is running: ollama serve")
    
    def understand_query(self, query):
        """
        Use Ollama (llama3.2:3b) for intent detection and TOOL ALIGNMENT scoring.

        GATE LOGIC:
        -----------
        1. Ollama classifies the query into one of 7 intent categories
        2. Each category maps to a specific tool
        3. Confidence score = How well the query aligns with that tool
           - HIGH (>0.7): Clear tool fit (e.g., "Compare clusters"  compare_clusters)
           - MEDIUM (0.4-0.7): Some tool fit (e.g., "What about demographics?"  analyze)
           - LOW (<0.4): Out-of-scope (e.g., "Tell me a joke"  No tool fits)

        Returns
        -------
        dict with keys:
          intent     – human-readable label detected byOllama
          route      – routing key for tool dispatcher
          confidence – tool alignment strength (0-1), used by gate
          query      – original query string
        """
        print(f"\n [LLAMA/Ollama] Classifying query: '{query}'")

        if self.classifier != "ollama":
            print(" Ollama classifier not available — using keyword fallback")
            return self._keyword_intent(query)

        try:
            result = self._call_ollama_api(query, INTENT_LABELS)
            
            if result is None:
                print(" Ollama API returned no result — using keyword fallback")
                return self._keyword_intent(query)
            
            top_label = result['label']
            confidence = result['confidence']
            route = _map_intent(top_label)

            # Calculate tool alignment score (0-1)
            # Higher = query better aligns with a registered tool
            tool_alignment = self._calculate_tool_alignment(query, top_label, confidence)

            print(f"   Intent: {top_label}")
            print(f"   Route: {route}")
            print(f"   LLAMA confidence: {confidence:.1%}")
            print(f"   Tool alignment: {tool_alignment:.1%}")

            return {
                'intent': top_label,
                'route': route,
                'confidence': tool_alignment,  # Use alignment score for gating decision
                'query': query,
            }
        except Exception as e:
            print(f" Ollama classification failed: {e}")
            return self._keyword_intent(query)

    def _calculate_tool_alignment(self, query, detected_intent, base_confidence):
        """
        Calculate how well this query aligns with registered tools.

        Returns
        -------
        float: Alignment score 0-1
               > 0.4  = Tool-alignable (will be routed to tool)
               <= 0.4 = Out-of-scope (will route to Groq/ReActAgent)
        """
        query_lower = query.lower()

        # Check for explicit tool keywords
        tool_keywords = {
            "allocate or distribute": ["give", "distribut", "allocat", "provid"],
            "compare clusters": ["compare", "contrast", "differ", "vs"],
            "cluster query": ["cluster", "group", "segment"],
            "insights": ["insight", "trend", "pattern", "what are"],
            "demographics": ["demographic", "statistic", "age", "gender", "sector"],
            "outliers": ["outlier", "unusual", "anomal", "extreme"],
        }

        # Count keyword matches
        max_matches = 0
        for tool, keywords in tool_keywords.items():
            matches = sum(1 for kw in keywords if kw in query_lower)
            max_matches = max(max_matches, matches)

        # Alignment score based on keyword matches
        if max_matches >= 2:
            alignment = min(0.95, base_confidence + 0.3)  # Strong signal
        elif max_matches == 1:
            alignment = base_confidence + 0.15  # Moderate signal
        else:
            alignment = base_confidence * 0.7  # Weak signal

        return min(1.0, alignment)


    def _call_ollama_api(self, text, candidate_labels):
        """
        Call Ollama (llama3.2:3b) for zero-shot intent classification.
        Uses a simple classification prompt to classify the query against candidate labels.
        """
        # Create a simpler, more direct prompt
        intent_descriptions = {
            "allocate or distribute resources to people": "Give, distribute, provide, allocate, or assign items (laptops, money, food) to specific people",
            "compare clusters or segments": "Compare, contrast, show differences between clusters, groups, or population segments",
            "find records in a specific cluster": "Get information about, describe, or ask questions about a specific cluster or group",
            "get insights or trends in the data": "What are the trends, patterns, insights, observations, or findings in the data",
            "analyze demographic or statistical patterns": "Show statistics, counts, percentages, distributions, or demographic breakdowns",
            "identify outliers or unusual records": "Find unusual, extreme, anomalous, or outlier cases in the data",
            "get cluster statistics or summary": "Show cluster summary, statistics, distribution, or overview",
        }
        
        prompt = f"""Classify this user query into ONE of these intent categories:

INTENT CATEGORIES:
{chr(10).join([f"{i+1}. {label} - {intent_descriptions[label]}" for i, label in enumerate(candidate_labels)])}

USER QUERY: "{text}"

Respond with ONLY the category number (1-7) or the exact intent keyword. Be decisive."""

        try:
            response = requests.post(
                OLLAMA_API_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.1,  # Very low temperature for deterministic results
                },
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            response_text = result.get('response', '').strip()
            
            # Try to extract intent from response
            detected_label = None
            
            # First try: look for exact matches
            for candidate in candidate_labels:
                if candidate.lower() in response_text.lower():
                    detected_label = candidate
                    break
            
            # Second try: look for category numbers (1-7)
            if not detected_label:
                for i, match in enumerate(response_text.lower()):
                    if match in '1234567':
                        idx = int(match) - 1
                        if 0 <= idx < len(candidate_labels):
                            detected_label = candidate_labels[idx]
                            break
            
            # Third try: look for keywords from descriptions
            if not detected_label:
                response_lower = response_text.lower()
                keywords = {
                    "allocate or distribute resources to people": ["give", "distribut", "allocat", "provid", "assign"],
                    "compare clusters or segments": ["compare", "contrast", "differ", "segment"],
                    "find records in a specific cluster": ["cluster", "group", "specific", "about"],
                    "get insights or trends in the data": ["insight", "trend", "pattern", "observ"],
                    "analyze demographic or statistical patterns": ["statistic", "count", "percent", "distribut", "demograp"],
                    "identify outliers or unusual records": ["outlier", "unusual", "anomal", "extreme"],
                    "get cluster statistics or summary": ["cluster", "summary", "statistic", "overview"],
                }
                
                for intent, keywords_list in keywords.items():
                    if any(kw in response_lower for kw in keywords_list):
                        detected_label = intent
                        break
            
            # Final fallback: use keyword matching directly on the input query
            if not detected_label:
                detected_label = self._keyword_intent(text)['intent']
            
            # Estimate confidence
            confidence = 0.8 if any(label in response_text for label in candidate_labels) else 0.6
            
            return {
                'label': detected_label,
                'confidence': confidence,
                'raw_response': response_text
            }
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Ollama API request failed: {e}")
            raise


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

        print(f"\n Executing query...")

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
        print("\n Outlier Detection:")

        if self.scaler is not None:
            # Use the exact same scaling as K-Means training
            X_scaled = self.scaler.transform(self.df[self.features])
        else:
            # Fallback only if scaler.pkl was not found — log a warning
            print("  Using fallback scaler — results may differ from cluster assignments")
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
                print(" Goodbye!")
                break
            
            if query:
                self.query_clusters(query)
