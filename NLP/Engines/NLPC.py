
import os
import re
import pickle
import numpy as np
import sys
from dotenv import load_dotenv

# SkillDev class stub for pickle unpickling
class SkillDev:
    """Minimal stub to support unpickling SkillDev instances."""
    pass


class CompatibleUnpickler(pickle.Unpickler):
    """Custom unpickler that handles pandas StringDtype and SkillDev compatibility."""

    @staticmethod
    def _make_string_dtype_proxy():
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


load_dotenv()

try:
    from .constants import COLUMN_DESCRIPTIONS, SECTOR_MAP, DISTRICT_MAP, EMPLOYMENT_STATUS
except ImportError:
    from constants import COLUMN_DESCRIPTIONS, SECTOR_MAP, DISTRICT_MAP, EMPLOYMENT_STATUS


# ── Intent routing map ────────────────────────────────────────────────────────

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
    NLP-based query engine for LFS-2023 cluster data.

    Intent detection uses keyword matching only — no LLM required.
    All natural language generation is delegated to Groq via LLMQueryEngine.
    """

    def __init__(self, model_path):
        print(" Loading trained clustering model...")
        self.skilldev_model = load_model_safely(model_path)

        self.df = self.skilldev_model.df
        self.kmeans = self.skilldev_model.kmeans
        self.features = self.skilldev_model.features

        # Load scaler for outlier detection
        self.scaler = None
        possible_paths = [
            os.path.join(os.path.dirname(model_path), 'scaler.pkl'),
            os.path.join(os.path.dirname(model_path), '..', 'model', 'scaler.pkl'),
        ]
        for path in possible_paths:
            normalized_path = os.path.normpath(path)
            if os.path.exists(normalized_path):
                with open(normalized_path, 'rb') as sf:
                    self.scaler = pickle.load(sf)
                print(f" Scaler loaded from {normalized_path}")
                break

        if self.scaler is None:
            print("⚠️  scaler.pkl not found — outlier detection will be approximate")

        print(" Model loaded successfully!")
        print(" Intent detection: keyword routing (no Ollama required)")

    # ── Public interface ───────────────────────────────────────────────────────

    def understand_query(self, query: str) -> dict:
        """
        Classify query intent using keyword matching.

        Returns a dict with keys: intent, route, confidence, query.
        Confidence is fixed at 0.85 for clear keyword matches, 0.5 for fallback.
        """
        print(f"\n[ROUTER] Classifying: '{query}'")
        result = self._keyword_intent(query)
        print(f"[ROUTER] → intent={result['intent']} | route={result['route']} | confidence={result['confidence']}")
        return result

    def query_clusters(self, query: str):
        """Execute query against the cluster data."""
        intent_result = self.understand_query(query)
        route = intent_result['route']

        if route == "cluster_query":
            return self._get_cluster_records(query)
        elif route == "compare_clusters":
            return self._compare_clusters()
        elif route == "insights":
            return self._analyze_patterns(query)
        elif route == "general_analysis":
            if any(kw in query.lower() for kw in ['outlier', 'extreme', 'unusual', 'anomal']):
                return self._find_outliers()
            return self._get_cluster_stats()
        else:
            return self._get_cluster_stats()

    # ── Keyword intent classifier ──────────────────────────────────────────────

    def _keyword_intent(self, query: str) -> dict:
        """
        Keyword-based intent detection. Fast and deterministic.
        Covers all 7 intent categories without any LLM call.
        """
        q = query.lower()

        # Resource allocation — highest priority, check first
        if any(kw in q for kw in [
            'give', 'distribut', 'allocat', 'provide', 'deliver', 'hand out',
            'send', 'assign', 'target', 'beneficiar', 'most vulnerable',
            'most in need', 'who should receive', 'neediest', 'poorest',
            'sewing', 'laptop', 'computer', 'scholarship', 'phone', 'tablet',
            'device', 'aid', 'ration', 'food packet', 'grant',
        ]):
            return {
                'intent': 'allocate or distribute resources to people',
                'route': 'resource_allocation',
                'confidence': 0.90,
                'query': query,
            }

        # Cluster comparison
        if any(kw in q for kw in [
            'compare', 'difference', 'vs ', 'versus', 'differ', 'contrast',
            'compare cluster', 'all cluster', 'between group',
        ]):
            return {
                'intent': 'compare clusters or segments',
                'route': 'compare_clusters',
                'confidence': 0.85,
                'query': query,
            }

        # Specific cluster query
        if any(kw in q for kw in [
            'cluster 0', 'cluster 1', 'cluster 2', 'cluster 3', 'cluster 4',
            'which cluster', 'tell me about cluster', 'show cluster',
            'cluster stats', 'cluster statistic', 'cluster summary',
        ]):
            return {
                'intent': 'find records in a specific cluster',
                'route': 'cluster_query',
                'confidence': 0.85,
                'query': query,
            }

        # Insights / trends
        if any(kw in q for kw in [
            'insight', 'trend', 'pattern', 'overview', 'summary', 'highlight',
            'key finding', 'tell me about', 'what is happening',
        ]):
            return {
                'intent': 'get insights or trends in the data',
                'route': 'insights',
                'confidence': 0.80,
                'query': query,
            }

        # Outlier detection
        if any(kw in q for kw in [
            'outlier', 'extreme', 'unusual', 'anomal', 'irregular',
            'does not fit', 'edge case',
        ]):
            return {
                'intent': 'identify outliers or unusual records',
                'route': 'general_analysis',
                'confidence': 0.80,
                'query': query,
            }

        # General demographic / statistical analysis (default)
        return {
            'intent': 'analyze demographic or statistical patterns',
            'route': 'general_analysis',
            'confidence': 0.60,
            'query': query,
        }

    # ── Cluster data methods ───────────────────────────────────────────────────

    def _get_cluster_records(self, query: str):
        cluster_nums = re.findall(r'\d+', query)
        cluster_id = (
            int(cluster_nums[0]) % self.skilldev_model.n_clusters
            if cluster_nums else 0
        )
        cluster_data = self.df[self.df['cluster_id'] == cluster_id]
        print(f" Cluster {cluster_id}: {len(cluster_data)} records")
        print(cluster_data[self.features[:5]].head(10))
        return cluster_data

    def _compare_clusters(self):
        print("\n Cluster Comparison:")
        for cluster_id in range(self.skilldev_model.n_clusters):
            cluster_data = self.df[self.df['cluster_id'] == cluster_id]
            print(f"\nCluster {cluster_id}:")
            print(f"  Records: {len(cluster_data)}")
            print(f"  Mean values: {cluster_data[self.features[:3]].mean().round(2).to_dict()}")

    def _analyze_patterns(self, query: str):
        print("\n🔬 Pattern Analysis:")
        for feature in self.features[:5]:
            cluster_means = self.df.groupby('cluster_id')[feature].mean()
            print(f"\n{feature}:")
            print(cluster_means.round(2))
        return self.df.groupby('cluster_id')[self.features[:5]].mean()

    def _find_outliers(self):
        """Identify outlier records using the saved scaler from training."""
        print("\n Outlier Detection:")

        if self.scaler is not None:
            X_scaled = self.scaler.transform(self.df[self.features])
        else:
            print("  Using fallback scaler — results may differ from cluster assignments")
            from sklearn.preprocessing import StandardScaler
            fallback_scaler = StandardScaler()
            X_scaled = fallback_scaler.fit_transform(self.df[self.features])

        outliers = np.where((np.abs(X_scaled) > 3).any(axis=1))[0]
        print(f"Found {len(outliers)} outlier records")
        if len(outliers) > 0:
            print(self.df.iloc[outliers[:10]][self.features[:5]])

        return self.df.iloc[outliers]

    def _get_cluster_stats(self) -> dict:
        """Get comprehensive cluster statistics."""
        print("\n Cluster Statistics:")
        stats = {
            'Total Records': len(self.df),
            'Clusters': self.skilldev_model.n_clusters,
            'Distribution': self.df['cluster_id'].value_counts().to_dict(),
        }
        for key, value in stats.items():
            print(f"  {key}: {value}")
        return stats

    def interactive_query(self):
        """Interactive query loop."""
        print("\n" + "=" * 60)
        print(" NLP Cluster Query Engine — Interactive Mode")
        print("=" * 60)
        print("Type 'quit' to exit\n")

        while True:
            query = input(" Your query: ").strip()
            if query.lower() == 'quit':
                print(" Goodbye!")
                break
            if query:
                self.query_clusters(query)