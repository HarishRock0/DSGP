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
