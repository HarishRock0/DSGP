# import pandas as pd
#
# FEATURES_COLS = {
#     'R1_avg_burden':    ['Avg_cases'],
#     'R2_recent_trend':  ['2022', '2023', '2024'],
#     'R3_growth_rate':   ['2010', '2024'],
#     'R4_surge_penalty': ['2020', '2021', '2022', '2023', '2024', 'Avg_cases'],
#     'R5_recovery_gap':  ['2021', '2024'],
# }
#
# ALLOCATION_RULES_COLS = {
#     'tier_thresholds': {
#         'CRITICAL': {'score_min': 65, 'score_max': 100, 'tier_weight': 4.0},
#         'HIGH':     {'score_min': 45, 'score_max':  64, 'tier_weight': 2.5},
#         'MODERATE': {'score_min': 25, 'score_max':  44, 'tier_weight': 1.5},
#         'LOW':      {'score_min':  0, 'score_max':  24, 'tier_weight': 1.0},
#     },
#     'allocation_formula':  'Allocated_LKR = (Risk_Score x Tier_Weight / sum) x Budget',
#     'recent_year_weights': {'2022': 0.30, '2023': 0.35, '2024': 0.35},
#     'min_floor_pct':       1.0,
#     'score_range':         (0, 100),
#     'n_districts':         25,
# }
#
#
# class ChildResourceAllocationModel:
#
#     def __init__(self, scaler, feature_cols, rules, encoder=None):
#         self.scaler = scaler
#         self.feature_cols = feature_cols
#         self.rules = rules
#         self.encoder = encoder
#
#         # NLP artefacts — attached after construction
#         self.corpus_embeddings = None
#         self.corpus_labels = None
#         self.corpus_sentences = None
#         self.district_aliases = None
#
#         # Data artefacts
#         self.risk_df = None
#         self.case_df = None
#
#         # Metadata
#         self.meta = {}
#
#     # ------------------------------------------------------------------
#     # Attachment helpers
#     # ------------------------------------------------------------------
#     def attach_nlp(self, corpus_embeddings, corpus_labels,
#                    corpus_sentences, district_aliases):
#         """Attach pre-computed sentence-transformer NLP artefacts."""
#         self.corpus_embeddings = corpus_embeddings
#         self.corpus_labels = corpus_labels
#         self.corpus_sentences = corpus_sentences
#         self.district_aliases = district_aliases
#
#     def attach_data(self, risk_df, case_df):
#         """Attach the scored risk DataFrame and raw case data."""
#         self.risk_df = risk_df
#         self.case_df = case_df
#
#     # ------------------------------------------------------------------
#     # Encoder management
#     # ------------------------------------------------------------------
#     def load_encoder(self, model_name=None):
#
#         from sentence_transformers import SentenceTransformer
#         name = model_name or self.meta.get('model_name', 'all-MiniLM-L6-v2')
#         self.encoder = SentenceTransformer(name)
#         print(f'Encoder loaded: {name}')
#         return self.encoder
#
#     def release_encoder(self):
#         """Set encoder to None and free GPU/CPU memory."""
#         import gc
#         self.encoder = None
#         gc.collect()
#         try:
#             import torch
#             torch.cuda.empty_cache()
#         except Exception:
#             pass
#         print('Encoder released')
#
#     # ------------------------------------------------------------------
#     # Inference helpers
#     # ------------------------------------------------------------------
#     def detect_intent(self, query: str) -> tuple:
#
#         if self.encoder is None:
#             raise RuntimeError(
#                 'Encoder is None. Call load_encoder() before inference.'
#             )
#         if self.corpus_embeddings is None or self.corpus_labels is None:
#             raise RuntimeError('NLP artefacts not attached. Call attach_nlp().')
#
#         q_emb = self.encoder.encode([query.lower()], convert_to_numpy=True)
#         sims = cosine_similarity(q_emb, self.corpus_embeddings).flatten()
#         idx = int(np.argmax(sims))
#         return self.corpus_labels[idx], float(sims[idx])
#
#     def transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
#         """
#         Apply the fitted scalers to a DataFrame and return a scores
#         DataFrame with one column per rule.
#
#         Parameters
#         ----------
#         df : pd.DataFrame
#             Must contain the columns listed in self.feature_cols.
#         """
#         df = df.copy()
#         df.columns = [str(c) for c in df.columns]
#         scores = pd.DataFrame({'District': df['District']})
#
#         for rule, sc in self.scaler.items():
#             cfg = self.feature_cols[rule]
#
#             if rule == 'R1_avg_burden':
#                 raw = df[cfg]
#             elif rule == 'R2_recent_trend':
#                 raw = (df['2022'] * 0.30 + df['2023'] * 0.35 + df['2024'] * 0.35
#                        ).values.reshape(-1, 1)
#             elif rule == 'R3_growth_rate':
#                 base = df['2010'].replace(0, 1)
#                 raw = ((df['2024'] - df['2010']) / base * 100).values.reshape(-1, 1)
#             elif rule == 'R4_surge_penalty':
#                 peak = df[['2020', '2021', '2022', '2023', '2024']].max(axis=1)
#                 raw = ((peak - df['Avg_cases']) / df['Avg_cases'].replace(0, 1) * 100
#                        ).clip(lower=0).values.reshape(-1, 1)
#             elif rule == 'R5_recovery_gap':
#                 drop = ((df['2021'] - df['2024']) / df['2021'].replace(0, 1) * 100
#                         ).clip(lower=0).values.reshape(-1, 1)
#                 scores[rule] = (10 - sc.transform(drop).flatten()).round(2)
#                 continue
#             else:
#                 continue
#
#             scores[rule] = sc.transform(raw).round(2)
#
#         return scores
#
#     def classify_tier(self, score: float) -> str:
#         """Return the risk tier string for a given composite score."""
#         for tier, cfg in self.rules['tier_thresholds'].items():
#             if cfg['score_min'] <= score <= cfg['score_max']:
#                 return tier
#         return 'LOW'
#
#     # ------------------------------------------------------------------
#     # Summary
#     # ------------------------------------------------------------------
#     def summary(self):
#         """Print a quick inventory of all loaded artefacts."""
#         print('ChildResourceAllocationModel')
#         print(f'  encoder           : {type(self.encoder).__name__}')
#         print(f'  scaler keys       : {list(self.scaler.keys())}')
#         print(f'  feature_cols keys : {list(self.feature_cols.keys())}')
#         print(f'  tier thresholds   : {list(self.rules["tier_thresholds"].keys())}')
#         print(f'  min_floor_pct     : {self.rules["min_floor_pct"]}%')
#         emb_shape = (self.corpus_embeddings.shape
#                      if self.corpus_embeddings is not None else None)
#         print(f'  corpus_embeddings : {emb_shape}')
#         risk_shape = self.risk_df.shape if self.risk_df is not None else None
#         print(f'  risk_df           : {risk_shape}')
#         print(f'  meta              : {self.meta}')

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

FEATURES_COLS = {
    'R1_avg_burden':    ['Avg_cases'],
    'R2_recent_trend':  ['2022', '2023', '2024'],
    'R3_growth_rate':   ['2010', '2024'],
    'R4_surge_penalty': ['2020', '2021', '2022', '2023', '2024', 'Avg_cases'],
    'R5_recovery_gap':  ['2021', '2024'],
}

ALLOCATION_RULES_COLS = {
    'tier_thresholds': {
        'CRITICAL': {'score_min': 65, 'score_max': 100, 'tier_weight': 4.0},
        'HIGH':     {'score_min': 45, 'score_max':  64, 'tier_weight': 2.5},
        'MODERATE': {'score_min': 25, 'score_max':  44, 'tier_weight': 1.5},
        'LOW':      {'score_min':  0, 'score_max':  24, 'tier_weight': 1.0},
    },
    'allocation_formula':  'Allocated_LKR = (Risk_Score x Tier_Weight / sum) x Budget',
    'recent_year_weights': {'2022': 0.30, '2023': 0.35, '2024': 0.35},
    'min_floor_pct':       1.0,
    'score_range':         (0, 100),
    'n_districts':         25,
}


class ChildResourceAllocationModel:

    def __init__(self, scaler, feature_cols, rules, encoder=None):
        self.scaler = scaler
        self.feature_cols = feature_cols
        self.rules = rules
        self.encoder = encoder

        # NLP artefacts — attached after construction
        self.corpus_embeddings = None
        self.corpus_labels = None
        self.corpus_sentences = None
        self.district_aliases = None

        # Data artefacts
        self.risk_df = None
        self.case_df = None

        # Metadata
        self.meta = {}

    # ------------------------------------------------------------------
    # Attachment helpers
    # ------------------------------------------------------------------
    def attach_nlp(self, corpus_embeddings, corpus_labels,
                   corpus_sentences, district_aliases):
        """Attach pre-computed sentence-transformer NLP artefacts."""
        self.corpus_embeddings = corpus_embeddings
        self.corpus_labels = corpus_labels
        self.corpus_sentences = corpus_sentences
        self.district_aliases = district_aliases

    def attach_data(self, risk_df, case_df):
        """Attach the scored risk DataFrame and raw case data."""
        self.risk_df = risk_df
        self.case_df = case_df

    # ------------------------------------------------------------------
    # Encoder management
    # ------------------------------------------------------------------
    def load_encoder(self, model_name=None):

        from sentence_transformers import SentenceTransformer
        name = model_name or self.meta.get('model_name', 'all-MiniLM-L6-v2')
        self.encoder = SentenceTransformer(name)
        print(f'Encoder loaded: {name}')
        return self.encoder

    def release_encoder(self):
        """Set encoder to None and free GPU/CPU memory."""
        import gc
        self.encoder = None
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        print('Encoder released')

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------
    def detect_intent(self, query: str) -> tuple:

        if self.encoder is None:
            raise RuntimeError(
                'Encoder is None. Call load_encoder() before inference.'
            )
        if self.corpus_embeddings is None or self.corpus_labels is None:
            raise RuntimeError('NLP artefacts not attached. Call attach_nlp().')

        q_emb = self.encoder.encode([query.lower()], convert_to_numpy=True)
        sims = cosine_similarity(q_emb, self.corpus_embeddings).flatten()
        idx = int(np.argmax(sims))
        return self.corpus_labels[idx], float(sims[idx])

    def transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the fitted scalers to a DataFrame and return a scores
        DataFrame with one column per rule.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain the columns listed in self.feature_cols.
        """
        df = df.copy()
        df.columns = [str(c) for c in df.columns]
        scores = pd.DataFrame({'District': df['District']})

        for rule, sc in self.scaler.items():
            cfg = self.feature_cols[rule]

            if rule == 'R1_avg_burden':
                raw = df[cfg]
            elif rule == 'R2_recent_trend':
                raw = (df['2022'] * 0.30 + df['2023'] * 0.35 + df['2024'] * 0.35
                       ).values.reshape(-1, 1)
            elif rule == 'R3_growth_rate':
                base = df['2010'].replace(0, 1)
                raw = ((df['2024'] - df['2010']) / base * 100).values.reshape(-1, 1)
            elif rule == 'R4_surge_penalty':
                peak = df[['2020', '2021', '2022', '2023', '2024']].max(axis=1)
                raw = ((peak - df['Avg_cases']) / df['Avg_cases'].replace(0, 1) * 100
                       ).clip(lower=0).values.reshape(-1, 1)
            elif rule == 'R5_recovery_gap':
                drop = ((df['2021'] - df['2024']) / df['2021'].replace(0, 1) * 100
                        ).clip(lower=0).values.reshape(-1, 1)
                scores[rule] = (10 - sc.transform(drop).flatten()).round(2)
                continue
            else:
                continue

            scores[rule] = sc.transform(raw).round(2)

        return scores

    def classify_tier(self, score: float) -> str:
        """Return the risk tier string for a given composite score."""
        for tier, cfg in self.rules['tier_thresholds'].items():
            if cfg['score_min'] <= score <= cfg['score_max']:
                return tier
        return 'LOW'

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    def summary(self):
        """Print a quick inventory of all loaded artefacts."""
        print('ChildResourceAllocationModel')
        print(f'  encoder           : {type(self.encoder).__name__}')
        print(f'  scaler keys       : {list(self.scaler.keys())}')
        print(f'  feature_cols keys : {list(self.feature_cols.keys())}')
        print(f'  tier thresholds   : {list(self.rules["tier_thresholds"].keys())}')
        print(f'  min_floor_pct     : {self.rules["min_floor_pct"]}%')
        emb_shape = (self.corpus_embeddings.shape
                     if self.corpus_embeddings is not None else None)
        print(f'  corpus_embeddings : {emb_shape}')
        risk_shape = self.risk_df.shape if self.risk_df is not None else None
        print(f'  risk_df           : {risk_shape}')
        print(f'  meta              : {self.meta}')