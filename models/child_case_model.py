import re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util


YEAR_RANGE = range(2010, 2025)
SIMILARITY_THRESHOLD = 0.35


# ─────────────────────────────────────────────────────────────────────────────
# Rule Functions — defined here so pickle can always resolve them
# ─────────────────────────────────────────────────────────────────────────────

def rule_high_cases(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    return df.sort_values('average_child_cases', ascending=False).head(n)


def rule_low_cases(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    return df.sort_values('average_child_cases', ascending=True).head(n)


def rule_moderate_risk(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    mean_val = df['average_child_cases'].mean()
    return df.iloc[(df['average_child_cases'] - mean_val).abs().argsort()[:n]]


def rule_top5_critical(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    return df.sort_values('average_child_cases', ascending=False).head(n)


def rule_by_risk_tier(df: pd.DataFrame, tier: str) -> pd.DataFrame:
    tier = tier.capitalize()
    return df[df['risk_tier'] == tier].sort_values(
        'average_child_cases', ascending=(tier == 'Low')
    )


def rule_district_lookup(df: pd.DataFrame, district_name: str) -> pd.DataFrame:
    mask = df['District'].str.contains(district_name.strip().upper(), na=False)
    return df[mask]


def rule_year_trend(df: pd.DataFrame, year: int) -> pd.DataFrame:
    year_col = next(
        (c for c in df.columns if str(c).replace('*', '').strip() == str(year)), None
    )
    if year_col is None:
        return df.head(0)
    return df[['District', year_col, 'risk_tier']].sort_values(year_col, ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# ChildCaseRuleEngine — mirrors PovertyRiskModel pattern
# encoder (st_model) is NOT stored — attached fresh at runtime in the dataloader
# ─────────────────────────────────────────────────────────────────────────────

class ChildCaseRuleEngine:
    def __init__(self, templates, template_queries, template_embeddings, df_main, known_districts):
        self.templates           = templates
        self.template_queries    = template_queries
        self.template_embeddings = template_embeddings  # numpy array — version-safe
        self.df_main             = df_main
        self.known_districts     = known_districts
        self.st_model            = None  # attached fresh at runtime

    def precheck_query(self, query: str):
        q_upper = query.upper()
        year_match = re.search(r'\b(20[12][0-9])\b', q_upper)
        if year_match:
            year = int(year_match.group(1))
            if year in YEAR_RANGE:
                return rule_year_trend, {'year': year}
        for district in self.known_districts:
            if district in q_upper:
                return rule_district_lookup, {'district_name': district}
        return None, None

    def recommend(self, user_query: str, top_n: int = 10) -> pd.DataFrame:
        import torch

        action_fn, params = self.precheck_query(user_query)
        if action_fn is not None:
            return action_fn(self.df_main.copy(), **params)

        query_embedding = self.st_model.encode(user_query, convert_to_tensor=True)
        te = torch.tensor(self.template_embeddings).to(query_embedding.device)
        similarities = util.cos_sim(query_embedding, te)[0]

        best_idx      = int(similarities.argmax())
        best_score    = float(similarities[best_idx])
        best_template = self.templates[best_idx]

        if best_score < SIMILARITY_THRESHOLD:
            return self.df_main.copy().sort_values('average_child_cases', ascending=False)

        merged_params = {**best_template['params']}
        if 'n' in merged_params:
            merged_params['n'] = top_n

        return best_template['action'](self.df_main.copy(), **merged_params)