import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class PovertyRiskModel:
    def __init__(self, scaler, feature_cols, rules, encoder, district_data, embeddings):
        self.scaler        = scaler
        self.feature_cols  = feature_cols
        self.rules         = rules
        self.encoder       = encoder
        self.district_data = district_data
        self.embeddings    = embeddings

    def predict_risk(self, district_name):
        row = self.district_data[self.district_data['District'].str.lower() == district_name.lower()]
        if row.empty:
            return None
        r = row.iloc[0]
        return {
            'district':   r['District'],
            'risk_index': r['risk_index'],
            'risk_tier':  r['risk_tier'],
            'allocation': self.rules[r['risk_tier']]
        }

    def query(self, user_input: str, top_k: int = 3) -> list:
        q_emb   = self.encoder.encode([user_input], convert_to_numpy=True)
        scores  = cosine_similarity(q_emb, self.embeddings)[0]
        top_idx = np.argsort(scores)[::-1][:top_k]
        results = []
        for rank, idx in enumerate(top_idx, 1):
            row  = self.district_data.iloc[idx]
            tier = row['risk_tier']
            results.append({
                'rank':       rank,
                'district':   row['District'],
                'risk_index': row['risk_index'],
                'risk_tier':  tier,
                'similarity': round(float(scores[idx]), 4),
                'actions':    self.rules[tier]['recommended_actions'],
            })
        return results

    def risk_table(self):
        cols = ['District', 'risk_index', 'risk_tier',
                'mean_household_income_per_month',
                'mean_household_per_capita_expenditure_per_month',
                'gini_coefficient_income', 'affordability_ratio']
        return self.district_data[cols].sort_values('risk_index', ascending=False).reset_index(drop=True)

    def filter_by_tier(self, tier: str):
        tier = tier.upper()
        sub  = self.district_data[self.district_data['risk_tier'] == tier]
        return sub[['District', 'risk_index', 'mean_household_income_per_month',
                    'gini_coefficient_income']].reset_index(drop=True)