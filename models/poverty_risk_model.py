import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Keyword aliases for tier detection
TIER_KEYWORDS = {
    "CRITICAL": ["critical", "extreme", "very high risk", "highest risk", "worst", "most vulnerable"],
    "HIGH":     ["high", "high risk", "elevated", "serious", "severe"],
    "MODERATE": ["moderate", "medium", "mid", "average risk", "middle"],
    "LOW":      ["low", "low risk", "safe", "least vulnerable", "best", "minimal risk"],
}


def _detect_tier(user_input: str) -> str | None:
    """Return the first matching risk tier keyword found in user_input, or None."""
    lower = user_input.lower()
    for tier, keywords in TIER_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return tier
    return None


class PovertyRiskModel:
    def __init__(self, scaler, feature_cols, rules, encoder, district_data, embeddings):
        self.scaler        = scaler
        self.feature_cols  = feature_cols
        self.rules         = rules
        self.encoder       = encoder
        self.district_data = district_data
        self.embeddings    = embeddings


    def predict_risk(self, district_name: str) -> dict | None:
        row = self.district_data[
            self.district_data["District"].str.lower() == district_name.lower()
        ]
        if row.empty:
            return None
        r = row.iloc[0]
        return {
            "district":   r["District"],
            "risk_index": r["risk_index"],
            "risk_tier":  r["risk_tier"],
            "allocation": self.rules[r["risk_tier"]],
        }

    def query(self, user_input: str, top_k: int = 3) -> list:
        detected_tier = _detect_tier(user_input)

        if detected_tier:
            mask      = self.district_data["risk_tier"] == detected_tier
            subset_df = self.district_data[mask].reset_index(drop=True)
            # Gather the original integer positions so we can index embeddings
            orig_indices = self.district_data.index[mask].tolist()
        else:
            subset_df    = self.district_data.reset_index(drop=True)
            orig_indices = list(range(len(self.district_data)))

        if subset_df.empty:
            return []

        q_emb          = self.encoder.encode([user_input], convert_to_numpy=True)
        subset_embeddings = self.embeddings[orig_indices]
        scores         = cosine_similarity(q_emb, subset_embeddings)[0]

        actual_k = min(top_k, len(subset_df))
        top_idx  = np.argsort(scores)[::-1][:actual_k]

        results = []
        for rank, idx in enumerate(top_idx, 1):
            row  = subset_df.iloc[idx]
            tier = row["risk_tier"]
            results.append({
                "rank":       rank,
                "district":   row["District"],
                "risk_index": row["risk_index"],
                "risk_tier":  tier,
                "similarity": round(float(scores[idx]), 4),
                "actions":    self.rules[tier]["recommended_actions"],
            })
        return results

    def risk_table(self):
        cols = [
            "District", "risk_index", "risk_tier",
            "mean_household_income_per_month",
            "mean_household_per_capita_expenditure_per_month",
            "gini_coefficient_income", "affordability_ratio",
        ]
        return (
            self.district_data[cols]
            .sort_values("risk_index", ascending=False)
            .reset_index(drop=True)
        )

    def filter_by_tier(self, tier: str):
        tier = tier.upper()
        sub  = self.district_data[self.district_data["risk_tier"] == tier]
        return sub[
            ["District", "risk_index", "mean_household_income_per_month", "gini_coefficient_income"]
        ].reset_index(drop=True)