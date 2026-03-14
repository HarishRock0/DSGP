from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ── Constants ───────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "poverty_proxy",
    "inequality_score",
    "expenditure_burden",
    "hh_size_pressure",
    "price_pressure",
    "price_trend",
    "price_volatility",
]

ALLOCATION_RULES = {
    "base_weights": {
        "poverty_proxy": 0.30,
        "inequality_score": 0.20,
        "expenditure_burden": 0.20,
        "price_pressure": 0.15,
        "price_trend": 0.10,
        "hh_size_pressure": 0.05,
    },
    "penalties": {
        "extreme_poverty_threshold": 15_000,
        "extreme_poverty_bonus": 0.10,
        "high_gini_threshold": 0.45,
        "high_gini_bonus": 0.08,
        "expenditure_burden_threshold": 0.90,
        "expenditure_burden_bonus": 0.07,
        "large_hh_size_threshold": 4.0,
        "large_hh_income_threshold": 55_000,
        "large_hh_bonus": 0.06,
        "rising_price_poverty_bonus": 0.05,
    },
    "equity_floor_pct": 0.015,
}


def _compute_enhanced_risk_score(row_norm, row_raw, rules):
    p = rules["penalties"]
    base = sum(row_norm[f] * w for f, w in rules["base_weights"].items())
    bonus = 0.0

    if row_raw["mean_per_capita_income"] < p["extreme_poverty_threshold"]:
        bonus += p["extreme_poverty_bonus"]
    if row_raw["gini_income"] > p["high_gini_threshold"]:
        bonus += p["high_gini_bonus"]
    if (row_raw["mean_hh_expenditure"] / row_raw["mean_hh_income"]) > p["expenditure_burden_threshold"]:
        bonus += p["expenditure_burden_bonus"]
    if row_raw["avg_hh_size"] >= p["large_hh_size_threshold"] and row_raw["mean_hh_income"] < p["large_hh_income_threshold"]:
        bonus += p["large_hh_bonus"]
    if row_norm["price_trend"] > 0.5 and row_norm["poverty_proxy"] > 0.5:
        bonus += p["rising_price_poverty_bonus"]

    return round(min(base + bonus, 1.0), 4)


def _classify_tier(score: float):
    if score >= 0.70:
        return "CRITICAL"
    if score >= 0.55:
        return "HIGH"
    if score >= 0.40:
        return "MODERATE"
    return "LOW"


# ── Resource Allocation Model ────────────────────────────────────────────────

class ResourceAllocationModel:
    """
    Accepts preprocessed raw and normalized dataframes to compute
    enhanced risk scores, tiers, and budget allocations.
    """

    def __init__(self, df_raw: pd.DataFrame, df_norm: pd.DataFrame):
        self._df_raw = df_raw.reset_index(drop=True)
        df_alloc = df_norm.reset_index(drop=True).copy()

        df_alloc["enhanced_risk_score"] = [
            _compute_enhanced_risk_score(df_norm.loc[i], df_raw.loc[i], ALLOCATION_RULES)
            for i in df_norm.index
        ]

        df_alloc["risk_tier"] = df_alloc["enhanced_risk_score"].apply(_classify_tier)

        total_risk = df_alloc["enhanced_risk_score"].sum()
        df_alloc["risk_pct"] = (df_alloc["enhanced_risk_score"] / total_risk * 100).round(3)

        self.risk_df = df_alloc[["district", "enhanced_risk_score", "risk_tier", "risk_pct"]].copy()

    def allocate(self, total_budget: float):
        n = len(self.risk_df)
        floor_amount = round(total_budget * ALLOCATION_RULES["equity_floor_pct"], 0)
        total_floor = floor_amount * n
        remaining = total_budget - total_floor

        out = self.risk_df.copy()
        risk_sum = out["enhanced_risk_score"].sum()
        out["prop_share"] = out["enhanced_risk_score"] / risk_sum
        out["prop_allocation"] = (out["prop_share"] * remaining).round(0)
        out["floor_allocation"] = floor_amount
        out["total_allocation"] = out["floor_allocation"] + out["prop_allocation"]
        out["allocation_pct"] = (out["total_allocation"] / total_budget * 100).round(3)
        out["alloc_per_hh"] = (out["total_allocation"].values / self._df_raw["avg_hh_size"].values).round(0)

        diff = total_budget - out["total_allocation"].sum()
        if abs(diff) > 0:
            out.loc[out["enhanced_risk_score"].idxmax(), "total_allocation"] += diff

        return out.sort_values("total_allocation", ascending=False).reset_index(drop=True)