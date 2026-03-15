from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ── Feature columns (order must match MinMaxScaler fit in PovertyDataLoader) ─

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
        "poverty_proxy":      0.30,
        "inequality_score":   0.20,
        "expenditure_burden": 0.20,
        "price_pressure":     0.15,
        "price_trend":        0.10,
        "hh_size_pressure":   0.05,
    },
    "penalties": {
        # Districts below this per-capita income (LKR/month) get an extra bonus
        "extreme_poverty_threshold": 15_000,
        "extreme_poverty_bonus":     0.10,
        # High income inequality bonus
        "high_gini_threshold":       0.45,
        "high_gini_bonus":           0.08,
        # Households spending >90 % of income bonus
        "expenditure_burden_threshold": 0.90,
        "expenditure_burden_bonus":     0.07,
        # Large households (≥4 persons) with below-median income bonus
        "large_hh_size_threshold":   4.0,
        "large_hh_income_threshold": 55_000,
        "large_hh_bonus":            0.06,
        # Rising prices in already-poor districts bonus
        "rising_price_poverty_bonus": 0.05,
    },
    "equity_floor_pct": 0.015,   # 1.5 % of total budget per district, guaranteed
}


def _compute_enhanced_risk_score(
    row_norm: pd.Series,
    row_raw:  pd.Series,
    rules:    dict,
) -> float:
    p = rules["penalties"]

    # Weighted base score from normalised features
    base = sum(row_norm[feat] * weight for feat, weight in rules["base_weights"].items())

    # Threshold bonuses evaluated on raw values
    bonus = 0.0

    if row_raw["mean_per_capita_income"] < p["extreme_poverty_threshold"]:
        bonus += p["extreme_poverty_bonus"]

    if row_raw["gini_income"] > p["high_gini_threshold"]:
        bonus += p["high_gini_bonus"]

    if (row_raw["mean_hh_expenditure"] / row_raw["mean_hh_income"]) > p["expenditure_burden_threshold"]:
        bonus += p["expenditure_burden_bonus"]

    if (
        row_raw["avg_hh_size"] >= p["large_hh_size_threshold"]
        and row_raw["mean_hh_income"] < p["large_hh_income_threshold"]
    ):
        bonus += p["large_hh_bonus"]

    if row_norm["price_trend"] > 0.5 and row_norm["poverty_proxy"] > 0.5:
        bonus += p["rising_price_poverty_bonus"]

    return round(min(base + bonus, 1.0), 4)


def _classify_tier(score: float) -> str:
    """Map a risk score to a named tier."""
    if score >= 0.70:
        return "CRITICAL"
    if score >= 0.55:
        return "HIGH"
    if score >= 0.40:
        return "MODERATE"
    return "LOW"


class ResourceAllocationModel:

    def __init__(self, df_raw: pd.DataFrame, df_norm: pd.DataFrame):
        self._df_raw = df_raw.reset_index(drop=True)
        df_alloc     = df_norm.reset_index(drop=True).copy()

        # Score every district
        df_alloc["enhanced_risk_score"] = [
            _compute_enhanced_risk_score(
                df_norm.loc[i], df_raw.loc[i], ALLOCATION_RULES
            )
            for i in df_norm.index
        ]

        df_alloc["risk_tier"] = df_alloc["enhanced_risk_score"].apply(_classify_tier)

        total_risk = df_alloc["enhanced_risk_score"].sum()
        df_alloc["risk_pct"] = (df_alloc["enhanced_risk_score"] / total_risk * 100).round(3)

        self.risk_df = df_alloc[
            ["district", "enhanced_risk_score", "risk_tier", "risk_pct"]
        ].copy()

    def allocate(self, total_budget: float) -> pd.DataFrame:

        n            = len(self.risk_df)
        floor_amount = round(total_budget * ALLOCATION_RULES["equity_floor_pct"], 0)
        remaining    = total_budget - floor_amount * n

        out           = self.risk_df.copy()
        risk_sum      = out["enhanced_risk_score"].sum()
        out["prop_share"]      = out["enhanced_risk_score"] / risk_sum
        out["prop_allocation"] = (out["prop_share"] * remaining).round(0)
        out["floor_allocation"] = floor_amount
        out["total_allocation"] = out["floor_allocation"] + out["prop_allocation"]
        out["allocation_pct"]   = (out["total_allocation"] / total_budget * 100).round(3)

        # alloc_per_hh uses raw avg_hh_size, aligned by index
        out["alloc_per_hh"] = (
            out["total_allocation"].values / self._df_raw["avg_hh_size"].values
        ).round(0)

        # Correct rounding residual on highest-risk district
        diff = total_budget - out["total_allocation"].sum()
        if abs(diff) > 0:
            out.loc[out["enhanced_risk_score"].idxmax(), "total_allocation"] += diff

        return (
            out.sort_values("total_allocation", ascending=False)
            .reset_index(drop=True)
        )