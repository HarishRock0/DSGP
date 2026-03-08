import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

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
        "extreme_poverty_threshold":    15_000,
        "extreme_poverty_bonus":        0.10,
        "high_gini_threshold":          0.45,
        "high_gini_bonus":              0.08,
        "expenditure_burden_threshold": 0.90,
        "expenditure_burden_bonus":     0.07,
        "large_hh_size_threshold":      4.0,
        "large_hh_income_threshold":    55_000,
        "large_hh_bonus":               0.06,
        "rising_price_poverty_bonus":   0.05,
    },
    "equity_floor_pct": 0.015,
}


class PovertyDataLoader:
    def __init__(self, project_root):
        self.project_root = project_root
        self.scaler = MinMaxScaler()

    def load(self):
        """
        Loads the pickled PovertyRiskModel and preprocesses Povertylines.xlsx
        into the two DataFrames the model expects:
          - df_raw  : cleaned raw data (for penalty conditions)
          - df_norm : normalised feature data (for base score computation)

        Returns
        -------
        model     : PovertyRiskModel  (loaded from poverty_model.pkl)
        df_raw    : pd.DataFrame      (cleaned, unrenamed socioeconomic columns)
        df_norm   : pd.DataFrame      (MinMax-normalised feature columns)
        """
        model_path = os.path.join(self.project_root, "model","resource", "poverty_model.pkl")
        poverty_line_path = os.path.join(self.project_root, "data", "resource", "Povertylines.xlsx")

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        df_raw, df_norm = self._preprocess(poverty_line_path)

        return model, df_raw, df_norm

    def _preprocess(self, poverty_line_path: str):
        """
        Full preprocessing pipeline from the notebook (Sections 2 & 3).
        Returns df_raw and df_norm ready to pass into model.compute_risk().
        """
        # ── Section 2: Load & Clean ───────────────────────────────────────────
        df = pd.read_excel(poverty_line_path)

        df.columns = (
            df.columns.str.strip().str.lower()
            .str.replace(r"\s+", "_", regex=True)
            .str.replace(r"[().]", "", regex=True)
            .str.replace(r"_+", "_", regex=True)
        )

        rename_map = {
            "mean_household_income_per_month":                 "mean_hh_income",
            "median_household_income_per_month_rs":            "median_hh_income",
            "average_household_size":                          "avg_hh_size",
            "gini_coefficient_income":                         "gini_income",
            "mean_per_capita_income_per_month_rs":             "mean_per_capita_income",
            "mean_household_expenditure_per_month_rs":         "mean_hh_expenditure",
            "median_household_expenditure_per_month_rs":       "median_hh_expenditure",
            "gini_coefficient_expenditure":                    "gini_expenditure",
            "mean_household_per_capita_expenditure_per_month": "mean_hh_per_capita_expenditure",
        }
        df.rename(columns=rename_map, inplace=True)

        for col in [c for c in df.columns if c != "district"]:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", "", regex=False).str.strip(),
                errors="coerce",
            )

        # Fix known anomalies (Kilinochchi Gini=0, Gampaha per-capita=269)
        median_gini = df.loc[df["gini_income"] > 0, "gini_income"].median()
        df.loc[df["gini_income"] == 0, "gini_income"] = median_gini

        median_gini_exp = df.loc[df["gini_expenditure"] > 0, "gini_expenditure"].median()
        df.loc[df["gini_expenditure"] == 0, "gini_expenditure"] = median_gini_exp

        mask = df["district"] == "Gampaha"
        df.loc[mask, "mean_per_capita_income"] = (
            df.loc[mask, "mean_hh_income"] / df.loc[mask, "avg_hh_size"]
        ).round(0)

        df_raw = df.reset_index(drop=True)

        # ── Section 3: Feature Engineering & Normalisation ───────────────────
        price_cols = [c for c in df.columns if c.startswith("202")]
        month_x    = np.arange(len(price_cols))

        df["price_volatility"]   = df[price_cols].std(axis=1)
        df["poverty_proxy"]      = 1 / df["mean_per_capita_income"]
        df["inequality_score"]   = df["gini_income"]
        df["expenditure_burden"] = df["mean_hh_expenditure"] / df["mean_hh_income"]
        df["hh_size_pressure"]   = df["avg_hh_size"]
        df["price_pressure"]     = df[price_cols[-1]]
        df["price_trend"]        = df[price_cols].apply(
            lambda row: np.polyfit(month_x, row.values.astype(float), 1)[0],
            axis=1,
        )

        df_norm = df.copy()
        df_norm[FEATURE_COLS] = self.scaler.fit_transform(df[FEATURE_COLS])

        return df_raw.reset_index(drop=True), df_norm.reset_index(drop=True)