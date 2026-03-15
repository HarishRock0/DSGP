from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from models.resource_allocation_model import FEATURE_COLS

warnings.filterwarnings("ignore")


class PovertyDataLoader:

    DATA_REL_PATH = os.path.join("data", "resource", "Povertylines.xlsx")

    def __init__(self, project_root: str):
        self.project_root = project_root
        self._scaler = MinMaxScaler()

    def load(self):
        """Returns (df_raw, df_norm)."""
        data_path = os.path.join(self.project_root, self.DATA_REL_PATH)
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at: {data_path}")
        return self._preprocess(data_path)

    def _preprocess(self, xlsx_path: str):
        df = pd.read_excel(xlsx_path)

        df.columns = (
            df.columns.str.strip()
            .str.lower()
            .str.replace(r"\s+", "_", regex=True)   # all whitespace runs → _
            .str.replace(r"[().]", "", regex=True)   # strip punctuation
            .str.replace(r"_+", "_", regex=True)     # collapse double underscores
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

        # ── Coerce all numeric columns ───────────────────────────────────────
        for col in [c for c in df.columns if c != "district"]:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", "").str.strip(),
                errors="coerce",
            )

        # ── Impute zero Gini values with column median ───────────────────────
        # Kilinochchi has gini_income = 0 (missing data, not a true zero).
        median_gini = df.loc[df["gini_income"] > 0, "gini_income"].median()
        df.loc[df["gini_income"] == 0, "gini_income"] = median_gini

        median_gini_exp = df.loc[df["gini_expenditure"] > 0, "gini_expenditure"].median()
        df.loc[df["gini_expenditure"] == 0, "gini_expenditure"] = median_gini_exp

        # ── FIX: Gampaha per-capita income data-entry error ──────────────────
        # Source value is 269 LKR/month — clearly wrong for a high-income district.
        # Correct value derived from mean_hh_income / avg_hh_size ≈ 27,150.
        mask = df["district"] == "Gampaha"
        df.loc[mask, "mean_per_capita_income"] = (
            df.loc[mask, "mean_hh_income"] / df.loc[mask, "avg_hh_size"]
        ).round(0)

        df_raw = df.reset_index(drop=True)

        # ── Derive engineered features ───────────────────────────────────────
        # Price columns are date-prefixed (e.g. "2024_jan", "2025_sep") and come
        # first in the sheet.  After column-name cleaning they all start with "202".
        price_cols = [c for c in df.columns if c.startswith("202")]
        month_x = np.arange(len(price_cols))

        df["price_volatility"]   = df[price_cols].std(axis=1)
        df["poverty_proxy"]      = 1 / df["mean_per_capita_income"]
        df["inequality_score"]   = df["gini_income"]
        df["expenditure_burden"] = df["mean_hh_expenditure"] / df["mean_hh_income"]
        df["hh_size_pressure"]   = df["avg_hh_size"]
        df["price_pressure"]     = df[price_cols[-1]]          # most-recent month
        df["price_trend"]        = df[price_cols].apply(
            lambda row: np.polyfit(month_x, row.values.astype(float), 1)[0],
            axis=1,
        )

        # ── Min-Max normalise feature columns ───────────────────────────────
        df_norm = df.copy()
        df_norm[FEATURE_COLS] = self._scaler.fit_transform(df[FEATURE_COLS])

        return df_raw.reset_index(drop=True), df_norm.reset_index(drop=True)