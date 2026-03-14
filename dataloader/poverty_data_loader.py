import os
import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer
from models.poverty_risk_model import PovertyRiskModel  # noqa: F401 — required for pickle.load()


class PovertyDataLoader:
    def __init__(self, project_root):
        self.project_root = project_root

    def load(self):
        # -------------------------------
        # Paths
        # -------------------------------
        model_path        = os.path.join(self.project_root, "model", "poverty_risk_model.pkl")
        demographic_path  = os.path.join(self.project_root, "data", "demographic_district_wise.xlsx")
        poverty_line_path = os.path.join(self.project_root, "data", "Povertylines.xlsx")

        # -------------------------------
        # Load trained model
        # PovertyRiskModel is imported above so pickle.load() can resolve it
        # -------------------------------
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        # -------------------------------
        # Attach SentenceTransformer fresh at runtime
        # Version-safe — not stored in pkl
        # -------------------------------
        model.encoder = SentenceTransformer("all-MiniLM-L6-v2")

        # -------------------------------
        # Load datasets
        # -------------------------------
        region_data  = pd.read_excel(demographic_path)
        poverty_data = pd.read_excel(poverty_line_path)

        # -------------------------------
        # Poverty Feature Engineering
        # -------------------------------
        poverty_data.columns = poverty_data.columns.str.strip()

        poverty_cols = [
            c for c in poverty_data.columns
            if ("2024" in str(c) or "2025" in str(c))
        ]

        feature_cols = [
            "mean_household_income_per_month",
            "median_household_income_per_month_rs",
            "average_household_size",
            "gini_coefficient_income",
            "mean_per_capita_income_per_month_rs",
            "mean_household_expenditure_per_month_rs",
            "median_household_expenditure_per_month_rs",
            "gini_coefficient_expenditure",
            "mean_household_per_capita_expenditure_per_month",
        ]

        # -------------------------------
        # Derived Features
        # -------------------------------
        poverty_data["poverty_line_latest"] = poverty_data[poverty_cols].iloc[:, -1]
        poverty_data["poverty_line_mean"]   = poverty_data[poverty_cols].mean(axis=1)
        poverty_data["poverty_line_trend"]  = (
            poverty_data[poverty_cols].iloc[:, -1]
            - poverty_data[poverty_cols].iloc[:, 0]
        )
        poverty_data["income_expenditure_gap"] = (
            poverty_data["mean_household_income_per_month"]
            - poverty_data["mean_household_expenditure_per_month_rs"]
        )
        poverty_data["affordability_ratio"] = (
            poverty_data["mean_household_per_capita_expenditure_per_month"]
            / poverty_data["poverty_line_latest"]
        )
        poverty_data["income_inequality"] = (
            poverty_data["mean_household_income_per_month"]
            - poverty_data["median_household_income_per_month_rs"]
        )

        extended_features = feature_cols + [
            "poverty_line_latest",
            "poverty_line_mean",
            "poverty_line_trend",
            "income_expenditure_gap",
            "affordability_ratio",
            "income_inequality",
        ]

        poverty_features = poverty_data[["District"] + extended_features]

        # -------------------------------
        # Population Data
        # -------------------------------
        district_pop = (
            region_data.groupby("DISTRICT_N")["PPROJ_22"]
            .sum()
            .reset_index()
        )
        district_pop.rename(
            columns={"DISTRICT_N": "District", "PPROJ_22": "Population"},
            inplace=True,
        )

        # -------------------------------
        # Merge poverty + population
        # -------------------------------
        merged_df = pd.merge(
            district_pop,
            poverty_features,
            on="District",
            how="inner",
        )

        # -------------------------------
        # Text representation for NLP
        # -------------------------------
        merged_df["text"] = (
            merged_df["District"]
            + " Population: "
            + merged_df["Population"].astype(str)
            + " PovertyIndexInputs: "
            + merged_df[extended_features].astype(str).agg(" ".join, axis=1)
        )

        return model, merged_df