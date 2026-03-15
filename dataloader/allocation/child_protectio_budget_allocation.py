# from __future__ import annotations
#
# import pickle
# import sys
# import warnings
# import os
# import sys
#
# _HERE = os.path.dirname(os.path.abspath(__file__))
# _PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
# if _PROJECT_ROOT not in sys.path:
#     sys.path.insert(0, _PROJECT_ROOT)
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
#
# from models.child_protection_allocation_model import (
#     ChildResourceAllocationModel,
#     FEATURES_COLS,
#     ALLOCATION_RULES_COLS,
# )
#
# warnings.filterwarnings("ignore")
#
# _PKL_REL_PATH  = os.path.join("model", "resource allocation models", "child_welfare_pipeline.pkl")
# _DATA_REL_PATH = os.path.join("data", "childcases.xlsx")
# _YEAR_COLS     = [str(y) for y in range(2010, 2025)]
#
#
# class ChildProtectionBudgetDataLoader:
#
#     DATA_REL_PATH = _DATA_REL_PATH
#
#     def __init__(self, project_root: str) -> None:
#         self.project_root = project_root
#         self._scaler = MinMaxScaler()
#
#     def load(self) -> tuple[pd.DataFrame, pd.DataFrame]:
#         data_path = os.path.join(self.project_root, self.DATA_REL_PATH)
#         if not os.path.exists(data_path):
#             raise FileNotFoundError(f"Data file not found at: {data_path}")
#         return self._preprocess(data_path)
#
#     def _preprocess(self, xlsx_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
#         df = pd.read_excel(xlsx_path)
#
#         df.columns = (
#             df.columns.str.strip()
#             .str.lower()
#             .str.replace(r"\s+", "_", regex=True)
#             .str.replace(r"[().]", "", regex=True)
#             .str.replace(r"_+", "_", regex=True)
#         )
#
#         if "district" in df.columns:
#             df.rename(columns={"district": "District"}, inplace=True)
#         df["District"] = df["District"].astype(str).str.strip().str.title()
#
#         year_cols_present = [c for c in _YEAR_COLS if c in df.columns]
#         for col in year_cols_present:
#             df[col] = pd.to_numeric(
#                 df[col].astype(str).str.replace(",", "").str.strip(),
#                 errors="coerce",
#             )
#
#         if "avg_cases" not in df.columns and year_cols_present:
#             df["Avg_cases"] = df[year_cols_present].mean(axis=1).round(2)
#         elif "avg_cases" in df.columns:
#             df.rename(columns={"avg_cases": "Avg_cases"}, inplace=True)
#
#         df.columns = [str(c) for c in df.columns]
#
#         df_raw  = df.reset_index(drop=True)
#         df_norm = df_raw.copy()
#
#         return df_raw, df_norm
#
#
#     def _load_pkl(project_root: str) -> ChildResourceAllocationModel:
#         """
#         Load the pkl with a __main__ shim so pickle can find ChildResourceAllocationModel
#         regardless of whether the pkl was dumped from a Jupyter notebook (__main__)
#         or from the package (models.child_protection_allocation_model).
#         """
#         pkl_path = os.path.join(project_root, _PKL_REL_PATH)
#         if not os.path.exists(pkl_path):
#             raise FileNotFoundError(
#                 f"Pipeline pickle not found: {pkl_path}\n"
#                 "Run the notebook dump cell to generate it."
#             )
#
#         # Pickle stores the class path used at dump time.
#         # The notebook dumped it under __main__.ChildResourceAllocationModel.
#         # We register it there so pickle.load() can resolve it.
#         import types
#         shim = types.ModuleType("__main__")
#         shim.ChildResourceAllocationModel = ChildResourceAllocationModel
#         sys.modules.setdefault("__main__", shim)
#         # Also cover the case where pkl was dumped from the module path directly
#         sys.modules["__main__"].ChildResourceAllocationModel = ChildResourceAllocationModel
#
#         with open(pkl_path, "rb") as f:
#             model: ChildResourceAllocationModel = pickle.load(f)
#
#         model.feature_cols = FEATURES_COLS
#         model.rules        = ALLOCATION_RULES_COLS
#         return model
#
#
#     def _score_districts(model: ChildResourceAllocationModel, df: pd.DataFrame) -> pd.DataFrame:
#         scores = model.transform_features(df)
#
#         rule_cols = ["R1_avg_burden", "R2_recent_trend", "R3_growth_rate",
#                      "R4_surge_penalty", "R5_recovery_gap"]
#         scores["Risk_Score"]  = sum(
#             scores[col] for col in rule_cols if col in scores.columns
#         ).round(2)
#         scores["Risk_Tier"]   = scores["Risk_Score"].apply(model.classify_tier)
#         scores["Tier_Weight"] = scores["Risk_Tier"].map({
#             t: cfg["tier_weight"]
#             for t, cfg in model.rules["tier_thresholds"].items()
#         })
#
#         return scores.reset_index(drop=True)
#
#
#     def get_child_protection_pipeline(
#         project_root: str,
#         restore_encoder: bool = True,
#         encoder_name: str | None = None,
#     ) -> tuple[ChildResourceAllocationModel, pd.DataFrame, pd.DataFrame]:
#
#         loader = ChildProtectionBudgetDataLoader(project_root)
#         df_raw, df_norm = loader.load()
#         print(f"Child cases data loaded: {df_raw.shape[0]} districts")
#
#         model = _load_pkl(project_root)
#         print("child_welfare_pipeline.pkl loaded")
#
#         if restore_encoder:
#             model.load_encoder(encoder_name)
#
#         risk_df = _score_districts(model, df_norm)
#         model.attach_data(risk_df=risk_df, case_df=df_raw)
#         print(f"Districts scored. Tier distribution:\n{risk_df['Risk_Tier'].value_counts().to_string()}\n")
#
#         return model, df_raw, df_norm
from __future__ import annotations

import pickle
import sys
import types
import warnings
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from models.child_protection_allocation_model import (
    ChildResourceAllocationModel,
    FEATURES_COLS,
    ALLOCATION_RULES_COLS,
)

warnings.filterwarnings("ignore")

_PKL_REL_PATH = os.path.join("model", "resource allocation models", "child_welfare_pipeline.pkl")
_DATA_REL_PATH = os.path.join("data", "childcases.xlsx")
_YEAR_COLS = [str(y) for y in range(2010, 2025)]


class ChildProtectionBudgetDataLoader:
    DATA_REL_PATH = _DATA_REL_PATH

    def __init__(self, project_root: str) -> None:
        self.project_root = project_root
        self._scaler = MinMaxScaler()

    def load(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        data_path = os.path.join(self.project_root, self.DATA_REL_PATH)
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at: {data_path}")
        return self._preprocess(data_path)

    def _preprocess(self, xlsx_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = pd.read_excel(xlsx_path)
        df.columns = (
            df.columns.astype(str).str.strip()
            .str.lower()
            .str.replace(r"\s+", "_", regex=True)
            .str.replace(r"[().]", "", regex=True)
            .str.replace(r"_+", "_", regex=True)
        )
        if "district" in df.columns:
            df.rename(columns={"district": "District"}, inplace=True)
        df["District"] = df["District"].astype(str).str.strip().str.title()
        year_cols_present = [c for c in _YEAR_COLS if c in df.columns]
        for col in year_cols_present:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", "").str.strip(),
                errors="coerce",
            )
        if "avg_cases" not in df.columns and year_cols_present:
            df["Avg_cases"] = df[year_cols_present].mean(axis=1).round(2)
        elif "avg_cases" in df.columns:
            df.rename(columns={"avg_cases": "Avg_cases"}, inplace=True)
        df.columns = [str(c) for c in df.columns]
        df_raw = df.reset_index(drop=True)
        df_norm = df_raw.copy()
        return df_raw, df_norm


def _load_pkl(project_root: str) -> ChildResourceAllocationModel:
    pkl_path = os.path.join(project_root, _PKL_REL_PATH)
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(
            f"Pipeline pickle not found: {pkl_path}\n"
            "Run rebuild_child_welfare_pkl.py to generate it."
        )
    shim = types.ModuleType("__main__")
    shim.ChildResourceAllocationModel = ChildResourceAllocationModel
    sys.modules.setdefault("__main__", shim)
    sys.modules["__main__"].ChildResourceAllocationModel = ChildResourceAllocationModel
    with open(pkl_path, "rb") as f:
        model: ChildResourceAllocationModel = pickle.load(f)
    model.feature_cols = FEATURES_COLS
    model.rules = ALLOCATION_RULES_COLS
    return model


def _score_districts(model: ChildResourceAllocationModel, df: pd.DataFrame) -> pd.DataFrame:
    scores = model.transform_features(df)
    rule_cols = [
        "R1_avg_burden", "R2_recent_trend", "R3_growth_rate",
        "R4_surge_penalty", "R5_recovery_gap",
    ]
    scores["Risk_Score"] = sum(
        scores[col] for col in rule_cols if col in scores.columns
    ).round(2)
    scores["Risk_Tier"] = scores["Risk_Score"].apply(model.classify_tier)
    scores["Tier_Weight"] = scores["Risk_Tier"].map({
        t: cfg["tier_weight"]
        for t, cfg in model.rules["tier_thresholds"].items()
    })
    return scores.reset_index(drop=True)


def get_child_protection_pipeline(
        project_root: str,
        restore_encoder: bool = True,
        encoder_name: str | None = None,
) -> tuple[ChildResourceAllocationModel, pd.DataFrame, pd.DataFrame]:
    loader = ChildProtectionBudgetDataLoader(project_root)
    df_raw, df_norm = loader.load()
    print(f"Child cases data loaded: {df_raw.shape[0]} districts")
    model = _load_pkl(project_root)
    print("child_welfare_pipeline.pkl loaded")
    if restore_encoder:
        model.load_encoder(encoder_name)
    risk_df = _score_districts(model, df_norm)
    model.attach_data(risk_df=risk_df, case_df=df_raw)
    print(f"Districts scored. Tier distribution:\n{risk_df['Risk_Tier'].value_counts().to_string()}\n")
    return model, df_raw, df_norm