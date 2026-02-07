# import os, pandas as pd, pickle
#
# class ChildCaseDataLoader:
#     def __init__(self, project_root: str):
#         self.project_root = project_root
#
#         self.child_case_data = os.path.join(project_root, "data", "childcases.xlsx")
#         self.demographic_path = os.path.join(project_root, "data", "demographic_district_wise.xlsx")
#
#     def load(self):
#         # ---------- CHILD CASE DATA ----------
#         child_case_df = pd.read_excel(self.child_case_data)
#         child_case_df.columns = child_case_df.columns.str.strip()
#
#         # Set District as index
#         if "District" in child_case_df.columns:
#             child_case_df["District"] = (
#                 child_case_df["District"].astype(str).str.strip()
#             )
#             child_case_df = child_case_df.set_index("District")
#
#         # Clean year columns (2010–2024)
#         cleaned_cols = {}
#         for col in child_case_df.columns:
#             try:
#                 year = int(str(col).strip())
#                 cleaned_cols[col] = year
#             except ValueError:
#                 cleaned_cols[col] = col
#
#         child_case_df = child_case_df.rename(columns=cleaned_cols)
#
#         # Convert case values to numeric (remove commas)
#         child_case_df = child_case_df.apply(
#             lambda c: (
#                 c.astype(str)
#                 .str.replace(",", "", regex=False)
#                 .astype(float)
#             )
#             if pd.api.types.is_object_dtype(c)
#             else c
#         )
#
#         # Ensure years are sorted
#         # Keep ONLY year columns (2010–2024)
#         year_cols = sorted(
#             [c for c in child_case_df.columns if isinstance(c, int)]
#         )
#         child_case_df = child_case_df[year_cols]
#
#         # ---------- DEMOGRAPHIC DATA ----------
#         demo_df = pd.read_excel(self.demographic_path)
#         demo_df.columns = demo_df.columns.str.strip()
#
#         if "DISTRICT_N" in demo_df.columns:
#             demo_df["DISTRICT_N"] = (
#                 demo_df["DISTRICT_N"].astype(str).str.strip()
#             )
#
#         return {
#             "child_case_df": child_case_df,
#             "demo_df": demo_df
#         }
#

import os
import pandas as pd


class ChildCaseDataLoader:
    def __init__(self, project_root: str):
        self.project_root = project_root
        # Paths to the data files
        self.child_case_data = os.path.join(project_root, "data", "childcases.xlsx")
        self.demographic_path = os.path.join(project_root, "data", "demographic_district_wise.xlsx")

    def load(self):
        # ---------- CHILD CASE DATA ----------
        # Load the Excel file
        child_case_df = pd.read_excel(self.child_case_data)

        # Clean column names: remove whitespace and convert to string first
        child_case_df.columns = child_case_df.columns.astype(str).str.strip()

        # Set 'District' as the index if it exists in the columns
        if "District" in child_case_df.columns:
            child_case_df["District"] = child_case_df["District"].astype(str).str.strip()
            child_case_df = child_case_df.set_index("District")

        # Clean and Identify Year Columns (2010–2024)
        cleaned_cols = {}
        for col in child_case_df.columns:
            try:
                # Attempt to parse column name as a float first to handle "2010.0" formats,
                # then convert to integer.
                val = float(col)
                if val.is_integer():
                    cleaned_cols[col] = int(val)
                else:
                    cleaned_cols[col] = col
            except ValueError:
                # If parsing fails (e.g., "Avg_cases"), keep the original name
                cleaned_cols[col] = col

        # Rename columns to their integer year representation
        child_case_df = child_case_df.rename(columns=cleaned_cols)

        # Convert case values to numeric (remove commas if present)
        child_case_df = child_case_df.apply(
            lambda c: (
                c.astype(str)
                .str.replace(",", "", regex=False)
                .astype(float)
            )
            if pd.api.types.is_object_dtype(c)
            else c
        )

        # Filter: Keep ONLY columns that are Integers (Years).
        # This step explicitly removes "Avg_cases" and other non-year columns.
        year_cols = sorted(
            [c for c in child_case_df.columns if isinstance(c, int)]
        )

        # Reconstruct DataFrame with only the sorted year columns
        child_case_df = child_case_df[year_cols]

        # ---------- DEMOGRAPHIC DATA ----------
        demo_df = pd.read_excel(self.demographic_path)
        demo_df.columns = demo_df.columns.astype(str).str.strip()

        if "DISTRICT_N" in demo_df.columns:
            demo_df["DISTRICT_N"] = demo_df["DISTRICT_N"].astype(str).str.strip()

        return {
            "child_case_df": child_case_df,
            "demo_df": demo_df
        }