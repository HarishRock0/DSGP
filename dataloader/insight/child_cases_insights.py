import os, pandas as pd, pickle

class ChildCaseDataLoader:
    def __init__(self, project_root: str):
        self.project_root = project_root

        self.child_case_data = os.path.join(project_root, "data", "childcases.xlsx")
        self.demographic_path = os.path.join(project_root, "data", "demographic_district_wise.xlsx")

    def load(self):
        child_case_df = pd.read_excel(self.child_case_data)
        child_case_df.columns = child_case_df.columns.str.strip()

        if "District" in child_case_df.columns:
            child_case_df = child_case_df.set_index("District")

        demo_df = pd.read_excel(self.demographic_path)
        demo_df.columns = demo_df.columns.str.strip()

        if "DISTRICT_N" in demo_df.columns:
            demo_df["DISTRICT_N"] = demo_df["DISTRICT_N"].astype(str).str.strip()

        return {
            "child_case_df": child_case_df,
            "demo_df": demo_df
        }


