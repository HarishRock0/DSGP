import os, pandas as pd, pickle

class ChildCaseDataLoader:
    def __init__(self, project_root: str):
        self.project_root = project_root

        self.child_case_data = os.path.join(project_root, "data", "childcases.xlsx")

    def load(self):
        child_case_df = pd.read_excel(self.child_case_data)
        child_case_df.columns = child_case_df.columns.str.strip()
        if "District" in child_case_df.columns:
            child_case_df = child_case_df.set_index("District")

        return child_case_df
