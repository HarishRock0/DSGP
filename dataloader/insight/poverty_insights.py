import os
import pandas as pd


class PovertyInsightsDataLoader:
    def __init__(self, project_root: str):
        self.project_root = project_root

        self.poverty_line_path = os.path.join(project_root, "data", "Povertylines.xlsx")
        self.demographic_path = os.path.join(project_root, "data", "demographic_district_wise.xlsx")

    def load(self):
        poverty_df = pd.read_excel(self.poverty_line_path)
        poverty_df.columns = poverty_df.columns.str.strip()
        if "District" in poverty_df.columns:
            poverty_df = poverty_df.set_index("District")

        demo_df = pd.read_excel(self.demographic_path)
        demo_df.columns = demo_df.columns.str.strip()

        # If you have DISTRICT_N, keep it normalized
        if "DISTRICT_N" in demo_df.columns:
            demo_df["DISTRICT_N"] = demo_df["DISTRICT_N"].astype(str).str.strip()

        return {"poverty_df": poverty_df, "demo_df": demo_df}
