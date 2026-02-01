import os
import pickle
import pandas as pd


class PovertyDataLoader:
    def __init__(self, project_root):
        self.project_root = project_root

    def load(self):
        model_path = os.path.join(self.project_root, "model", "poverty_model.pkl")
        demographic_path = os.path.join(self.project_root, "data", "demographic_district_wise.xlsx")
        poverty_line_path = os.path.join(self.project_root, "data", "Povertylines.xlsx")

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        region_data = pd.read_excel(demographic_path)
        poverty_data = pd.read_excel(poverty_line_path)
        poverty_data['average_poverty_line'] = poverty_data.iloc[:, 1:].mean(axis=1)

        district_pop = region_data.groupby('DISTRICT_N')['PPROJ_22'].sum().reset_index()
        district_pop.rename(columns={'DISTRICT_N':'District','PPROJ_22':'Population'}, inplace=True)

        merged_df = pd.merge(
            district_pop,
            poverty_data[['District','average_poverty_line']],
            on='District',
            how='inner'
        )

        merged_df['text'] = (
            merged_df['District'] +
            " Population: " + merged_df['Population'].astype(str) +
            " Poverty: " + merged_df['average_poverty_line'].astype(str)
        )

        return model, merged_df
