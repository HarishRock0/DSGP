import os
import re
import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer
from models.child_case_model import ChildCaseRuleEngine  # noqa: F401 — required for pickle.load()


class ChildCasesDataLoader:
    def __init__(self, project_root):
        self.project_root = project_root

    def load(self):

        model_path      = os.path.join(self.project_root, "model", "child_case_nlp.pkl")
        child_case_path = os.path.join(self.project_root, "data", "childcases.xlsx")

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        model.st_model = SentenceTransformer("all-MiniLM-L6-v2")

        child_data = pd.read_excel(child_case_path, skiprows=2)

        if len(child_data.columns) > 1:
            child_data = child_data.rename(columns={child_data.columns[1]: 'District'})
        else:
            raise ValueError("Expected a 'District' column at index 1 after skipping rows.")

        child_data['_sno'] = pd.to_numeric(child_data.iloc[:, 0], errors='coerce')
        child_data = child_data.dropna(subset=['_sno']).copy()

        child_data['District'] = child_data['District'].astype(str).str.strip().str.upper()
        child_data['District'] = child_data['District'].apply(
            lambda x: re.sub(r'[^A-Z\s]', '', x).strip()
        )

        year_cols = [c for c in child_data.columns if str(c).replace('*', '').strip().isdigit()]
        for col in year_cols:
            child_data[col] = pd.to_numeric(child_data[col], errors='coerce')

        child_data['average_child_cases'] = child_data[year_cols].mean(axis=1).round(2)

        p33 = child_data['average_child_cases'].quantile(0.33)
        p66 = child_data['average_child_cases'].quantile(0.66)

        def assign_risk(val):
            if val >= p66:   return 'High'
            elif val >= p33: return 'Moderate'
            else:            return 'Low'

        child_data['risk_tier'] = child_data['average_child_cases'].apply(assign_risk)
        child_data = child_data[
            ['District', 'average_child_cases', 'risk_tier'] + year_cols
        ].reset_index(drop=True)

        child_data['text'] = (
            "District " + child_data['District'].astype(str) +
            " has an average of " + child_data['average_child_cases'].astype(str) +
            " child cases and is classified as " + child_data['risk_tier'].astype(str) + " risk"
        )

        return model, child_data