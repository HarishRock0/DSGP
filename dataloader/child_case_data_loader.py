import os
import pickle
import pandas as pd


class ChildCasesDataLoader:
    def __init__(self, project_root):
        self.project_root = project_root

    def load(self):
        model_path = os.path.join(self.project_root, "model", "child_case_nlp.pkl")
        child_case_path = os.path.join(self.project_root, "data", "childcases.xlsx")

        with open(model_path, "rb") as f:
            child_case_model = pickle.load(f)


        child_data = pd.read_excel(child_case_path)
        child_data['Avg_cases'] = child_data.iloc[:, 1:].mean(axis=1)
        child_data['text'] = (
                "District " + child_data['District'].astype(str) +
                " has an average of " + child_data['Avg_cases'].astype(str) +
                " child cases"
        )

        return child_case_model, child_data
