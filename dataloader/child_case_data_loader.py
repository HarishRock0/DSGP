class PovertyDataLoader:
    def __init__(self, project_root):
        self.project_root = project_root

    def load(self):
        model_path = os.path.join(self.project_root, "model", "child_case_nlp.pkl")
        child_case_path = os.path.join(self.project_root, "data", "childcases.xlsx")

        with open(model_path, "rb") as f:
            child_case_model = pickle.load(f)


        child_data = pd.read_excel(child_case_path)
        child_data[['District','Avg_cases']] = child_data.iloc[:, 1:].mean(axis=1)

        return child_data , child_case_model