import os
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer

class MentalHealthDataLoader:
    def __init__(self, project_root):
        self.project_root = project_root
        self.data_path = os.path.join(project_root,"data","MentalHealth.xlsx")
        self.model_path = os.path.join(project_root,"model","mental_health_model.pkl")

    def load(self):
        # Load model (or create if missing)
        if os.path.exists(self.model_path):
            with open(self.model_path, "rb") as f:
                model = pickle.load(f)
        else:
            model = SentenceTransformer("all-MiniLM-L6-v2")
            with open(self.model_path, "wb") as f:
                pickle.dump(model, f)

        # Load and preprocess data
        df = pd.read_excel(self.data_path, skiprows=1)
        df.columns = ["District", "Disorder", "Gender", "Count"]

        df = df.dropna(subset=["District", "Disorder"])
        df["District"] = df["District"].str.strip().str.upper()
        df["Disorder"] = df["Disorder"].str.strip()
        df["Count"] = pd.to_numeric(df["Count"], errors="coerce").fillna(0)

        # Aggregate disorder counts
        stats = (
            df.groupby(["District", "Disorder"])["Count"]
            .sum()
            .reset_index()
        )

        # Build NLP-friendly district descriptions
        profiles = []
        for district in stats["District"].unique():
            subset = stats[stats["District"] == district] \
                        .sort_values("Count", ascending=False)

            top_disorders = subset["Disorder"].head(10).tolist()
            text = (
                f"In {district}, the most commonly treated mental health "
                f"conditions include {', '.join(top_disorders)}."
            )

            profiles.append({
                "District": district,
                "text": text
            })

        profile_df = pd.DataFrame(profiles)

        return model, profile_df
