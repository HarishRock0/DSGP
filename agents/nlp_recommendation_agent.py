import sentence_transformers
from sentence_transformers import util
import pickle
import pandas as pd
import os
import torch
from agents.base_agent import BaseAgent


class NLPRecommendationAgent(BaseAgent):
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)

        model_path = os.path.join(project_root, "model", "poverty_model.pkl")
        demographic_path = os.path.join(project_root, "data", "demographic_district_wise.xlsx")
        poverty_line_path = os.path.join(project_root, "data", "Povertylines.xlsx")

        try:
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)

            self.region_data = pd.read_excel(demographic_path)
            self.poverty_data = pd.read_excel(poverty_line_path)

            self.data_descriptions = self.region_data.astype(str).apply(lambda x: ' '.join(x), axis=1).tolist()
            self.corpus_embeddings = self.model.encode(self.data_descriptions, convert_to_tensor=True)

        except FileNotFoundError as e:
            print(
                f"Error: Missing files. Checked paths:\nModel: {model_path}\nDemographic: {demographic_path}\nPoverty: {poverty_line_path}")
            raise e

    def run(self, input_data: dict):
        user_text = input_data.get("preference", "")

        query_embedding = self.model.encode(user_text, convert_to_tensor=True)

        cos_scores = util.cos_sim(query_embedding, self.corpus_embeddings)[0]

        self.region_data["score"] = cos_scores.cpu().numpy()

        top_regions = (
            self.region_data
            .sort_values("score", ascending=False)
            .head(10)
            .reset_index(drop=True)
        )

        return top_regions