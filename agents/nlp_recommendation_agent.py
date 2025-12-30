import pickle
import pandas as pd
from agents.base_agent import BaseAgent

class NLPRecommendationAgent(BaseAgent):
    def __init__(self):
        with open("model/poverty_model.pkl", "rb") as f:
            self.model = pickle.load(f)

        self.region_data = pd.read_excel("data/demograpic_district wise.xlsx")

    def run(self, input_data: dict):
        user_text = input_data["preference"]

        # Example vectorization / embedding logic
        scores = self.model.predict_proba([user_text])[0]

        self.region_data["score"] = scores[:len(self.region_data)]

        top_regions = (
            self.region_data
            .sort_values("score", ascending=False)
            .head(10)
            .reset_index(drop=True)
        )

        return top_regions
