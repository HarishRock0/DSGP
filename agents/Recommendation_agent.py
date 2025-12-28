# agents/recommendation_agent.py
import pickle
import pandas as pd  # Assuming you use pandas for data handling


class RecommendationAgent:
    def __init__(self):
        # Load your existing NLP/model/resources
        self.model = pickle.load(open('model/poverty_model.pkl', 'rb'))  # Adjust path if needed
        self.demographics = pd.read_excel('data/demographic_district_wise.xlsx')
        self.poverty_lines = pd.read_excel('data/Povertylines.xlsx')
        # Assume your NLP logic is here or imported from service/utils

    def process_preference(self, preference: str) -> list:
        features = self._extract_features(
            preference)
        matched_regions = self.demographics[self.demographics['some_column'].str.contains(features[0])].head(10)[
            'district'].tolist()
        ranked_regions = sorted(matched_regions, key=lambda r: self.model.predict(
            self._get_region_features(r)))  # Use your model for scoring
        return ranked_regions[:10]

    def _extract_features(self, preference: str) -> list:
        return preference.lower().split()

    def _get_region_features(self, region: str):
        return self.demographics[self.demographics['district'] == region].iloc[0].to_dict()  # Placeholder