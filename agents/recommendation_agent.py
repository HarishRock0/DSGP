# agents/recommendation_agent.py

import pickle
import pandas as pd
import os
import random


class RecommendationAgent:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, '..'))


        demographics_path = os.path.join(project_root, 'data', 'demograpic_district_wise.xlsx')
        poverty_lines_path = os.path.join(project_root, 'data', 'Povertylines.xlsx')
        model_path = os.path.join(project_root, 'model', 'poverty_model.pkl')

        print(f"Looking for data in: {demographics_path}")  # Temporary debug print
        print(f"Looking for model in: {model_path}")  # Temporary debug print

        # Handle missing model gracefully
        if os.path.exists(model_path):
            self.model = pickle.load(open(model_path, 'rb'))
        else:
            print("Warning: Model file not found. Using random ranking.")
            self.model = lambda x: random.random()

        # Load data with better error messages
        if not os.path.exists(demographics_path):
            raise FileNotFoundError(f"Demographics file not found at: {demographics_path}")
        if not os.path.exists(poverty_lines_path):
            raise FileNotFoundError(f"Poverty lines file not found at: {poverty_lines_path}")

        try:
            self.demographics = pd.read_excel(demographics_path)
            self.poverty_lines = pd.read_excel(poverty_lines_path)
            print("Data files loaded successfully!")
        except FileNotFoundError as e:
            print(f"WARNING: {e}")
            print("Continuing with empty/fallback data...")
            # Create empty DataFrames as fallback
            self.demographics = pd.DataFrame({'DISTRICT_N': ['Colombo', 'Gampaha', 'Kandy', 'Jaffna']})
            self.poverty_lines = pd.DataFrame()

    def process_preference(self, preference: str) -> list:
        features = self._extract_features(preference)
        keyword = features[0] if features else ""

        # Case-insensitive search on district name
        mask = self.demographics['DISTRICT_N'].str.contains(keyword, case=False, na=False)
        matched = self.demographics[mask]['DISTRICT_N'].unique().tolist()

        # Fallback: if no match, return random districts
        if not matched:
            matched = self.demographics['DISTRICT_N'].unique().tolist()

        # Shuffle for mock ranking
        random.shuffle(matched)
        return matched[:10]

    def _extract_features(self, preference: str) -> list:
        return [word for word in preference.lower().split() if len(word) > 3]