# agents/coordinator_agent.py
from .recommendation_agent import RecommendationAgent

class CoordinatorAgent:
    def __init__(self):
        self.rec_agent = RecommendationAgent()

    def get_recommendations(self, preference: str) -> list:
        return self.rec_agent.process_preference(preference)