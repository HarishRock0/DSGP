# agents/coordinator_agent.py
from recommendation_agent import RecommendationAgent

class CoordinatorAgent:
    def __init__(self):
        self.rec_agent = RecommendationAgent()
        self.insight_agent = InsightAgent()

    def get_recommendations(self, preference: str) -> list:
        # Delegate to recommendation agent
        return self.rec_agent.process_preference(preference)
