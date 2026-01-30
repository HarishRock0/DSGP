from agents.coordinator_agent import CoordinatorAgent
import os

class RecommendationService:
    def __init__(self):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.coordinator = CoordinatorAgent(project_root)

    def get_recommendations(self, preference: str):
        return self.coordinator.invoke(preference)

    def get_insights(self, district: str):
        return self.coordinator.get_insights_for_district(district)

