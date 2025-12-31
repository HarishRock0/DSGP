from agents.orchestrator import OrchestratorAgent

class RecommendationService:
    def __init__(self):
        self.orchestrator = OrchestratorAgent()

    def get_recommendations(self, preference: str):
        return self.orchestrator.run(preference)