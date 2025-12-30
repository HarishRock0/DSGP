from agents.nlp_recommendation_agent import NLPRecommendationAgent

class OrchestratorAgent:
    def __init__(self):
        self.recommendation_agent = NLPRecommendationAgent()

    def run(self, user_input: str):
        context = {"preference": user_input}

        recommendations = self.recommendation_agent.run(context)

        return {
            "recommendations": recommendations
        }
