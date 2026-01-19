from langchain_core.runnables import Runnable
from signals.nlp_signals import NLPQuerySignal
from agents.nlp_recommendation_agent import NLPRecommendationAgent

class CoordinatorAgent(Runnable):
    def __init__(self, project_root):
        self.recommender = NLPRecommendationAgent(project_root)

    def invoke(self, user_input: str):
        signal = NLPQuerySignal(preference=user_input)
        rec_signal = self.recommender.invoke(signal)

        return {
            "recommendations": rec_signal.districts
        }
