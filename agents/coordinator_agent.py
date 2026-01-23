# from langchain_core.runnables import Runnable
# from signals.nlp_signals import NLPQuerySignal
# from agents.nlp_recommendation_agent import NLPRecommendationAgent
#
# class CoordinatorAgent(Runnable):
#     def __init__(self, project_root):
#         self.recommender = NLPRecommendationAgent(project_root)
#
#     def invoke(self, user_input: str):
#         signal = NLPQuerySignal(preference=user_input)
#         rec_signal = self.recommender.invoke(signal)
#
#         return {
#             "recommendations": rec_signal.districts
#         }

from langchain_core.runnables import Runnable

from signals.nlp_signals import NLPQuerySignal
from signals.insight_signals.poverty_insight_signals import InsightQuerySignal

from agents.nlp_recommendation_agent import NLPRecommendationAgent
from agents.insight_generator_agent import InsightGeneratorAgent


class CoordinatorAgent(Runnable):
    def __init__(self, project_root):
        self.recommender = NLPRecommendationAgent(project_root)
        self.insight_generator = InsightGeneratorAgent(project_root)

    def invoke(self, user_input: str):
        # Only return recommendations here
        nlp_signal = NLPQuerySignal(preference=user_input)
        rec_signal = self.recommender.invoke(nlp_signal)

        return {"recommendations": rec_signal.districts}

    def get_insights_for_district(self, district: str):
        # Called only when user selects a district
        sig = InsightQuerySignal(district=district)
        out = self.insight_generator.invoke(sig)
        return out.insights
