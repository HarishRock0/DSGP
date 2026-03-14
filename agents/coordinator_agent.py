from langchain_core.runnables import Runnable

from signals.nlp_signals import NLPQuerySignal
from signals.insight_signals.poverty_insight_signals import InsightQuerySignal

from signals.child_nlp_signals import ChildNLPSignals
from signals.insight_signals.child_cases_insight_signals import ChildInsightQuerySignal
from signals.mental_health_nlp_signals import MentalHealthNLPSignals

from signals.allocation.poverty_resource_signals import BudgetAllocationRequest   # ← fixed path
from agents.resource_allocation_agent import ResourceAllocationAgent

from agents.nlp_recommendation_agent import NLPRecommendationAgent
from agents.insight_generator_agent import InsightGeneratorAgent


class CoordinatorAgent(Runnable):
    def __init__(self, project_root):
        self.recommender       = NLPRecommendationAgent(project_root)
        self.insight_generator = InsightGeneratorAgent(project_root)
        self.allocation_agent  = ResourceAllocationAgent(project_root)

    def invoke(self, user_input: str):
        nlp_signal = NLPQuerySignal(preference=user_input)
        rec_signal = self.recommender.invoke(nlp_signal)
        return {"recommendations": rec_signal.districts}

    def get_insights_for_district(self, district: str):
        sig = InsightQuerySignal(district=district)
        out = self.insight_generator.invoke(sig)
        return out.insights

    def get_child_cases_insights(self, district: str):
        sig = ChildInsightQuerySignal(district=district)
        out = self.insight_generator.child_case_invoke(sig)
        return out.insights

    def invoke_child_cases(self, user_input: str):
        signal     = ChildNLPSignals(preference=user_input)
        rec_signal = self.recommender.child_case_invoke(signal)
        return {"recommendations": rec_signal.districts}

    def allocate_budget(self, total_budget: float) -> dict:
        response = self.allocation_agent.invoke(total_budget)
        return {
            "total_budget": response.total_budget,
            "floor_per_district": response.floor_per_district,
            "allocations": [a.model_dump() for a in response.allocations],
        }

    def get_risk_summary(self) -> dict:
        df = self.allocation_agent.risk_model.risk_df
        return {
            "critical_districts": df[df["risk_tier"] == "CRITICAL"]["district"].tolist(),
            "high_districts": df[df["risk_tier"] == "HIGH"]["district"].tolist(),
            "moderate_districts": df[df["risk_tier"] == "MODERATE"]["district"].tolist(),
            "low_districts": df[df["risk_tier"] == "LOW"]["district"].tolist(),
        }
