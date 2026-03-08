from langchain_core.runnables import Runnable

from signals.nlp_signals import NLPQuerySignal
from signals.insight_signals.poverty_insight_signals import InsightQuerySignal

from signals.child_nlp_signals import ChildNLPSignals
from signals.insight_signals.child_cases_insight_signals import ChildInsightQuerySignal
from signals.mental_health_nlp_signals import MentalHealthNLPSignals

from agents.nlp_recommendation_agent import NLPRecommendationAgent
from agents.insight_generator_agent import InsightGeneratorAgent
from agents.resource_allocation_agent import ResourceAllocationAgent
from signals.allocation.poverty_resource_signals import BudgetAllocationRequest



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

    def get_child_cases_insights(self , district: str):
        sig = ChildInsightQuerySignal(district=district)
        out = self.insight_generator.child_case_invoke(sig)
        return out.insights

    def invoke_child_cases(self, user_input: str):
        signal = ChildNLPSignals(preference=user_input)
        rec_signal = self.recommender.child_case_invoke(signal)

        return {
            "recommendations": rec_signal.districts
        }

    def invoke_mental_health(self, user_input: str):
        signal = MentalHealthNLPSignals(preference=user_input)
        rec_signal = self.recommender.mental_health_invoke(signal)

        return {
            "recommendations": rec_signal.districts
        }

    def allocate_budget(self, total_budget: float, top_n=None, tier_filter=None) -> dict:
        """
        Allocate a budget across districts and return the allocation breakdown.

        Parameters
        ----------
        total_budget : float   Total budget in Rs.
        top_n        : int     Optional - return only top-N highest-allocation districts.
        tier_filter  : str     Optional - 'CRITICAL', 'HIGH', 'MODERATE', 'LOW'.
        """
        request = BudgetAllocationRequest(
            total_budget=total_budget, top_n=top_n, tier_filter=tier_filter)
        response = self.allocation_agent.invoke(request)
        return {
            "total_budget": response.total_budget,
            "equity_floor_pct": response.equity_floor_pct,
            "floor_per_district": response.floor_per_district,
            "allocations": [a.model_dump() for a in response.allocations],
            "summary": response.summary,
        }

    def get_risk_summary(self) -> dict:
        """Return all districts grouped by risk tier."""
        sig = self.allocation_agent.get_risk_summary()
        return sig.model_dump()
