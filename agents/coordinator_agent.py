from langchain_core.runnables import Runnable

from signals.nlp_signals import NLPQuerySignal
from signals.insight_signals.poverty_insight_signals import InsightQuerySignal
from signals.child_nlp_signals import ChildNLPSignals
from signals.insight_signals.child_cases_insight_signals import ChildInsightQuerySignal

from signals.allocation.poverty_resource_signals import BudgetAllocationRequest
from signals.allocation.child_resource_signals import BudgetAllocationRequest as ChildBudgetAllocationRequest


class CoordinatorAgent(Runnable):
    def __init__(self, project_root):
        self.project_root          = project_root
        self._recommender          = None
        self._insight_generator    = None
        self._allocation_agent     = None
        self._child_allocation_agent = None

    @property
    def recommender(self):
        if self._recommender is None:
            from agents.nlp_recommendation_agent import NLPRecommendationAgent
            self._recommender = NLPRecommendationAgent(self.project_root)
        return self._recommender

    @property
    def insight_generator(self):
        if self._insight_generator is None:
            from agents.insight_generator_agent import InsightGeneratorAgent
            self._insight_generator = InsightGeneratorAgent(self.project_root)
        return self._insight_generator

    @property
    def allocation_agent(self):
        if self._allocation_agent is None:
            from agents.resource_allocation_agent import ResourceAllocationAgent
            self._allocation_agent = ResourceAllocationAgent(self.project_root)
        return self._allocation_agent

    @property
    def child_allocation_agent(self):
        if self._child_allocation_agent is None:
            from agents.resource_allocation_agent import ChildAllocationAgent
            self._child_allocation_agent = ChildAllocationAgent(self.project_root)
        return self._child_allocation_agent

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
            "total_budget":       response.total_budget,
            "floor_per_district": response.floor_per_district,
            "allocations":        [a.model_dump() for a in response.allocations],
        }

    def get_risk_summary(self) -> dict:
        df = self.allocation_agent.risk_model.risk_df
        return {
            "critical_districts": df[df["risk_tier"] == "CRITICAL"]["district"].tolist(),
            "high_districts":     df[df["risk_tier"] == "HIGH"]["district"].tolist(),
            "moderate_districts": df[df["risk_tier"] == "MODERATE"]["district"].tolist(),
            "low_districts":      df[df["risk_tier"] == "LOW"]["district"].tolist(),
        }

    def allocate_child_budget(self, total_budget: float, query: str = None, selected_districts: list = None) -> dict:
        request  = ChildBudgetAllocationRequest(total_budget=total_budget, query=query, selected_districts=selected_districts)
        response = self.child_allocation_agent.invoke(request)
        return {
            "total_budget": response.total_budget,
            "tier_distribution": response.tier_distribution,
            "total_verified": response.total_verified(),
            "min_floor_pct": response.min_floor_pct,
            "allocations": [a.model_dump() for a in response.allocations],
        }

    def get_child_risk_summary(self) -> dict:
        df = self.child_allocation_agent.model.risk_df
        return {
            "critical_districts": df[df["Risk_Tier"] == "CRITICAL"]["District"].tolist(),
            "high_districts":     df[df["Risk_Tier"] == "HIGH"]["District"].tolist(),
            "moderate_districts": df[df["Risk_Tier"] == "MODERATE"]["District"].tolist(),
            "low_districts":      df[df["Risk_Tier"] == "LOW"]["District"].tolist(),
        }