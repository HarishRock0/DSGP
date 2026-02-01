from typing import Dict, Any
from langchain_core.runnables import Runnable

from dataloader.insight.poverty_insights import PovertyInsightsDataLoader
from dataloader.insight.child_cases_insights import ChildCaseDataLoader
from signals.insight_signals.poverty_insight_signals import InsightQuerySignal, InsightSignal
from signals.insight_signals.child_cases_insight_signals import ChildInsightQuerySignal , ChildInsightSignal


class InsightGeneratorAgent(Runnable):
    def __init__(self, project_root: str):
        data = PovertyInsightsDataLoader(project_root).load()
        child_data = ChildCaseDataLoader(project_root).load()
        self.poverty_df = data["poverty_df"]
        self.demo_df = data["demo_df"]

        self.child_df = child_data["child_case_df"]

    def _poverty_insight(self, district: str) -> Dict[str, Any]:
        if district not in self.poverty_df.index:
            return {"available": False, "trend": {}, "latest": None}

        s = self.poverty_df.loc[district].dropna()
        if s.empty:
            return {"available": True, "trend": {}, "latest": None}
        return {
            "available": True,
            "trend": s.to_dict(),
            "latest": float(s.iloc[-1]),
            "latest_period": str(s.index[-1]),
        }

    def _child_case_insight(self, district: str) -> Dict[str, Any]:
        if district not in self.child_df.index:
            return {"available": False, "trend": {}, "latest": None}
        s = self.child_df.loc[district].dropna()
        if s.empty:
            return {"available": True, "trend": {}, "latest": None}
        return {
            "available": True,
            "trend": s.to_dict(),
            "latest_period": str(s.index[-1]),
            "latest_period_period": str(s.index[-2]),
        }

    def _demo_insight(self, district: str) -> Dict[str, Any]:
        if "DISTRICT_N" not in self.demo_df.columns:
            return {"available": False, "metrics": {}}

        row = self.demo_df[self.demo_df["DISTRICT_N"] == district]
        if row.empty:
            return {"available": False, "metrics": {}}

        r = row.iloc[0].to_dict()
        metrics = {}
        for key in ["TOT_POP", "MALE", "FEMALE", "POP_DENSITY", "AREA"]:
            if key in r:
                metrics[key] = r[key]
        return {
            "available": True,
            "metrics": metrics
        }

    def invoke(self, signal: InsightQuerySignal) -> InsightSignal:
        district = signal.district
        insights = {
            "district": district,
            "poverty": self._poverty_insight(district),
            "demographics": self._demo_insight(district),
        }
        return InsightSignal(district=district, insights=insights)

    def child_case_invoke(self, signal: ChildInsightQuerySignal) -> ChildInsightSignal:
        district = signal.district
        insights = {
            "district": district,
            "child_cases": self._child_case_insight(district),
            "demographics": self._demo_insight(district),
        }
        return ChildInsightSignal(district=district, insights=insights)



