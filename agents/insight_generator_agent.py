from typing import Dict, Any
from langchain_core.runnables import Runnable

from dataloader.insight.poverty_insights import PovertyInsightsDataLoader
from signals.insight_signals.poverty_insight_signals import InsightQuerySignal, InsightSignal


class InsightGeneratorAgent(Runnable):
    def __init__(self, project_root: str):
        data = PovertyInsightsDataLoader(project_root).load()
        self.poverty_df = data["poverty_df"]
        self.demo_df = data["demo_df"]

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

    def _demo_insight(self, district: str) -> Dict[str, Any]:
        # Adjust keys here to match your demographic columns
        if "DISTRICT_N" not in self.demo_df.columns:
            return {"available": False, "row": {}}

        row = self.demo_df[self.demo_df["DISTRICT_N"] == district]
        if row.empty:
            return {"available": False, "row": {}}

        r = row.iloc[0].to_dict()

        # Optional: pick a few common fields if present
        picked = {}
        for key in ["TOT_POP", "MALE", "FEMALE", "POP_DENSITY", "AREA"]:
            if key in r:
                picked[key] = r[key]

        return {"available": True, "row": picked or r}

    def invoke(self, signal: InsightQuerySignal) -> InsightSignal:
        district = signal.district

        insights = {
            "district": district,
            "poverty": self._poverty_insight(district),
            "demographics": self._demo_insight(district),
        }

        return InsightSignal(district=district, insights=insights)
