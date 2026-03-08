"""
ResourceAllocationAgent
=======================
LangChain Runnable that accepts a BudgetAllocationRequest signal and returns
a full BudgetAllocationResponse using the PovertyRiskModel allocation engine.
"""
from __future__ import annotations

import os
from langchain_core.runnables import Runnable

from signals.poverty_resource_signals import (
    BudgetAllocationRequest,
    BudgetAllocationResponse,
    DistrictAllocation,
    RiskSummarySignal,
)
from models.poverty_risk_model import PovertyRiskModel, ALLOCATION_RULES


class ResourceAllocationAgent(Runnable):
    """
    Allocates a given budget across Sri Lanka districts based on poverty risk scores.

    Parameters
    ----------
    project_root : str
        Absolute path to the project root (used to locate the data file).
    """

    DATA_RELATIVE_PATH = os.path.join("data", "resource", "Povertylines.xlsx")

    def __init__(self, project_root: str) -> None:
        self.project_root = project_root
        xlsx_path         = os.path.join(project_root, self.DATA_RELATIVE_PATH)
        self.risk_model   = PovertyRiskModel(xlsx_path)

    # ── main invoke ───────────────────────────────────────────────────────────

    def invoke(self, signal: BudgetAllocationRequest, config=None) -> BudgetAllocationResponse:
        """Run the budget allocation engine and return structured response."""
        result_df = self.risk_model.allocate(signal.total_budget)

        # Optional filters
        if signal.tier_filter:
            result_df = result_df[
                result_df["risk_tier"] == signal.tier_filter.upper()
            ]
        if signal.top_n:
            result_df = result_df.head(signal.top_n)

        allocations = [
            DistrictAllocation(
                district=row["district"],
                risk_tier=row["risk_tier"],
                enhanced_risk_score=row["enhanced_risk_score"],
                risk_pct=row["risk_pct"],
                total_allocation=row["total_allocation"],
                allocation_pct=row["allocation_pct"],
                alloc_per_hh=row["alloc_per_hh"],
                floor_allocation=row["floor_allocation"],
                prop_allocation=row["prop_allocation"],
            )
            for _, row in result_df.iterrows()
        ]

        floor_per_district = round(
            signal.total_budget * ALLOCATION_RULES["equity_floor_pct"], 0
        )

        # Tier-based summary
        summary = {}
        for tier in ["CRITICAL", "HIGH", "MODERATE", "LOW"]:
            tier_rows = result_df[result_df["risk_tier"] == tier]
            if not tier_rows.empty:
                summary[tier] = {
                    "count": len(tier_rows),
                    "total_allocation": float(tier_rows["total_allocation"].sum()),
                    "allocation_pct": float(tier_rows["allocation_pct"].sum()),
                }

        return BudgetAllocationResponse(
            total_budget=signal.total_budget,
            equity_floor_pct=ALLOCATION_RULES["equity_floor_pct"],
            floor_per_district=floor_per_district,
            allocations=allocations,
            summary=summary,
        )

    # ── convenience

    def get_risk_summary(self) -> RiskSummarySignal:
        """Return a summary of all districts grouped by risk tier."""
        df = self.risk_model.risk_df
        return RiskSummarySignal(
            critical_districts=df[df["risk_tier"] == "CRITICAL"]["district"].tolist(),
            high_districts=df[df["risk_tier"] == "HIGH"]["district"].tolist(),
            moderate_districts=df[df["risk_tier"] == "MODERATE"]["district"].tolist(),
            low_districts=df[df["risk_tier"] == "LOW"]["district"].tolist(),
        )