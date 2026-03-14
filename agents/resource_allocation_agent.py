from __future__ import annotations

import os
from langchain_core.runnables import Runnable

from dataloader.allocation.poverty_allocation import PovertyDataLoader
from models.resource_allocation_model import ResourceAllocationModel, ALLOCATION_RULES
from signals.allocation.poverty_resource_signals import (
    BudgetAllocationResponse,
    DistrictAllocation,
)


class ResourceAllocationAgent(Runnable):

    def __init__(self, project_root: str) -> None:
        self.project_root = project_root

        # Load and preprocess xlsx — no pkl, no pickle
        loader = PovertyDataLoader(project_root)
        df_raw, df_norm = loader.load()

        # Build model with preprocessed dataframes
        self.risk_model = ResourceAllocationModel(df_raw, df_norm)

    def invoke(self, total_budget: float, config=None) -> BudgetAllocationResponse:
        result_df = self.risk_model.allocate(total_budget)
        floor_amount = round(total_budget * ALLOCATION_RULES["equity_floor_pct"], 0)

        allocations = [
            DistrictAllocation(
                district=row["district"],
                risk_tier=row["risk_tier"],
                enhanced_risk_score=row["enhanced_risk_score"],
                risk_pct=row["risk_pct"],
                total_allocation=row["total_allocation"],
                allocation_pct=row["allocation_pct"],
                alloc_per_hh=row["alloc_per_hh"],
            )
            for _, row in result_df.iterrows()
        ]

        return BudgetAllocationResponse(
            total_budget=total_budget,
            floor_per_district=floor_amount,
            allocations=allocations,
        )