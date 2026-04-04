from __future__ import annotations

from langchain_core.runnables import Runnable

from dataloader.allocation.poverty_allocation import PovertyDataLoader
from models.resource_allocation_model import ResourceAllocationModel, ALLOCATION_RULES
from signals.allocation.poverty_resource_signals import (
    BudgetAllocationResponse,
    DistrictAllocation,
)


class ResourceAllocationAgent(Runnable):
    """Poverty budget allocation. Loads Povertylines.xlsx at init."""

    def __init__(self, project_root: str) -> None:
        self.project_root = project_root
        loader = PovertyDataLoader(project_root)
        df_raw, df_norm = loader.load()
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
                floor_allocation=row["floor_allocation"],
                prop_allocation=row["prop_allocation"],
                total_allocation=row["total_allocation"],
                allocation_pct=row["allocation_pct"],
                alloc_per_hh=row["alloc_per_hh"],
            )
            for _, row in result_df.iterrows()
        ]

        return BudgetAllocationResponse(
            total_budget=total_budget,
            equity_floor_pct=ALLOCATION_RULES["equity_floor_pct"],
            floor_per_district=floor_amount,
            allocations=allocations,
        )


class ChildAllocationAgent(Runnable):
    """Child protection budget allocation. Loads child_welfare_pipeline.pkl at init.
    Does NOT touch PovertyDataLoader or Povertylines.xlsx."""

    def __init__(self, project_root: str) -> None:
        from dataloader.allocation.child_protectio_budget_allocation import get_child_protection_pipeline
        self.project_root = project_root
        self.model, self.df_raw, self.df_norm = get_child_protection_pipeline(project_root)

    def invoke(self, request, config=None):
        from signals.allocation.child_resource_signals import (
            BudgetAllocationResponse as ChildBudgetAllocationResponse,
            ChildDistrictAllocation,
        )

        risk_df = self.model.risk_df.copy()
        budget = request.total_budget
        selected = request.selected_districts

        if request.query and not selected:
            lower = request.query.lower()
            selected = [v for k, v in (self.model.district_aliases or {}).items() if k in lower] or None

        if selected:
            risk_df = risk_df[risk_df["District"].isin(selected)].copy()

        # Merge Avg_cases from case_df if not already present in risk_df.
        # case_df column may be "Avg_cases" or "avg_cases" depending on cleaning.
        if "Avg_cases" not in risk_df.columns:
            case = self.model.case_df.copy()
            # Normalise column name to Avg_cases regardless of source casing
            avg_col = next((c for c in case.columns if c.lower() == "avg_cases"), None)
            if avg_col and avg_col != "Avg_cases":
                case = case.rename(columns={avg_col: "Avg_cases"})
            # Also normalise District column
            dist_col = next((c for c in case.columns if c.lower() == "district"), None)
            if dist_col and dist_col != "District":
                case = case.rename(columns={dist_col: "District"})
            if avg_col:
                avg = case[["District", "Avg_cases"]].drop_duplicates("District")
                risk_df = risk_df.merge(avg, on="District", how="left")
            else:
                risk_df["Avg_cases"] = None

        risk_df["Alloc_Weight"] = risk_df["Risk_Score"] * risk_df["Tier_Weight"]
        risk_df["Budget_Share_Pct"] = (risk_df["Alloc_Weight"] / risk_df["Alloc_Weight"].sum() * 100).round(4)
        risk_df["Budget_Share_Pct"] = risk_df["Budget_Share_Pct"].clip(lower=1.0)
        risk_df["Budget_Share_Pct"] = (risk_df["Budget_Share_Pct"] / risk_df["Budget_Share_Pct"].sum() * 100).round(2)
        risk_df["Allocated_LKR"] = (risk_df["Budget_Share_Pct"] / 100 * budget).round(0)
        risk_df["Per_Case_LKR"] = (risk_df["Allocated_LKR"] / risk_df["Avg_cases"].replace(0, 1)).round(0)

        allocations = [
            ChildDistrictAllocation(
                district=row["District"],
                risk_tier=row["Risk_Tier"],
                risk_score=row["Risk_Score"],
                budget_share_pct=row["Budget_Share_Pct"],
                allocated_lkr=row["Allocated_LKR"],
                tier_weight=row["Tier_Weight"],
                avg_cases=row.get("Avg_cases"),
                per_case_lkr=row.get("Per_Case_LKR"),
            )
            for _, row in risk_df.iterrows()
        ]

        tier_counts = {}
        for a in allocations:
            tier_counts[a.risk_tier] = tier_counts.get(a.risk_tier, 0) + 1

        return ChildBudgetAllocationResponse(
            total_budget=budget,
            min_floor_pct=1.0,
            allocations=allocations,
            tier_distribution=tier_counts,
        )