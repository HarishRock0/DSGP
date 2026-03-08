"""
Poverty resource signals — budget & allocation I/O contracts.

These signals carry budget inputs and allocation outputs between agents.
"""
from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel


class BudgetAllocationRequest(BaseModel):
    """Input: user wants to allocate a specific budget."""
    total_budget: float
    top_n: Optional[int] = None          # limit results to top N districts
    tier_filter: Optional[str] = None    # e.g. "CRITICAL", "HIGH"

    class Config:
        arbitrary_types_allowed = True


class DistrictAllocation(BaseModel):
    """Allocation details for one district."""
    district: str
    risk_tier: str
    enhanced_risk_score: float
    risk_pct: float
    total_allocation: float
    allocation_pct: float
    alloc_per_hh: float
    floor_allocation: float
    prop_allocation: float

    class Config:
        arbitrary_types_allowed = True


class BudgetAllocationResponse(BaseModel):
    """Output: full allocation result for all/filtered districts."""
    total_budget: float
    equity_floor_pct: float
    floor_per_district: float
    allocations: List[DistrictAllocation]
    summary: dict = {}

    class Config:
        arbitrary_types_allowed = True


class RiskSummarySignal(BaseModel):
    """Snapshot of risk tiers across all districts."""
    critical_districts: List[str]
    high_districts: List[str]
    moderate_districts: List[str]
    low_districts: List[str]

    class Config:
        arbitrary_types_allowed = True