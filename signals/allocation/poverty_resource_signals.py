from typing import List
from pydantic import BaseModel


class BudgetAllocationRequest(BaseModel):
    total_budget: float


class DistrictAllocation(BaseModel):

    district: str
    risk_tier: str
    enhanced_risk_score: float
    allocation: float
    allocation_pct: float


class BudgetAllocationResponse(BaseModel):
    total_budget: float
    equity_floor_pct: float  # add this
    floor_per_district: float
    allocations: List[DistrictAllocation]