from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Input signal
# ---------------------------------------------------------------------------
class BudgetAllocationRequest(BaseModel):

    total_budget:        float            = Field(..., gt=0, description="Total LKR budget")
    query:               Optional[str]    = Field(None,  description="Natural language budget query")
    selected_districts:  Optional[List[str]] = Field(None, description="Explicit district filter")
    restore_encoder:     bool             = Field(True,  description="Load SentenceTransformer encoder")


# ---------------------------------------------------------------------------
# Per-district allocation row
# ---------------------------------------------------------------------------
class ChildDistrictAllocation(BaseModel):

    district:          str
    risk_tier:         str   = Field(..., description="CRITICAL | HIGH | MODERATE | LOW")
    risk_score:        float = Field(..., ge=0, le=100)
    budget_share_pct:  float = Field(..., ge=0, le=100, description="Percentage of total budget")
    allocated_lkr:     float = Field(..., ge=0, description="Allocated amount in LKR")
    tier_weight:       float = Field(..., description="Tier multiplier used in allocation formula")
    avg_cases:         Optional[float] = Field(None, description="Long-term average child cases")
    per_case_lkr:      Optional[float] = Field(None, description="LKR allocated per average case")


# ---------------------------------------------------------------------------
# Output response signal
# ---------------------------------------------------------------------------
class BudgetAllocationResponse(BaseModel):

    total_budget:      float
    min_floor_pct:     float                       = Field(1.0)
    allocations:       List[ChildDistrictAllocation]
    tier_distribution: dict                        = Field(default_factory=dict)

    def total_verified(self) -> float:
        """Sum of all allocated_lkr values — should equal total_budget."""
        return sum(a.allocated_lkr for a in self.allocations)

class ChildRiskSummaryResponse(BaseModel):

    critical_districts: List[str] = Field(default_factory=list)
    high_districts:     List[str] = Field(default_factory=list)
    moderate_districts: List[str] = Field(default_factory=list)
    low_districts:      List[str] = Field(default_factory=list)

    def all_districts(self) -> List[str]:
        return (
            self.critical_districts + self.high_districts +
            self.moderate_districts + self.low_districts
        )