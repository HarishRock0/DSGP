import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from service.child_protection_service import ChildProtectionService
from service.recommendation_service import RecommendationService

app = FastAPI()

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

child_protection_service = ChildProtectionService(project_root)
recommendation_service = RecommendationService()


# ── Request Models ─────────────────────────────────────────────────────────────

class CaseRequest(BaseModel):
    user_input: str

class ChildInsightsRequest(BaseModel):
    district: str

class RecommendationRequest(BaseModel):
    preference: str

class InsightsRequest(BaseModel):
    district: str

class ChildBudgetRequest(BaseModel):
    total_budget: float
    query: Optional[str] = None
    selected_districts: Optional[list[str]] = None

class BudgetRequest(BaseModel):
    total_budget: float


# ── Child Protection Endpoints ─────────────────────────────────────────────────

@app.post("/child-cases", tags=["Child Protection"])
def get_child_case_recommendations(request: CaseRequest):
    return child_protection_service.get_child_case_recommendations(request.user_input)

@app.post("/child-insights", tags=["Child Protection"])
def get_child_insights(request: ChildInsightsRequest):
    return child_protection_service.get_child_insights(request.district)

@app.get("/child-risk-summary", tags=["Child Protection"])
def get_child_risk_summary():
    return child_protection_service.get_risk_summary()

@app.post("/child-budget", tags=["Child Protection"])
def allocate_child_budget(request: ChildBudgetRequest):
    return child_protection_service.allocate_budget(
        total_budget=request.total_budget,
        query=request.query,
        selected_districts=request.selected_districts,
    )


# ── Poverty Endpoints ──────────────────────────────────────────────────────────

@app.post("/recommendations", tags=["Poverty"])
def get_recommendations(request: RecommendationRequest):
    return recommendation_service.get_recommendations(request.preference)

@app.post("/insights", tags=["Poverty"])
def get_insights(request: InsightsRequest):
    return recommendation_service.get_insights(request.district)

@app.get("/risk-summary", tags=["Poverty"])
def get_risk_summary():
    return recommendation_service.get_risk_summary()

@app.post("/budget", tags=["Poverty"])
def allocate_budget(request: BudgetRequest):
    return recommendation_service.allocate_budget(request.total_budget)