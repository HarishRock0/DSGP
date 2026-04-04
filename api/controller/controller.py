import os
from fastapi import FastAPI
from pydantic import BaseModel
from service.child_protection_service import ChildProtectionService
from service.recommendation_service import RecommendationService

app = FastAPI()

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

child_protection_service = ChildProtectionService(project_root)
recommendation_service = RecommendationService()


class CaseRequest(BaseModel):
    user_input: str

class ChildInsightsRequest(BaseModel):
    district: str

class RecommendationRequest(BaseModel):
    preference: str

class InsightsRequest(BaseModel):
    district: str


@app.post("/child-cases", tags=["Child Protection"])
def get_child_case_recommendations(request: CaseRequest):
    return child_protection_service.get_child_case_recommendations(request.user_input)

@app.post("/child-insights", tags=["Child Protection"])
def get_child_insights(request: ChildInsightsRequest):
    return child_protection_service.get_child_insights(request.district)

@app.post("/recommendations", tags=["Poverty"])
def get_recommendations(request: RecommendationRequest):
    return recommendation_service.get_recommendations(request.preference)

@app.post("/insights", tags=["Poverty"])
def get_insights(request: InsightsRequest):
    return recommendation_service.get_insights(request.district)