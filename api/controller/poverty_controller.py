from fastapi import FastAPI
from pydantic import BaseModel
from recommendation_service import RecommendationService

app = FastAPI()
recommendation_service = RecommendationService()


class RecommendationRequest(BaseModel):
    preference: str


class InsightsRequest(BaseModel):
    district: str


@app.post("/recommendations")
def get_recommendations(request: RecommendationRequest):
    return recommendation_service.get_recommendations(request.preference)


@app.post("/insights")
def get_insights(request: InsightsRequest):
    return recommendation_service.get_insights(request.district)

