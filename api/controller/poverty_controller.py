# from fastapi import APIRouter
# from pydantic import BaseModel
# from agents.coordinator_agent import CoordinatorAgent
#
# router = APIRouter()
# coordinator = CoordinatorAgent()
#
# class PreferenceInput(BaseModel):
#     preference: str
#
# @router.post("/recommend")
# def recommend_regions(input: PreferenceInput):
#     regions = coordinator.get_recommendations(input.preference)
#     return {"top_regions": regions}