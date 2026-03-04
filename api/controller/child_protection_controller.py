import os
from fastapi import FastAPI
from pydantic import BaseModel
from service.child_protection_service import ChildProtectionService

app = FastAPI()

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
child_protection_service = ChildProtectionService(project_root)


class CaseRequest(BaseModel):
    user_input: str


class ChildInsightsRequest(BaseModel):
    district: str


@app.post("/child-cases")
def get_child_case_recommendations(request: CaseRequest):
    return child_protection_service.get_child_case_recommendations(request.user_input)


@app.post("/child-insights")
def get_child_insights(request: ChildInsightsRequest):
    return child_protection_service.get_child_insights(request.district)