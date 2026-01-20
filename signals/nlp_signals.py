from pydantic import BaseModel

class NLPQuerySignal(BaseModel):
    preference: str

class RecommendationSignal(BaseModel):
    districts: list
