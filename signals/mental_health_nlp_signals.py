from pydantic import BaseModel

class MentalHealthNLPSignals(BaseModel):
    preference: str

class MentalHealthRecommenderSignals(MentalHealthNLPSignals):
    districts : list