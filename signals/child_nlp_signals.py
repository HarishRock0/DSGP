from pydantic import BaseModel

class ChildNLPSignals(BaseModel):
    preference : str

class ChildRecommenderSignals(ChildNLPSignals):
    districts : list

