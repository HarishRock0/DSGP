from pydinamic import BaseModel

class ChildNLPSignals(BaseModel):
    preference : str

class ChildRecomenderSignals(ChildNLPSignals):
    districts : list

