import torch
from langchain_core.runnables import Runnable
from dataloader.poverty_data_loader import PovertyDataLoader
from dataloader.child_case_data_loader import ChildCasesDataLoader
from sentence_transformers.util import normalize_embeddings
from signals.nlp_signals import NLPQuerySignal, RecommendationSignal
from signals.child_nlp_signals import ChildNLPSignals,ChildRecommenderSignals

class NLPRecommendationAgent(Runnable):
    def __init__(self, project_root):
        self.model, self.df = PovertyDataLoader(project_root).load()
        self.embeddings = self.model.encode(self.df['text'].tolist(), convert_to_tensor=True)
        
        self.child_case_model, self.child_case_df = ChildCasesDataLoader(project_root).load()
        self.child_case_embeddings = self.child_case_model.encode(
            self.child_case_df['text'].tolist(),
            convert_to_tensor=True,
            normalize_embeddings=True
        )
    """
    This function is get the preferences from the poverty nlp and then return the top 10  regions
    to the service through the Coordinate agent 
    """
    def invoke(self, signal: NLPQuerySignal) -> RecommendationSignal:
        with torch.no_grad():
            q = self.model.encode(signal.preference, convert_to_tensor=True, normalize_embeddings=True)

        scores = torch.matmul(q, self.embeddings.T).cpu().numpy()
        top_idx = scores.argsort()[-10:][::-1]

        districts = self.df.iloc[top_idx][['District','average_poverty_line']].to_dict(orient="records")

        return RecommendationSignal(districts=districts)


    """
    This function is get the preferences from the Child cases nlp and then return the top 10  regions
    to the service through the Coordinate agent
    """
    def child_case_invoke(self, signal: ChildNLPSignals) -> ChildRecommenderSignals:
        with torch.no_grad():
            q = self.child_case_model.encode(
                signal.preference,
                convert_to_tensor=True,
                normalize_embeddings=True
            )

        scores = torch.matmul(q, self.child_case_embeddings.T)
        scores = scores.cpu().numpy().flatten()

        if scores.size == 0:
            return ChildRecommenderSignals(
                preference=signal.preference,
                districts=[]
            )

        top_k = min(10, len(scores))
        top_idx = scores.argsort()[-top_k:][::-1]

        districts = self.child_case_df.iloc[top_idx][
            ['District', 'Avg_cases']
        ].to_dict(orient="records")

        return ChildRecommenderSignals(
            preference=signal.preference,
            districts=districts
        )
