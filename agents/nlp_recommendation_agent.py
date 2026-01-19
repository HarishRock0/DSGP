import torch
from langchain_core.runnables import Runnable
from dataloader.poverty_data_loader import PovertyDataLoader
from signals.nlp_signals import NLPQuerySignal, RecommendationSignal

class NLPRecommendationAgent(Runnable):
    def __init__(self, project_root):
        self.model, self.df = PovertyDataLoader(project_root).load()
        self.embeddings = self.model.encode(self.df['text'].tolist(), convert_to_tensor=True)

    def invoke(self, signal: NLPQuerySignal) -> RecommendationSignal:
        with torch.no_grad():
            q = self.model.encode(signal.preference, convert_to_tensor=True, normalize_embeddings=True)

        scores = torch.matmul(q, self.embeddings.T).cpu().numpy()
        top_idx = scores.argsort()[-10:][::-1]

        districts = self.df.iloc[top_idx][['District','average_poverty_line']].to_dict(orient="records")

        return RecommendationSignal(districts=districts)

