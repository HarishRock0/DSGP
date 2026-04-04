import torch
from langchain_core.runnables import Runnable
from dataloader.poverty_data_loader import PovertyDataLoader
from dataloader.child_case_data_loader import ChildCasesDataLoader

from signals.nlp_signals import NLPQuerySignal, RecommendationSignal
from signals.child_nlp_signals import ChildNLPSignals, ChildRecommenderSignals
from signals.mental_health_nlp_signals import MentalHealthNLPSignals, MentalHealthRecommenderSignals


class NLPRecommendationAgent(Runnable):
    def __init__(self, project_root):
        # ── Poverty ───────────────────────────────────────────────────────
        self.model, self.df = PovertyDataLoader(project_root).load()
        self.embeddings = self.model.encoder.encode(
            self.df['text'].tolist(),
            convert_to_tensor=True
        )

        # ── Child Cases ───────────────────────────────────────────────────
        # ChildCaseRuleEngine has pre-computed embeddings inside it
        # recommend() handles precheck + semantic match internally — no re-encoding needed
        self.child_case_model, self.child_case_df = ChildCasesDataLoader(project_root).load()


    """
    This function gets the preferences from the poverty nlp and returns the top 10
    regions to the service through the Coordinator agent.
    """
    def invoke(self, signal: NLPQuerySignal) -> RecommendationSignal:
        with torch.no_grad():
            q = self.model.encoder.encode(
                signal.preference,
                convert_to_tensor=True,
                normalize_embeddings=True
            )

        scores = torch.matmul(q, self.embeddings.T).cpu().numpy()
        top_idx = scores.argsort()[-10:][::-1]

        districts = self.model.district_data.iloc[top_idx][
            ['District', 'risk_index', 'risk_tier']
        ].to_dict(orient="records")

        return RecommendationSignal(districts=districts)

    """
    This function gets the preferences from the Child cases nlp and returns the top 10
    regions to the service through the Coordinator agent.
    Uses ChildCaseRuleEngine.recommend() which handles everything internally.
    """
    def child_case_invoke(self, signal: ChildNLPSignals) -> ChildRecommenderSignals:
        result_df = self.child_case_model.recommend(signal.preference, top_n=10)

        if result_df.empty:
            return ChildRecommenderSignals(preference=signal.preference, districts=[])

        col = 'average_child_cases' if 'average_child_cases' in result_df.columns else 'Avg_cases'
        districts = result_df[['District', col]].to_dict(orient="records")

        return ChildRecommenderSignals(preference=signal.preference, districts=districts)
