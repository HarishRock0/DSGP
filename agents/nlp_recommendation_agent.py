# import sentence_transformers
# from sentence_transformers import util
# import pickle
# import pandas as pd
# import os
# import torch
# from agents.base_agent import BaseAgent
#
#
# class NLPRecommendationAgent(BaseAgent):
#     def __init__(self):
#         current_dir = os.path.dirname(os.path.abspath(__file__))
#         project_root = os.path.dirname(current_dir)
#
#         model_path = os.path.join(project_root, "model", "poverty_model.pkl")
#         demographic_path = os.path.join(project_root, "data", "demographic_district_wise.xlsx")
#         poverty_line_path = os.path.join(project_root, "data", "Povertylines.xlsx")
#
#         try:
#             with open(model_path, "rb") as f:
#                 self.model = pickle.load(f)
#
#             self.region_data = pd.read_excel(demographic_path)
#             self.poverty_data = pd.read_excel(poverty_line_path)
#
#             self.data_descriptions = self.region_data.astype(str).apply(lambda x: ' '.join(x), axis=1).tolist()
#             self.corpus_embeddings = self.model.encode(self.data_descriptions, convert_to_tensor=True)
#
#         except FileNotFoundError as e:
#             print(
#                 f"Error: Missing files. Checked paths:\nModel: {model_path}\nDemographic: {demographic_path}\nPoverty: {poverty_line_path}")
#             raise e
#
#     def run(self, input_data: dict):
#         user_text = input_data.get("preference", "")
#
#         with torch.no_grad():
#             query_embedding = self.model.encode(
#                 user_text,
#                 convert_to_tensor=True,
#                 normalize_embeddings=True
#             )
#
#         # Fast dot product (cosine similarity)
#         cos_scores = torch.matmul(query_embedding, self.corpus_embeddings.T)
#
#         scores = cos_scores.cpu().numpy()
#
#         top_idx = scores.argsort()[-10:][::-1]
#
#         top_regions = (
#             self.region_data.iloc[top_idx]
#             .assign(score=scores[top_idx])
#             .reset_index(drop=True)
#         )
#
#         return top_regions

# nlp_controller.py
import os
import pandas as pd
import torch
import pickle
from agents.base_agent import BaseAgent

class NLPRecommendationAgent(BaseAgent):
    """
    Uses your pretrained model to return district-level recommendations
    """

    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)

        # Paths
        model_path = os.path.join(project_root, "model", "poverty_model.pkl")
        demographic_path = os.path.join(project_root, "data", "demographic_district_wise.xlsx")
        poverty_line_path = os.path.join(project_root, "data", "Povertylines.xlsx")

        # Load model
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        # Load data
        self.region_data = pd.read_excel(demographic_path)
        self.poverty_data = pd.read_excel(poverty_line_path)
        self.poverty_data['average_poverty_line'] = self.poverty_data.iloc[:, 1:].mean(axis=1)

        # Aggregate population by district
        district_pop = self.region_data.groupby('DISTRICT_N')['PPROJ_22'].sum().reset_index()
        district_pop.rename(columns={'DISTRICT_N': 'District', 'PPROJ_22': 'Population'}, inplace=True)

        # Merge datasets on district
        self.merged_df = pd.merge(
            district_pop,
            self.poverty_data[['District', 'average_poverty_line']],
            on='District',
            how='inner'
        )

        # Prepare textual descriptions per district for embedding
        self.merged_df['text'] = self.merged_df['District'] + " Population: " + self.merged_df['Population'].astype(str) \
                                 + " Poverty: " + self.merged_df['average_poverty_line'].astype(str)

        self.corpus_embeddings = self.model.encode(
            self.merged_df['text'].tolist(),
            convert_to_tensor=True
        )

    def run(self, input_data: dict, top_n: int = 10):
        user_text = input_data.get("preference", "")

        # Encode the user query
        with torch.no_grad():
            query_embedding = self.model.encode(
                user_text,
                convert_to_tensor=True,
                normalize_embeddings=True
            )

        # Cosine similarity
        cos_scores = torch.matmul(query_embedding, self.corpus_embeddings.T)
        scores = cos_scores.cpu().numpy()

        # Top districts
        top_idx = scores.argsort()[-top_n:][::-1]
        top_districts = self.merged_df.iloc[top_idx].copy()
        top_districts['score'] = scores[top_idx]

        return top_districts[['District', 'Population', 'average_poverty_line', 'score']].reset_index(drop=True)
