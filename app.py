# from fastapi import FastAPI
# from api.controller.poverty_controller import router  # ← Full correct path
#
# app = FastAPI(title="DSGP Multi-Agent Recommendation System")
# app.include_router(router, prefix="/api")

import pickle
import os
from sentence_transformers import SentenceTransformer

# Define your paths (update these to match your project_root)
model_paths = [
    "model/poverty_model.pkl",
    "model/child_case_nlp.pkl",
    "model/mental_health_model.pkl"
]

def update_model(path):
    if os.path.exists(path):
        print(f"Updating {path}...")
        # This will likely fail with the same error if using pickle.load directly
        # Instead, we identify which base model it was and reload it fresh
        # OR if you can, load it, then re-save:
        try:
            with open(path, "rb") as f:
                model = pickle.load(f)
            with open(path, "wb") as f:
                pickle.dump(model, f)
            print("Success.")
        except AttributeError:
            print(f"Cannot load {path} due to version mismatch. Please re-download or re-initialize this model.")

# If they are standard models, you can also just re-initialize them:
# model = SentenceTransformer('all-MiniLM-L6-v2')
# with open("model/poverty_model.pkl", "wb") as f:
#     pickle.dump(model, f)