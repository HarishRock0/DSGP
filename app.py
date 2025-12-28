# app.py
from fastapi import FastAPI
from api.routes import router

app = FastAPI(title="DSGP Multi-Agent Recommendation System")

app.include_router(router, prefix="/api")