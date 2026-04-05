import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
import sys
import warnings
warnings.filterwarnings('ignore')

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import time

# ── Path setup ─────────────────────────────────────────────────────────────────
# SERVICE_DIR = os.path.dirname(os.path.abspath(__file__))
# NLP_DIR     = os.path.join(SERVICE_DIR, 'NLP')
# MODEL_PATH  = os.path.join(SERVICE_DIR, 'model', 'skilldev_model.pkl')

SERVICE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SERVICE_DIR)   # one level up = DSGP/
NLP_DIR     = os.path.join(PROJECT_DIR, 'NLP')
MODEL_PATH  = os.path.join(PROJECT_DIR, 'model', 'skilldev_model.pkl')

# Allow running from different working directories
sys.path.insert(0, NLP_DIR)
sys.path.insert(0, SERVICE_DIR)

# SkillDev stub required before any pickle loads
class SkillDev:
    pass


# ── App bootstrap ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="LFS-2023 Resource Allocation AI",
    description=(
        "AI-powered resource allocation and workforce analysis for "
        "Sri Lanka's Labour Force Survey 2023 (18,937 respondents)."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Lazy-load agent (expensive — do once at startup) ──────────────────────────
_agent = None


# def get_agent():
#     global _agent
#     if _agent is None:
#         if not os.path.exists(MODEL_PATH):
#             raise HTTPException(
#                 status_code=503,
#                 detail=f"Model not found at {MODEL_PATH}. Run train_model.py first.",
#             )
#         os.chdir(NLP_DIR)
#         from NLP.Engines.agent import LFSAgent
#         _agent = LFSAgent(model_path=MODEL_PATH, verbose=False)
#     return _agent

def get_agent():
    global _agent
    if _agent is None:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(
                status_code=503,
                detail=f"Model not found at {MODEL_PATH}. Run train_model.py first.",
            )
        # Add NLP dir to path so Engines is directly importable
        if NLP_DIR not in sys.path:
            sys.path.insert(0, NLP_DIR)
        from Engines.agent import LFSAgent   # NOT NLP.Engines.agent
        _agent = LFSAgent(model_path=MODEL_PATH, verbose=False)
    return _agent

@app.on_event("startup")
async def _startup():
    """Pre-load agent so first request is fast."""
    try:
        get_agent()
        print("✅ LFS Agent loaded and ready.")
    except Exception as e:
        print(f"⚠️  Agent pre-load failed: {e}")


# ── Request / Response models ──────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, example="How many people work in the estate sector?")


class ChatResponse(BaseModel):
    query: str
    response: str
    route: Optional[str] = None
    elapsed_ms: int


class AllocateRequest(BaseModel):
    num_items: int = Field(..., ge=1, le=10000,example=50, description="Number of items to distribute")
    item_type: str = Field(..., min_length=1, max_length=50,example="laptops", description="Type of resource")
    context: Optional[str] = Field(None, max_length=200,example="prioritise estate sector women",description="Optional targeting criteria appended to the query",)


class AllocateResponse(BaseModel):
    query_sent: str
    result: str
    elapsed_ms: int


class AnalyzeRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=400, example="What is the average income by district?")


class AnalyzeResponse(BaseModel):
    question: str
    result: str
    elapsed_ms: int


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health_check():
    """Quick health check — returns 200 if service is running."""
    agent_ready = _agent is not None
    return {
        "status": "ok",
        "agent_loaded": agent_ready,
        "model_path": MODEL_PATH,
        "model_exists": os.path.exists(MODEL_PATH),
    }


@app.post("/chat", response_model=ChatResponse, tags=["General"])
def chat(body: ChatRequest):
    agent = get_agent()
    t0 = time.time()

    try:
        # Capture the route from the nlpc engine before calling chat
        intent = agent.nlpc_engine.understand_query(body.query)
        route  = intent.get('route', 'unknown')

        response = agent.chat(body.query)

        return ChatResponse(
            query=body.query,
            response=response,
            route=route,
            elapsed_ms=int((time.time() - t0) * 1000),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/allocate", response_model=AllocateResponse, tags=["Allocation"])
def allocate(body: AllocateRequest):
    agent = get_agent()
    t0 = time.time()

    # Build a natural-language query the engine understands
    query = f"Give {body.num_items} {body.item_type} to the most vulnerable people"
    if body.context:
        query += f", {body.context}"

    try:
        result = agent.llm_engine.handle_allocation(
            question=query,
            num_items=body.num_items,
            item_type=body.item_type,
        )
        return AllocateResponse(
            query_sent=query,
            result=result,
            elapsed_ms=int((time.time() - t0) * 1000),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/clusters", tags=["Clusters"])
def get_clusters():
    agent = get_agent()
    t0 = time.time()
    try:
        stats = agent.nlpc_engine._get_cluster_stats()
        return {
            "clusters": stats,
            "elapsed_ms": int((time.time() - t0) * 1000),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/insights", tags=["Analysis"])
def get_insights(
    topic: Optional[str] = Query(None,description="Focus topic: income | education | disability | gender | employment | digital",example="income",)):

    agent = get_agent()
    t0 = time.time()
    try:
        result = agent.llm_engine.get_insights(topic=topic)
        return {
            "topic": topic or "general",
            "insights": result,
            "elapsed_ms": int((time.time() - t0) * 1000),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze", response_model=AnalyzeResponse, tags=["Analysis"])
def analyze(body: AnalyzeRequest):
    agent = get_agent()
    t0 = time.time()
    try:
        result = agent.llm_engine.analyze_data(body.question)
        return AnalyzeResponse(
            question=body.question,
            result=result,
            elapsed_ms=int((time.time() - t0) * 1000),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/compare-clusters", tags=["Clusters"])
def compare_clusters():
    agent = get_agent()
    t0 = time.time()
    try:
        result = agent.llm_engine.compare_clusters()
        return {
            "comparison": result,
            "elapsed_ms": int((time.time() - t0) * 1000),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/schema", tags=["System"])
def get_schema():
    try:
        from constants import (
            COLUMN_DESCRIPTIONS, SECTOR_MAP, DISTRICT_MAP,
            EMPLOYMENT_STATUS, ETHNICITY_MAP, RELIGION_MAP,
            MARITAL_MAP, PROVINCE_DISTRICTS,
        )
        return {
            "columns": COLUMN_DESCRIPTIONS,
            "value_maps": {
                "SECTOR":   SECTOR_MAP,
                "DISTRICT": DISTRICT_MAP,
                "Q16_employment": EMPLOYMENT_STATUS,
                "ETH_ethnicity":  ETHNICITY_MAP,
                "REL_religion":   RELIGION_MAP,
                "MARITAL":        MARITAL_MAP,
                "PROVINCE_DISTRICTS": PROVINCE_DISTRICTS,
            },
        }
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="constants.py not found — check your working directory.",
        )