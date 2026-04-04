"""
app.py — Streamlit UI for the LFS-2023 Resource Allocation AI

Run:
    streamlit run app.py

Requires the FastAPI service to be running:
    uvicorn service:app --reload --port 8000
"""

import streamlit as st
import requests
import json
import time
from typing import Optional

# ── Config ─────────────────────────────────────────────────────────────────────
API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="LFS-2023 AI · Sri Lanka Workforce",
    page_icon="🇱🇰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Page background */
.stApp {
    background: #f7f5f0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #1a1a2e;
    color: white;
}
section[data-testid="stSidebar"] * {
    color: white !important;
}
section[data-testid="stSidebar"] .stRadio label {
    color: white !important;
    font-size: 0.95rem;
    padding: 6px 0;
}

/* Header */
.main-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
    color: white;
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(229,57,53,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.main-header h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    margin: 0 0 0.4rem 0;
    color: white;
}
.main-header p {
    font-size: 1rem;
    opacity: 0.75;
    margin: 0;
    color: #ccc;
}

/* Cards */
.metric-card {
    background: white;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    border-left: 4px solid #e53935;
    margin-bottom: 1rem;
}
.metric-card .metric-val {
    font-size: 2rem;
    font-weight: 600;
    color: #1a1a2e;
    line-height: 1;
}
.metric-card .metric-lbl {
    font-size: 0.8rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 4px;
}

/* Chat bubbles */
.chat-user {
    background: #1a1a2e;
    color: white;
    padding: 0.9rem 1.2rem;
    border-radius: 16px 16px 4px 16px;
    margin: 0.5rem 0 0.5rem 20%;
    font-size: 0.95rem;
    line-height: 1.5;
}
.chat-ai {
    background: white;
    color: #1a1a2e;
    padding: 0.9rem 1.2rem;
    border-radius: 16px 16px 16px 4px;
    margin: 0.5rem 20% 0.5rem 0;
    font-size: 0.9rem;
    line-height: 1.6;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
}
.chat-label {
    font-size: 0.72rem;
    color: #aaa;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin: 0.3rem 0;
}

/* Status badge */
.badge-ok   { background:#e8f5e9; color:#2e7d32; padding:2px 10px; border-radius:20px; font-size:0.8rem; font-weight:500; }
.badge-err  { background:#ffebee; color:#c62828; padding:2px 10px; border-radius:20px; font-size:0.8rem; font-weight:500; }

/* Allocation table */
.alloc-result {
    background: white;
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    font-family: 'DM Sans', monospace;
    font-size: 0.82rem;
    overflow-x: auto;
    white-space: pre-wrap;
    line-height: 1.6;
    color: #333;
}

/* Section title */
.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.4rem;
    color: #1a1a2e;
    margin-bottom: 1rem;
    border-bottom: 2px solid #e53935;
    padding-bottom: 0.4rem;
    display: inline-block;
}

/* Primary button override */
.stButton > button {
    background: #1a1a2e !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.5rem !important;
    font-weight: 500 !important;
    font-size: 0.95rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: #e53935 !important;
    transform: translateY(-1px) !important;
}
</style>
""", unsafe_allow_html=True)


# ── Helper functions ───────────────────────────────────────────────────────────

def api_post(endpoint: str, payload: dict) -> dict:
    try:
        r = requests.post(f"{API_BASE}{endpoint}", json=payload, timeout=120)
        r.raise_for_status()
        return {"ok": True, "data": r.json()}
    except requests.exceptions.ConnectionError:
        return {"ok": False, "error": "Cannot connect to API. Is `uvicorn service:app` running?"}
    except requests.exceptions.Timeout:
        return {"ok": False, "error": "Request timed out (>120s). Try a smaller allocation."}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def api_get(endpoint: str, params: dict = None) -> dict:
    try:
        r = requests.get(f"{API_BASE}{endpoint}", params=params, timeout=60)
        r.raise_for_status()
        return {"ok": True, "data": r.json()}
    except requests.exceptions.ConnectionError:
        return {"ok": False, "error": "Cannot connect to API. Is `uvicorn service:app` running?"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def check_health() -> bool:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        return r.status_code == 200 and r.json().get("agent_loaded", False)
    except Exception:
        return False


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🇱🇰 LFS-2023 AI")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["💬 Chat", "🎯 Allocate Resources", "📊 Insights", "🔍 Cluster Analysis", "⚙️ System"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # API health indicator
    healthy = check_health()
    if healthy:
        st.markdown('<span class="badge-ok">● API Online</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-err">● API Offline</span>', unsafe_allow_html=True)
        st.caption("Run: `uvicorn service:app --reload --port 8000`")

    st.markdown("---")
    st.caption("Sri Lanka Labour Force Survey 2023")
    st.caption("18,937 respondents · 128 features")
    st.caption("Powered by Groq · llama-3.3-70b")


# ── Header ─────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>Labour Force Survey 2023</h1>
    <p>AI-powered workforce analysis and resource allocation for Sri Lanka</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Chat
# ══════════════════════════════════════════════════════════════════════════════

if page == "💬 Chat":
    st.markdown('<span class="section-title">Ask anything about the data</span>', unsafe_allow_html=True)

    # Session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-label">You</div><div class="chat-user">{msg["content"]}</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-label">AI · {msg.get("route","")}</div>'
                        f'<div class="chat-ai">{msg["content"]}</div>',
                        unsafe_allow_html=True)

    # Quick examples
    if not st.session_state.messages:
        st.markdown("**Try asking:**")
        cols = st.columns(3)
        examples = [
            "How many people work in the estate sector?",
            "What is the average income by district?",
            "Give 30 laptops to the most vulnerable workers",
            "Show education level distribution by gender",
            "What are the key insights from this data?",
            "Compare all clusters",
        ]
        for i, ex in enumerate(examples):
            with cols[i % 3]:
                if st.button(ex, key=f"ex_{i}"):
                    st.session_state.pending_query = ex

    # Input
    query = st.chat_input("Type your question…")

    # Handle quick-example click
    if "pending_query" in st.session_state:
        query = st.session_state.pop("pending_query")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})

        with st.spinner("Thinking…"):
            result = api_post("/chat", {"query": query})

        if result["ok"]:
            data = result["data"]
            st.session_state.messages.append({
                "role": "ai",
                "content": data["response"],
                "route": data.get("route", ""),
            })
        else:
            st.session_state.messages.append({
                "role": "ai",
                "content": f"❌ {result['error']}",
                "route": "error",
            })

        st.rerun()

    if st.session_state.messages:
        if st.button("🗑 Clear chat"):
            st.session_state.messages = []
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Allocate Resources
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🎯 Allocate Resources":
    st.markdown('<span class="section-title">Resource Allocation</span>', unsafe_allow_html=True)
    st.markdown(
        "Score all 18,937 respondents by vulnerability and identify the most "
        "deserving beneficiaries for any resource type."
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### Configure")
        num_items = st.number_input("Number of items to distribute", min_value=1, max_value=5000,
                                    value=50, step=10)
        item_type = st.text_input("Resource type", value="laptops",
                                  help="e.g. laptops, food rations, scholarships, sewing machines")
        context   = st.text_area(
            "Optional targeting criteria",
            placeholder="e.g. prioritise estate sector women with disabilities",
            height=80,
        )

        # Scoring weights info
        with st.expander("📐 Need score weights"):
            st.markdown("""
| Factor | Weight |
|---|---|
| Income | 40% |
| Education | 15% |
| Disability burden | 15% |
| Informality | 10% |
| Sector deprivation | 10% |
| Item-specific fit | 10% |
            """)

        run = st.button("🚀 Run Allocation", use_container_width=True)

    with col2:
        if run:
            if not item_type.strip():
                st.error("Please enter a resource type.")
            else:
                with st.spinner(f"Scoring {18937:,} respondents and selecting top {num_items}…"):
                    t0 = time.time()
                    result = api_post("/allocate", {
                        "num_items": num_items,
                        "item_type": item_type.strip(),
                        "context":   context.strip() or None,
                    })
                elapsed = time.time() - t0

                if result["ok"]:
                    data = result["data"]
                    st.success(f"✅ Allocation complete in {elapsed:.1f}s")
                    st.markdown(f"**Query sent:** `{data['query_sent']}`")
                    st.markdown('<div class="alloc-result">' +
                                data["result"].replace("\n", "<br>") +
                                '</div>', unsafe_allow_html=True)

                    st.download_button(
                        "⬇️ Download result",
                        data=data["result"],
                        file_name=f"allocation_{item_type}_{num_items}.txt",
                        mime="text/plain",
                    )
                else:
                    st.error(f"❌ {result['error']}")
        else:
            st.info("Configure the allocation on the left and click **Run Allocation**.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Insights
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Insights":
    st.markdown('<span class="section-title">Dataset Insights</span>', unsafe_allow_html=True)

    topic_options = {
        "General overview (all topics)": None,
        "Income & wages":   "income",
        "Employment":       "employment",
        "Education":        "education",
        "Disability":       "disability",
        "Gender":           "gender",
        "Digital literacy": "digital",
        "Regional disparities": "district",
    }

    choice = st.selectbox("Select insight topic", list(topic_options.keys()))
    topic  = topic_options[choice]

    if st.button("📈 Generate Insights"):
        with st.spinner("Analysing dataset…"):
            result = api_get("/insights", params={"topic": topic} if topic else None)

        if result["ok"]:
            data = result["data"]
            st.markdown("---")
            st.markdown(data["insights"])
            st.caption(f"Topic: {data['topic']} · {data['elapsed_ms']}ms")
        else:
            st.error(f"❌ {result['error']}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Cluster Analysis
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔍 Cluster Analysis":
    st.markdown('<span class="section-title">Population Cluster Analysis</span>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📋 Cluster Overview", "⚖️ Compare Clusters"])

    with tab1:
        if st.button("Load cluster statistics"):
            with st.spinner("Fetching cluster data…"):
                result = api_get("/clusters")

            if result["ok"]:
                data = result["data"]["clusters"]
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-val">{data.get('Total Records', 'N/A'):,}</div>
                        <div class="metric-lbl">Total respondents</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-val">{data.get('Clusters', 'N/A')}</div>
                        <div class="metric-lbl">Clusters identified</div>
                    </div>
                    """, unsafe_allow_html=True)

                dist = data.get("Distribution", {})
                with col3:
                    largest = max(dist.values()) if dist else 0
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-val">{largest:,}</div>
                        <div class="metric-lbl">Largest cluster size</div>
                    </div>
                    """, unsafe_allow_html=True)

                if dist:
                    st.markdown("### Cluster distribution")
                    import pandas as pd
                    df = pd.DataFrame(
                        list(dist.items()), columns=["Cluster ID", "Count"]
                    ).sort_values("Count", ascending=False)
                    st.bar_chart(df.set_index("Cluster ID"))
            else:
                st.error(f"❌ {result['error']}")

    with tab2:
        if st.button("Compare all clusters"):
            with st.spinner("Comparing clusters…"):
                result = api_post("/compare-clusters", {})

            if result["ok"]:
                st.markdown(result["data"]["comparison"])
                st.caption(f"{result['data']['elapsed_ms']}ms")
            else:
                st.error(f"❌ {result['error']}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: System
# ══════════════════════════════════════════════════════════════════════════════

elif page == "⚙️ System":
    st.markdown('<span class="section-title">System Status</span>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### API Health")
        result = api_get("/health")
        if result["ok"]:
            h = result["data"]
            st.json(h)
        else:
            st.error(result["error"])

    with col2:
        st.markdown("### Architecture")
        st.markdown("""
| Component | Technology |
|---|---|
| LLM | Groq · llama-3.3-70b |
| Intent router | Keyword matching (no LLM) |
| Clustering | K-Means (scikit-learn) |
| Need scoring | Python (pre-computed) |
| API layer | FastAPI |
| Frontend | Streamlit |
| Dataset | LFS-2023 Sri Lanka |
        """)

    st.markdown("### Dataset Schema")
    if st.button("Load column descriptions"):
        result = api_get("/schema")
        if result["ok"]:
            data = result["data"]
            with st.expander("📋 All columns", expanded=True):
                import pandas as pd
                rows = [{"Column": k, "Description": v}
                        for k, v in data["columns"].items()]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, height=400)

            with st.expander("🗺️ Value mappings"):
                st.json(data["value_maps"])
        else:
            st.error(result["error"])

    st.markdown("### Quick start")
    st.code("""
# 1. Start the API
uvicorn service:app --reload --port 8000

# 2. Start the UI (separate terminal)
streamlit run app.py

# 3. Open in browser
http://localhost:8501
    """, language="bash")