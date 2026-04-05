import streamlit as st
import requests
import json
import time
from typing import Optional

# ── Config ─────────────────────────────────────────────────────────────────────
API_BASE = "http://localhost:8001"

st.set_page_config(
    page_title="LFS-2023 AI · Sri Lanka Workforce",
    page_icon="🇱🇰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;0,700;1,400;1,600&family=Outfit:wght@300;400;500;600;700&display=swap');

:root {
    --ink:  #0f0a1e;
    --muted:#6b5f8a;
    --viv:  #6d28d9;
    --mid:  #8b5cf6;
    --lite: #a78bfa;
    --pale: #ede9fe;
    --wash: #f5f3ff;
    --bd:   rgba(109,40,217,.13);
    --bdd:  rgba(109,40,217,.26);
}

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

/* Page background */
.stApp {
    background: var(--wash);
}
[data-testid="stHeader"], footer, [data-testid="stToolbar"],
[data-testid="stDecoration"] { display:none !important; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--ink);
    color: white;
}
section[data-testid="stSidebar"] * {
    color: white !important;
}
section[data-testid="stSidebar"] .stRadio label {
    color: white !important;
    font-size: 0.9rem;
    padding: 6px 0;
}

/* Header */
.main-header {
    background: radial-gradient(ellipse 65% 55% at 65% -5%, rgba(109,40,217,.18) 0%, transparent 70%),
                radial-gradient(ellipse 45% 35% at 5% 85%, rgba(167,139,250,.10) 0%, transparent 60%),
                var(--ink);
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
    background: radial-gradient(circle, rgba(109,40,217,0.25) 0%, transparent 70%);
    border-radius: 50%;
}
.main-header h1 {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.2rem;
    font-weight: 700;
    margin: 0 0 0.4rem 0;
    color: white;
    letter-spacing: -0.02em;
}
.main-header p {
    font-size: 0.95rem;
    font-weight: 300;
    opacity: 0.75;
    margin: 0;
    color: var(--lite);
}

/* Cards */
.metric-card {
    background: white;
    border-radius: 13px;
    padding: 1.2rem 1.5rem;
    box-shadow: 0 4px 18px var(--bd);
    border-left: 4px solid var(--viv);
    margin-bottom: 1rem;
}
.metric-card .metric-val {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: var(--viv);
    line-height: 1;
}
.metric-card .metric-lbl {
    font-size: 0.72rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 500;
    margin-top: 4px;
}

/* Chat bubbles */
.chat-user {
    background: var(--viv);
    color: white;
    padding: 0.9rem 1.2rem;
    border-radius: 16px 16px 4px 16px;
    margin: 0.5rem 0 0.5rem 20%;
    font-size: 0.92rem;
    line-height: 1.55;
    box-shadow: 0 4px 16px rgba(109,40,217,.28);
}
.chat-ai {
    background: white;
    color: var(--ink);
    padding: 0.9rem 1.2rem;
    border-radius: 16px 16px 16px 4px;
    margin: 0.5rem 20% 0.5rem 0;
    font-size: 0.88rem;
    line-height: 1.65;
    box-shadow: 0 2px 12px var(--bd);
    border: 1px solid var(--bd);
}
.chat-label {
    font-size: 0.68rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 600;
    margin: 0.3rem 0;
}

/* Status badge */
.badge-ok  { background:#ecfdf5; color:#059669; padding:2px 10px; border-radius:20px; font-size:0.78rem; font-weight:600; border:1px solid rgba(5,150,105,.2); }
.badge-err { background:#fff1f2; color:#e11d48; padding:2px 10px; border-radius:20px; font-size:0.78rem; font-weight:600; border:1px solid rgba(225,29,72,.2); }

/* Allocation table */
.alloc-result {
    background: white;
    border-radius: 13px;
    padding: 1.5rem;
    box-shadow: 0 4px 18px var(--bd);
    border: 1px solid var(--bdd);
    font-family: 'Outfit', monospace;
    font-size: 0.82rem;
    overflow-x: auto;
    white-space: pre-wrap;
    line-height: 1.65;
    color: var(--ink);
}

/* Section title */
.section-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--ink);
    margin-bottom: 1rem;
    border-bottom: 2px solid var(--viv);
    padding-bottom: 0.4rem;
    display: inline-block;
    letter-spacing: -0.01em;
}

/* Primary button override */
.stButton > button {
    background: var(--viv) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em !important;
    padding: 0.78rem 1.5rem !important;
    transition: all 0.2s cubic-bezier(.34,1.56,.64,1) !important;
    box-shadow: 0 4px 16px rgba(109,40,217,.28) !important;
    cursor: pointer !important;
}
.stButton > button:hover {
    background: #5b21b6 !important;
    transform: translateY(-3px) scale(1.02) !important;
    box-shadow: 0 12px 32px rgba(109,40,217,.40) !important;
}
.stButton > button:active {
    transform: translateY(0px) scale(0.98) !important;
    box-shadow: 0 2px 8px rgba(109,40,217,.25) !important;
    background: #4c1d95 !important;
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

    st.markdown("---")
    if st.button("🏠 Back to Home", use_container_width=True):
        st.switch_page("../home.py")


# ── Header ─────────────────────────────────────────────────────────────────────

# Back to Home — scoped to just this button's container so it doesn't override
# the sidebar buttons which inherit the global .stButton rule
st.markdown("""
<style>
div[data-testid="stButton"].home-back-btn > button {
    background: #6d28d9 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em !important;
    padding: 0.72rem 1.5rem !important;
    box-shadow: 0 4px 16px rgba(109,40,217,.30) !important;
    transition: all 0.2s cubic-bezier(.34,1.56,.64,1) !important;
    width: auto !important;
}
div[data-testid="stButton"].home-back-btn > button:hover {
    background: #5b21b6 !important;
    transform: translateY(-2px) scale(1.02) !important;
    box-shadow: 0 10px 28px rgba(109,40,217,.42) !important;
}
</style>
""", unsafe_allow_html=True)

# Inject a class onto the NEXT stButton element
st.markdown('<div class="home-back-btn" style="display:contents">', unsafe_allow_html=True)
if st.button("🏠 Back to Home"):
    st.switch_page("home.py")
st.markdown('</div>', unsafe_allow_html=True)

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