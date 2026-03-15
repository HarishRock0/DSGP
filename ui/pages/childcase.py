#
# import sys
# import os
# import streamlit as st
# import pandas as pd
# import plotly.express as px
#
# # ---------------- PATH SETUP ----------------
# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# if PROJECT_ROOT not in sys.path:
#     sys.path.insert(0, PROJECT_ROOT)
#
# # ---------------- SERVICE ----------------
# from service.child_protection_service import ChildProtectionService
#
# st.set_page_config(page_title="Child Protection Risk Dashboard", layout="wide", initial_sidebar_state="collapsed")
#
# # ---------------- CSS (MATCH poverty.py incl. DEMOGRAPHICS) ----------------
# st.markdown("""
# <style>
# /* ---------- GLOBAL ---------- */
# html, body, [class*="css"] {
#     font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont;
# }
#
# .stApp {
#     background: linear-gradient(135deg, #0f172a 0%, #020617 100%);
#     color: #e5e7eb;
# }
#
# /* ---------- HEADERS ---------- */
# h1, h2, h3 {
#     font-weight: 700;
#     letter-spacing: 0.3px;
# }
#
# h1 { color: #38bdf8; }
# h3 { color: #e5e7eb; }
#
# /* ---------- INPUTS ---------- */
# .stTextInput > div > div > input {
#     background-color: #020617;
#     border-radius: 12px;
#     border: 1px solid #1e293b;
#     color: #e5e7eb;
#     padding: 12px;
# }
#
# /* ---------- BUTTONS ---------- */
# .stButton > button {
#     background: linear-gradient(90deg, #38bdf8, #0ea5e9);
#     color: black;
#     border-radius: 14px;
#     padding: 10px 22px;
#     font-weight: 600;
#     border: none;
#     transition: transform 0.15s ease;
# }
#
# .stButton > button:hover {
#     transform: scale(1.03);
# }
#
# /* ---------- METRIC CARDS ---------- */
# [data-testid="metric-container"] {
#     background: rgba(2, 6, 23, 0.9);
#     border: 1px solid #1e293b;
#     padding: 20px;
#     border-radius: 18px;
#     box-shadow: 0 10px 30px rgba(0,0,0,0.4);
# }
#
# [data-testid="metric-container"] label {
#     color: #94a3b8;
# }
#
# [data-testid="metric-container"] div {
#     color: #38bdf8;
#     font-weight: 700;
# }
#
# /* ---------- DATAFRAME ---------- */
# .stDataFrame {
#     background-color: #020617;
#     border-radius: 14px;
#     border: 1px solid #1e293b;
# }
#
# /* ---------- SELECT / RADIO ---------- */
# .stSelectbox, .stRadio {
#     background-color: #020617;
#     border-radius: 12px;
# }
#
# /* ---------- DIVIDERS ---------- */
# hr {
#     border: none;
#     border-top: 1px solid #1e293b;
#     margin: 30px 0;
# }
#
# /* ---------- POPULATION STATS (FROM poverty.py) ---------- */
# .pop-card {
#     background: linear-gradient(145deg, #020617, #020617);
#     border: 1px solid #1e293b;
#     border-radius: 18px;
#     padding: 18px 22px;
#     margin-bottom: 14px;
#     box-shadow: 0 10px 25px rgba(0,0,0,0.45);
# }
#
# .pop-label {
#     color: #94a3b8;
#     font-size: 0.85rem;
#     letter-spacing: 0.4px;
# }
#
# .pop-value {
#     font-size: 1.8rem;
#     font-weight: 800;
#     margin-top: 4px;
# }
#
# .pop-male { color: #38bdf8; }
# .pop-female { color: #f472b6; }
# .pop-total { color: #22c55e; }
# </style>
# """, unsafe_allow_html=True)
#
# # ---------------- SERVICE CACHE ----------------
# @st.cache_resource
# def load_service():
#     return ChildProtectionService(PROJECT_ROOT)
#
# service = load_service()
#
# # ---------------- HEADER ----------------
# st.markdown("""
# ## Child Protection Risk Recommendation Dashboard
# <span style="color:#94a3b8">
# NLP-driven regional risk analysis with interactive child case & demographic insights
# </span>
# """, unsafe_allow_html=True)
#
# # ---------------- SESSION STATE ----------------
# if "child_recs" not in st.session_state:
#     st.session_state.child_recs = None
#
# # ---------------- INPUT ----------------
# user_input = st.text_input(
#     "Describe your concern:",
#     placeholder="e.g. child abuse risk, neglected children, vulnerable regions"
# )
#
# run_btn = st.button("Analyze Child Protection Risk")
#
# if run_btn:
#     if not user_input.strip():
#         st.warning("Please describe a concern.")
#     else:
#         with st.spinner("Analyzing child protection signals..."):
#             result = service.get_child_case_recommendations(user_input)
#         st.session_state.child_recs = result.get("recommendations", [])
#
# # ---------------- RECOMMENDATIONS ----------------
# recs = st.session_state.child_recs
# if not recs:
#     st.info("Enter a concern and run the analysis to see recommendations.")
#     st.stop()
#
# rec_df = pd.DataFrame(recs)
# st.subheader("Top At-Risk Districts")
# st.dataframe(rec_df, use_container_width=True, height=260)
#
# districts = rec_df["District"].tolist()
# st.divider()
#
# # ---------------- DASHBOARD ----------------
# st.subheader("Child Protection Risk Dashboard")
#
# left, right = st.columns([1, 2])
#
# with left:
#     selected_district = st.selectbox("Select District", districts)
#     chart_type = st.radio("Chart Type", ["Line", "Bar"], horizontal=True)
#     rolling_window = st.selectbox("Trend Smoothing (years)", [1, 2, 3], index=0)
#
#
#
# # ---------------- INSIGHTS ----------------
# with st.spinner("Generating district insights..."):
#     insights = service.get_child_insights(selected_district)
#
# child_cases = insights.get("child_cases", {})
# demo = insights.get("demographics", {})
# metrics = demo.get("metrics", {})
# trend = child_cases.get("trend", {}) or {}
#
# # ---------------- KPIs ----------------
# k1, k2, k3 = st.columns(3)
# latest_cases = list(trend.values())[-1] if trend else None
#
# k1.metric("Selected District", selected_district)
# k2.metric("Latest Reported Cases", latest_cases if latest_cases else "N/A")
# k3.metric("Months Available", len(trend))
#
# # ---------------- TREND VISUALIZATION ----------------
# if trend:
#     chart_df = pd.DataFrame({"Year": list(trend.keys()), "Reported Cases": list(trend.values())})
#     chart_df["Reported Cases"] = pd.to_numeric(chart_df["Reported Cases"], errors="coerce")
#
#     if rolling_window > 1:
#         chart_df["Smoothed"] = chart_df["Reported Cases"].rolling(rolling_window, min_periods=1).mean()
#         y_col = "Smoothed"
#     else:
#         y_col = "Reported Cases"
#
#     fig = px.line(chart_df, x="Year", y=y_col, markers=True)
#     fig.update_layout(height=420, hovermode="x unified")
#
#     with right:
#         st.plotly_chart(fig, use_container_width=True)
#
#
# # ---------------- DEMOGRAPHICS (NOW IDENTICAL TO poverty.py) ----------------
# # ---------------------------
# # Demographics
# # ---------------------------
# st.markdown("### Demographics Snapshot")
#
# if metrics:
#     col1, col2 = st.columns([1, 1.5])
#
#     male = metrics.get("MALE", 0)
#     female = metrics.get("FEMALE", 0)
#     total = metrics.get("TOT_POP", male + female)
#
#     # -------- COL 1 : Metrics --------
#     with col1:
#         st.subheader("Population Stats")
#
#         st.markdown(f"""
#         <div class="pop-card">
#             <div class="pop-label">Male Population</div>
#             <div class="pop-value pop-male">{int(male):,}</div>
#         </div>
#
#         <div class="pop-card">
#             <div class="pop-label">Female Population</div>
#             <div class="pop-value pop-female">{int(female):,}</div>
#         </div>
#
#         <div class="pop-card">
#             <div class="pop-label">Total Population</div>
#             <div class="pop-value pop-total">{int(total):,}</div>
#         </div>
#         """, unsafe_allow_html=True)
#
#     # -------- COL 2 : Pie Chart --------
#     with col2:
#         st.subheader("Gender Distribution")
#
#         pie_df = pd.DataFrame({
#             "Gender": ["Male", "Female"],
#             "Population": [male, female]
#         })
#
#         pie_fig = px.pie(
#             pie_df,
#             names="Gender",
#             values="Population",
#             hole=0.4,
#             title="Male vs Female Population (%)"
#         )
#
#         pie_fig.update_traces(
#             textinfo="percent+label",
#             pull=[0.03, 0.03]
#         )
#
#         pie_fig.update_layout(
#             height=380,
#             margin=dict(t=50, b=20, l=20, r=20)
#         )
#
#         st.plotly_chart(pie_fig, use_container_width=True)
#
# else:
#     st.info("No demographic data available for this district.")



import sys
import os
import streamlit as st
import pandas as pd
import plotly.express as px

# ---------------- PATH SETUP ----------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------- SERVICE ----------------
from service.child_protection_service import ChildProtectionService

st.set_page_config(page_title="Child Protection Risk Dashboard", layout="wide", initial_sidebar_state="collapsed")


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Base ── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main, .block-container {
    background-color: #f4f2fb !important;
    color: #1e1b4b !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stHeader"], footer { display: none !important; }
[data-testid="block-container"] {
    padding: 2rem 2.5rem 4rem !important;
    max-width: 1400px;
    margin: 0 auto;
}

/* ── All text dark ── */
p, span, div, label, li, td, th, h1, h2, h3, h4, h5, h6,
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] * {
    color: #1e1b4b !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── Page header ── */
.page-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-bottom: 1.4rem;
    border-bottom: 1px solid #e5e7eb;
    margin-bottom: 2rem;
}
.page-title { font-size: 1.55rem; font-weight: 700; color: #1e1b4b !important; margin: 0; }
.page-title span { color: #7c3aed !important; }
.page-subtitle { font-size: 0.875rem; color: #6b7280 !important; margin-top: 0.2rem; }
.page-badge {
    font-size: 0.75rem; font-weight: 600;
    color: #7c3aed !important;
    background: #ede9fe;
    padding: 0.35rem 0.9rem;
    border-radius: 100px;
}

/* ── Cards ── */
.search-card, .white-card {
    background: #ffffff !important;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.card-label {
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #6b7280 !important;
    margin-bottom: 0.5rem;
    display: block;
}

/* ── Input ── */
input, .stTextInput input, [data-baseweb="input"] input {
    background: #ffffff !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 8px !important;
    color: #1e1b4b !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
}
input::placeholder { color: #9ca3af !important; }
input:focus {
    border-color: #7c3aed !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.1) !important;
}
.stTextInput label, .stTextInput label * { color: #6b7280 !important; font-size: 0.8rem !important; }

/* ── Button ── */
.stButton button {
    background: #7c3aed !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    font-family: 'Inter', sans-serif !important;
    box-shadow: 0 2px 8px rgba(124,58,237,0.3) !important;
    transition: background 0.2s !important;
}
.stButton button:hover { background: #6d28d9 !important; }
.stButton button p, .stButton button * { color: #ffffff !important; }

/* ── Selectbox ── */
[data-baseweb="select"] > div {
    background: #ffffff !important;
    border-color: #e5e7eb !important;
}
[data-baseweb="select"] span,
[data-baseweb="select"] div[data-value],
[data-baseweb="select"] input,
[data-baseweb="select"] [data-baseweb="select-option"] {
    background: #ffffff !important;
    color: #1e1b4b !important;
    font-family: 'Inter', sans-serif !important;
}
[data-baseweb="select"] svg,
[data-baseweb="select"] svg * {
    fill: #6b7280 !important;
    color: inherit !important;
    font-family: unset !important;
}
[data-baseweb="popover"] *, [data-baseweb="menu"] * {
    background: #ffffff !important; color: #1e1b4b !important;
}
[data-baseweb="option"]:hover { background: #f3e8ff !important; }
.stSelectbox label, .stSelectbox label * { color: #6b7280 !important; font-size: 0.8rem !important; }

/* ── Radio ── */
.stRadio label *, .stRadio [data-testid="stMarkdownContainer"] p {
    color: #374151 !important; font-size: 0.85rem !important;
}

/* ── Metric KPI ── */
[data-testid="metric-container"] {
    background: #ffffff !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 12px !important;
    padding: 1rem 1.4rem !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06) !important;
}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {
    color: #6b7280 !important; font-size: 0.72rem !important;
    font-weight: 600 !important; letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}
[data-testid="stMetricValue"], [data-testid="stMetricValue"] * {
    color: #7c3aed !important; font-size: 1.45rem !important; font-weight: 700 !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] > div {
    background: #ffffff !important;
    border-radius: 12px !important;
    border: 1px solid #e5e7eb !important;
    overflow: hidden;
}

/* ── Alert / Info ── */
[data-testid="stAlert"], [data-testid="stAlert"] * {
    background: #faf5ff !important; color: #1e1b4b !important;
    border: 1px solid #ddd6fe !important; border-radius: 10px !important;
}

/* ── Population cards ── */
.pop-card {
    background: #ffffff !important;
    border: 1px solid #e5e7eb;
    border-left: 4px solid #7c3aed;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.75rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.pop-card.female { border-left-color: #a78bfa; }
.pop-card.total  { border-left-color: #10b981; }
.pop-label {
    font-size: 0.72rem !important; font-weight: 600 !important;
    letter-spacing: 0.08em !important; text-transform: uppercase !important;
    color: #6b7280 !important;
}
.pop-value { font-size: 1.7rem !important; font-weight: 700 !important; margin-top: 0.15rem; line-height: 1; }
.pop-male   { color: #7c3aed !important; }
.pop-female { color: #a78bfa !important; }
.pop-total  { color: #059669 !important; }

/* ── control-card ── */
.control-card {
    background: #ffffff !important;
    border: 1px solid #e5e7eb;
    padding: 1.4rem 1.6rem;
    border-radius: 12px;
    margin-bottom: 1rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}

hr { border: none !important; border-top: 1px solid #e5e7eb !important; margin: 1.5rem 0 !important; }
[data-testid="column"] { padding: 0 0.4rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Plotly light theme ──────────────────────────────────────────────────────
PLOTLY_LIGHT = dict(
    paper_bgcolor="#ffffff",
    plot_bgcolor="#f4f2fb",
    font=dict(family="Inter, sans-serif", color="#374151", size=12),
    title_font=dict(family="Inter, sans-serif", color="#1e1b4b", size=15),
    xaxis=dict(gridcolor="#e5e7eb", linecolor="#e5e7eb",
               tickfont=dict(color="#374151"), title_font=dict(color="#374151")),
    yaxis=dict(gridcolor="#e5e7eb", linecolor="#e5e7eb",
               tickfont=dict(color="#374151"), title_font=dict(color="#374151")),
    legend=dict(bgcolor="#ffffff", bordercolor="#e5e7eb", borderwidth=1, font=dict(color="#374151")),
    hoverlabel=dict(bgcolor="#ffffff", font=dict(color="#1e1b4b", family="Inter")),
)

# ---------------- SERVICE CACHE ----------------
@st.cache_resource
def load_service():
    return ChildProtectionService(PROJECT_ROOT)

service = load_service()

# ---------------- CHILD NAV SWITCHER ----------------
st.markdown("""
<style>
.home-btn {
    position: fixed;
    top: 14px;
    left: 16px;
    z-index: 9999;
}
.home-btn a {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 7px 14px;
    border-radius: 100px;
    font-family: 'DM Sans', 'Inter', sans-serif;
    font-size: 0.78rem;
    font-weight: 600;
    text-decoration: none !important;
    color: #7c3aed;
    background: rgba(124,58,237,0.08);
    border: 1.5px solid rgba(124,58,237,0.25);
    transition: all 0.18s ease;
    white-space: nowrap;
    backdrop-filter: blur(8px);
}
.home-btn a:hover {
    background: linear-gradient(135deg,#7c3aed,#a855f7);
    color: #fff !important;
    border-color: transparent;
    box-shadow: 0 4px 12px rgba(124,58,237,0.35);
    text-decoration: none !important;
}
.child-nav-wrap {
    width: 100%;
    display: flex;
    justify-content: center;
    padding: 32px 0 20px;
}
.child-nav {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: rgba(124,58,237,0.07);
    border: 1px solid rgba(124,58,237,0.18);
    border-radius: 100px;
    padding: 4px;
}
.child-nav a {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 8px 20px;
    border-radius: 100px;
    font-family: 'DM Sans', 'Inter', sans-serif;
    font-size: 0.84rem;
    font-weight: 500;
    text-decoration: none !important;
    color: #7c3aed;
    background: transparent;
    border: none;
    transition: all 0.18s ease;
    cursor: pointer;
    white-space: nowrap;
}
.child-nav a:hover {
    background: rgba(124,58,237,0.10);
    color: #6d28d9;
    text-decoration: none !important;
}
.child-nav a.active {
    background: linear-gradient(135deg,#7c3aed,#a855f7);
    color: #fff !important;
    box-shadow: 0 4px 14px rgba(124,58,237,0.35);
}
</style>

<div class="home-btn">
  <a href="/" target="_self">🏠&nbsp; Home</a>
</div>

<div class="child-nav-wrap">
  <div class="child-nav">
    <a href="/childcase" class="active" target="_self">🛡️&nbsp; Regional Insights</a>
    <a href="/childprotection" target="_self">💰&nbsp; Resource Allocation</a>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------- PAGE HEADER ----------------
st.markdown("""
<div class="page-header">
    <div>
        <div class="page-title">Child Protection <span>Risk Dashboard</span></div>
        <div class="page-subtitle">NLP-driven regional risk analysis with interactive child case &amp; demographic insights</div>
    </div>
    <div class="page-badge">🛡️ Child Protection</div>
</div>
""", unsafe_allow_html=True)

# ---------------- SESSION STATE ----------------
if "child_recs" not in st.session_state:
    st.session_state.child_recs = None

# ---------------- INPUT — styled, logic original ----------------
st.markdown('<div class="search-card">', unsafe_allow_html=True)
st.markdown('<span class="card-label">Describe your concern</span>', unsafe_allow_html=True)
scol1, scol2 = st.columns([4, 1])
with scol1:
    user_input = st.text_input(
        "Describe your concern:",
        placeholder="e.g. child abuse risk, neglected children, vulnerable regions",
        label_visibility="collapsed"
    )
with scol2:
    st.markdown("<div style='height:0.1rem'></div>", unsafe_allow_html=True)
    run_btn = st.button("Analyze Risk", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

if run_btn:
    if not user_input.strip():
        st.warning("Please describe a concern.")
    else:
        with st.spinner("Analyzing child protection signals..."):
            result = service.get_child_case_recommendations(user_input)
        st.session_state.child_recs = result.get("recommendations", [])

# ---------------- RECOMMENDATIONS — styled, logic original ----------------
recs = st.session_state.child_recs
if not recs:
    st.markdown("""
    <div style="text-align:center;padding:3rem 1rem;">
        <div style="font-size:0.95rem;color:#6b7280;font-family:Inter,sans-serif;">
            Enter a concern and click
            <strong style="color:#7c3aed;">Analyze Risk</strong> to see recommendations.
        </div>
    </div>""", unsafe_allow_html=True)
    st.stop()

rec_df = pd.DataFrame(recs)
st.markdown('<span class="card-label" style="margin-bottom:0.5rem;">Top At-Risk Districts</span>', unsafe_allow_html=True)
st.dataframe(rec_df, use_container_width=True, height=260)

districts = rec_df["District"].tolist()
# Normalise to Title Case so it matches child_case_df index and demo DISTRICT_N
districts = [d.strip().title() for d in districts]
st.divider()

# ---------------- DASHBOARD — ORIGINAL UNTOUCHED ----------------
st.subheader("Child Protection Risk Dashboard")

left, right = st.columns([1, 2])

with left:
    selected_district = st.selectbox("Select District", districts)
    chart_type = st.radio("Chart Type", ["Line", "Bar"], horizontal=True)
    rolling_window = st.selectbox("Trend Smoothing (years)", [1, 2, 3], index=0)

# ---------------- INSIGHTS — ORIGINAL UNTOUCHED ----------------
with st.spinner("Generating district insights..."):
    # Normalise case: service data uses Title Case (e.g. "Mannar" not "MANNAR")
    lookup_district = selected_district.strip().title()
    # Fix known spelling variants returned by the recommendation service
    DISTRICT_NAME_MAP = {
        "Monaragala": "Moneragala",
        "Rathnapura":  "Ratnapura",
    }
    lookup_district = DISTRICT_NAME_MAP.get(lookup_district, lookup_district)
    insights = service.get_child_insights(lookup_district)

child_cases = insights.get("child_cases", {})
demo = insights.get("demographics", {})
metrics = demo.get("metrics", {})
trend = child_cases.get("trend", {}) or {}

# ---------------- KPIs — styled ----------------
k1, k2, k3 = st.columns(3)
latest_cases = list(trend.values())[-1] if trend else None

k1.metric("Selected District", lookup_district)
k2.metric("Latest Reported Cases", latest_cases if latest_cases else "N/A")
k3.metric("Months Available", len(trend))

# ---------------- TREND CHART — ORIGINAL UNTOUCHED ----------------
if trend:
    chart_df = pd.DataFrame({"Year": list(trend.keys()), "Reported Cases": list(trend.values())})
    chart_df["Reported Cases"] = pd.to_numeric(chart_df["Reported Cases"], errors="coerce")

    if rolling_window > 1:
        chart_df["Smoothed"] = chart_df["Reported Cases"].rolling(rolling_window, min_periods=1).mean()
        y_col = "Smoothed"
    else:
        y_col = "Reported Cases"

    if chart_type == "Bar":
        fig = px.bar(chart_df, x="Year", y=y_col, color_discrete_sequence=["#7c3aed"])
    else:
        fig = px.line(chart_df, x="Year", y=y_col, markers=True,
                      color_discrete_sequence=["#7c3aed"])
        fig.update_traces(line=dict(width=2.5), marker=dict(size=5, color="#7c3aed"))

    fig.update_layout(height=420, hovermode="x unified", **PLOTLY_LIGHT)

    with right:
        st.plotly_chart(fig, use_container_width=True)

# ---------------- DEMOGRAPHICS — styled, logic original ----------------
st.markdown("---")
st.markdown('<p style="font-size:1rem;font-weight:700;color:#1e1b4b;margin-bottom:1rem;font-family:Inter,sans-serif;">Demographics Snapshot</p>', unsafe_allow_html=True)

if metrics:
    col1, col2 = st.columns([1, 1.5])

    male   = metrics.get("MALE", 0)
    female = metrics.get("FEMALE", 0)
    total  = metrics.get("TOT_POP", male + female)

    with col1:
        st.markdown('<span class="card-label">Population Stats</span>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="pop-card male">
            <div class="pop-label">Male Population</div>
            <div class="pop-value pop-male">{int(male):,}</div>
        </div>
        <div class="pop-card female">
            <div class="pop-label">Female Population</div>
            <div class="pop-value pop-female">{int(female):,}</div>
        </div>
        <div class="pop-card total">
            <div class="pop-label">Total Population</div>
            <div class="pop-value pop-total">{int(total):,}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<span class="card-label">Gender Distribution</span>', unsafe_allow_html=True)
        pie_df = pd.DataFrame({
            "Gender": ["Male", "Female"],
            "Population": [male, female]
        })
        pie_fig = px.pie(
            pie_df, names="Gender", values="Population",
            hole=0.4,
            title="Male vs Female Population (%)",
            color_discrete_sequence=["#7c3aed", "#a78bfa"],
        )
        pie_fig.update_traces(
            textinfo="percent+label",
            pull=[0.03, 0.03],
            textfont=dict(family="Inter, sans-serif", size=13, color="#1e1b4b"),
        )
        pie_fig.update_layout(
            height=380,
            margin=dict(t=50, b=20, l=20, r=20),
            **PLOTLY_LIGHT,
        )
        st.plotly_chart(pie_fig, use_container_width=True)

else:
    st.info("No demographic data available for this district.")