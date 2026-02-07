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
# st.set_page_config(page_title="Child Protection Risk Dashboard", layout="wide")
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
# ##  Child Protection Risk Recommendation Dashboard
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
#     selected_district = st.selectbox(" Select District", districts)
#     chart_type = st.radio("Chart Type", ["Line", "Bar"], horizontal=True)
#     rolling_window = st.selectbox("Trend Smoothing (years)", [1, 2, 3], index=0)
#
# # ---------------- INSIGHTS ----------------
# with st.spinner("Generating district insights..."):
#     insights = service.get_child_insights(selected_district)
#
# child_cases = insights.get("child_cases", {})
# demo = insights.get("demographics", {})
# metrics = demo.get("metrics", {})
#
# trend = child_cases.get("trend", {}) or {}
#
# # ---------------- KPIs ----------------
# k1, k2, k3 = st.columns(3)
#
# latest_cases = list(trend.values())[-1] if trend else None
# risk_level = "High" if latest_cases and latest_cases > 150 else "Moderate" if latest_cases else "Unknown"
#
# k1.metric("Selected District", selected_district)
# k2.metric("Latest Reported Cases", latest_cases if latest_cases else "N/A")
# k3.metric("Risk Level", risk_level)
#
# # ---------------- TREND VISUALIZATION ----------------
# if not trend:
#     st.info("No historical child case trend data available.")
# else:
#     chart_df = pd.DataFrame({
#         "Year": list(trend.keys()),
#         "Reported Cases": list(trend.values())
#     })
#
#     chart_df["Reported Cases"] = pd.to_numeric(chart_df["Reported Cases"], errors="coerce")
#
#     if rolling_window > 1:
#         chart_df["Smoothed"] = chart_df["Reported Cases"].rolling(
#             rolling_window, min_periods=1
#         ).mean()
#         y_col = "Smoothed"
#     else:
#         y_col = "Reported Cases"
#
#     if chart_type == "Line":
#         fig = px.line(
#             chart_df,
#             x="Year",
#             y=y_col,
#             title=f"Child Protection Case Trend — {selected_district}",
#             markers=True
#         )
#     else:
#         fig = px.bar(
#             chart_df,
#             x="Year",
#             y=y_col,
#             title=f"Child Protection Case Trend — {selected_district}"
#         )
#
#     fig.update_layout(height=420, hovermode="x unified")
#
#     with right:
#         st.plotly_chart(fig, use_container_width=True)
#
# # ---------------- DEMOGRAPHICS ----------------
# st.divider()
# st.subheader(" Demographic Snapshot")
#
# if metrics:
#     col1, col2 = st.columns([1, 1.5])
#
#     male = metrics.get("MALE", 0)
#     female = metrics.get("FEMALE", 0)
#     total = metrics.get("TOT_POP", male + female)
#
#     with col1:
#         st.metric("Male Population", f"{int(male):,}")
#         st.metric("Female Population", f"{int(female):,}")
#         st.metric("Total Population", f"{int(total):,}")
#
#     with col2:
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
#             title="Gender Distribution"
#         )
#
#         st.plotly_chart(pie_fig, use_container_width=True)
# else:
#     st.info("No demographic data available for this district.")
#

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

st.set_page_config(page_title="Child Protection Risk Dashboard", layout="wide")

# ---------------- CSS (MATCH poverty.py incl. DEMOGRAPHICS) ----------------
st.markdown("""
<style>
/* ---------- GLOBAL ---------- */
html, body, [class*="css"] {
    font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont;
}

.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #020617 100%);
    color: #e5e7eb;
}

/* ---------- HEADERS ---------- */
h1, h2, h3 {
    font-weight: 700;
    letter-spacing: 0.3px;
}

h1 { color: #38bdf8; }
h3 { color: #e5e7eb; }

/* ---------- INPUTS ---------- */
.stTextInput > div > div > input {
    background-color: #020617;
    border-radius: 12px;
    border: 1px solid #1e293b;
    color: #e5e7eb;
    padding: 12px;
}

/* ---------- BUTTONS ---------- */
.stButton > button {
    background: linear-gradient(90deg, #38bdf8, #0ea5e9);
    color: black;
    border-radius: 14px;
    padding: 10px 22px;
    font-weight: 600;
    border: none;
    transition: transform 0.15s ease;
}

.stButton > button:hover {
    transform: scale(1.03);
}

/* ---------- METRIC CARDS ---------- */
[data-testid="metric-container"] {
    background: rgba(2, 6, 23, 0.9);
    border: 1px solid #1e293b;
    padding: 20px;
    border-radius: 18px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.4);
}

[data-testid="metric-container"] label {
    color: #94a3b8;
}

[data-testid="metric-container"] div {
    color: #38bdf8;
    font-weight: 700;
}

/* ---------- DATAFRAME ---------- */
.stDataFrame {
    background-color: #020617;
    border-radius: 14px;
    border: 1px solid #1e293b;
}

/* ---------- SELECT / RADIO ---------- */
.stSelectbox, .stRadio {
    background-color: #020617;
    border-radius: 12px;
}

/* ---------- DIVIDERS ---------- */
hr {
    border: none;
    border-top: 1px solid #1e293b;
    margin: 30px 0;
}

/* ---------- POPULATION STATS (FROM poverty.py) ---------- */
.pop-card {
    background: linear-gradient(145deg, #020617, #020617);
    border: 1px solid #1e293b;
    border-radius: 18px;
    padding: 18px 22px;
    margin-bottom: 14px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.45);
}

.pop-label {
    color: #94a3b8;
    font-size: 0.85rem;
    letter-spacing: 0.4px;
}

.pop-value {
    font-size: 1.8rem;
    font-weight: 800;
    margin-top: 4px;
}

.pop-male { color: #38bdf8; }
.pop-female { color: #f472b6; }
.pop-total { color: #22c55e; }
</style>
""", unsafe_allow_html=True)

# ---------------- SERVICE CACHE ----------------
@st.cache_resource
def load_service():
    return ChildProtectionService(PROJECT_ROOT)

service = load_service()

# ---------------- HEADER ----------------
st.markdown("""
## Child Protection Risk Recommendation Dashboard
<span style="color:#94a3b8">
NLP-driven regional risk analysis with interactive child case & demographic insights
</span>
""", unsafe_allow_html=True)

# ---------------- SESSION STATE ----------------
if "child_recs" not in st.session_state:
    st.session_state.child_recs = None

# ---------------- INPUT ----------------
user_input = st.text_input(
    "Describe your concern:",
    placeholder="e.g. child abuse risk, neglected children, vulnerable regions"
)

run_btn = st.button("Analyze Child Protection Risk")

if run_btn:
    if not user_input.strip():
        st.warning("Please describe a concern.")
    else:
        with st.spinner("Analyzing child protection signals..."):
            result = service.get_child_case_recommendations(user_input)
        st.session_state.child_recs = result.get("recommendations", [])

# ---------------- RECOMMENDATIONS ----------------
recs = st.session_state.child_recs
if not recs:
    st.info("Enter a concern and run the analysis to see recommendations.")
    st.stop()

rec_df = pd.DataFrame(recs)
st.subheader("Top At-Risk Districts")
st.dataframe(rec_df, use_container_width=True, height=260)

districts = rec_df["District"].tolist()
st.divider()

# ---------------- DASHBOARD ----------------
st.subheader("Child Protection Risk Dashboard")

left, right = st.columns([1, 2])

with left:
    selected_district = st.selectbox("Select District", districts)
    chart_type = st.radio("Chart Type", ["Line", "Bar"], horizontal=True)
    rolling_window = st.selectbox("Trend Smoothing (years)", [1, 2, 3], index=0)

# ---------------- INSIGHTS ----------------
with st.spinner("Generating district insights..."):
    insights = service.get_child_insights(selected_district)

child_cases = insights.get("child_cases", {})
demo = insights.get("demographics", {})
metrics = demo.get("metrics", {})
trend = child_cases.get("trend", {}) or {}

# ---------------- KPIs ----------------
k1, k2, k3 = st.columns(3)
latest_cases = list(trend.values())[-1] if trend else None
risk_level = "High" if latest_cases and latest_cases > 150 else "Moderate" if latest_cases else "Unknown"

k1.metric("Selected District", selected_district)
k2.metric("Latest Reported Cases", latest_cases if latest_cases else "N/A")
k3.metric("Risk Level", risk_level)

# ---------------- TREND VISUALIZATION ----------------
if trend:
    chart_df = pd.DataFrame({"Year": list(trend.keys()), "Reported Cases": list(trend.values())})
    chart_df["Reported Cases"] = pd.to_numeric(chart_df["Reported Cases"], errors="coerce")

    if rolling_window > 1:
        chart_df["Smoothed"] = chart_df["Reported Cases"].rolling(rolling_window, min_periods=1).mean()
        y_col = "Smoothed"
    else:
        y_col = "Reported Cases"

    fig = px.line(chart_df, x="Year", y=y_col, markers=True)
    fig.update_layout(height=420, hovermode="x unified")

    with right:
        st.plotly_chart(fig, use_container_width=True)

# ---------------- DEMOGRAPHICS (NOW IDENTICAL TO poverty.py) ----------------
st.divider()
st.subheader("Demographics Snapshot")

if metrics:
    col1, col2 = st.columns([1, 1.5])

    male = metrics.get("MALE", 0)
    female = metrics.get("FEMALE", 0)
    total = metrics.get("TOT_POP", male + female)

    with col1:
        st.markdown(f"""
        <div class="pop-card">
            <div class="pop-label">Male Population</div>
            <div class="pop-value pop-male">{int(male):,}</div>
        </div>

        <div class="pop-card">
            <div class="pop-label">Female Population</div>
            <div class="pop-value pop-female">{int(female):,}</div>
        </div>

        <div class="pop-card">
            <div class="pop-label">Total Population</div>
            <div class="pop-value pop-total">{int(total):,}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        pie_df = pd.DataFrame({
            "Gender": ["Male", "Female"],
            "Population": [male, female]
        })

        pie_fig = px.pie(
            pie_df,
            names="Gender",
            values="Population",
            hole=0.4,
            title="Male vs Female Population (%)"
        )

        pie_fig.update_traces(textinfo="percent+label", pull=[0.03, 0.03])
        pie_fig.update_layout(height=380)

        st.plotly_chart(pie_fig, use_container_width=True)
else:
    st.info("No demographic data available for this district.")
