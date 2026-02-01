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

# ---------------- SERVICE CACHE ----------------
@st.cache_resource
def load_service():
    return ChildProtectionService(PROJECT_ROOT)

service = load_service()

# ---------------- HEADER ----------------
st.markdown("""
##  Child Protection Risk Recommendation Dashboard
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
st.subheader("🔍 Top At-Risk Districts")
st.dataframe(rec_df, use_container_width=True, height=260)

districts = rec_df["District"].tolist()
st.divider()

# ---------------- DASHBOARD ----------------
st.subheader("Child Protection Risk Dashboard")

left, right = st.columns([1, 2])

with left:
    selected_district = st.selectbox(" Select District", districts)
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
if not trend:
    st.info("No historical child case trend data available.")
else:
    chart_df = pd.DataFrame({
        "Year": list(trend.keys()),
        "Reported Cases": list(trend.values())
    })

    chart_df["Reported Cases"] = pd.to_numeric(chart_df["Reported Cases"], errors="coerce")

    if rolling_window > 1:
        chart_df["Smoothed"] = chart_df["Reported Cases"].rolling(
            rolling_window, min_periods=1
        ).mean()
        y_col = "Smoothed"
    else:
        y_col = "Reported Cases"

    if chart_type == "Line":
        fig = px.line(
            chart_df,
            x="Year",
            y=y_col,
            title=f"Child Protection Case Trend — {selected_district}",
            markers=True
        )
    else:
        fig = px.bar(
            chart_df,
            x="Year",
            y=y_col,
            title=f"Child Protection Case Trend — {selected_district}"
        )

    fig.update_layout(height=420, hovermode="x unified")

    with right:
        st.plotly_chart(fig, use_container_width=True)

# ---------------- DEMOGRAPHICS ----------------
st.divider()
st.subheader(" Demographic Snapshot")

if metrics:
    col1, col2 = st.columns([1, 1.5])

    male = metrics.get("MALE", 0)
    female = metrics.get("FEMALE", 0)
    total = metrics.get("TOT_POP", male + female)

    with col1:
        st.metric("Male Population", f"{int(male):,}")
        st.metric("Female Population", f"{int(female):,}")
        st.metric("Total Population", f"{int(total):,}")

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
            title="Gender Distribution"
        )

        st.plotly_chart(pie_fig, use_container_width=True)
else:
    st.info("No demographic data available for this district.")
