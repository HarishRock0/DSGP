import sys
import os
import streamlit as st
import pandas as pd
import plotly.express as px

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from service.recommendation_service import RecommendationService

st.set_page_config(page_title="Region Recommendation Dashboard", layout="wide")

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

h1 {
    color: #38bdf8;
}

h3 {
    color: #e5e7eb;
}

/* ---------- INPUTS ---------- */
.stTextInput > div > div > input {
    background-color: #020617;
    border-radius: 12px;
    border: 1px solid #1e293b;
    color: #e5e7eb;
    padding: 12px;
}

.stButton > button {
    background: linear-gradient(90deg, #38bdf8, #0ea5e9);
    color: black;
    border-radius: 14px;
    padding: 10px 22px;
    font-weight: 600;
    border: none;
}

.stButton > button:hover {
    transform: scale(1.03);
    background: linear-gradient(90deg, #0ea5e9, #38bdf8);
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

/* ---------- SELECT / RADIO / TOGGLE ---------- */
.stSelectbox, .stRadio, .stToggle {
    background-color: #020617;
    border-radius: 12px;
}

/* ---------- SIDEBAR-LIKE CARDS ---------- */
.control-card {
    background: rgba(2, 6, 23, 0.85);
    border: 1px solid #1e293b;
    padding: 18px;
    border-radius: 18px;
    margin-bottom: 16px;
}

/* ---------- DIVIDERS ---------- */
hr {
    border: none;
    border-top: 1px solid #1e293b;
    margin: 30px 0;
}
/* ---------- POPULATION STATS CARDS ---------- */
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

.pop-male {
    color: #38bdf8;
}

.pop-female {
    color: #f472b6;
}

.pop-total {
    color: #22c55e;
}

</style>
""", unsafe_allow_html=True)



@st.cache_resource
def load_service():
    return RecommendationService()


service = load_service()


st.markdown("""
## 🧠 Intelligent Region Recommendation Dashboard
<span style="color:#94a3b8">
NLP-driven recommendations with interactive poverty & demographic analytics
</span>
""", unsafe_allow_html=True)


# Persist recommendations across reruns
if "recommendations" not in st.session_state:
    st.session_state.recommendations = None

# ---------------------------
# Inputs
# ---------------------------
user_input = st.text_input(
    "Describe your preference:",
    placeholder="e.g., low poverty, high population, urban areas"
)
run_btn = st.button("Get Recommendations")

if run_btn:
    if user_input.strip() == "":
        st.warning("Please enter a preference.")
    else:
        with st.spinner("Analyzing preferences..."):
            result = service.get_recommendations(user_input)
        st.session_state.recommendations = result.get("recommendations", [])

# ---------------------------
# Recommendations Table
# ---------------------------
recs = st.session_state.recommendations
if not recs:
    st.info("Enter a preference and click **Get Recommendations** to begin.")
    st.stop()

rec_df = pd.DataFrame(recs)
st.subheader("Top Recommended Regions")
st.dataframe(rec_df, use_container_width=True, height=260)

districts = [r["District"] for r in recs if isinstance(r, dict) and "District" in r]
if not districts:
    st.error("No District names found in recommendations output.")
    st.stop()

st.divider()

# ---------------------------
# Monthly Dashboard
# ---------------------------
st.subheader("Monthly Poverty & Demographics Dashboard")

left, right = st.columns([1, 2])

with left:
    st.markdown('<div class="control-card">', unsafe_allow_html=True)

    selected_district = st.selectbox("📍 Select a region", districts)
    st.markdown("#####  Chart Controls")
    chart_type = st.radio("Chart type", ["Bar", "Line", "Bar + Line"], horizontal=True)
    show_rangeslider = st.toggle("Show range slider", value=True)
    use_log_y = st.toggle("Log scale (Y)", value=False)
    rolling_window = st.selectbox("Smoothing (rolling months)", [1, 2, 3, 6], index=0)

    st.markdown('</div>', unsafe_allow_html=True)


# Generate insights ONLY after a region is selected
with st.spinner("Generating insights..."):
    insights = service.get_insights(selected_district)

poverty = insights.get("poverty", {})
demo = insights.get("demographics", {})
metrics = demo.get("metrics") or demo.get("row", {})


trend_dict = poverty.get("trend", {}) or {}
latest = poverty.get("latest", None)

# ---------------------------
# KPIs
# ---------------------------
k1, k2, k3 = st.columns(3)
k1.metric("Selected Region", selected_district)
k2.metric("Latest Poverty Line", "N/A" if latest is None else f"{latest:,.2f}")
k3.metric("Months Available", len(trend_dict))

# ---------------------------
# Chart data (MONTHLY)
# Expect Period like: "2025-01", "2025-02", ... OR any parseable month string
# ---------------------------
if not trend_dict:
    st.info("No poverty trend data available for this district.")
else:
    chart_df = pd.DataFrame(
        {"Month": list(trend_dict.keys()), "PovertyLine": list(trend_dict.values())}
    )

    # Parse month
    chart_df["Month"] = pd.to_datetime(chart_df["Month"], errors="coerce")
    chart_df["PovertyLine"] = pd.to_numeric(chart_df["PovertyLine"], errors="coerce")
    chart_df = chart_df.dropna(subset=["Month", "PovertyLine"]).sort_values("Month")

    if chart_df.empty:
        st.info("Monthly poverty trend values are missing or not parseable (expect formats like 2025-01).")
    else:
        # smoothing (monthly rolling mean)
        if rolling_window > 1:
            chart_df["Smoothed"] = chart_df["PovertyLine"].rolling(rolling_window, min_periods=1).mean()
            y_col = "Smoothed"
            y_name = f"Poverty Line (smoothed {rolling_window}m)"
        else:
            y_col = "PovertyLine"
            y_name = "Poverty Line"

        # % change for hover
        chart_df["PctChange"] = chart_df[y_col].pct_change() * 100

        labels = {"Month": "Month", y_col: y_name}

        if chart_type == "Line":
            fig = px.line(
                chart_df,
                x="Month",
                y=y_col,
                title=f"Monthly Poverty Trend — {selected_district}",
                labels=labels,
                hover_data={"PctChange": ":.2f"},
            )

        elif chart_type == "Bar":
            fig = px.bar(
                chart_df,
                x="Month",
                y=y_col,
                title=f"Monthly Poverty Trend — {selected_district}",
                labels=labels,
                hover_data={y_col: ":,.2f", "PctChange": ":.2f"},
            )

        else:  # Bar + Line
            fig = px.bar(
                chart_df,
                x="Month",
                y=y_col,
                title=f"Monthly Poverty Trend — {selected_district}",
                labels=labels,
                hover_data={y_col: ":,.2f", "PctChange": ":.2f"},
            )
            fig.add_scatter(
                x=chart_df["Month"],
                y=chart_df[y_col],
                mode="lines+markers",
                name="Trend Line",
            )

        fig.update_layout(height=460, hovermode="x unified", margin=dict(l=20, r=20, t=60, b=20))

        # Monthly-friendly tick labels
        fig.update_xaxes(
            tickformat="%Y-%m",
            dtick="M1",
            rangeslider=dict(visible=show_rangeslider),
        )

        if use_log_y:
            fig.update_yaxes(type="log")

        with right:
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("Show monthly poverty trend data"):
            st.dataframe(chart_df, use_container_width=True)


# ---------------------------
# Demographics
# ---------------------------
st.markdown("### Demographics Snapshot")

if metrics:
    col1, col2 = st.columns([1, 1.5])

    male = metrics.get("MALE", 0)
    female = metrics.get("FEMALE", 0)
    total = metrics.get("TOT_POP", male + female)

    # -------- COL 1 : Metrics --------
    with col1:
        st.subheader("📊 Population Stats")

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

    # -------- COL 2 : Pie Chart --------
    with col2:
        st.subheader("Gender Distribution")

        pie_df = pd.DataFrame({
            "Gender": ["Male", "Female"],
            "Population": [male, female]
        })

        pie_fig = px.pie(
            pie_df,
            names="Gender",
            values="Population",
            hole=0.4,   # donut style
            title="Male vs Female Population (%)"
        )

        pie_fig.update_traces(
            textinfo="percent+label",
            pull=[0.03, 0.03]
        )

        pie_fig.update_layout(
            height=380,
            margin=dict(t=50, b=20, l=20, r=20)
        )

        st.plotly_chart(pie_fig, use_container_width=True)

else:
    st.info("No demographic data available for this district.")
