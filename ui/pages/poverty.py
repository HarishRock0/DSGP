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

@st.cache_resource
def load_service():
    return RecommendationService()

service = load_service()

st.title(" Intelligent Region Recommendation Dashboard")
st.caption("NLP recommendations + interactive poverty line insights (poverty + demographics only)")

# Persist recommendations across reruns
if "recommendations" not in st.session_state:
    st.session_state.recommendations = None

# ---------------------------
# Inputs
# ---------------------------
with st.container():
    user_input = st.text_input("Describe your preference:", placeholder="e.g., low poverty, high population, urban areas")

    colA, colB = st.columns([1, 4])
    with colA:
        run_btn = st.button("Get Recommendations", use_container_width=True)

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
if recs:
    rec_df = pd.DataFrame(recs)

    st.subheader(" Top Recommended Regions")
    st.dataframe(rec_df, use_container_width=True, height=280)

    districts = [r["District"] for r in recs if isinstance(r, dict) and "District" in r]
    if not districts:
        st.error("No District names found in recommendations output.")
        st.stop()

    st.divider()

    # ---------------------------
    # Dashboard Controls
    # ---------------------------
    st.subheader(" Poverty & Demographics Dashboard")

    left, right = st.columns([1, 2])

    with left:
        selected_district = st.selectbox("Select a region:", districts)

        st.markdown("##### Chart controls")
        show_points = st.toggle("Show data points", value=False)
        show_rangeslider = st.toggle("Show range slider", value=True)
        use_log_y = st.toggle("Log scale (Y)", value=False)
        rolling_window = st.selectbox("Smoothing (rolling mean)", [1, 3, 5, 7], index=0)

    # Generate insights ONLY after a region is selected
    with st.spinner("Generating insights..."):
        insights = service.get_insights(selected_district)

    poverty = insights.get("poverty", {})
    demo = insights.get("demographics", {})

    # ---------------------------
    # KPIs
    # ---------------------------
    k1, k2, k3, k4 = st.columns(4)

    latest = poverty.get("latest", None)
    trend_dict = poverty.get("trend", {}) or {}

    # compute change if possible
    change = None
    if isinstance(trend_dict, dict) and len(trend_dict) >= 2:
        try:
            vals = [float(v) for v in trend_dict.values() if v is not None]
            if len(vals) >= 2:
                change = vals[-1] - vals[0]
        except Exception:
            change = None

    k1.metric("Selected Region", selected_district)
    k2.metric("Latest Poverty Line", "N/A" if latest is None else f"{latest:,.2f}")
    k3.metric("Periods Available", len(trend_dict))
    k4.metric("Change (first → last)", "N/A" if change is None else f"{change:,.2f}")

    # ---------------------------
    # Build Poverty Chart DataFrame
    # ---------------------------
    if not trend_dict:
        st.info("No poverty trend data available for this district.")
    else:
        chart_df = pd.DataFrame(
            {"Period": list(trend_dict.keys()), "PovertyLine": list(trend_dict.values())}
        )

        # Try to sort Period intelligently (year-like), otherwise keep order
        def _to_year(x):
            try:
                # handles "2019" or 2019
                return int(str(x).strip()[:4])
            except Exception:
                return None

        years = chart_df["Period"].apply(_to_year)
        if years.notna().all():
            chart_df["Year"] = years
            chart_df = chart_df.sort_values("Year")
            x_col = "Year"
        else:
            # fallback: keep as string order
            chart_df["Period"] = chart_df["Period"].astype(str)
            x_col = "Period"

        # Numeric
        chart_df["PovertyLine"] = pd.to_numeric(chart_df["PovertyLine"], errors="coerce")
        chart_df = chart_df.dropna(subset=["PovertyLine"])

        if chart_df.empty:
            st.info("Poverty trend values are not numeric or are missing.")
        else:
            # Optional smoothing
            if rolling_window > 1:
                chart_df["Smoothed"] = chart_df["PovertyLine"].rolling(rolling_window, min_periods=1).mean()
                y_col = "Smoothed"
                y_name = f"Poverty Line (smoothed, window={rolling_window})"
            else:
                y_col = "PovertyLine"
                y_name = "Poverty Line"

            # Plotly interactive line chart
            fig = px.line(
                chart_df,
                x=x_col,
                y=y_col,
                markers=show_points,
                title=f"Poverty Line Trend — {selected_district}",
                labels={x_col: "Year" if x_col == "Year" else "Period", y_col: y_name},
            )

            fig.update_layout(
                height=420,
                margin=dict(l=20, r=20, t=60, b=20),
                hovermode="x unified",
            )

            if use_log_y:
                fig.update_yaxes(type="log")

            if show_rangeslider:
                fig.update_xaxes(rangeslider=dict(visible=True))

            with right:
                st.plotly_chart(fig, use_container_width=True)

            # Optional: show raw table under chart
            with st.expander("Show poverty trend data"):
                st.dataframe(chart_df, use_container_width=True)

    # ---------------------------
    # Demographics Panel
    # ---------------------------
    st.markdown("### Demographics Snapshot")
    if isinstance(demo, dict) and demo:
        # If your demographics has huge columns, you can filter here
        demo_df = pd.DataFrame([demo])
        st.dataframe(demo_df, use_container_width=True)
    else:
        st.info("No demographic data available for this district.")

else:
    st.info("Enter a preference and click **Get Recommendations** to begin.")
