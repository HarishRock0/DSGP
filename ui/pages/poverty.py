import sys
import os
import streamlit as st
import pandas as pd
import plotly.express as px

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from service.recommendation_service import RecommendationService

st.set_page_config(page_title="Poverty Analysis — Sri Lanka District Insights", layout="wide", initial_sidebar_state="collapsed")


# ── Force light theme via JS (overrides any user/system dark preference) ──────
st.markdown("""
<script>
(function() {
    // Remove dark class from html/body
    document.documentElement.removeAttribute('data-theme');
    document.documentElement.style.colorScheme = 'light';
    // Watch for dynamic dark-mode injection and strip it
    const obs = new MutationObserver(() => {
        document.documentElement.removeAttribute('data-theme');
        document.documentElement.style.colorScheme = 'light';
    });
    obs.observe(document.documentElement, { attributes: true });
})();
</script>
""", unsafe_allow_html=True)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ════════════════════════════════════════
   FORCE LIGHT — targets every layer Streamlit uses
   ════════════════════════════════════════ */
:root, html, body,
.stApp, .main, .block-container,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > section,
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
[data-testid="stVerticalBlock"],
[data-testid="stHorizontalBlock"],
[data-testid="column"],
[data-testid="stForm"],
[class*="css"] {
    background-color: #f4f2fb !important;
    color: #1e1b4b !important;
    font-family: 'Inter', sans-serif !important;
    color-scheme: light !important;
}

/* White surfaces */
[data-testid="stHeader"],
footer,
[data-testid="stDecoration"] { display: none !important; }

/* ════════ ALL TEXT ELEMENTS ════════ */
p, span, div, li, td, th, label,
h1, h2, h3, h4, h5, h6,
small, strong, em, code, pre,
[class*="st-"], [data-testid*="st"] {
    color: #1e1b4b !important;
    font-family: 'Inter', sans-serif !important;
}

/* Muted overrides — apply after above */
.muted { color: #6b7280 !important; }

/* ════════ STREAMLIT MARKDOWN ════════ */
.stMarkdown, .stMarkdown *,
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] * {
    color: #1e1b4b !important;
}

/* ════════ METRIC ════════ */
[data-testid="metric-container"] {
    background: #ffffff !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 12px !important;
    padding: 1rem 1.4rem !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06) !important;
}
[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] *,
[data-testid="stMetricLabel"] p {
    color: #6b7280 !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] * {
    color: #7c3aed !important;
    font-size: 1.45rem !important;
    font-weight: 700 !important;
}
[data-testid="stMetricDelta"],
[data-testid="stMetricDelta"] * { color: #6b7280 !important; }

/* ════════ INPUT ════════ */
input, textarea,
.stTextInput input,
[data-baseweb="input"] input,
[data-baseweb="textarea"] textarea {
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
    outline: none !important;
}
.stTextInput label, .stTextInput label * { color: #1e1b4b !important; }

/* ════════ BUTTON ════════ */
.stButton button {
    background: #7c3aed !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    font-family: 'Inter', sans-serif !important;
    box-shadow: 0 2px 8px rgba(124,58,237,0.3) !important;
}
.stButton button:hover { background: #6d28d9 !important; }
.stButton button *, .stButton button p { color: #ffffff !important; }

/* ════════ SELECT ════════ */
[data-baseweb="select"] > div,
[data-baseweb="select"] * {
    background: #ffffff !important;
    color: #1e1b4b !important;
    border-color: #e5e7eb !important;
    font-family: 'Inter', sans-serif !important;
}
[data-baseweb="popover"],
[data-baseweb="popover"] *,
[data-baseweb="menu"],
[data-baseweb="menu"] li,
[data-baseweb="menu"] ul {
    background: #ffffff !important;
    color: #1e1b4b !important;
}
[data-baseweb="option"]:hover { background: #f3e8ff !important; }
.stSelectbox label, .stSelectbox label * { color: #6b7280 !important; font-size: 0.8rem !important; font-weight: 500 !important; }
.stTextInput label, .stTextInput label * { color: #6b7280 !important; font-size: 0.8rem !important; font-weight: 500 !important; }

/* ════════ RADIO ════════ */
.stRadio label,
.stRadio label *,
.stRadio [data-testid="stMarkdownContainer"] p {
    color: #374151 !important;
    font-size: 0.85rem !important;
}

/* ════════ TOGGLE ════════ */
.stToggle label, .stToggle label * { color: #374151 !important; font-size: 0.85rem !important; }
[data-testid="stToggle"] p { color: #374151 !important; }

/* ════════ DATAFRAME / TABLE ════════ */
[data-testid="stDataFrame"],
[data-testid="stDataFrame"] > div,
[data-testid="stDataFrame"] iframe,
.stDataFrame, .stDataFrame > div {
    background: #ffffff !important;
    border-radius: 12px !important;
    border: 1px solid #e5e7eb !important;
    overflow: hidden;
}
/* Glide table cells (Streamlit uses this internally) */
.glide-data-grid *,
.dvn-scroller * { color: #1e1b4b !important; }



/* ════════ ALERT / INFO ════════ */
[data-testid="stAlert"],
[data-testid="stAlert"] * {
    background: #faf5ff !important;
    color: #1e1b4b !important;
    border: 1px solid #ddd6fe !important;
    border-radius: 10px !important;
}

/* ════════ SPINNER ════════ */
.stSpinner > div { color: #7c3aed !important; }

/* ════════════════════════════
   CUSTOM COMPONENTS
   ════════════════════════════ */
.page-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid #e5e7eb;
    margin-bottom: 2rem;
}
.page-title {
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    color: #1e1b4b !important;
    margin: 0;
}
.page-title span { color: #7c3aed !important; }
.page-subtitle { font-size: 0.875rem !important; color: #6b7280 !important; margin-top: 0.2rem; }
.page-badge {
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    color: #7c3aed !important;
    background: #ede9fe !important;
    padding: 0.35rem 0.9rem;
    border-radius: 100px;
}
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
    margin-bottom: 0.6rem;
    display: block;
}
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
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: #6b7280 !important;
}
.pop-value { font-size: 1.7rem !important; font-weight: 700 !important; margin-top: 0.15rem; line-height: 1; }
.pop-male   { color: #7c3aed !important; }
.pop-female { color: #7c3aed !important; }
.pop-total  { color: #059669 !important; }

hr { border: none !important; border-top: 1px solid #e5e7eb !important; margin: 1.5rem 0 !important; }
[data-testid="column"] { padding: 0 0.4rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Plotly light theme ──────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="#ffffff",
    plot_bgcolor="#f4f2fb",
    font=dict(family="Inter, sans-serif", color="#374151", size=12),
    title_font=dict(family="Inter, sans-serif", color="#1e1b4b", size=15, weight="bold"),
    xaxis=dict(gridcolor="#e5e7eb", linecolor="#e5e7eb",
               tickfont=dict(color="#374151"), title_font=dict(color="#374151")),
    yaxis=dict(gridcolor="#e5e7eb", linecolor="#e5e7eb",
               tickfont=dict(color="#374151"), title_font=dict(color="#374151")),
    legend=dict(bgcolor="#ffffff", bordercolor="#e5e7eb", borderwidth=1,
                font=dict(color="#374151")),
    hoverlabel=dict(bgcolor="#ffffff", font=dict(color="#1e1b4b", family="Inter")),
)

# ==================== SERVICE ====================
@st.cache_resource
def load_service():
    return RecommendationService()

service = load_service()

# ==================== POVERTY NAV SWITCHER ====================
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
.pov-nav-wrap {
    width: 100%;
    display: flex;
    justify-content: center;
    padding: 32px 0 20px;
}
.pov-nav {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: rgba(124,58,237,0.07);
    border: 1px solid rgba(124,58,237,0.18);
    border-radius: 100px;
    padding: 4px;
}
.pov-nav a {
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
.pov-nav a:hover {
    background: rgba(124,58,237,0.10);
    color: #6d28d9;
    text-decoration: none !important;
}
.pov-nav a.active {
    background: linear-gradient(135deg,#7c3aed,#a855f7);
    color: #fff !important;
    box-shadow: 0 4px 14px rgba(124,58,237,0.35);
}
</style>
<div class="home-btn"><a href="/" target="_self">🏠&nbsp; Home</a></div>
<div class="pov-nav-wrap">
  <div class="pov-nav">
    <a href="/poverty" class="active" target="_self">📊&nbsp; Regional Insights</a>
    <a href="/poverty_resource" target="_self">💰&nbsp; Resource Allocation</a>
  </div>
</div>
""", unsafe_allow_html=True)

# ==================== PAGE HEADER ====================
st.markdown("""
<div class="page-header">
    <div>
        <div class="page-title">Poverty <span>Analysis</span></div>
        <div class="page-subtitle">NLP-driven region recommendations with poverty &amp; demographic analytics</div>
    </div>
    <div class="page-badge">💰 Economic Analysis</div>
</div>
""", unsafe_allow_html=True)

if "recommendations" not in st.session_state:
    st.session_state.recommendations = None

# ==================== SEARCH ====================
st.markdown('<div class="search-card">', unsafe_allow_html=True)
st.markdown('<span class="card-label">Describe your preference</span>', unsafe_allow_html=True)
scol1, scol2 = st.columns([4, 1])
with scol1:
    user_input = st.text_input(
        "Preference",
        placeholder="e.g., low poverty, high population, urban areas",
        label_visibility="hidden"
    )
with scol2:
    st.markdown("<div style='height:1.65rem'></div>", unsafe_allow_html=True)
    run_btn = st.button("Get Recommendations", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

if run_btn:
    if not user_input.strip():
        st.warning("Please enter a preference.")
    else:
        with st.spinner("Analysing preferences..."):
            result = service.get_recommendations(user_input)
        st.session_state.recommendations = result.get("recommendations", [])

# ==================== RECOMMENDATIONS ====================
recs = st.session_state.recommendations
if not recs:
    st.markdown("""
    <div style="text-align:center;padding:3rem 1rem;">
        <div style="font-size:0.95rem;color:#6b7280;font-family:Inter,sans-serif;">
            Enter a preference and click
            <strong style="color:#7c3aed;">Get Recommendations</strong> to begin.
        </div>
    </div>""", unsafe_allow_html=True)
    st.stop()

rec_df = pd.DataFrame(recs)
st.markdown('<span class="card-label">Top Recommended Regions</span>', unsafe_allow_html=True)
st.dataframe(rec_df, use_container_width=True, height=240)

districts = [r["District"] for r in recs if isinstance(r, dict) and "District" in r]
if not districts:
    st.error("No District names found in recommendations output.")
    st.stop()

st.markdown("<hr>", unsafe_allow_html=True)

# ==================== MONTHLY DASHBOARD ====================
st.markdown('<p style="font-size:1rem;font-weight:700;color:#1e1b4b;margin-bottom:1rem;font-family:Inter,sans-serif;">Monthly Poverty &amp; Demographics Dashboard</p>', unsafe_allow_html=True)

left, right = st.columns([1, 2.3])

with left:
    st.markdown('<div class="white-card">', unsafe_allow_html=True)
    st.markdown('<span class="card-label">Select Region</span>', unsafe_allow_html=True)
    selected_district = st.selectbox("Region", districts, label_visibility="hidden")
    st.markdown('<span class="card-label" style="margin-top:1rem;display:block;">Chart Controls</span>', unsafe_allow_html=True)
    chart_type = st.radio("Chart type", ["Bar", "Line", "Bar + Line"], horizontal=True, key="chart_type_radio")
    show_rangeslider = st.toggle("Range slider", value=True, key="rangeslider_toggle")
    use_log_y = st.toggle("Log scale (Y)", value=False, key="logy_toggle")
    rolling_window = st.selectbox("Smoothing (months)", [1, 2, 3, 6], index=0, label_visibility="visible", key="smoothing_select")
    st.markdown('</div>', unsafe_allow_html=True)

with st.spinner("Loading insights..."):
    insights = service.get_insights(selected_district)

poverty    = insights.get("poverty", {})
demo       = insights.get("demographics", {})
metrics    = demo.get("metrics") or demo.get("row", {})
trend_dict = poverty.get("trend", {}) or {}
latest     = poverty.get("latest", None)

k1, k2, k3 = st.columns(3)
k1.metric("Selected Region", selected_district)
k2.metric("Latest Poverty Line", "N/A" if latest is None else f"{latest:,.2f}")
k3.metric("Months Available", len(trend_dict))

st.markdown("<div style='margin-bottom:1rem'></div>", unsafe_allow_html=True)

# ==================== CHART ====================
if not trend_dict:
    st.info("No poverty trend data available for this district.")
else:
    chart_df = pd.DataFrame({"Month": list(trend_dict.keys()), "PovertyLine": list(trend_dict.values())})
    chart_df["Month"] = pd.to_datetime(chart_df["Month"], errors="coerce")
    chart_df["PovertyLine"] = pd.to_numeric(chart_df["PovertyLine"], errors="coerce")
    chart_df = chart_df.dropna(subset=["Month", "PovertyLine"]).sort_values("Month")

    if chart_df.empty:
        st.info("Monthly poverty trend values are not parseable (expected format: 2025-01).")
    else:
        if rolling_window > 1:
            chart_df["Smoothed"] = chart_df["PovertyLine"].rolling(rolling_window, min_periods=1).mean()
            y_col, y_name = "Smoothed", f"Poverty Line (smoothed {rolling_window}m)"
        else:
            y_col, y_name = "PovertyLine", "Poverty Line"

        chart_df["PctChange"] = chart_df[y_col].pct_change() * 100
        labels = {"Month": "Month", y_col: y_name}

        if chart_type == "Line":
            fig = px.line(chart_df, x="Month", y=y_col,
                          title=f"Monthly Poverty Trend — {selected_district}",
                          labels=labels, hover_data={"PctChange": ":.2f"},
                          color_discrete_sequence=["#7c3aed"])
            fig.update_traces(line=dict(width=2.5), mode="lines+markers",
                              marker=dict(size=5, color="#7c3aed"))
        elif chart_type == "Bar":
            fig = px.bar(chart_df, x="Month", y=y_col,
                         title=f"Monthly Poverty Trend — {selected_district}",
                         labels=labels, hover_data={y_col: ":,.2f", "PctChange": ":.2f"},
                         color_discrete_sequence=["#7c3aed"])
        else:
            fig = px.bar(chart_df, x="Month", y=y_col,
                         title=f"Monthly Poverty Trend — {selected_district}",
                         labels=labels, hover_data={y_col: ":,.2f", "PctChange": ":.2f"},
                         color_discrete_sequence=["#ede9fe"])
            fig.add_scatter(x=chart_df["Month"], y=chart_df[y_col],
                            mode="lines+markers", name="Trend",
                            line=dict(color="#7c3aed", width=2.5),
                            marker=dict(size=5, color="#7c3aed"))

        fig.update_layout(height=440, hovermode="x unified",
                          margin=dict(l=10, r=10, t=50, b=10), **PLOTLY_LAYOUT)
        fig.update_xaxes(tickformat="%Y-%m", dtick="M1",
                         rangeslider=dict(visible=show_rangeslider))
        if use_log_y:
            fig.update_yaxes(type="log")

        with right:
            st.plotly_chart(fig, use_container_width=True)



st.markdown("<hr>", unsafe_allow_html=True)

# ==================== DEMOGRAPHICS ====================
st.markdown('<p style="font-size:1rem;font-weight:700;color:#1e1b4b;margin-bottom:1rem;font-family:Inter,sans-serif;">Demographics Snapshot</p>', unsafe_allow_html=True)

if metrics:
    col1, col2 = st.columns([1, 1.6])
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
        pie_df = pd.DataFrame({"Gender": ["Male", "Female"], "Population": [male, female]})
        pie_fig = px.pie(pie_df, names="Gender", values="Population", hole=0.48,
                         color_discrete_sequence=["#7c3aed", "#a78bfa"])
        pie_fig.update_traces(textinfo="percent+label", pull=[0.03, 0.03],
                              textfont=dict(family="Inter, sans-serif", size=13, color="#1e1b4b"))
        pie_fig.update_layout(height=340, margin=dict(t=20, b=20, l=20, r=20), **PLOTLY_LAYOUT)
        st.plotly_chart(pie_fig, use_container_width=True)
else:
    st.info("No demographic data available for this district.")