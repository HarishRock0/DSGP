from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from service.recommendation_service import RecommendationService

st.set_page_config(
    page_title="Poverty Resource Allocation",
    page_icon="🇱🇰",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;0,700;1,400;1,600&family=Outfit:wght@300;400;500;600;700&display=swap');
:root {
    --bg:       #ffffff; --surface: #ffffff; --border: #d8c9f5;
    --accent:   #7c3aed; --text:    #000000; --muted:  #555555;
    --critical: #9b1dff; --high:    #6d28d9; --moderate:#a78bfa; --low:#4f46e5;
}
    html,body,[class*="css"] { 
        font-family:'DM Sans',sans-serif; 
        background:var(--bg); color:var(--text); 
    }
    .stApp, .stApp > div, [data-testid="stAppViewContainer"], 
    [data-testid="stAppViewBlockContainer"], [data-testid="block-container"] {
        background:#ffffff !important;
    }
    .stApp, .stApp p, .stApp div, .stApp span, .stApp label {
        color:#000000 !important;
    }
    [data-testid="stPlotlyChart"] > div, .js-plotly-plot .plotly,
    .js-plotly-plot .plotly .bg { 
        background:#ffffff !important; 
    }
    .metric-card { background:#ffffff !important; }
    .control-panel { background:#ffffff !important; }
    .tier-box { background:#ffffff !important; }
    #MainMenu,footer,header {
        visibility:hidden; 
    }
    section[data-testid="stSidebar"] { 
        display:none!important; 
    }
    button[data-testid="collapsedControl"] {
        display:none!important; 
    }
    .block-container { 
            padding:1.8rem 2rem 4rem!important; 
            max-width:100%!important; 
    }

    .page-header { 
        border-bottom:1px solid var(--border); 
        padding-bottom:1rem; margin-bottom:1.5rem; 
    }
    .page-header h1 { 
        font-family:'Syne',sans-serif; 
        font-weight:800; font-size:1.6rem;
        color:var(--text); margin:0; 
        letter-spacing:-0.03em; 
    }
    .page-header .sub { 
        font-size:0.8rem; 
        color:var(--muted); 
    }
    .control-panel { 
        background:var(--surface); 
        border:1px solid var(--border);
        border-radius:12px; padding:1.4rem 1.2rem; 
    }
    .panel-title { 
        font-family:'Syne',sans-serif; 
        font-weight:700; font-size:0.85rem;
        text-transform:uppercase; 
        letter-spacing:0.1em; 
        color:var(--muted); 
        margin-bottom:1.2rem; 
    }
    .panel-divider { 
        border:none; 
        border-top:1px solid var(--border); 
        margin:1.1rem 0; 
    }


    .tier-box { 
        border-radius:8px; 
        padding:0.7rem 0.9rem; 
        margin-bottom:0.5rem; 
        border:1px solid var(--border); 
    }

    .tier-box-header { display:flex; align-items:center; justify-content:space-between; margin-bottom:0.3rem; }
    .tier-box-name { font-family:'Syne',sans-serif; font-size:0.7rem; font-weight:700;
        text-transform:uppercase; letter-spacing:0.07em; }
    .tier-box-count { font-family:'Syne',sans-serif; font-size:1rem; font-weight:700; }
    .tier-box-districts { font-size:0.72rem; color:var(--muted); line-height:1.55; }

.metric-row { display:flex; gap:0.8rem; margin-bottom:1.2rem; flex-wrap:wrap; }
.metric-card { flex:1 1 140px; background:var(--surface); border:1px solid var(--border);
    border-radius:10px; padding:1rem 1.2rem; position:relative; overflow:hidden; }
.metric-card::before { content:""; position:absolute; top:0;left:0;right:0; height:3px;
    background:var(--card-accent,var(--accent)); border-radius:10px 10px 0 0; }
.metric-card .mc-label { font-size:0.65rem; font-weight:600; text-transform:uppercase;
    letter-spacing:0.08em; color:var(--muted); margin-bottom:0.3rem; }
.metric-card .mc-value { font-family:'Syne',sans-serif; font-size:1.6rem; font-weight:700;
    line-height:1; color:var(--card-accent,var(--text)); }
.metric-card .mc-sub { font-size:0.7rem; color:var(--muted); margin-top:0.3rem; }

.section-title { font-family:'Syne',sans-serif; font-weight:700; font-size:0.95rem;
    color:var(--text); margin:0 0 0.8rem; display:flex; align-items:center; gap:0.45rem; }
.section-title .dot { width:7px; height:7px; border-radius:50%;
    background:var(--accent); display:inline-block; flex-shrink:0; }

.badge { display:inline-block; padding:2px 9px; border-radius:20px;
    font-size:0.65rem; font-weight:600; letter-spacing:0.06em; text-transform:uppercase; }
.badge-CRITICAL { background:rgba(155,29,255,.12); color:var(--critical); border:1px solid rgba(155,29,255,.35); }
.badge-HIGH     { background:rgba(109,40,217,.12); color:var(--high);     border:1px solid rgba(109,40,217,.35); }
.badge-MODERATE { background:rgba(167,139,250,.2); color:var(--moderate); border:1px solid rgba(167,139,250,.45); }
.badge-LOW      { background:rgba(79,70,229,.12);  color:var(--low);      border:1px solid rgba(79,70,229,.35); }

.risk-bar-wrap { width:100%; background:var(--border); border-radius:4px; height:5px; }
.risk-bar-fill  { height:5px; border-radius:4px; }

.alloc-table { width:100%; border-collapse:collapse; font-size:0.8rem; }
.alloc-table th { text-align:left; padding:0.45rem 0.75rem; font-size:0.63rem; font-weight:600;
    text-transform:uppercase; letter-spacing:0.07em; color:var(--muted);
    border-bottom:1px solid var(--border); white-space:nowrap; }
.alloc-table td { padding:0.6rem 0.75rem; border-bottom:1px solid rgba(33,38,45,.5); vertical-align:middle; }
.alloc-table tr:hover td { background:rgba(255,255,255,.02); }
.alloc-table .mono { font-family:'Syne',sans-serif; font-weight:600; font-size:0.82rem; }

div[data-testid="stButton"] button[kind="primary"] {
    background:var(--accent)!important; color:#ffffff!important; border:none!important;
    font-weight:700!important; font-family:'Syne',sans-serif!important;
    border-radius:8px!important; width:100%!important; }
</style>
""", unsafe_allow_html=True)


# ── Service ───────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def get_service():
    return RecommendationService()


TIER_COLORS = {"CRITICAL": "#9b1dff", "HIGH": "#6d28d9", "MODERATE": "#a78bfa", "LOW": "#4f46e5"}
TIER_BG = {"CRITICAL": "rgba(155,29,255,.07)", "HIGH": "rgba(109,40,217,.07)",
           "MODERATE": "rgba(167,139,250,.1)", "LOW": "rgba(79,70,229,.07)"}
PB = "#ffffff"


def fmt_rs(v):
    if v >= 1_000_000_000: return f"Rs. {v / 1_000_000_000:.2f} B"
    if v >= 1_000_000:     return f"Rs. {v / 1_000_000:.1f} M"
    return f"Rs. {v:,.0f}"


def badge(tier):
    return f'<span class="badge badge-{tier}">{tier}</span>'


def risk_bar(score, tier):
    color = TIER_COLORS.get(tier, "#8b949e")
    return (f'<div class="risk-bar-wrap"><div class="risk-bar-fill" '
            f'style="width:{int(score * 100)}%;background:{color}"></div></div>'
            f'<div style="font-size:0.65rem;color:{color};margin-top:2px">{score:.4f}</div>')


# ── Poverty Nav Switcher ──────────────────────────────────────────────────────
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
    <a href="/poverty" target="_self">📊&nbsp; Regional Insights</a>
    <a href="/poverty_resource" class="active" target="_self">💰&nbsp; Resource Allocation</a>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
  <h1>🇱🇰 Poverty Resource Allocation</h1>
  <div class="sub">Sri Lanka District-Level Budget Distribution</div>
</div>""", unsafe_allow_html=True)

# ── Layout ────────────────────────────────────────────────────────────────────
col_panel, col_main = st.columns([2, 7], gap="large")

# ════════════════════════════
# LEFT PANEL
# ════════════════════════════
with col_panel:
    st.markdown('<div class="control-panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">⚙ Budget</div>', unsafe_allow_html=True)

    budget_input = st.number_input(
        "budget", min_value=100_000_000, max_value=100_000_000_000,
        value=5_000_000_000, step=500_000_000, format="%d",
        label_visibility="collapsed",
    )
    st.caption(f"≈ {fmt_rs(budget_input)}")

    st.markdown('<hr class="panel-divider">', unsafe_allow_html=True)
    run_btn = st.button("▶  Run Allocation", type="primary")
    st.markdown('<hr class="panel-divider">', unsafe_allow_html=True)

    # Risk summary
    st.markdown('<div class="panel-title">Risk Summary</div>', unsafe_allow_html=True)
    try:
        svc = get_service()
        summary = svc.get_risk_summary()
        for tier, key, color in [
            ("CRITICAL", "critical_districts", "#9b1dff"),
            ("HIGH", "high_districts", "#6d28d9"),
            ("MODERATE", "moderate_districts", "#a78bfa"),
            ("LOW", "low_districts", "#4f46e5"),
        ]:
            districts = summary.get(key, [])
            st.markdown(
                f'<div class="tier-box" style="background:{TIER_BG[tier]}">'
                f'<div class="tier-box-header">'
                f'<span class="tier-box-name" style="color:{color}">{tier}</span>'
                f'<span class="tier-box-count" style="color:{color}">{len(districts)}</span>'
                f'</div>'
                f'<div class="tier-box-districts">{", ".join(districts) or "—"}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    except Exception as e:
        st.error(f"Risk summary error: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

# ════════════════════════════
# RIGHT — RESULTS
# ════════════════════════════
with col_main:
    if run_btn:
        with st.spinner("Running allocation…"):
            try:
                svc = get_service()
                result = svc.allocate_budget(float(budget_input))
                st.session_state["alloc_result"] = result
            except Exception as e:
                st.error(f"Allocation failed: {e}")
                st.stop()

    if "alloc_result" not in st.session_state:
        # Landing — tier overview chart
        try:
            svc = get_service()
            summary = svc.get_risk_summary()
            all_d = []
            for tier, key in [("CRITICAL", "critical_districts"), ("HIGH", "high_districts"),
                              ("MODERATE", "moderate_districts"), ("LOW", "low_districts")]:
                for d in summary.get(key, []):
                    all_d.append({"District": d, "Tier": tier})
            if all_d:
                tc = pd.DataFrame(all_d)["Tier"].value_counts().reset_index()
                tc.columns = ["Tier", "Count"]
                tc["Color"] = tc["Tier"].map(TIER_COLORS)
                fig = go.Figure(go.Bar(
                    x=tc["Tier"], y=tc["Count"],
                    marker_color=tc["Color"].tolist(),
                    text=tc["Count"], textposition="outside",
                    textfont=dict(color="#000000", size=14),
                ))
                fig.update_layout(plot_bgcolor=PB, paper_bgcolor=PB,
                                  xaxis=dict(showgrid=False, color="#7a6d94", tickfont=dict(size=13, color="#000000")),
                                  yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                  margin=dict(l=0, r=0, t=20, b=10), height=260, font=dict(family="DM Sans"))
                st.markdown('<div class="section-title"><span class="dot"></span> District Risk Overview</div>',
                            unsafe_allow_html=True)
                st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass
        st.stop()

    # ── Results ───────────────────────────────────────────────────────────────
    result = st.session_state["alloc_result"]
    allocations = result["allocations"]
    df = pd.DataFrame(allocations)

    # Metric cards
    tier_counts = df["risk_tier"].value_counts()
    cards = f'<div class="metric-row">'
    cards += f"""
    <div class="metric-card" style="--card-accent:#7c3aed">
      <div class="mc-label">Total Budget</div>
      <div class="mc-value">{fmt_rs(result['total_budget'])}</div>
      <div class="mc-sub">{len(df)} districts</div>
    </div>
    <div class="metric-card" style="--card-accent:#a78bfa">
      <div class="mc-label">Floor / District</div>
      <div class="mc-value">{fmt_rs(result['floor_per_district'])}</div>
      <div class="mc-sub">1.5% equity floor</div>
    </div>"""
    for tier, color in TIER_COLORS.items():
        n = int(tier_counts.get(tier, 0))
        if n:
            tier_total = df[df["risk_tier"] == tier]["total_allocation"].sum()
            cards += f"""
    <div class="metric-card" style="--card-accent:{color}">
      <div class="mc-label">{tier}</div>
      <div class="mc-value">{n}</div>
      <div class="mc-sub">{fmt_rs(tier_total)}</div>
    </div>"""
    cards += "</div>"
    st.markdown(cards, unsafe_allow_html=True)

    # Charts
    ch1, ch2 = st.columns([3, 2], gap="medium")

    with ch1:
        st.markdown('<div class="section-title"><span class="dot"></span> Allocation by District</div>',
                    unsafe_allow_html=True)
        df_bar = df.sort_values("total_allocation", ascending=True)
        fig_bar = go.Figure(go.Bar(
            x=df_bar["total_allocation"], y=df_bar["district"], orientation="h",
            marker_color=[TIER_COLORS.get(t, "#8b949e") for t in df_bar["risk_tier"]],
            text=[fmt_rs(v) for v in df_bar["total_allocation"]],
            textposition="outside", textfont=dict(size=9, color="#000000"),
        ))
        fig_bar.update_layout(plot_bgcolor=PB, paper_bgcolor=PB,
                              xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, color="#7a6d94"),
                              yaxis=dict(showgrid=False, color="#000000", tickfont=dict(size=10)),
                              margin=dict(l=0, r=70, t=5, b=5), height=420, font=dict(family="DM Sans"))
        st.plotly_chart(fig_bar, use_container_width=True)

    with ch2:
        st.markdown('<div class="section-title"><span class="dot"></span> Share by Tier</div>',
                    unsafe_allow_html=True)
        tier_totals = df.groupby("risk_tier")["total_allocation"].sum().reindex(
            ["CRITICAL", "HIGH", "MODERATE", "LOW"]).dropna()
        fig_pie = go.Figure(go.Pie(
            labels=tier_totals.index, values=tier_totals.values, hole=0.58,
            marker=dict(colors=[TIER_COLORS[t] for t in tier_totals.index],
                        line=dict(color="#ffffff", width=2)),
            textinfo="percent", textfont=dict(size=11, color="#ffffff"),
            hovertemplate="<b>%{label}</b><br>%{customdata}<extra></extra>",
            customdata=[fmt_rs(v) for v in tier_totals.values],
        ))
        fig_pie.update_layout(plot_bgcolor=PB, paper_bgcolor=PB,
                              legend=dict(font=dict(color="#000000", size=10), bgcolor=PB),
                              margin=dict(l=5, r=5, t=5, b=5), height=420, font=dict(family="DM Sans"),
                              annotations=[dict(text=f"<b>{len(df)}</b><br>districts",
                                                x=0.5, y=0.5, showarrow=False, font=dict(size=15, color="#000000"))])
        st.plotly_chart(fig_pie, use_container_width=True)

    # Full table — matches the screenshot columns exactly
    st.markdown('<div class="section-title" style="margin-top:1rem">'
                '<span class="dot"></span> Full Allocation Breakdown</div>',
                unsafe_allow_html=True)

    rows = ""
    for i, (_, row) in enumerate(df.iterrows(), 1):
        rows += f"""
        <tr>
          <td><b>{i}</b></td>
          <td><b>{row['district']}</b></td>
          <td>{badge(row['risk_tier'])}</td>
          <td style="min-width:90px">{risk_bar(row['enhanced_risk_score'], row['risk_tier'])}</td>
          <td style="color:#7a6d94">{row['risk_pct']:.2f}%</td>
          <td class="mono">{fmt_rs(row['total_allocation'])}</td>
          <td style="color:#7a6d94">{row['allocation_pct']:.2f}%</td>
          <td style="color:#7a6d94">{fmt_rs(row['alloc_per_hh'])}</td>
        </tr>"""

    st.markdown(f"""
    <div style="background:var(--surface);border:1px solid var(--border);
                border-radius:10px;overflow:hidden;overflow-x:auto;margin-bottom:1rem;">
      <table class="alloc-table">
        <thead><tr>
          <th>#</th><th>District</th><th>Risk Tier</th><th>Risk Score</th>
          <th>Risk %</th><th>Allocation (Rs.)</th><th>Alloc %</th><th>Per HH (Rs.)</th>
        </tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>""", unsafe_allow_html=True)

    csv = df[["district", "risk_tier", "enhanced_risk_score", "risk_pct",
              "total_allocation", "allocation_pct", "alloc_per_hh"]].to_csv(index=False).encode()
    st.markdown("""
<style>
[data-testid="stDownloadButton"] > button {
    background: #ffffff !important;
    color: #7c3aed !important;
    border: 1.5px solid rgba(124,58,237,0.35) !important;
    border-radius: 10px !important;
    font-family: 'DM Sans','Inter',sans-serif !important;
    font-size: 0.875rem !important;
    font-weight: 600 !important;
    padding: 10px 22px !important;
    box-shadow: 0 2px 8px rgba(124,58,237,0.10) !important;
    transition: all 0.18s ease !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: linear-gradient(135deg,#7c3aed,#a855f7) !important;
    color: #ffffff !important;
    border-color: transparent !important;
    box-shadow: 0 4px 14px rgba(124,58,237,0.35) !important;
}
</style>
""", unsafe_allow_html=True)
    st.download_button("⬇  Export CSV", data=csv,
                       file_name="resource_allocation.csv", mime="text/csv")