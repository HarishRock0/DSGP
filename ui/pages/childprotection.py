from __future__ import annotations
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from service.child_protection_service import ChildProtectionService

PROJECT_ROOT = _PROJECT_ROOT

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title            = "Child Protection — Budget Allocation",
    page_icon             = "🛡️",
    layout                = "wide",
    initial_sidebar_state = "collapsed",
)


# ---------------------------------------------------------------------------
# Design system — ported from poverty_resource.py
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;0,700;1,400;1,600&family=Outfit:wght@300;400;500;600;700&display=swap');
:root {
    --bg:       #ffffff;
    --surface:  #ffffff;
    --border:   #d8c9f5;
    --accent:   #7c3aed;
    --text:     #000000;
    --muted:    #555555;
    --critical: #9b1dff;
    --high:     #6d28d9;
    --moderate: #a78bfa;
    --low:      #4f46e5;
}

/* ── Base ─────────────────────────────────────────────── */
.stApp, .stApp > div,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
[data-testid="block-container"] {
    background: var(--bg) !important;
}
.stApp, .stApp p, .stApp div, .stApp span, .stApp label {
    color: #000000 !important;
}
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: var(--bg);
    color: var(--text);
}
#MainMenu, footer, header          { visibility: hidden; }
section[data-testid="stSidebar"]   { display: none !important; }
button[data-testid="collapsedControl"] { display: none !important; }
.block-container { padding: 1.8rem 2rem 4rem !important; max-width: 100% !important; }

/* ── Page header ──────────────────────────────────────── */
.page-header {
    border-bottom: 1px solid var(--border);
    padding-bottom: 1rem;
    margin-bottom: 1.5rem;
}
.page-header h1 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.6rem;
    color: var(--text);
    margin: 0;
    letter-spacing: -0.03em;
}
.page-header .sub { font-size: 0.8rem; color: var(--muted); }

/* ── Left control panel ───────────────────────────────── */
.control-panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.4rem 1.2rem;
}
.panel-title {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--muted);
    margin-bottom: 1.2rem;
}
.panel-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1.1rem 0;
}

/* ── Tier summary boxes ───────────────────────────────── */
.tier-box {
    border-radius: 8px;
    padding: 0.7rem 0.9rem;
    margin-bottom: 0.5rem;
    border: 1px solid var(--border);
}
.tier-box-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.3rem; }
.tier-box-name   { font-family: 'Syne', sans-serif; font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.07em; }
.tier-box-count  { font-family: 'Syne', sans-serif; font-size: 1rem; font-weight: 700; }
.tier-box-districts { font-size: 0.72rem; color: var(--muted); line-height: 1.55; }

/* ── Metric cards ─────────────────────────────────────── */
.metric-row  { display: flex; gap: 0.8rem; margin-bottom: 1.2rem; flex-wrap: wrap; }
.metric-card {
    flex: 1 1 140px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--card-accent, var(--accent));
    border-radius: 10px 10px 0 0;
}
.metric-card .mc-label {
    font-size: 0.65rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.08em; color: var(--muted); margin-bottom: 0.3rem;
}
.metric-card .mc-value {
    font-family: 'Syne', sans-serif; font-size: 1.6rem; font-weight: 700;
    line-height: 1; color: var(--card-accent, var(--text));
}
.metric-card .mc-sub { font-size: 0.7rem; color: var(--muted); margin-top: 0.3rem; }

/* ── Section titles ───────────────────────────────────── */
.section-title {
    font-family: 'Syne', sans-serif; font-weight: 700; font-size: 0.95rem;
    color: var(--text); margin: 0 0 0.8rem;
    display: flex; align-items: center; gap: 0.45rem;
}
.section-title .dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--accent); display: inline-block; flex-shrink: 0;
}

/* ── Tier badges ──────────────────────────────────────── */
.badge {
    display: inline-block; padding: 2px 9px; border-radius: 20px;
    font-size: 0.65rem; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase;
}
.badge-CRITICAL { background: rgba(155,29,255,.12);  color: var(--critical); border: 1px solid rgba(155,29,255,.35); }
.badge-HIGH     { background: rgba(109,40,217,.12);  color: var(--high);     border: 1px solid rgba(109,40,217,.35); }
.badge-MODERATE { background: rgba(167,139,250,.2);  color: var(--moderate); border: 1px solid rgba(167,139,250,.45); }
.badge-LOW      { background: rgba(79,70,229,.12);   color: var(--low);      border: 1px solid rgba(79,70,229,.35); }

/* ── Risk score bar ───────────────────────────────────── */
.risk-bar-wrap { width: 100%; background: var(--border); border-radius: 4px; height: 5px; }
.risk-bar-fill { height: 5px; border-radius: 4px; }

/* ── Allocation table ─────────────────────────────────── */
.alloc-table { width: 100%; border-collapse: collapse; font-size: 0.8rem; }
.alloc-table th {
    text-align: left; padding: 0.45rem 0.75rem;
    font-size: 0.63rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.07em;
    color: var(--muted); border-bottom: 1px solid var(--border); white-space: nowrap;
}
.alloc-table td { padding: 0.6rem 0.75rem; border-bottom: 1px solid rgba(226,229,234,.6); vertical-align: middle; }
.alloc-table tr:hover td { background: rgba(0,0,0,.02); }
.alloc-table .mono { font-family: 'Syne', sans-serif; font-weight: 600; font-size: 0.82rem; }

/* ── Primary button ───────────────────────────────────── */
div[data-testid="stButton"] button[kind="primary"] {
    background: var(--accent) !important;
    color: #ffffff !important;
    border: none !important;
    font-weight: 700 !important;
    font-family: 'Syne', sans-serif !important;
    border-radius: 8px !important;
    width: 100% !important;
}

/* ── Plotly transparent bg ────────────────────────────── */
[data-testid="stPlotlyChart"] > div,
.js-plotly-plot .plotly,
.js-plotly-plot .plotly .bg { background: #ffffff !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TIER_COLORS = {
    "CRITICAL": "#9b1dff",
    "HIGH":     "#6d28d9",
    "MODERATE": "#a78bfa",
    "LOW":      "#4f46e5",
}
TIER_BG = {
    "CRITICAL": "rgba(155,29,255,.07)",
    "HIGH":     "rgba(109,40,217,.07)",
    "MODERATE": "rgba(167,139,250,.1)",
    "LOW":      "rgba(79,70,229,.07)",
}
PB = "#ffffff"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def fmt_lkr(v: float) -> str:
    if v >= 1_000_000_000: return f"LKR {v / 1_000_000_000:.2f} B"
    if v >= 1_000_000:     return f"LKR {v / 1_000_000:.1f} M"
    return f"LKR {v:,.0f}"


def badge(tier: str) -> str:
    return f'<span class="badge badge-{tier}">{tier}</span>'


def risk_bar(score: float, tier: str) -> str:
    color = TIER_COLORS.get(tier, "#8b949e")
    pct   = min(int(score * 100), 100)
    return (
        f'<div class="risk-bar-wrap">'
        f'<div class="risk-bar-fill" style="width:{pct}%;background:{color}"></div>'
        f'</div>'
        f'<div style="font-size:0.65rem;color:{color};margin-top:2px">{score:.4f}</div>'
    )


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading pipeline…")
def get_service() -> ChildProtectionService:
    return ChildProtectionService(PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
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
<div class="home-btn"><a href="/" target="_self">🏠&nbsp; Home</a></div>
<div class="child-nav-wrap">
  <div class="child-nav">
    <a href="/childcase" target="_self">🛡️&nbsp; Regional Insights</a>
    <a href="/childprotection" class="active" target="_self">💰&nbsp; Resource Allocation</a>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="page-header">
  <h1>🛡️ Child Protection — Budget Allocation</h1>
  <div class="sub">Sri Lanka District-Level Child Protection Risk &amp; Resource Distribution</div>
</div>""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Two-column layout
# ---------------------------------------------------------------------------
col_panel, col_main = st.columns([2, 7], gap="large")

# ════════════════════════════════════════
# LEFT PANEL — controls + risk summary
# ════════════════════════════════════════
with col_panel:
    st.markdown('<div class="control-panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">⚙ Budget</div>', unsafe_allow_html=True)

    budget_input = st.number_input(
        "budget",
        min_value         = 1_000_000,
        max_value         = 100_000_000_000,
        value             = 500_000_000,
        step              = 10_000_000,
        format            = "%d",
        label_visibility  = "collapsed",
    )
    st.caption(f"≈ {fmt_lkr(budget_input)}")

    st.markdown('<hr class="panel-divider">', unsafe_allow_html=True)

    # District filter
    st.markdown('<div class="panel-title">🗺 District Filter</div>', unsafe_allow_html=True)
    try:
        svc     = get_service()
        summary = svc.get_risk_summary()
        all_districts = sorted(
            summary.get("critical_districts", []) +
            summary.get("high_districts",     []) +
            summary.get("moderate_districts", []) +
            summary.get("low_districts",      [])
        )
        selected = st.multiselect(
            "Restrict to districts",
            options = all_districts,
            default = [],
            help    = "Leave blank to allocate across all 25 districts",
            label_visibility = "collapsed",
        )
    except Exception:
        selected = []

    st.markdown('<hr class="panel-divider">', unsafe_allow_html=True)
    run_btn = st.button("▶  Run Allocation", type="primary")
    st.markdown('<hr class="panel-divider">', unsafe_allow_html=True)

    # Risk summary boxes
    st.markdown('<div class="panel-title">Risk Summary</div>', unsafe_allow_html=True)
    try:
        svc     = get_service()
        summary = svc.get_risk_summary()
        for tier, key in [
            ("CRITICAL", "critical_districts"),
            ("HIGH",     "high_districts"),
            ("MODERATE", "moderate_districts"),
            ("LOW",      "low_districts"),
        ]:
            color     = TIER_COLORS[tier]
            bg        = TIER_BG[tier]
            districts = summary.get(key, [])
            st.markdown(
                f'<div class="tier-box" style="background:{bg}">'
                f'<div class="tier-box-header">'
                f'<span class="tier-box-name"  style="color:{color}">{tier}</span>'
                f'<span class="tier-box-count" style="color:{color}">{len(districts)}</span>'
                f'</div>'
                f'<div class="tier-box-districts">{", ".join(districts) or "—"}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    except Exception as e:
        st.error(f"Risk summary error: {e}")

    st.markdown("</div>", unsafe_allow_html=True)  # close control-panel


# ════════════════════════════════════════
# RIGHT — results
# ════════════════════════════════════════
with col_main:

    # ── Run allocation ────────────────────────────────────────────────────────
    if run_btn:
        with st.spinner("Running allocation…"):
            try:
                svc    = get_service()
                result = svc.allocate_budget(
                    total_budget       = float(budget_input),
                    query              = None,
                    selected_districts = selected or None,
                )
                st.session_state["cp_alloc_result"] = result
            except Exception as e:
                st.error(f"Allocation failed: {e}")
                st.stop()

    # ── Landing state — overview chart ────────────────────────────────────────
    if "cp_alloc_result" not in st.session_state:
        try:
            svc     = get_service()
            summary = svc.get_risk_summary()
            all_d   = []
            for tier, key in [
                ("CRITICAL", "critical_districts"),
                ("HIGH",     "high_districts"),
                ("MODERATE", "moderate_districts"),
                ("LOW",      "low_districts"),
            ]:
                for d in summary.get(key, []):
                    all_d.append({"District": d, "Tier": tier})

            if all_d:
                tc = pd.DataFrame(all_d)["Tier"].value_counts().reset_index()
                tc.columns = ["Tier", "Count"]
                tc["Color"] = tc["Tier"].map(TIER_COLORS)
                fig = go.Figure(go.Bar(
                    x             = tc["Tier"],
                    y             = tc["Count"],
                    marker_color  = tc["Color"].tolist(),
                    text          = tc["Count"],
                    textposition  = "outside",
                    textfont      = dict(color="#000000", size=14),
                ))
                fig.update_layout(
                    plot_bgcolor  = PB, paper_bgcolor = PB,
                    xaxis         = dict(showgrid=False, color="#555555",
                                         tickfont=dict(size=13, color="#000000")),
                    yaxis         = dict(showgrid=False, zeroline=False, showticklabels=False),
                    margin        = dict(l=0, r=0, t=20, b=10),
                    height        = 280,
                    font          = dict(family="DM Sans"),
                )
                st.markdown(
                    '<div class="section-title"><span class="dot"></span>'
                    ' District Risk Overview</div>',
                    unsafe_allow_html=True,
                )
                st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass

        st.info("Enter a budget and click **▶ Run Allocation** to see the full district breakdown.")
        st.stop()

    # ── Results ───────────────────────────────────────────────────────────────
    result      = st.session_state["cp_alloc_result"]
    allocations = result.get("allocations", [])
    df          = pd.DataFrame(allocations)

    if df.empty:
        st.error("No allocations returned. Check service logs.")
        st.stop()

    # Ensure optional columns exist
    for col in ("avg_cases", "per_case_lkr", "risk_score", "budget_share_pct",
                "allocated_lkr", "risk_tier"):
        if col not in df.columns:
            df[col] = None

    df = df.sort_values("risk_score", ascending=False).reset_index(drop=True)

    # ── Metric cards ──────────────────────────────────────────────────────────
    tier_counts = df["risk_tier"].value_counts()
    tier_dist   = result.get("tier_distribution", {})

    cards  = '<div class="metric-row">'
    cards += f"""
    <div class="metric-card" style="--card-accent:#7c3aed">
      <div class="mc-label">Total Budget</div>
      <div class="mc-value">{fmt_lkr(result['total_budget'])}</div>
      <div class="mc-sub">{len(df)} district{'s' if len(df)!=1 else ''}</div>
    </div>
    <div class="metric-card" style="--card-accent:#a78bfa">
      <div class="mc-label">Total Verified</div>
      <div class="mc-value">{fmt_lkr(result.get('total_verified', result['total_budget']))}</div>
      <div class="mc-sub">min floor {result.get('min_floor_pct', 1.0)}%</div>
    </div>"""

    for tier, color in TIER_COLORS.items():
        n = int(tier_counts.get(tier, 0))
        if n:
            tier_total = df[df["risk_tier"] == tier]["allocated_lkr"].sum()
            cards += f"""
    <div class="metric-card" style="--card-accent:{color}">
      <div class="mc-label">{tier}</div>
      <div class="mc-value">{n}</div>
      <div class="mc-sub">{fmt_lkr(tier_total)}</div>
    </div>"""
    cards += "</div>"
    st.markdown(cards, unsafe_allow_html=True)

    # ── Charts ────────────────────────────────────────────────────────────────
    ch1, ch2 = st.columns([3, 2], gap="medium")

    with ch1:
        st.markdown(
            '<div class="section-title"><span class="dot"></span>'
            ' Allocation by District</div>',
            unsafe_allow_html=True,
        )
        df_bar = df.sort_values("allocated_lkr", ascending=True)
        fig_bar = go.Figure(go.Bar(
            x            = df_bar["allocated_lkr"],
            y            = df_bar["district"],
            orientation  = "h",
            marker_color = [TIER_COLORS.get(t, "#8b949e") for t in df_bar["risk_tier"]],
            text         = [fmt_lkr(v) for v in df_bar["allocated_lkr"]],
            textposition = "outside",
            textfont     = dict(size=9, color="#000000"),
        ))
        fig_bar.update_layout(
            plot_bgcolor  = PB, paper_bgcolor = PB,
            xaxis         = dict(showgrid=False, zeroline=False,
                                  showticklabels=False, color="#555555"),
            yaxis         = dict(showgrid=False, color="#000000",
                                  tickfont=dict(size=10)),
            margin        = dict(l=0, r=75, t=5, b=5),
            height        = 450,
            font          = dict(family="DM Sans"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with ch2:
        st.markdown(
            '<div class="section-title"><span class="dot"></span>'
            ' Share by Tier</div>',
            unsafe_allow_html=True,
        )
        tier_totals = (
            df.groupby("risk_tier")["allocated_lkr"]
              .sum()
              .reindex(["CRITICAL", "HIGH", "MODERATE", "LOW"])
              .dropna()
        )
        fig_pie = go.Figure(go.Pie(
            labels        = tier_totals.index,
            values        = tier_totals.values,
            hole          = 0.58,
            marker        = dict(
                colors=[TIER_COLORS[t] for t in tier_totals.index],
                line=dict(color="#ffffff", width=2),
            ),
            textinfo      = "percent",
            textfont      = dict(size=11, color="#ffffff"),
            hovertemplate = "<b>%{label}</b><br>%{customdata}<extra></extra>",
            customdata    = [fmt_lkr(v) for v in tier_totals.values],
        ))
        fig_pie.update_layout(
            plot_bgcolor = PB, paper_bgcolor = PB,
            legend       = dict(font=dict(color="#000000", size=10), bgcolor=PB),
            margin       = dict(l=5, r=5, t=5, b=5),
            height       = 450,
            font         = dict(family="DM Sans"),
            annotations  = [dict(
                text      = f"<b>{len(df)}</b><br>districts",
                x=0.5, y=0.5, showarrow=False,
                font      = dict(size=15, color="#000000"),
            )],
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # ── Full allocation table ─────────────────────────────────────────────────
    st.markdown(
        '<div class="section-title" style="margin-top:1rem">'
        '<span class="dot"></span> Full Allocation Breakdown</div>',
        unsafe_allow_html=True,
    )

    TIER_WEIGHT = {"CRITICAL": 4.0, "HIGH": 2.5, "MODERATE": 1.5, "LOW": 1.0}

    rows = ""
    for i, (_, row) in enumerate(df.iterrows(), 1):
        avg_c = (
            f"{row['avg_cases']:,.1f}"
            if pd.notna(row.get("avg_cases")) else "—"
        )
        per_c = (
            fmt_lkr(row["per_case_lkr"])
            if pd.notna(row.get("per_case_lkr")) else "—"
        )
        score = row.get("risk_score", 0) or 0
        rows += f"""
        <tr>
          <td><b>{i}</b></td>
          <td><b>{row['district']}</b></td>
          <td>{badge(row['risk_tier'])}</td>
          <td style="min-width:100px">{risk_bar(float(score), row['risk_tier'])}</td>
          <td style="color:#6b7280">{TIER_WEIGHT.get(row['risk_tier'], '—')}</td>
          <td style="color:#6b7280">{row['budget_share_pct']:.2f}%</td>
          <td class="mono">{fmt_lkr(row['allocated_lkr'])}</td>
          <td style="color:#6b7280">{avg_c}</td>
          <td style="color:#6b7280">{per_c}</td>
        </tr>"""

    st.markdown(f"""
    <div style="background:var(--surface);border:1px solid var(--border);
                border-radius:10px;overflow:hidden;overflow-x:auto;margin-bottom:1rem;">
      <table class="alloc-table">
        <thead><tr>
          <th>#</th>
          <th>District</th>
          <th>Risk Tier</th>
          <th>Risk Score</th>
          <th>Tier Wt</th>
          <th>Share %</th>
          <th>Allocated (LKR)</th>
          <th>Avg Cases</th>
          <th>Per Case</th>
        </tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>""", unsafe_allow_html=True)

    # Totals row
    total_share = df["budget_share_pct"].sum()
    total_alloc = df["allocated_lkr"].sum()
    st.markdown(
        f"**Totals** — Share: `{total_share:.2f}%` &nbsp;|&nbsp; "
        f"Allocated: `{fmt_lkr(total_alloc)}`",
        unsafe_allow_html=True,
    )

    # ── Download ──────────────────────────────────────────────────────────────
    st.divider()
    csv = df.to_csv(index=False).encode("utf-8")
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
    st.download_button(
        label     = "⬇  Export CSV",
        data      = csv,
        file_name = "child_protection_allocation.csv",
        mime      = "text/csv",
    )