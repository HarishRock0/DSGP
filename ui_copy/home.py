# import streamlit as st
#
# st.set_page_config(
#     page_title="Sri Lanka District Insights",
#     layout="wide"
# )
#
# # -------------------- CUSTOM CSS --------------------
# st.markdown("""
# <style>
#     body {
#         background-color: #0e1117;
#     }
#     .card {
#         background: #161b22;
#         padding: 2rem;
#         border-radius: 16px;
#         text-align: center;
#         transition: all 0.3s ease;
#         height: 100%;
#     }
#     .card:hover {
#         transform: translateY(-8px);
#         box-shadow: 0 20px 40px rgba(88,166,255,0.15);
#         border: 1px solid #58a6ff;
#     }
#     .card h2 {
#         margin-bottom: 0.5rem;
#     }
#     .card p {
#         color: #8b949e;
#         font-size: 1.05rem;
#     }
#     .card-btn {
#         margin-top: 1.2rem;
#     }
# </style>
# """, unsafe_allow_html=True)
#
# # -------------------- HEADER --------------------
# st.markdown("""
# <div style="text-align:center; padding: 2.5rem 0;">
#     <h1 style="color:lightblue;">Sri Lanka District Insights</h1>
#     <p style="font-size:1.2rem; color:#8b949e;">
#         Poverty • Child Protection • Mental Health — Interactive Decision Support
#     </p>
# </div>
# """, unsafe_allow_html=True)
#
# # -------------------- SESSION STATE --------------------
# if "page" not in st.session_state:
#     st.session_state.page = "home"
#
# # -------------------- HOME PAGE CONTENT --------------------
# col1, col2, col3 = st.columns(3, gap="large")
#
# with col1:
#     st.markdown("""
#     <div class="card">
#         <h2>Poverty Analysis</h2>
#         <p>
#             Identify high-poverty districts and allocate
#             development projects where impact is highest.
#         </p>
#     </div>
#     """, unsafe_allow_html=True)
#
#     if st.button("Explore Poverty Data", key="poverty_btn", use_container_width=True):
#         # Updated path to include pages/ folder
#         st.switch_page("pages/poverty.py")
#
# with col2:
#     st.markdown("""
#     <div class="card">
#         <h2>Child Protection</h2>
#         <p>
#             Analyze vulnerable child populations and
#             optimize child protection service deployment.
#         </p>
#     </div>
#     """, unsafe_allow_html=True)
#
#     if st.button("View Child Protection Insights", key="child_btn", use_container_width=True):
#         # Updated path to include pages/ folder
#         st.switch_page("pages/childcase.py")
#
# with col3:
#     st.markdown("""
#     <div class="card">
#         <h2>Mental Health Services</h2>
#         <p>
#             Understand district-level mental health needs
#             and resource gaps.
#         </p>
#     </div>
#     """, unsafe_allow_html=True)
#
#     if st.button("Mental Health Dashboard", key="mental_btn", use_container_width=True):
#         # Updated path to include pages/ folder
#         st.switch_page("pages/mentalhealth.py")


import streamlit as st
import time

st.set_page_config(
    page_title="Sri Lanka District Insights",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------- CUSTOM CSS --------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=DM+Sans:wght@300;400;500;600&display=swap');

    :root {
        --purple-deep:    #1a0533;
        --purple-mid:     #3d1170;
        --purple-vivid:   #7c3aed;
        --purple-light:   #a855f7;
        --purple-glow:    #c084fc;
        --white-pure:     #ffffff;
        --white-soft:     #f3e8ff;
        --white-muted:    #d8b4fe;
        --accent-gold:    #fbbf24;
    }

    /* ---- Base ---- */
    html, body, [data-testid="stAppViewContainer"] {
        background: var(--purple-deep) !important;
        font-family: 'DM Sans', sans-serif;
    }
    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(ellipse 80% 50% at 50% -10%, rgba(124,58,237,0.45) 0%, transparent 70%),
            radial-gradient(ellipse 60% 40% at 85% 80%, rgba(168,85,247,0.20) 0%, transparent 60%),
            radial-gradient(ellipse 50% 50% at 10% 90%, rgba(109,40,217,0.25) 0%, transparent 60%),
            var(--purple-deep) !important;
    }
    [data-testid="stHeader"], footer { display: none !important; }
    [data-testid="block-container"] {
        padding: 0 3rem 4rem 3rem !important;
        max-width: 1300px;
        margin: 0 auto;
    }

    /* ---- Animated noise overlay ---- */
    body::before {
        content: '';
        position: fixed; inset: 0;
        background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.04'/%3E%3C/svg%3E");
        pointer-events: none;
        z-index: 9999;
    }

    /* ---- Hero ---- */
    .hero-wrap {
        text-align: center;
        padding: 5rem 1rem 3.5rem;
        position: relative;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(124,58,237,0.25);
        border: 1px solid rgba(168,85,247,0.5);
        color: var(--purple-glow);
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        padding: 0.4rem 1.1rem;
        border-radius: 100px;
        margin-bottom: 1.4rem;
        backdrop-filter: blur(8px);
        animation: fadeDown 0.7s ease both;
    }
    .hero-title {
        font-family: 'Playfair Display', serif;
        font-size: clamp(2.6rem, 5vw, 4.2rem);
        font-weight: 900;
        color: var(--white-pure);
        line-height: 1.12;
        margin: 0 0 0.6rem;
        animation: fadeDown 0.85s ease 0.1s both;
    }
    .hero-title span {
        background: linear-gradient(135deg, var(--purple-light), var(--purple-glow), var(--accent-gold));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .hero-subtitle {
        font-size: 1.15rem;
        color: var(--white-muted);
        font-weight: 300;
        letter-spacing: 0.02em;
        max-width: 560px;
        margin: 0 auto 0.5rem;
        animation: fadeDown 0.95s ease 0.2s both;
    }
    .hero-divider {
        width: 80px;
        height: 3px;
        background: linear-gradient(90deg, transparent, var(--purple-vivid), transparent);
        margin: 2rem auto 0;
        border-radius: 2px;
        animation: fadeDown 1s ease 0.3s both;
    }

    /* ---- Stats row ---- */
    .stats-row {
        display: flex;
        justify-content: center;
        gap: 3rem;
        padding: 1.5rem 0 2.5rem;
        animation: fadeUp 0.9s ease 0.4s both;
    }
    .stat-item { text-align: center; }
    .stat-number {
        font-family: 'Playfair Display', serif;
        font-size: 2rem;
        font-weight: 700;
        color: var(--purple-glow);
        line-height: 1;
    }
    .stat-label {
        font-size: 0.78rem;
        color: var(--white-muted);
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-top: 0.25rem;
    }
    .stat-sep {
        width: 1px;
        background: rgba(168,85,247,0.3);
        align-self: stretch;
    }

    /* ---- Cards ---- */
    .cards-section {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1.6rem;
        animation: fadeUp 0.9s ease 0.55s both;
    }
    .module-card {
        position: relative;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(168,85,247,0.2);
        border-radius: 20px;
        padding: 2.2rem 1.8rem 1.6rem;
        overflow: hidden;
        transition: transform 0.35s ease, border-color 0.35s ease, box-shadow 0.35s ease;
        cursor: default;
    }
    .module-card::before {
        content: '';
        position: absolute;
        inset: 0;
        background: linear-gradient(135deg, rgba(124,58,237,0.12) 0%, transparent 60%);
        opacity: 0;
        transition: opacity 0.35s ease;
    }
    .module-card:hover {
        transform: translateY(-10px);
        border-color: rgba(168,85,247,0.7);
        box-shadow: 0 24px 60px rgba(124,58,237,0.35), 0 0 0 1px rgba(168,85,247,0.15);
    }
    .module-card:hover::before { opacity: 1; }
    .card-icon-wrap {
        width: 52px; height: 52px;
        border-radius: 14px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.5rem;
        margin-bottom: 1.3rem;
    }
    .icon-poverty  { background: linear-gradient(135deg, #4c1d95, #7c3aed); }
    .icon-child    { background: linear-gradient(135deg, #5b21b6, #a855f7); }
    .icon-mental   { background: linear-gradient(135deg, #6d28d9, #c084fc); }
    .card-number {
        position: absolute;
        top: 1.4rem; right: 1.6rem;
        font-family: 'Playfair Display', serif;
        font-size: 3.5rem;
        font-weight: 900;
        color: rgba(168,85,247,0.07);
        line-height: 1;
        user-select: none;
    }
    .card-tag {
        display: inline-block;
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--purple-glow);
        background: rgba(124,58,237,0.2);
        border: 1px solid rgba(168,85,247,0.3);
        padding: 0.22rem 0.7rem;
        border-radius: 100px;
        margin-bottom: 0.85rem;
    }
    .card-title {
        font-family: 'Playfair Display', serif;
        font-size: 1.45rem;
        font-weight: 700;
        color: var(--white-pure);
        margin: 0 0 0.7rem;
        line-height: 1.25;
    }
    .card-desc {
        font-size: 0.93rem;
        color: var(--white-muted);
        line-height: 1.65;
        margin-bottom: 1.5rem;
    }
    .card-metric {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.82rem;
        color: rgba(216,180,254,0.7);
        margin-bottom: 0.35rem;
    }
    .card-metric::before {
        content: '';
        display: inline-block;
        width: 6px; height: 6px;
        border-radius: 50%;
        background: var(--purple-light);
        flex-shrink: 0;
    }
    .card-footer {
        margin-top: 1.4rem;
        padding-top: 1.2rem;
        border-top: 1px solid rgba(168,85,247,0.12);
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .card-status {
        font-size: 0.75rem;
        color: #6ee7b7;
        background: rgba(110,231,183,0.1);
        border: 1px solid rgba(110,231,183,0.25);
        padding: 0.2rem 0.65rem;
        border-radius: 100px;
    }
    .card-arrow {
        font-size: 1.1rem;
        color: var(--purple-light);
        transition: transform 0.2s ease;
    }
    .module-card:hover .card-arrow { transform: translateX(5px); }

    /* ---- Bottom info bar ---- */
    .info-bar {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 2.5rem;
        margin-top: 3rem;
        padding: 1.2rem 2rem;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(168,85,247,0.15);
        border-radius: 14px;
        animation: fadeUp 0.9s ease 0.7s both;
    }
    .info-item {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        font-size: 0.82rem;
        color: var(--white-muted);
    }
    .info-dot {
        width: 8px; height: 8px;
        border-radius: 50%;
        background: var(--purple-light);
        box-shadow: 0 0 8px var(--purple-vivid);
        animation: pulse 2s ease-in-out infinite;
    }

    /* ---- Streamlit button overrides ---- */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, var(--purple-vivid), var(--purple-light)) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.04em !important;
        padding: 0.65rem 1.2rem !important;
        cursor: pointer !important;
        transition: all 0.25s ease !important;
        box-shadow: 0 4px 20px rgba(124,58,237,0.4) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 28px rgba(124,58,237,0.6) !important;
        background: linear-gradient(135deg, var(--purple-light), var(--purple-glow)) !important;
    }
    .stButton > button:active {
        transform: translateY(0) !important;
    }

    /* ---- Column gaps ---- */
    [data-testid="column"] { padding: 0 0.5rem !important; }

    /* ---- Animations ---- */
    @keyframes fadeDown {
        from { opacity: 0; transform: translateY(-22px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(22px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50%       { opacity: 0.5; transform: scale(1.3); }
    }
    @keyframes shimmer {
        0%   { background-position: -200% center; }
        100% { background-position:  200% center; }
    }
</style>
""", unsafe_allow_html=True)

# -------------------- HERO --------------------
st.markdown("""
<div class="hero-wrap">
    <div class="hero-badge">🇱🇰 &nbsp; Interactive Decision Support Platform</div>
    <h1 class="hero-title">Sri Lanka<br><span>District Insights</span></h1>
    <p class="hero-subtitle">
        Evidence-based analytics across poverty, child protection, and mental health services — empowering smarter resource allocation.
    </p>
    <div class="hero-divider"></div>
</div>

<div class="stats-row">
    <div class="stat-item">
        <div class="stat-number">25</div>
        <div class="stat-label">Districts Covered</div>
    </div>
    <div class="stat-sep"></div>
    <div class="stat-item">
        <div class="stat-number">3</div>
        <div class="stat-label">Analytics Modules</div>
    </div>
    <div class="stat-sep"></div>
    <div class="stat-item">
        <div class="stat-number">Live</div>
        <div class="stat-label">Data Status</div>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------- MODULE CARDS --------------------
col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    st.markdown("""
    <div class="module-card">
        <div class="card-number">01</div>
        <div class="card-icon-wrap icon-poverty">💰</div>
        <div class="card-tag">Economic Analysis</div>
        <h3 class="card-title">Poverty Analysis</h3>
        <p class="card-desc">
            Identify high-poverty districts and allocate development projects where social impact is greatest.
        </p>
        <div class="card-metric">District-level poverty mapping</div>
        <div class="card-metric">Resource allocation scoring</div>
        <div class="card-metric">Trend forecasting</div>
        <div class="card-footer">
            <span class="card-status">● Active</span>
            <span class="card-arrow">→</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Explore Poverty Data →", key="poverty_btn", use_container_width=True):
        st.switch_page("pages/poverty.py")

with col2:
    st.markdown("""
    <div class="module-card">
        <div class="card-number">02</div>
        <div class="card-icon-wrap icon-child">🛡️</div>
        <div class="card-tag">Social Protection</div>
        <h3 class="card-title">Child Protection</h3>
        <p class="card-desc">
            Analyze vulnerable child populations and optimize child protection service deployment across regions.
        </p>
        <div class="card-metric">Vulnerability heat-mapping</div>
        <div class="card-metric">Service gap identification</div>
        <div class="card-metric">Intervention prioritization</div>
        <div class="card-footer">
            <span class="card-status">● Active</span>
            <span class="card-arrow">→</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("View Child Protection Insights →", key="child_btn", use_container_width=True):
        st.switch_page("pages/childcase.py")

with col3:
    st.markdown("""
    <div class="module-card">
        <div class="card-number">03</div>
        <div class="card-icon-wrap icon-mental">🧠</div>
        <div class="card-tag">Health Services</div>
        <h3 class="card-title">Mental Health</h3>
        <p class="card-desc">
            Understand district-level mental health needs, resource gaps, and service coverage disparities.
        </p>
        <div class="card-metric">Needs assessment dashboard</div>
        <div class="card-metric">Facility coverage analysis</div>
        <div class="card-metric">Workforce gap metrics</div>
        <div class="card-footer">
            <span class="card-status">● Active</span>
            <span class="card-arrow">→</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Mental Health Dashboard →", key="mental_btn", use_container_width=True):
        st.switch_page("pages/mentalhealth.py")

# -------------------- INFO BAR --------------------
st.markdown("""
<div class="info-bar">
    <div class="info-item"><div class="info-dot"></div> Real-time data integration</div>
    <div class="info-item"><div class="info-dot"></div> 25 districts nationwide</div>
    <div class="info-item"><div class="info-dot"></div> Decision-support optimized</div>
    <div class="info-item"><div class="info-dot"></div> Built for policymakers</div>
</div>
""", unsafe_allow_html=True)