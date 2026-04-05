import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Sri Lanka District Insights",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;0,700;1,400;1,600&family=Outfit:wght@300;400;500;600;700&display=swap');
[data-testid="stHeader"], footer, [data-testid="stToolbar"],
[data-testid="stDecoration"] { display:none !important; }
html, body, [data-testid="stAppViewContainer"] { background:#f5f3ff !important; }
[data-testid="block-container"] { padding:0 !important; max-width:100% !important; }
[data-testid="stVerticalBlock"] { gap:0 !important; }
[data-testid="stHorizontalBlock"] {
    padding: 0 5vw 5rem !important;
    max-width: 1260px !important;
    margin: 0 auto !important;
    background: #f5f3ff;
}
[data-testid="column"] { padding: 0 0.6rem !important; }
/* Real st.button — transparent overlay covering the full card */
.stButton > button {
    width:100% !important;
    background:transparent !important;
    color:transparent !important;
    border:none !important;
    border-radius:20px !important;
    padding:0 !important;
    cursor:pointer !important;
    position:relative !important;
    margin-top:-520px !important;
    height:520px !important;
    box-shadow:none !important;
    z-index:10 !important;
}
.stButton > button:hover { background:transparent !important; box-shadow:none !important; transform:none !important; }
.stButton > button:focus { outline:none !important; box-shadow:none !important; }
/* Eliminate all gaps */
[data-testid="stColumn"] > div {gap:0 !important;}
[data-testid="stColumn"] > div > div {gap:0 !important;}
[data-testid="stColumn"] > div > div > div {gap:0 !important;}
[data-testid="stColumn"] iframe {display:block !important; margin:0 !important; padding:0 !important;}
[data-testid="stColumn"] .stButton {margin:0 !important; padding:0 !important;}
</style>
""", unsafe_allow_html=True)

# ── HERO + NAV + STATS ───────────────────────────────────────────────────────
components.html("""<!DOCTYPE html><html><head><meta charset="UTF-8">
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,600;0,700;1,400&family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box;}
:root{--ink:#0f0a1e;--muted:#6b5f8a;--viv:#6d28d9;--mid:#8b5cf6;--lite:#a78bfa;--pale:#ede9fe;--wash:#f5f3ff;--bd:rgba(109,40,217,.13);--bdd:rgba(109,40,217,.26);}
body{font-family:'Outfit',sans-serif;background:#fff;color:var(--ink);}
nav{position:sticky;top:0;z-index:99;background:rgba(255,255,255,.93);backdrop-filter:blur(18px);border-bottom:1px solid var(--bd);display:flex;align-items:center;justify-content:space-between;padding:0 5vw;height:66px;}
.logo{display:flex;align-items:center;gap:.65rem;}
.emblem{width:34px;height:34px;background:var(--viv);border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:1rem;}
.wordmark{font-family:'Cormorant Garamond',serif;font-size:1.22rem;font-weight:700;color:var(--ink);}
.wordmark span{color:var(--viv);}
.links{display:flex;gap:2.2rem;align-items:center;}
.lnk{font-size:.84rem;font-weight:500;color:var(--muted);text-decoration:none;transition:color .2s;}
.lnk:hover{color:var(--viv);}
.cta{background:var(--viv);color:#fff !important;padding:.48rem 1.15rem;border-radius:8px;font-weight:600;font-size:.81rem;letter-spacing:.04em;transition:background .2s,transform .2s;}
.cta:hover{background:var(--ink) !important;transform:translateY(-1px);}
.hero{background:radial-gradient(ellipse 65% 55% at 65% -5%,rgba(109,40,217,.09) 0%,transparent 70%),radial-gradient(ellipse 45% 35% at 5% 85%,rgba(167,139,250,.07) 0%,transparent 60%),#fff;padding:6.5rem 5vw 5rem;display:flex;align-items:center;gap:4.5rem;max-width:1260px;margin:0 auto;}
.hl{flex:1;}
.hr{flex:0 0 400px;}
.eye{display:inline-flex;align-items:center;gap:.5rem;background:var(--pale);color:var(--viv);font-size:.72rem;font-weight:700;letter-spacing:.16em;text-transform:uppercase;padding:.38rem 1rem;border-radius:100px;border:1px solid var(--bdd);margin-bottom:1.7rem;animation:up .6s ease both;}
.edot{width:6px;height:6px;border-radius:50%;background:var(--viv);animation:pu 2s infinite;}
h1{font-family:'Cormorant Garamond',serif;font-size:clamp(3rem,5vw,4.8rem);font-weight:700;line-height:1.06;color:var(--ink);letter-spacing:-.02em;margin:0 0 1.4rem;animation:up .7s ease .1s both;}
h1 em{font-style:italic;color:var(--viv);}
.hp{font-size:1rem;font-weight:300;color:var(--muted);line-height:1.78;max-width:470px;margin:0 0 2.3rem;animation:up .7s ease .2s both;}
.btns{display:flex;gap:.9rem;align-items:center;animation:up .7s ease .3s both;}
.bp{background:var(--viv);color:#fff;border:none;padding:.82rem 1.9rem;border-radius:10px;font-family:'Outfit',sans-serif;font-size:.88rem;font-weight:600;letter-spacing:.03em;cursor:pointer;transition:all .25s;box-shadow:0 4px 20px rgba(109,40,217,.3);}
.bp:hover{transform:translateY(-2px);box-shadow:0 8px 28px rgba(109,40,217,.42);background:var(--ink);}
.bg{background:transparent;color:var(--muted);border:1px solid var(--bdd);padding:.82rem 1.6rem;border-radius:10px;font-family:'Outfit',sans-serif;font-size:.88rem;font-weight:500;cursor:pointer;transition:all .25s;}
.bg:hover{border-color:var(--mid);color:var(--viv);}
.panel{background:var(--wash);border:1px solid var(--bdd);border-radius:22px;padding:1.8rem;animation:lft .8s ease .35s both;}
.ph{display:flex;align-items:center;justify-content:space-between;margin-bottom:1.4rem;}
.pt{font-size:.74rem;font-weight:600;text-transform:uppercase;letter-spacing:.12em;color:var(--muted);}
.plv{display:flex;align-items:center;gap:.38rem;font-size:.7rem;font-weight:600;color:#059669;}
.ldot{width:6px;height:6px;border-radius:50%;background:#10b981;animation:pu 1.5s infinite;}
.sg{display:grid;grid-template-columns:1fr 1fr;gap:.9rem;margin-bottom:1.1rem;}
.sb{background:#fff;border:1px solid var(--bd);border-radius:13px;padding:.9rem 1.1rem;}
.sn{font-family:'Cormorant Garamond',serif;font-size:2.1rem;font-weight:700;color:var(--viv);line-height:1;}
.sl{font-size:.68rem;color:var(--muted);font-weight:500;margin-top:.22rem;text-transform:uppercase;letter-spacing:.08em;}
.br{display:flex;align-items:center;gap:.75rem;margin-bottom:.65rem;}
.bl{font-size:.7rem;color:var(--muted);width:76px;flex-shrink:0;}
.bt{flex:1;height:6px;background:var(--bd);border-radius:3px;overflow:hidden;}
.bf{height:100%;border-radius:3px;background:linear-gradient(90deg,var(--viv),var(--lite));}
.bv{font-size:.7rem;font-weight:600;color:#2e2250;width:28px;text-align:right;}
.strip{background:#fff;padding:2.4rem 5vw;border-top:1px solid var(--bd);border-bottom:1px solid var(--bd);}
.si{max-width:1260px;margin:0 auto;display:grid;grid-template-columns:repeat(4,1fr);background:var(--wash);border:1px solid var(--bdd);border-radius:14px;overflow:hidden;}
.sc{background:var(--wash);padding:1.9rem;text-align:center;}
.scn{font-family:'Cormorant Garamond',serif;font-size:2.7rem;font-weight:700;color:var(--viv);line-height:1;}
.scl{font-size:.74rem;color:var(--muted);font-weight:400;margin-top:.38rem;letter-spacing:.09em;text-transform:uppercase;}
@keyframes up{from{opacity:0;transform:translateY(26px);}to{opacity:1;transform:translateY(0);}}
@keyframes lft{from{opacity:0;transform:translateX(26px);}to{opacity:1;transform:translateX(0);}}
@keyframes pu{0%,100%{opacity:1;transform:scale(1);}50%{opacity:.4;transform:scale(1.4);}}
</style></head><body>
<nav>
  <div class="logo"><div class="emblem">🇱🇰</div><div class="wordmark">District <span>Insights</span></div></div>
  <div class="links">
    <a class="lnk" href="#">Platform</a>
    <a class="lnk" href="#">About</a>
    <a class="lnk" href="#">Modules</a>
    <a class="lnk cta" href="#">Get Access</a>
  </div>
</nav>
<div class="hero">
  <div class="hl">
    <div class="eye"><div class="edot"></div>Interactive Decision Support Platform</div>
    <h1>Sri Lanka<br><em>District</em><br>Insights</h1>
    <p class="hp">Evidence-based analytics across poverty, child protection, and skill development — empowering smarter resource allocation for 25 districts nationwide.</p>
    <div class="btns"><button class="bp">Explore the Platform →</button><button class="bg">View Methodology</button></div>
  </div>
  <div class="hr">
    <div class="panel">
      <div class="ph"><div class="pt">Live Dashboard Preview</div><div class="plv"><div class="ldot"></div>Real-time</div></div>
      <div class="sg">
        <div class="sb"><div class="sn">25</div><div class="sl">Districts</div></div>
        <div class="sb"><div class="sn">3</div><div class="sl">Modules</div></div>
        <div class="sb"><div class="sn">98%</div><div class="sl">Coverage</div></div>
        <div class="sb"><div class="sn">2024</div><div class="sl">Data Year</div></div>
      </div>
      <div class="br"><div class="bl">Poverty Index</div><div class="bt"><div class="bf" style="width:78%"></div></div><div class="bv">78%</div></div>
      <div class="br"><div class="bl">Child Welfare</div><div class="bt"><div class="bf" style="width:64%"></div></div><div class="bv">64%</div></div>
      <div class="br"><div class="bl">Skill Dev.</div><div class="bt"><div class="bf" style="width:52%"></div></div><div class="bv">52%</div></div>
    </div>
  </div>
</div>
<div class="strip">
  <div class="si">
    <div class="sc"><div class="scn">25</div><div class="scl">Districts Covered</div></div>
    <div class="sc"><div class="scn">3</div><div class="scl">Analytics Modules</div></div>
    <div class="sc"><div class="scn">100%</div><div class="scl">Data Coverage</div></div>
    <div class="sc"><div class="scn">Live</div><div class="scl">Integration Status</div></div>
  </div>
</div>
</body></html>""", height=920, scrolling=False)

# ── ABOUT ────────────────────────────────────────────────────────────────────
components.html("""<!DOCTYPE html><html><head><meta charset="UTF-8">
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,600;0,700&family=Outfit:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box;}
:root{--ink:#0f0a1e;--muted:#6b5f8a;--viv:#6d28d9;--mid:#8b5cf6;--pale:#ede9fe;--wash:#f5f3ff;--bd:rgba(109,40,217,.13);--bdd:rgba(109,40,217,.26);}
body{font-family:'Outfit',sans-serif;background:#fff;color:var(--ink);}
.about{padding:6rem 5vw;max-width:1260px;margin:0 auto;}
.sl{display:inline-flex;align-items:center;gap:.5rem;font-size:.7rem;font-weight:700;letter-spacing:.2em;text-transform:uppercase;color:var(--viv);margin-bottom:.9rem;}
.sl::before{content:'';display:block;width:22px;height:2px;background:var(--viv);}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:5.5rem;align-items:start;}
h2{font-family:'Cormorant Garamond',serif;font-size:clamp(1.9rem,3.2vw,3rem);font-weight:700;line-height:1.1;color:var(--ink);letter-spacing:-.02em;margin:0 0 1.3rem;}
h2 strong{color:var(--viv);}
p{font-size:.94rem;font-weight:300;color:var(--muted);line-height:1.8;margin-bottom:1.1rem;}
.pillars{display:flex;flex-direction:column;gap:.95rem;}
.pillar{display:flex;gap:1.1rem;align-items:flex-start;padding:1.1rem 1.2rem;background:var(--wash);border:1px solid var(--bd);border-radius:14px;transition:border-color .25s;}
.pillar:hover{border-color:var(--mid);}
.pico{width:40px;height:40px;border-radius:10px;background:#fff;border:1px solid var(--bdd);display:flex;align-items:center;justify-content:center;font-size:1.05rem;flex-shrink:0;}
.ptitle{font-weight:600;font-size:.86rem;color:var(--ink);margin-bottom:.18rem;}
.pdesc{font-size:.78rem;color:var(--muted);line-height:1.55;}
</style></head><body>
<div class="about">
  <div class="sl">About the Platform</div>
  <div class="grid">
    <div>
      <h2>Built for those who<br><strong>shape policy</strong><br>across Sri Lanka</h2>
      <p>Sri Lanka District Insights is a decision-support platform designed for government officials, policymakers, and development planners. It translates complex district-level data into clear, actionable intelligence.</p>
      <p>By consolidating socio-economic indicators across three critical domains, the platform enables faster, more equitable resource allocation — ensuring interventions reach the communities that need them most.</p>
    </div>
    <div class="pillars">
      <div class="pillar"><div class="pico">📊</div><div><div class="ptitle">Evidence-First Analytics</div><div class="pdesc">Every insight is grounded in verified district-level data. Transparent methodology, fully traceable sources.</div></div></div>
      <div class="pillar"><div class="pico">🎯</div><div><div class="ptitle">Equity-Centered Design</div><div class="pdesc">Built to surface disparities and prioritize underserved regions — ensuring no district is overlooked.</div></div></div>
      <div class="pillar"><div class="pico">⚡</div><div><div class="ptitle">Real-Time Integration</div><div class="pdesc">Dashboards reflect the latest data, enabling decision-makers to respond to emerging needs swiftly.</div></div></div>
    </div>
  </div>
</div>
</body></html>""", height=560, scrolling=False)

# ── MODULE CARDS ─────────────────────────────────────────────────────────────
# ── CARDS SECTION HEADER ─────────────────────────────────────────────────────
components.html("""<!DOCTYPE html><html><head><meta charset="UTF-8">
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@600;700&family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box;}
:root{--ink:#0f0a1e;--muted:#6b5f8a;--viv:#6d28d9;--mid:#8b5cf6;--lite:#a78bfa;--pale:#ede9fe;--wash:#f5f3ff;--bd:rgba(109,40,217,.13);--bdd:rgba(109,40,217,.26);}
body{font-family:'Outfit',sans-serif;background:var(--wash);color:var(--ink);}
.sec{padding:5.5rem 5vw 0;border-top:1px solid var(--bd);}
.inner{max-width:1260px;margin:0 auto;}
.hd{text-align:center;margin-bottom:3.5rem;}
.sl{display:inline-flex;align-items:center;justify-content:center;gap:.5rem;font-size:.7rem;font-weight:700;letter-spacing:.2em;text-transform:uppercase;color:var(--viv);margin-bottom:.9rem;}
.sl::before,.sl::after{content:'';display:block;width:22px;height:2px;background:var(--viv);}
h2{font-family:'Cormorant Garamond',serif;font-size:clamp(1.9rem,3.2vw,2.9rem);font-weight:700;line-height:1.1;color:var(--ink);letter-spacing:-.02em;margin:.5rem 0 .9rem;}
.sub{font-size:.94rem;color:var(--muted);font-weight:300;max-width:500px;margin:0 auto;line-height:1.7;}
</style></head><body>
<div class="sec"><div class="inner"><div class="hd">
  <div class="sl">Analytics Modules</div>
  <h2>Three domains.<br>One unified platform.</h2>
  <p class="sub">Each module delivers deep, district-level intelligence across a critical domain of social and economic development.</p>
</div></div></div>
</body></html>""", height=320, scrolling=False)

# ── CARDS GRID ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* card grid wrapper */
[data-testid="stHorizontalBlock"].cards-row {gap:1.6rem !important;}
div[data-card] {
    background:#fff;border:1px solid rgba(109,40,217,.26);border-radius:20px;
    overflow:hidden;display:flex;flex-direction:column;
    transition:transform .3s cubic-bezier(.34,1.4,.64,1),box-shadow .3s,border-color .3s;
    padding:0;
}
</style>
""", unsafe_allow_html=True)

CARD_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@700&family=Outfit:wght@300;400;500;600;700&display=swap');
*{margin:0;padding:0;box-sizing:border-box;}
:root{--ink:#0f0a1e;--muted:#6b5f8a;--viv:#6d28d9;--mid:#8b5cf6;--pale:#ede9fe;--wash:#f5f3ff;--bd:rgba(109,40,217,.13);--bdd:rgba(109,40,217,.26);}
html,body{height:100%;margin:0;}
body{font-family:'Outfit',sans-serif;background:var(--wash);color:var(--ink);}
.card{background:#fff;border:1px solid var(--bdd);border-radius:20px;overflow:hidden;display:flex;flex-direction:column;height:100%;transition:transform .3s cubic-bezier(.34,1.4,.64,1),box-shadow .3s,border-color .3s;}
.card:hover{transform:translateY(-8px);box-shadow:0 28px 60px rgba(109,40,217,.18);border-color:var(--mid);}
.ct{padding:1.9rem 1.9rem 1.4rem;border-bottom:1px solid var(--bd);position:relative;}
.cn{position:absolute;top:1.3rem;right:1.6rem;font-family:'Cormorant Garamond',serif;font-size:3.8rem;font-weight:700;color:var(--pale);line-height:1;user-select:none;}
.ci{width:48px;height:48px;border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:1.35rem;margin-bottom:1.2rem;border:1px solid var(--bdd);}
.i1{background:linear-gradient(135deg,#ede9fe,#ddd6fe);}
.i2{background:linear-gradient(135deg,#f5f3ff,#ede9fe);}
.i3{background:linear-gradient(135deg,#eef2ff,#e0e7ff);}
.tag{display:inline-block;font-size:.64rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--viv);background:var(--pale);border:1px solid var(--bdd);padding:.2rem .7rem;border-radius:100px;margin-bottom:.8rem;}
.ctitle{font-family:'Cormorant Garamond',serif;font-size:1.65rem;font-weight:700;color:var(--ink);letter-spacing:-.01em;}
.cb{padding:1.4rem 1.9rem 0;display:flex;flex-direction:column;flex:1;}
.desc{font-size:.85rem;color:var(--muted);line-height:1.7;margin-bottom:1.3rem;font-weight:300;}
.mts{display:flex;flex-direction:column;gap:.5rem;margin-bottom:1.3rem;}
.mt{display:flex;align-items:center;gap:.55rem;font-size:.78rem;color:var(--muted);}
.md{width:5px;height:5px;border-radius:50%;background:var(--mid);flex-shrink:0;}
.cf{display:flex;align-items:center;padding-top:1rem;border-top:1px solid var(--bd);margin-bottom:1.3rem;}
.act{display:flex;align-items:center;gap:.38rem;font-size:.7rem;font-weight:600;color:#059669;}
.ad{width:6px;height:6px;border-radius:50%;background:#10b981;animation:pu 1.8s infinite;}
/* Visual button inside card — just decoration, real st.button sits on top */
.btn-visual{display:block;width:100%;padding:.85rem 1rem;background:var(--viv);color:#fff;border:none;border-radius:12px;font-family:'Outfit',sans-serif;font-size:.875rem;font-weight:600;letter-spacing:.03em;text-align:center;box-shadow:0 4px 18px rgba(109,40,217,.28);margin-top:auto;margin-bottom:1.6rem;pointer-events:none;}
.arr{display:inline-block;}
@keyframes pu{0%,100%{opacity:1;transform:scale(1);}50%{opacity:.4;transform:scale(1.5);}}
</style>
"""

col1, col2, col3 = st.columns(3)

with col1:
    components.html(CARD_CSS + """<div class="card">
        <div class="ct"><div class="cn">01</div><div class="ci i1">💰</div><div class="tag">Economic Analysis</div><div class="ctitle">Poverty Analysis</div></div>
        <div class="cb">
          <p class="desc">Identify high-poverty districts and allocate development projects where social impact is greatest.</p>
          <div class="mts"><div class="mt"><div class="md"></div>District-level poverty mapping</div><div class="mt"><div class="md"></div>Resource allocation scoring</div><div class="mt"><div class="md"></div>Trend forecasting models</div></div>
          <div class="cf"><div class="act"><div class="ad"></div>Active</div></div>
          <div class="btn-visual">Explore Poverty Data <span class="arr">→</span></div>
        </div>
    </div>""", height=520, scrolling=False)
    if st.button("Explore Poverty Data →", key="btn_poverty", use_container_width=True):
        st.switch_page("pages/poverty.py")

with col2:
    components.html(CARD_CSS + """<div class="card">
        <div class="ct"><div class="cn">02</div><div class="ci i2">🛡️</div><div class="tag">Social Protection</div><div class="ctitle">Child Protection</div></div>
        <div class="cb">
          <p class="desc">Analyze vulnerable child populations and optimize child protection service deployment across regions.</p>
          <div class="mts"><div class="mt"><div class="md"></div>Vulnerability heat-mapping</div><div class="mt"><div class="md"></div>Service gap identification</div><div class="mt"><div class="md"></div>Intervention prioritization</div></div>
          <div class="cf"><div class="act"><div class="ad"></div>Active</div></div>
          <div class="btn-visual">View Child Protection Insights <span class="arr">→</span></div>
        </div>
    </div>""", height=520, scrolling=False)
    if st.button("View Child Protection Insights →", key="btn_child", use_container_width=True):
        st.switch_page("pages/childcase.py")

with col3:
    components.html(CARD_CSS + """<div class="card">
        <div class="ct"><div class="cn">03</div><div class="ci i3">🎓</div><div class="tag">Development Services</div><div class="ctitle">Skill Development</div></div>
        <div class="cb">
          <p class="desc">Understand district-level skill development program allocation and workforce readiness gaps.</p>
          <div class="mts"><div class="mt"><div class="md"></div>Needs assessment dashboard</div><div class="mt"><div class="md"></div>Facility coverage analysis</div><div class="mt"><div class="md"></div>Workforce gap metrics</div></div>
          <div class="cf"><div class="act"><div class="ad"></div>Active</div></div>
          <div class="btn-visual">Skill Allocation Dashboard <span class="arr">→</span></div>
        </div>
    </div>""", height=520, scrolling=False)
    if st.button("Skill Allocation Dashboard →", key="btn_skills", use_container_width=True):
        st.switch_page("pages/skill_dev.py")

# ── FOOTER ───────────────────────────────────────────────────────────────────
components.html("""<!DOCTYPE html><html><head><meta charset="UTF-8">
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@700&family=Outfit:wght@400&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box;}
body{font-family:'Outfit',sans-serif;background:#f5f3ff;}
footer{padding:2.6rem 5vw;border-top:1px solid rgba(109,40,217,.13);}
.inner{max-width:1260px;margin:0 auto;display:flex;align-items:center;justify-content:space-between;}
.logo{font-family:'Cormorant Garamond',serif;font-size:1.1rem;font-weight:700;color:#0f0a1e;}
.logo span{color:#6d28d9;}
.meta{font-size:.76rem;color:#6b5f8a;}
</style></head><body>
<footer><div class="inner">
  <div class="logo">District <span>Insights</span></div>
  <div class="meta">© 2024 Sri Lanka District Insights · Built for policymakers</div>
  <div style="font-size:1.4rem">🇱🇰</div>
</div></footer>
</body></html>""", height=88, scrolling=False)