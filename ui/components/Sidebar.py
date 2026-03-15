"""
ui/components/Sidebar.py
"""
import streamlit as st


def render_sidebar():
    if "sidebar_open" not in st.session_state:
        st.session_state.sidebar_open = True

    is_open = st.session_state.sidebar_open
    W       = 260 if is_open else 64
    Ws      = str(W)

    st.markdown(
        "<style>"
        "@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&display=swap');"

        # ── Hide Streamlit's default page-nav list at top of sidebar ──
        "[data-testid='stSidebarNav'] { display: none !important; }"

        # ── Sidebar shell — FIXED width so it never resizes ──
        "[data-testid='stSidebar'] {"
        "  background: linear-gradient(180deg,#1a0533 0%,#0d0220 100%) !important;"
        "  border-right: 1px solid rgba(168,85,247,.25) !important;"
        "  box-shadow: 4px 0 32px rgba(124,58,237,.18) !important;"
        "  min-width: " + Ws + "px !important;"
        "  max-width: " + Ws + "px !important;"
        "  width:     " + Ws + "px !important;"
        "  padding: 0 !important;"
        "  transition: none !important;"   # disable Streamlit's own transition
        "}"
        "[data-testid='stSidebar'] > div:first-child {"
        "  padding: 0 !important;"
        "  width: " + Ws + "px !important;"
        "}"
        "[data-testid='stSidebarContent'] {"
        "  padding: 0 !important; gap: 0 !important;"
        "  width: " + Ws + "px !important;"
        "  overflow-x: hidden !important;"
        "}"

        # ── Hide collapse handle ──
        "[data-testid='collapsedControl'] { display: none !important; }"

        # ── Logo ──
        ".sb-logo {"
        "  display: flex; align-items: center; gap: 10px;"
        "  padding: 20px 14px 16px;"
        "  border-bottom: 1px solid rgba(168,85,247,.15);"
        "  overflow: hidden; white-space: nowrap; flex-shrink: 0;"
        "}"
        ".sb-logo-icon { font-size: 1.3rem; flex-shrink: 0; }"
        ".sb-logo-text {"
        "  font-size: .80rem; font-weight: 700; color: #c084fc;"
        "  letter-spacing: .08em; text-transform: uppercase; white-space: nowrap;"
        "  display: " + ("block" if is_open else "none") + ";"
        "}"

        # ── Section labels ──
        ".sb-label {"
        "  font-size: .58rem; font-weight: 700; letter-spacing: .16em;"
        "  text-transform: uppercase; color: rgba(168,85,247,.55);"
        "  padding: 12px 14px 3px; white-space: nowrap;"
        "  display: " + ("block" if is_open else "none") + ";"
        "}"

        # ── Divider ──
        ".sb-div { height:1px; background:rgba(168,85,247,.15); margin:5px 10px; }"

        # ── Remove all extra spacing from Streamlit's widget wrappers ──
        "[data-testid='stSidebar'] [data-testid='stVerticalBlock'] { gap: 0 !important; }"
        "[data-testid='stSidebar'] .element-container { margin: 0 !important; padding: 0 !important; }"
        "[data-testid='stSidebar'] .stButton { margin: 0 !important; }"

        # ── Nav buttons ──
        "[data-testid='stSidebar'] .stButton > button {"
        "  width: 100% !important;"
        "  background: transparent !important;"
        "  border: none !important;"
        "  border-radius: 0 !important;"
        "  border-right: 3px solid transparent !important;"
        "  color: #e9d5ff !important;"
        "  font-family: 'DM Sans',sans-serif !important;"
        "  font-size: .875rem !important;"
        "  font-weight: 500 !important;"
        "  text-align: left !important;"
        "  justify-content: flex-start !important;"
        "  padding: 10px 14px !important;"
        "  box-shadow: none !important;"
        "  white-space: nowrap !important;"
        "  overflow: hidden !important;"
        "  transition: background .15s ease !important;"
        "  cursor: pointer !important;"
        "  min-height: unset !important;"
        "  line-height: 1.4 !important;"
        "}"
        "[data-testid='stSidebar'] .stButton > button:hover {"
        "  background: rgba(124,58,237,.25) !important; color: #fff !important;"
        "}"
        "[data-testid='stSidebar'] .stButton > button:focus {"
        "  box-shadow: none !important; outline: none !important;"
        "}"

        # ── Sub-item ──
        "[data-testid='stSidebar'] .sb-sub .stButton > button {"
        "  padding-left: " + ("30px" if is_open else "14px") + " !important;"
        "  font-size: .82rem !important;"
        "  color: #c4b5d4 !important;"
        "}"

        # ── Active highlight ──
        "[data-testid='stSidebar'] .sb-active .stButton > button {"
        "  background: rgba(124,58,237,.30) !important;"
        "  border-right: 3px solid #a855f7 !important;"
        "  color: #fff !important;"
        "}"

        # ── Toggle button — fixed to right edge of sidebar ──
        "[data-testid='stSidebar'] .sb-toggle .stButton > button {"
        "  position: fixed !important;"
        "  top: 50% !important;"
        "  left: " + str(W - 13) + "px !important;"
        "  transform: translateY(-50%) !important;"
        "  width: 26px !important; height: 52px !important;"
        "  min-height: unset !important; padding: 0 !important;"
        "  border-radius: 0 8px 8px 0 !important;"
        "  background: linear-gradient(180deg,#3d1170,#7c3aed) !important;"
        "  border: 1px solid rgba(168,85,247,.5) !important;"
        "  border-left: none !important;"
        "  color: #e9d5ff !important; font-size: .6rem !important;"
        "  z-index: 99999 !important;"
        "  box-shadow: 3px 0 12px rgba(124,58,237,.4) !important;"
        "  cursor: pointer !important;"
        "}"
        "[data-testid='stSidebar'] .sb-toggle .stButton > button:hover {"
        "  background: linear-gradient(180deg,#7c3aed,#a855f7) !important;"
        "}"
        "</style>",
        unsafe_allow_html=True
    )

    # ── Active page detection ─────────────────────────────────────────────
    try:
        current = st.context.pages.get("current", {}).get("script_path", "")
    except Exception:
        current = ""

    def _a(kw):
        return "sb-active" if kw in current else ""

    # ── Sidebar content ───────────────────────────────────────────────────
    with st.sidebar:

        # Logo
        st.markdown(
            '<div class="sb-logo">'
            '<span class="sb-logo-icon">&#127473;&#127472;</span>'
            '<span class="sb-logo-text">DISTRICT INSIGHTS</span>'
            '</div>',
            unsafe_allow_html=True
        )

        # Home
        st.markdown('<div class="' + _a("home") + '">', unsafe_allow_html=True)
        if st.button("🏠  Home" if is_open else "🏠", key="nav_home"):
            st.switch_page("home.py")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="sb-div"></div>', unsafe_allow_html=True)

        # Poverty Alleviation
        st.markdown('<div class="sb-label">Poverty Alleviation</div>', unsafe_allow_html=True)

        st.markdown('<div class="' + _a("poverty.py") + '">', unsafe_allow_html=True)
        if st.button("📊  Regional Insights" if is_open else "📊", key="nav_poverty"):
            st.switch_page("pages/poverty.py")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="sb-sub ' + _a("poverty_resource") + '">', unsafe_allow_html=True)
        if st.button("›  Resource Allocation" if is_open else "›", key="nav_poverty_res"):
            st.switch_page("pages/poverty_resource.py")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="sb-div"></div>', unsafe_allow_html=True)

        # Child Protection
        st.markdown('<div class="sb-label">Child Protection</div>', unsafe_allow_html=True)

        st.markdown('<div class="' + _a("childcase") + '">', unsafe_allow_html=True)
        if st.button("🛡️  Regional Insights" if is_open else "🛡️", key="nav_child"):
            st.switch_page("pages/childcase.py")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="sb-sub ' + _a("childprotection") + '">', unsafe_allow_html=True)
        if st.button("›  Resource Allocation" if is_open else "›", key="nav_child_res"):
            st.switch_page("pages/childprotection.py")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="sb-div"></div>', unsafe_allow_html=True)

        # Toggle
        st.markdown('<div class="sb-toggle">', unsafe_allow_html=True)
        if st.button("◀" if is_open else "▶", key="sb_toggle"):
            st.session_state.sidebar_open = not st.session_state.sidebar_open
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)