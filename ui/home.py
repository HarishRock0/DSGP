import streamlit as st

st.set_page_config(
    page_title="Sri Lanka District Insights",
    layout="wide"
)

# -------------------- CUSTOM CSS --------------------
st.markdown("""
<style>
    body {
        background-color: #0e1117;
    }
    .card {
        background: #161b22;
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        transition: all 0.3s ease;
        height: 100%;
    }
    .card:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 40px rgba(88,166,255,0.15);
        border: 1px solid #58a6ff;
    }
    .card h2 {
        margin-bottom: 0.5rem;
    }
    .card p {
        color: #8b949e;
        font-size: 1.05rem;
    }
    .card-btn {
        margin-top: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.markdown("""
<div style="text-align:center; padding: 2.5rem 0;">
    <h1 style="color:lightblue;">Sri Lanka District Insights</h1>
    <p style="font-size:1.2rem; color:#8b949e;">
        Poverty • Child Protection • Mental Health — Interactive Decision Support
    </p>
</div>
""", unsafe_allow_html=True)

# -------------------- SESSION STATE --------------------
if "page" not in st.session_state:
    st.session_state.page = "home"

# -------------------- HOME PAGE CONTENT --------------------
col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown("""
    <div class="card">
        <h2>Poverty Analysis</h2>
        <p>
            Identify high-poverty districts and allocate
            development projects where impact is highest.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Explore Poverty Data", key="poverty_btn", use_container_width=True):
        # Updated path to include pages/ folder
        st.switch_page("pages/poverty.py")

with col2:
    st.markdown("""
    <div class="card">
        <h2>Child Protection</h2>
        <p>
            Analyze vulnerable child populations and
            optimize child protection service deployment.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("View Child Protection Insights", key="child_btn", use_container_width=True):
        # Updated path to include pages/ folder
        st.switch_page("pages/childcase.py")

with col3:
    st.markdown("""
    <div class="card">
        <h2>Mental Health Services</h2>
        <p>
            Understand district-level mental health needs
            and resource gaps.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Mental Health Dashboard", key="mental_btn", use_container_width=True):
        # Updated path to include pages/ folder
        st.switch_page("pages/mentalhealth.py")