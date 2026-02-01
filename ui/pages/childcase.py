import sys
import os
import streamlit as st
import pandas as pd

# ---------------- PATH SETUP ----------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------- SERVICE IMPORT ----------------
from service.child_protection_service import ChildProtectionService

# ---------------- STREAMLIT CONFIG ----------------
st.set_page_config(
    page_title="Child Protection Recommendation System",
    layout="wide"
)

# ---------------- CACHING ----------------
@st.cache_resource
def load_service():
    return ChildProtectionService(PROJECT_ROOT)

@st.cache_data(show_spinner=False)
def get_cached_recommendations(text):
    return service.get_child_case_recommendations(text)

# ---------------- LOAD SERVICE ----------------
service = load_service()

# ---------------- UI ----------------
st.title("Child Protection Risk Recommendation System")
st.write(
    "Describe the type of child protection concern to identify districts with higher relevance."
)

user_input = st.text_input(
    "Describe your concern (e.g. high abuse risk, neglected children, vulnerable districts):"
)

if user_input:
    with st.spinner("Analyzing child case data..."):
        result = get_cached_recommendations(user_input)

    if result and "recommendations" in result:
        df = pd.DataFrame(result["recommendations"])

        st.subheader("🔍 Top 10 Relevant Districts")
        st.dataframe(
            df.rename(
                columns={
                    "District": "District",
                    "Avg_cases": "Average Child Cases"
                }
            ),
            use_container_width=True
        )
    else:
        st.warning("No recommendations found. Try refining your description.")
