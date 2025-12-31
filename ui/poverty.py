import sys
import os
import streamlit as st
import pandas as pd

# 1. Update the path FIRST to ensure project modules are findable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from service.recommendation_service import RecommendationService

st.set_page_config(
    page_title="Region Recommendation System",
    layout="wide"
)

# Use caching to prevent the slow model from reloading on every interaction
@st.cache_resource
def load_service():
    return RecommendationService()

st.title("📍 Intelligent Region Recommendation System")
st.write("Enter your preferences to get top 10 recommended regions")

# Initialize the service once
service = load_service()

user_input = st.text_input(
    "Describe your preference (e.g. low poverty, high population, urban areas):"
)

if st.button("Get Recommendations"):
    if user_input.strip() == "":
        st.warning("Please enter a preference.")
    else:
        with st.spinner("Analyzing preferences..."):
            result = service.get_recommendations(user_input)

        df = result["recommendations"]

        st.success("Top 10 Recommended Regions")
        st.dataframe(
            df,
            use_container_width=True
        )