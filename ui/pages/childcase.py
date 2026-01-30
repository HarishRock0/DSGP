
import sys
import os
import streamlit as st
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Now Python can find the 'service' folder in the PROJECT_ROOT
from service.recommendation_service import RecommendationService

st.set_page_config(
    page_title="Region Recommendation System",
    layout="wide"
)

# Use caching to prevent the slow model from reloading on every interaction
@st.cache_resource
def load_service():
    return RecommendationService()

@st.cache_data(show_spinner=False)
def get_cached_recommendations(text):
    return service.get_recommendations(text)


st.title("Intelligent Region Recommendation System")
st.write("Enter your preferences to get top 10 recommended regions")


user_input = st.text_input(
    "Describe your preference (e.g. low risk, high population, urban areas):"
)
