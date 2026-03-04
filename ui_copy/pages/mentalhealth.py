import sys
import os
import streamlit as st
import pandas as pd

# Project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from service.mental_health_service import MentalHealthService
from agents.coordinator_agent import CoordinatorAgent

# Streamlit page config
st.set_page_config(
    page_title="Region & Mental Health Recommendation System",
    layout="wide"
)

# Load RecommendationService with caching
@st.cache_resource
def load_service():
    return MentalHealthService(PROJECT_ROOT)

service = load_service()

@st.cache_data(show_spinner=False)
def get_cached_recommendations(text):
    return service.get_mental_health(text)

# Streamlit UI
st.title("Intelligent Region & Mental Health Recommendation System")
st.write("Enter your preferences to get top 10 recommended regions and mental health insights.")

user_input = st.text_input(
    "Describe your preference (e.g. low risk, high population, urban areas):"
)

if user_input:
    # Region recommendations
    recommendations = get_cached_recommendations(user_input)
    st.write("### Top 10 Recommended Regions")
    if recommendations:
        if isinstance(recommendations, list):
            st.table(pd.DataFrame(recommendations))
        else:
            st.write(recommendations)
    else:
        st.write("No recommendations found.")

    # Mental health recommendations
    mental_health_info = service.get_mental_health(user_input)

    st.write("### Mental Health Insights")
    if isinstance(mental_health_info, dict) and mental_health_info.get("districts"):
        df = pd.DataFrame(mental_health_info["districts"])
        st.table(df)
    else:
        st.write("No mental health recommendations available.")
