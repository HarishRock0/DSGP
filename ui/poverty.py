import streamlit as st
import pandas as pd
from service.recommendation_service import RecommendationService

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


st.set_page_config(
    page_title="Region Recommendation System",
    layout="wide"
)

st.title("📍 Intelligent Region Recommendation System")
st.write("Enter your preferences to get top 10 recommended regions")

service = RecommendationService()

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
