# ui/frontend.py
import streamlit as st
import requests

BASE_URL = "http://localhost:8000/api"  # FastAPI server

st.title("Region Recommender")

preference = st.text_input("Enter your preference (e.g., 'low poverty rural areas')")

if st.button("Get Recommendations"):
    response = requests.post(f"{BASE_URL}/recommend", json={"preference": preference})
    if response.status_code == 200:
        regions = response.json()["top_regions"]
        st.session_state["regions"] = regions
        st.write("Top 10 Regions:")
        for r in regions:
            st.write(r)

# Display insights on click (simulate click with selection)
if "regions" in st.session_state:
    selected_region = st.selectbox("Select a region for insights", st.session_state["regions"])
    if selected_region:
        response = requests.get(f"{BASE_URL}/insights/{selected_region}")
        if response.status_code == 200:
            insights = response.json()["insights"]
            st.write(f"Insights for {selected_region}:")
            st.json(insights)  # Or format as table/chart