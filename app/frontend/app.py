import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import requests

st.title("Data Dashboard")

# Fetch data from the FastAPI service
try:
    response = requests.get("http://backend:8000/data")
    data = pd.DataFrame(response.json())

    st.write("### Raw Datuh")
    st.dataframe(data)
    
    st.write("### Bar Chart")
    st.bar_chart(data.set_index("name"))
except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.write("Make sure the backend service is running.")