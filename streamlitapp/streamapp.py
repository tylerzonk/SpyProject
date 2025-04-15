import streamlit as st
from streamlit_navigation_bar import st_navbar
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="Market Dashboard", layout="wide")

# --- NAVIGATION BAR ---
page = st_navbar(["Home", "Volume", "Volatility", "Technical", "About"])

# --- PAGE: HOME ---
if page == "Home":
    st.title("SPY Real-Time Chart")

    # Simulate 1-min candle data (replace with live API data)
    now = datetime.now()
    times = [now - timedelta(minutes=i) for i in range(90)][::-1]
    df = pd.DataFrame({
        'time': times,
        'open': 440 + pd.Series(range(90)).apply(lambda x: x * 0.01),
        'high': 441,
        'low': 439,
        'close': 440.5,
    })

    fig = go.Figure(data=[go.Candlestick(
        x=df['time'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close']
    )])
    fig.update_layout(height=400)

    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Estimated Day Volatility", "1.2%")
    col2.metric("Estimated Day Volume", "88M")
    col3.metric("Remaining Estimated Volume", "37M")

# --- PAGE: VOLUME ---
elif page == "Volume":
    st.title("Volume Analysis")
    view = st.selectbox("Select Volume View", [
        "Past 10 Years",
        "Monthly Prediction Accuracy",
        "2025 Daily Prediction Accuracy"
    ])
    st.write(f"Showing volume data for: **{view}**")
    # TODO: Add your volume chart/data here

# --- PAGE: VOLATILITY ---
elif page == "Volatility":
    st.title("Volatility Analysis")
    view = st.selectbox("Select Volatility View", [
        "Past 10 Years",
        "Monthly Prediction Accuracy",
        "2025 Daily Prediction Accuracy"
    ])
    st.write(f"Showing volatility data for: **{view}**")
    # TODO: Add your volatility chart/data here

# --- PAGE: TECHNICAL ---
elif page == "Technical":
    st.title("Technical Breakdown")
    st.markdown("""
    ### Data Pipeline
    - Fetched real-time + historical 1-min candles using Polygon.io
    - Generated rolling statistical features

    ### Modeling
    - Trained XGBoost regressors for volume and volatility
    - Evaluated with MAE, R², and prediction interval accuracy

    ### Deployment
    - Frontend: Streamlit
    - Backend: FastAPI for serving model predictions
    - Containerized using Docker Compose
    """)

# --- PAGE: ABOUT ---
elif page == "About":
    st.title("About This Project")
    st.markdown("""
    This dashboard is part of a project to forecast SPY volume and volatility using real-time market data and machine learning models.

    ### Built With:
    - Python, Pandas, XGBoost, Streamlit
    - FastAPI backend
    - Docker Compose for container orchestration

    ### Goals:
    - Deliver interpretable and accurate daily forecasts
    - Provide a clean and responsive user interface
    - Lay groundwork for expanded multi-symbol support

    ### Author:
    [Your Name]
    """)

