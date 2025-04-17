from streamlit_navigation_bar import st_navbar
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from datetime import datetime, time
import functions.color as c
from matplotlib import style
import pytz
import joblib
import os

volume_model_path = os.path.join('model', 'volumemodel.pkl')
volatility_model_path = os.path.join('model', 'volatilitymodel.pkl')
base_dir = os.path.dirname(os.path.dirname(__file__))
volu_data_path = os.path.join(base_dir, 'cleaned_data', 'spy_volume.csv')
vola_data_path = os.path.join(base_dir, 'cleaned_data', 'spy_volatility.csv')
today_volume_datapath = os.path.join(base_dir, 'cleaned_data', 'spy_1min_current_day.csv')

# Today's predicted volume
volume_model = joblib.load(volume_model_path)
yesterday_volu = pd.read_csv(volu_data_path, parse_dates=True)
yesterday_volu.set_index('Date', inplace=True)
yesterday_volu = yesterday_volu.tail(1)
volu_X = yesterday_volu.drop(columns=['Next_Day_Volume'], errors='ignore')
today_volume = volume_model.predict(volu_X)
today_m_volume = today_volume/1000000
yesterday_volume = yesterday_volu['Volume']

# Today's predicted volatility
volatility_model = joblib.load(volatility_model_path)
yesterday_vola = pd.read_csv(vola_data_path, parse_dates=True)
yesterday_vola = yesterday_vola.tail(1)
vola_X = yesterday_vola[['Open', 'High', 'Low', 'Close', 'Volume', 'Day_of_Week', 'Week_of_Year', 'Month',
        'Daily_Return', 'Price_Range', 'Close_to_Open_Change', 'Average_Price', 'Lagged_Close',
        'Lagged_Volume', 'Volume_Ratio', 'Lagged_Return', 'Rolling_Volatility', 'Close_to_High_Ratio']]
today_volatility = volatility_model.predict(vola_X)
today_volatility = today_volatility * 100

# Estimated Remaining Volume
current_vola = pd.read_csv(today_volume_datapath, parse_dates=True)
current_vola_sum = current_vola['volume'].sum()
volu_total = today_volume - current_vola_sum
volu_m_total = volu_total/1000000

st.set_page_config(page_title="Market Dashboard", layout="wide")

# --- NAVIGATION BAR ---
page = st_navbar(["Home", "Volume", "Volatility", "Technical", "About"])

# --- PAGE: HOME ---
if page == "Home":
    # Get current time in US/Eastern
    eastern = pytz.timezone("US/Eastern")
    now = datetime.now(eastern).time()

    # Define your extended market window
    start_time = time(9, 27)   # 9:27 AM ET
    end_time = time(16, 3)     # 4:03 PM ET

    # Only refresh if within this time window
    if start_time <= now <= end_time:
        st_autorefresh(interval=60 * 1000, key="home_refresh")

    st.title("SPY Real-Time Chart")

    def get_data():
        df = pd.read_csv("../cleaned_data/spy_1min_current_day.csv", parse_dates=["timestamp", "time", "date"])
        df = df.sort_values("timestamp").tail(60)
        # Return only rows from today
        today = datetime.now().date()
        df = df[df["date"].dt.date == today]
        return df

    df = get_data()

    fig = go.Figure(data=[go.Candlestick(
        x=df['time'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close']
    )])

    fig.update_layout(
        height=500,
        xaxis_rangeslider_visible=False,
        xaxis=dict(tickformat='%I:%M %p')  # 12-hour AM/PM format
    )

    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Volatility", f"{today_volatility[0]:.2f}%")
    col2.metric("Predicted Volume: ", f"{today_m_volume[0]:.2f} M")
    col3.metric("Estimated Remaining Volume", f"{volu_m_total[0]:.2f} M")

# --- PAGE: VOLUME ---
elif page == "Volume":

    # Custom style helper, if used
    style.use('ggplot')

    st.title("Volume Analysis")
    view = st.selectbox("Select Volume View", [
        "Current Day's Volume",
        "2025 Daily Prediction Accuracy",
        "Past 8 Years monthly"
    ])

    try:
        df = pd.read_csv(volu_data_path, index_col=0, parse_dates=True)

        if view == "Past 8 Years monthly":
            X = df.drop(columns=['Next_Day_Volume'], errors='ignore')
            y = df['Next_Day_Volume']
            y_pred = volume_model.predict(X)
            results = pd.DataFrame({'Actual': y, 'Predicted': y_pred}, index=df.index)

            results_10y = results[results.index.year >= 2017]
            monthly_avg = results_10y.resample('ME').mean()

            fig, ax = plt.subplots(figsize=(16, 6))
            ax.set_title('Monthly Average Volume: Predicted vs Actual (2017–2025)', fontsize=14)
            ax.set_xlabel('Year')
            ax.set_ylabel('Average Volume x 10m')
            ax.grid(True, axis='y', linestyle='--', alpha=0.6)

            c.customize_graph(ax)  # Optional customization

            monthly_avg[['Predicted', 'Actual']].plot(kind='bar', ax=ax, width=0.8, color=['tab:blue', 'tab:red'])
            xticks = monthly_avg.index
            tick_positions = list(range(len(xticks)))
            tick_labels = [str(date.year) if date.month == 1 else '' for date in xticks]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=0)

            fig.tight_layout()
            st.pyplot(fig)

        elif view == "2025 Daily Prediction Accuracy":
            df_2025 = df[df.index.year == 2025].copy()
            X_2025 = df_2025.drop(columns=['Next_Day_Volume'], errors='ignore')
            df_2025['Predicted'] = volume_model.predict(X_2025)
            df_2025['Actual'] = df_2025['Volume']
            results_2025 = df_2025.sort_index()

            fig, ax = plt.subplots(figsize=(16, 6))
            x = range(len(results_2025))
            bar_width = 0.4

            tick_positions = [p + bar_width / 2 for p in x]
            tick_labels = results_2025.index.strftime('%b-%d')
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=90)

            ax.set_title('Daily Predicted vs Actual Volume (2025)', fontsize=14)
            ax.set_xlabel('Date')
            ax.set_ylabel('Volume')

            c.customize_graph(ax)

            ax.bar(x, results_2025['Predicted'], width=bar_width, label='Predicted Volume', color='tab:blue', align='center')
            ax.bar([p + bar_width for p in x], results_2025['Actual'], width=bar_width, label='Actual Volume', color='tab:red', align='center')
            ax.legend()
            ax.grid(True, axis='y', linestyle='--', alpha=0.6)

            fig.tight_layout()
            st.pyplot(fig)

        elif view == "Current Day's Volume":
            # Load current day 1-minute volume data
            current_day_df = pd.read_csv(today_volume_datapath, parse_dates=['timestamp', 'time', 'date'])
            current_day_df = current_day_df.sort_values('timestamp')

            # Filter today's data only
            today = datetime.now().date()
            current_day_df = current_day_df[current_day_df['date'].dt.date == today]

            # Plotting
            fig, ax = plt.subplots(figsize=(16, 6))
            ax.plot(current_day_df['time'], current_day_df['volume'], label="1-min Volume", color='tab:blue')
            ax.set_title("Current Day Minute-by-Minute Volume", fontsize=14)
            ax.set_xlabel("Time")
            ax.set_ylabel("Volume x 1M")
            ax.grid(True, axis='y', linestyle='--', alpha=0.6)
            

            c.customize_graph(ax)
            ax.legend()
            fig.tight_layout()
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Error loading model or data: {e}")

# --- PAGE: VOLATILITY ---
elif page == "Volatility":
    st.title("Volatility Analysis")
    view = st.selectbox("Select Volatility View", [ 
        "Current Day's Volatility",
        "2025 Daily Prediction Accuracy",
        "Past 8 Years monthly"
    ])

    try:
        df = pd.read_csv(vola_data_path, index_col=0, parse_dates=True)

        # Feature columns required by model
        feature_cols = [
            'Open', 'High', 'Low', 'Close', 'Volume', 'Day_of_Week', 'Week_of_Year', 'Month',
            'Daily_Return', 'Price_Range', 'Close_to_Open_Change', 'Average_Price', 'Lagged_Close',
            'Lagged_Volume', 'Volume_Ratio', 'Lagged_Return', 'Rolling_Volatility', 'Close_to_High_Ratio'
        ]

        if view == "Past 8 Years monthly":
            X = df[feature_cols]
            y = df['Next_Day_Volatility'] * 100  # 💡 Scale to percent if needed
            y_pred = volatility_model.predict(X) * 100

            results = pd.DataFrame({'Actual': y, 'Predicted': y_pred}, index=df.index)
            results_8y = results[results.index.year >= 2017]
            monthly_avg = results_8y.resample('ME').mean()

            fig, ax = plt.subplots(figsize=(16, 6))
            ax.set_title('Monthly Average Volatility: Predicted vs Actual (2017–2025)', fontsize=14)
            ax.set_xlabel('Year')
            ax.set_ylabel('Average Volatility (%)')
            ax.grid(True, axis='y', linestyle='--', alpha=0.6)

            c.customize_graph(ax)

            monthly_avg[['Predicted', 'Actual']].plot(kind='bar', ax=ax, width=0.8, color=['tab:blue', 'tab:red'])

            xticks = monthly_avg.index
            tick_positions = list(range(len(xticks)))
            tick_labels = [str(date.year) if date.month == 1 else '' for date in xticks]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=0)

            fig.tight_layout()
            st.pyplot(fig)

        elif view == "2025 Daily Prediction Accuracy":
            df_2025 = df[df.index.year == 2025].copy()
            X_2025 = df_2025[feature_cols]
            df_2025['Predicted'] = volatility_model.predict(X_2025) * 100
            df_2025['Actual'] = df_2025['Next_Day_Volatility'] * 100
            results_2025 = df_2025.sort_index()

            fig, ax = plt.subplots(figsize=(16, 6))
            x = range(len(results_2025))
            bar_width = 0.4

            tick_positions = [p + bar_width / 2 for p in x]
            tick_labels = results_2025.index.strftime('%b-%d')
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=90)

            ax.set_title('Daily Predicted vs Actual Volatility (2025)', fontsize=14)
            ax.set_xlabel('Date')
            ax.set_ylabel('Volatility (%)')

            c.customize_graph(ax)

            ax.bar(x, results_2025['Predicted'], width=bar_width, label='Predicted Volatility', color='tab:blue', align='center')
            ax.bar([p + bar_width for p in x], results_2025['Actual'], width=bar_width, label='Actual Volatility', color='tab:red', align='center')
            ax.legend()
            ax.grid(True, axis='y', linestyle='--', alpha=0.6)

            fig.tight_layout()
            st.pyplot(fig)

        elif view == "Current Day's Volatility":
            # Load current day 1-minute data
            current_day_df = pd.read_csv(today_volume_datapath, parse_dates=['timestamp', 'time', 'date'])
            current_day_df = current_day_df.sort_values('timestamp')

            # Filter for today's data only
            today = datetime.now().date()
            current_day_df = current_day_df[current_day_df['date'].dt.date == today]

            # Resample to 5-minute intervals, using the last close in each interval
            df_5min = current_day_df.set_index('timestamp').resample('5T').agg({'close': 'last'}).dropna()

            # Calculate % change (volatility proxy)
            df_5min['Pct_Change'] = df_5min['close'].pct_change() * 100
            df_5min = df_5min.dropna()

            # Plot
            fig, ax = plt.subplots(figsize=(16, 6))
            ax.plot(df_5min.index, df_5min['Pct_Change'], color='tab:orange', label="5-min % Change")
            ax.axhline(0, color='gray', linestyle='--', linewidth=1)
            ax.set_title("Current Day 5-Minute Volatility (Percent Change)", fontsize=14)
            ax.set_xlabel("Time")
            ax.set_ylabel("Percent Change (%)")
            ax.grid(True, axis='y', linestyle='--', alpha=0.6)

            c.customize_graph(ax)
            ax.legend()
            fig.tight_layout()
            st.pyplot(fig)


    except Exception as e:
        st.error(f"Error loading model or data: {e}")


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

