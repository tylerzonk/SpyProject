from streamlit_navigation_bar import st_navbar
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import numpy as np
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
spy_data_path = os.path.join(base_dir, 'cleaned_data', 'spy.csv')
spy_1min_data_path = os.path.join(base_dir, 'cleaned_data', 'spy_1min.csv')
spy_5min_data_path = os.path.join(base_dir, 'cleaned_data', 'spy_5min.csv')
volu_data_path = os.path.join(base_dir, 'cleaned_data', 'spy_volume.csv')
vola_data_path = os.path.join(base_dir, 'cleaned_data', 'spy_volatility.csv')
current_day_one = os.path.join(base_dir, 'cleaned_data', 'spy_1min_current_day.csv')
current_day_five = os.path.join(base_dir, 'cleaned_data', "spy_5min_current_day.csv")
prior_1 = os.path.join(base_dir, 'cleaned_data', 'spy_1min_yesterday.csv')
prior_5 = os.path.join(base_dir, 'cleaned_data', 'spy_5min_yesterday.csv')

# Today's predicted volume
volume_model = joblib.load(volume_model_path)
yesterday_volu = pd.read_csv(volu_data_path, parse_dates=True)
yesterday_volu.set_index('Date', inplace=True)
yesterday_volu = yesterday_volu.tail(1)
volu_X = yesterday_volu.drop(columns=['Next_Day_Volume'], errors='ignore')
today_volume = volume_model.predict(volu_X)
today_m_volume = today_volume/1000000
yesterday_volume = yesterday_volu['Volume']
yesterday_m_volume = yesterday_volume/1000000

# Today's predicted volatility
volatility_model = joblib.load(volatility_model_path)
yesterday_vola = pd.read_csv(vola_data_path, parse_dates=True)
yesterday_vola = yesterday_vola.tail(1)
vola_X = yesterday_vola[['Open', 'High', 'Low', 'Close', 'Volume', 'Day_of_Week', 'Week_of_Year', 'Month',
        'Daily_Return', 'Price_Range', 'Close_to_Open_Change', 'Average_Price', 'Lagged_Close',
        'Lagged_Volume', 'Volume_Ratio', 'Lagged_Return', 'Rolling_Volatility', 'Close_to_High_Ratio']]
today_volatility = volatility_model.predict(vola_X)
today_volatility = today_volatility * 100
yester_vola = yesterday_vola['Rolling_Volatility'].values[0] * 100

# Estimated Remaining Volume
current_vola = pd.read_csv(current_day_one, parse_dates=True)
current_volu_sum = current_vola['volume'].sum()
volu_total = today_volume - current_volu_sum
volu_m_total = volu_total/1000000
volu_m_sum = current_volu_sum/1000000

st.set_page_config(page_title="Market Dashboard", layout="wide", page_icon="💵")

# --- NAVIGATION BAR ---
page = st_navbar(["Home", "Volume", "Volatility", "Technical", "About"],
                options={"use_padding": False})

# --- PAGE: HOME ---
if page == "Home":
    # Get current time in US/Eastern
    eastern = pytz.timezone("US/Eastern")
    now = datetime.now(eastern).time()
    # Define your extended market window
    start_time = time(9, 29)   # 9:29 AM ET
    end_time = time(16, 1)     # 4:01 PM ET
    weekday = datetime.now(eastern).weekday()
    st1, st2 = st.columns(2)
    st1.title("SPY Real-Time Chart")
    plot = st2.selectbox("", ["1 Minute", "5 Minute", "Day Line"])
    fig = None
    # Only refresh if within this time window
    if weekday < 5 and start_time <= now <= end_time:
        st_autorefresh(interval=60 * 1000, key="home_refresh")
        st.write("Market is open. Live data will refresh.")
        spy_1min = "../cleaned_data/spy_1min_current_day.csv"
        spy_5min = "../cleaned_data/spy_5min_current_day.csv"
    else:
        st.write("Market is closed. Data will not refresh.")
        spy_1min = "../cleaned_data/spy_1min.csv"
        spy_5min = "../cleaned_data/spy_5min.csv"


    if plot == "1 Minute":
        def get_data():
            start_time = time(9, 29)   # 9:29 AM ET
            end_time = time(16, 1)     # 4:01 PM ET
            weekday = datetime.now(eastern).weekday()
            df = pd.read_csv(spy_1min, parse_dates=["timestamp", "time", "date"])
            df = df.sort_values("timestamp").tail(60)
            # Return only rows from today
            if weekday < 5 and start_time <= now <= end_time:
                today = datetime.now().date()
                df = df[df["date"].dt.date == today]
            else:
                # displays data only between the hours of 9:30 and 16:00
                df = pd.read_csv(prior_1, parse_dates=["timestamp", "time", "date"])
                df = df[df['market'] == 1]
                df = df.sort_values("timestamp").tail(60)

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
        yaxis=dict(
            range=[df['low'].min(), df['high'].max() + (df['high'].max() - df['low'].min()) * 0.1], 
            fixedrange=True),
        xaxis=dict(tickformat='%I:%M %p'),
        margin=dict(t=10, b=40, l=40, r=10)
        )
        st.plotly_chart(fig, use_container_width=True)
    elif plot == "5 Minute":
        def get_data():
            start_time = time(9, 29)   # 9:29 AM ET
            end_time = time(16, 1)     # 4:01 PM ET
            weekday = datetime.now(eastern).weekday()
            df = pd.read_csv(spy_5min, parse_dates=["timestamp", "time", "date"])
            df = df.sort_values("timestamp").tail(60)
            # Return only rows from today
            if weekday < 5 and start_time <= now <= end_time:
                today = datetime.now().date()
                df = df[df["date"].dt.date == today]
            else:
                # displays data only between the hours of 9:30 and 16:00
                df = pd.read_csv(prior_5, parse_dates=["timestamp", "time", "date"])
                df = df[df['market'] == 1]
                df = df.sort_values("timestamp").tail(60)

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
        yaxis=dict(
            range=[df['low'].min(), df['high'].max() + (df['high'].max() - df['low'].min()) * 0.1], 
            fixedrange=True),
        xaxis=dict(tickformat='%I:%M %p'),
        margin=dict(t=10, b=40, l=40, r=10)
        )
        st.plotly_chart(fig, use_container_width=True)
    elif plot == "Day Line":
        def get_data():


            start_time = time(9, 29)   # 9:29 AM ET
            end_time = time(16, 1)     # 4:01 PM ET
            weekday = datetime.now(eastern).weekday()
            df = pd.read_csv(spy_1min, parse_dates=["timestamp", "time", "date"])
            df = df.sort_values("timestamp")
            # Return only rows from today
            if weekday < 5 and start_time <= now <= end_time:
                today = datetime.now().date()
                df = df[df["date"].dt.date == today]
            else:
                # displays data only between the hours of 9:30 and 16:00
                df = pd.read_csv(prior_1, parse_dates=["timestamp", "time", "date"])
                df = df[df['market'] == 1]
                df = df.sort_values("timestamp")
                
            return df
        
        df = get_data()

        # Create line plot of close prices
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["time"],
            y=df["close"],
            mode="lines",
            name="Close Price",
            line=dict(color="white"),
        ))

        fig.update_layout(
            title="SPY 1-Min Close Prices – Today",
            xaxis_title="Time",
            yaxis_title="Price",
            xaxis=dict(tickformat='%I:%M %p'),
            height=500,
            margin=dict(t=40, b=40, l=40, r=10)
        )
        st.plotly_chart(fig, use_container_width=True)

    # Last Price and Percent Change
    df = pd.read_csv(spy_1min, parse_dates=["timestamp", "time", "date"])
    df_last = df.tail(1)
    
    last_close = df_last['close'].values[0]
    yester_df = pd.read_csv(spy_data_path, parse_dates=True)
    yester_df = yester_df.sort_values(by='Date', ascending=True)
    yester_df = yester_df.tail(1)
    yester_close = yester_df['Close'].values[0]
    percent_change = ((last_close / yester_close) - 1) * 100


    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Last Price", f"${last_close:.2f}")
    col1.metric("Percent Change Today", f"{percent_change:.2f}%")

    col2.metric("Predicted Volatility", f"{today_volatility[0]:.2f}%")
    col2.metric("Yesterday Volatility", f"{yester_vola:.2f}%")

    col3.metric("Predicted Volume: ", f"{today_m_volume[0]:.2f} M")
    col3.metric("Yesterday Volume: ", f"{yesterday_m_volume[0]:.2f} M")

    col4.metric("Current Volume", f"{volu_m_sum:.2f} M")
    col4.metric("Estimated Remaining Volume", f"{volu_m_total[0]:.2f} M")
# --- PAGE: VOLUME ---
elif page == "Volume":

    # Custom style helper, if used
    style.use('ggplot')

    st.title("Volume Analysis")
    view = st.selectbox("Select Volume View", [
        "Current Day's Volume",
        "2025 Daily Prediction Accuracy",
        "Monthly Average & Accuracy"
    ])

    try:
        df = pd.read_csv(volu_data_path, index_col=0, parse_dates=True)

        if view == "Monthly Average & Accuracy":
            X = df.drop(columns=['Next_Day_Volume'], errors='ignore')
            y = df['Next_Day_Volume']
            y_pred = volume_model.predict(X)
            results = pd.DataFrame({'Actual': y, 'Predicted': y_pred}, index=df.index)

            results_10y = results[results.index.year >= 2017]
            monthly_avg = results_10y.resample('ME').mean()

            fig, ax = plt.subplots(figsize=(15, 5))
            ax.set_title('Monthly Average Volume: Predicted vs Actual ', fontsize=14)
            ax.set_xlabel('Year')
            ax.set_ylabel('Average Volume x 10m')
            ax.grid(True, axis='y', linestyle='--', alpha=0.6)

            c.customize_graph(ax)  # Optional customization

            monthly_avg[['Predicted', 'Actual']].plot(kind='bar', ax=ax, width=0.8, color=['lightgrey', 'tab:red'])
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

            fig, ax = plt.subplots(figsize=(15, 5))
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

            ax.bar(x, results_2025['Predicted'], width=bar_width, label='Predicted Volume', color='lightgrey', align='center')
            ax.bar([p + bar_width for p in x], results_2025['Actual'], width=bar_width, label='Actual Volume', color='tab:red', align='center')
            ax.legend()
            ax.grid(True, axis='y', linestyle='--', alpha=0.6)

            fig.tight_layout()
            st.pyplot(fig)

        elif view == "Current Day's Volume":
            # Load current day 1-minute volume data
            current_day_df = pd.read_csv(current_day_one, parse_dates=['timestamp', 'time', 'date'])
            current_day_df = current_day_df.sort_values('timestamp')
            if current_day_df.empty:
                current_day_df = pd.read_csv(prior_1, parse_dates=['timestamp', 'time', 'date'])
                current_day_df = current_day_df[current_day_df['market'] == 1]
                current_day_df = current_day_df.sort_values('timestamp')

            # Filter today's data only
            today = datetime.now().date()
            current_day_df = current_day_df[current_day_df['date'].dt.date == today]

            # Plotting
            fig, ax = plt.subplots(figsize=(15, 5))
            
            ax.set_title("Current Day Minute-by-Minute Volume", fontsize=14)
            ax.set_xlabel("Time")
            ax.set_ylabel("Volume x 1M")
            ax.grid(True, axis='y', linestyle='--', alpha=0.6)
            c.customize_graph(ax)
            ax.plot(current_day_df['time'], current_day_df['volume'], label="1-min Volume", color='lightgrey')
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
        "Monthly Average & Accuracy"
    ])

    try:
        df = pd.read_csv(vola_data_path, index_col=0, parse_dates=True)

        # Feature columns required by model
        feature_cols = [
            'Open', 'High', 'Low', 'Close', 'Volume', 'Day_of_Week', 'Week_of_Year', 'Month',
            'Daily_Return', 'Price_Range', 'Close_to_Open_Change', 'Average_Price', 'Lagged_Close',
            'Lagged_Volume', 'Volume_Ratio', 'Lagged_Return', 'Rolling_Volatility', 'Close_to_High_Ratio'
        ]

        if view == "Monthly Average & Accuracy":
            X = df[feature_cols]
            y = df['Next_Day_Volatility'] * 100  # 💡 Scale to percent if needed
            y_pred = volatility_model.predict(X) * 100

            results = pd.DataFrame({'Actual': y, 'Predicted': y_pred}, index=df.index)
            results_8y = results[results.index.year >= 2017]
            monthly_avg = results_8y.resample('ME').mean()

            fig, ax = plt.subplots(figsize=(15, 5))
            ax.set_title('Monthly Average Volatility: Predicted vs Actual ', fontsize=14)
            ax.set_xlabel('Year')
            ax.set_ylabel('Average Volatility (%)')
            ax.grid(True, axis='y', linestyle='--', alpha=0.6)

            c.customize_graph(ax)

            monthly_avg[['Predicted', 'Actual']].plot(kind='bar', ax=ax, width=0.8, color=['lightgrey', 'tab:red'])

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

            fig, ax = plt.subplots(figsize=(15, 5))
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

            ax.bar(x, results_2025['Predicted'], width=bar_width, label='Predicted Volatility', color='lightgrey', align='center')
            ax.bar([p + bar_width for p in x], results_2025['Actual'], width=bar_width, label='Actual Volatility', color='tab:red', align='center')
            ax.legend()
            ax.grid(True, axis='y', linestyle='--', alpha=0.6)

            fig.tight_layout()
            st.pyplot(fig)

        elif view == "Current Day's Volatility":
            # Load current day 1-minute data
            current_day_df = pd.read_csv(current_day_one, parse_dates=['timestamp', 'time', 'date'])

            # check to see if the df is empty
            current_day_df = current_day_df.sort_values('timestamp')
            if current_day_df.empty:
                current_day_df = pd.read_csv(prior_1, parse_dates=['timestamp', 'time', 'date'])
                current_day_df = current_day_df[current_day_df['market'] == 1]
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
            fig, ax = plt.subplots(figsize=(15, 5))


            ax.set_title("Current Day 5-Minute Volatility (Percent Change)", fontsize=14)
            ax.set_xlabel("Time")
            ax.set_ylabel("Percent Change (%)")
            ax.grid(True, axis='y', linestyle='--', alpha=0.6)
            c.customize_graph(ax)
            ax.axhline(0, color='red', linestyle='-', linewidth=1)
            ax.plot(df_5min.index, df_5min['Pct_Change'], color='lightgrey', label="5-min % Change")
            ax.legend()
            fig.tight_layout()
            st.pyplot(fig)


    except Exception as e:
        st.error(f"Error loading model or data: {e}")


# --- PAGE: TECHNICAL ---
elif page == "Technical":
    st.title("Technical Breakdown")

    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "cleaned_data")
    ONE_MIN_FILE = os.path.join(DATA_DIR, "spy_1min_current_day.csv")
    SPY_DIR = os.path.join(DATA_DIR, "spy.csv")
    FIVE_MIN_FILE = os.path.join(DATA_DIR, "spy_5min_current_day.csv")
    VOLATILITY_FILE = os.path.join(DATA_DIR, "volatility_resids.csv")
    VOLUME_FILE = os.path.join(DATA_DIR, "volume_resids.csv")


    for key in ["data_set", "model_view", "deployment_step"]:
        if key not in st.session_state:
            st.session_state[key] = "Select"

    def on_data_change():
        st.session_state.model_view = "Select"
        st.session_state.deployment_step = "Select"

    def on_model_change():
        st.session_state.data_set = "Select"
        st.session_state.deployment_step = "Select"

    def on_deployment_change():
        st.session_state.data_set = "Select"
        st.session_state.model_view = "Select"

    # UI layout
    col1, col2, col3 = st.columns(3)

    # Content descriptions
    col1.markdown("""
    ### Data Pipeline
    - Downloaded about 30 years SPY daily history from Kaggle
    - Requested 1-min candles using Polygon.io API
    - Pulling nearly real-time data via yfinance
    - Generating rolling statistical features
    - Updating all datasets in real-time
    """)

    col2.markdown("""
    ### Modeling
    - Transformed Data into new features for training
    - Trained XGBoost regressors for volume & volatility
    - Evaluated using R², RMSE, and accuracy
    - Exported model for easy load and use
    """)

    col3.markdown("""
    ### Deployment
    - Frontend: Streamlit for UI and Graphics
    - Backend: FastAPI for Data Engineering and Predictions
    - Containerized with Docker Compose
    """)

    st.markdown("---")

    # --- BOTTOM: Three-column selectbox controls ---
    sb1, sb2, sb3 = st.columns(3)

    # Select boxes with on_change logic
    sb1.selectbox("Data", [
        "Select",
        "Daily Data",
        "Behind The Scenes"
    ], key="data_set", on_change=on_data_change)

    sb2.selectbox("Select Model", [
        "Select",
        "Features and Model",
        "Accuracy and Correlation"
    ], key="model_view", on_change=on_model_change)

    sb3.selectbox("Select Deployment Step", [
        "Select",
        "Structure",
        "Streamlit",
        "FastAPI",
        "Docker"
    ], key="deployment_step", on_change=on_deployment_change)

    # Example output based on last valid selection
    if st.session_state.data_set != "Select":
        st.write(f"Showing data for: {st.session_state.data_set}")

        if st.session_state.data_set == "Daily Data":
            dat1, dat2, dat3 = st.columns(3)
            dat1.markdown("## Daily")
            df1 = pd.read_csv(SPY_DIR, parse_dates=['Date'])
            df1 = df1.sort_values(by='Date', ascending=False)
            df1 = df1.drop(columns=['Year', 'Month', 'Week', 'Weekday', 'Day'])
            dat1.dataframe(df1)

            dat2.markdown("## One Minute")
            df2 = pd.read_csv(ONE_MIN_FILE, parse_dates=['timestamp'])
            df2 = df2.sort_values(by='timestamp', ascending=False)
            df2 = df2.drop(columns=['time_diff', 'pre_market', 'market', 'after_market', 'timestamp'])
            df2 = df2[['date', 'time', 'open', 'high', 'low', 'close', 'volume']]
            dat2.dataframe(df2)

            dat3.markdown("## Five Minute ")
            df3 = pd.read_csv(FIVE_MIN_FILE, parse_dates=['timestamp', 'date'])
            df3 = df3.sort_values(by='timestamp', ascending=False)
            df3 = df3.drop(columns=['time_diff', 'pre_market', 'market', 'after_market', 'timestamp'])
            df3 = df3[['date', 'time', 'open', 'high', 'low', 'close', 'volume']]
            # df3['date'] = df3['date'].dt.date
            dat3.dataframe(df3)
            
        elif st.session_state.data_set == "Behind The Scenes":
            # Code Here
            # Example: Show how the data was processed
            st.markdown("""
            - Loaded from local CSV
            - Parsed time columns
            - Engineered lag and rolling features
            """)

    elif st.session_state.model_view != "Select":
        st.write(f"Showing model: {st.session_state.model_view}")

        if st.session_state.model_view == "Features and Model":
            sec1, sec2 = st.columns(2)
            sec1.markdown("## Volume")

            with sec1.expander("Features Code"):
                st.code("""
                def create_volume_features(filepath):
                    print("Generating volume features...")
                    spy_volume_csv = os.path.join(DATA_DIR, "spy_volume.csv")

                    if not os.path.exists(filepath):
                        print(f"File not found: {filepath}")
                        return pd.DataFrame()

                    df = pd.read_csv(filepath)
                    
                    if 'Date' not in df.columns:
                        print("Missing 'Date' column in input data. Aborting volume feature creation.")
                        return pd.DataFrame()

                    df['Date'] = pd.to_datetime(df['Date'])
                    df.set_index('Date', inplace=True)
                    df.sort_index(inplace=True)

                    # Features
                    df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
                    df['Lag_Volume_1'] = df['Volume'].shift(1)
                    df['Next_Day_Volume'] = df['Volume'].shift(-1)

                    df.dropna(inplace=True)

                    # Append or create
                    if os.path.exists(spy_volume_csv):
                        df_existing = pd.read_csv(spy_volume_csv, parse_dates=['Date'])
                        df_combined = pd.concat([df_existing, df.reset_index()], ignore_index=True)
                        df_combined.drop_duplicates(subset='Date', keep='last', inplace=True)
                    else:
                        df_combined = df.reset_index()

                    df_combined.to_csv(spy_volume_csv, index=False)
                    print(f"Saved volume features to {spy_volume_csv} ({len(df_combined)} rows).")
                    return df_combined
                """, language="python")
            with sec1.expander("Model Code"):
                st.code("""
                    X = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Day_of_Week', 'Week_of_Year', 'Month',
                            'Daily_Return', 'Price_Range', 'Close_to_Open_Change', 'Average_Price', 'Lagged_Close',
                            'Lagged_Volume', 'Volume_Ratio', 'Lagged_Return', 'Rolling_Volatility', 'Close_to_High_Ratio']]
                    y = df['Next_Day_Volatility']

                    # Step 4: Train-test split (80% train, 20% test)
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    # Step 5: Initialize the XGBoost model
                    v_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=5)

                    # Step 6: Train the model
                    v_model.fit(X_train, y_train)

                    # Step 7: Make predictions
                    y_pred = v_model.predict(X_test)
                """, language="python")
            sec1.markdown('#### Features \n'
                            '- Open - price at market open \n'   
                            '- High - highest price of day \n'   
                            '- Low - lowest price of day \n' 
                            '- Close - price at market close \n' 
                            '- Volume - shares traded that day \n' 
                            '- Day - calendar day \n' 
                            '- Weekday - day of the week \n' 
                            '- Week	- week of the year \n'  
                            '- Month - month number \n' 
                            '- Year	- year \n' 
                            '- Volume_MA5 - 5-day avg volume \n' 
                            '- Lag_Volume_1 - previous day volume \n'
                            '#### Target \n'
                            '- Next_Day_Volume - following day volume \n')

# -------------------------------------------------------------------------------------------
            sec2.markdown("## Volatility")

            with sec2.expander("Features Code"):
                st.code("""
            def create_volatility_features(filepath):
                print("Generating volatility features...")
                output_file = os.path.join(DATA_DIR, "spy_volatility.csv")

                if not os.path.exists(filepath):
                    print(f"File not found: {filepath}")
                    return pd.DataFrame()

                df = pd.read_csv(filepath, parse_dates=['Date'])
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
                df.sort_index(inplace=True)

                # === Feature Engineering ===
                df['Daily_Return'] = df['Close'].pct_change()
                df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
                df['Price_Range'] = df['High'] - df['Low']
                df['Close_to_Open_Change'] = df['Close'] - df['Open']
                df['Average_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
                df['Lagged_Close'] = df['Close'].shift(1)
                df['Lagged_Volume'] = df['Volume'].shift(1)
                df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
                df['Volume_Ratio'] = df['Volume'] / df['Volume_MA5']
                df['Lagged_Return'] = df['Daily_Return'].shift(1)
                df['Rolling_Volatility'] = df['Daily_Return'].rolling(window=5).std()
                df['Day_of_Week'] = df.index.dayofweek
                df['Week_of_Year'] = df.index.isocalendar().week
                df['Month'] = df.index.month
                df['Close_to_High_Ratio'] = (df['Close'] - df['High']) / (df['High'] - df['Low'])
                df['Next_Day_Volatility'] = df['Daily_Return'].shift(-1).rolling(window=5).std()

                df.dropna(inplace=True)

                # === Save and combine logic ===
                if os.path.exists(output_file):
                    df_existing = pd.read_csv(output_file, parse_dates=['Date'])
                    df_combined = pd.concat([df_existing, df.reset_index()], ignore_index=True)
                    df_combined.drop_duplicates(subset='Date', keep='last', inplace=True)
                else:
                    df_combined = df.reset_index()

                df_combined.to_csv(output_file, index=False)
                print(f"Saved volatility features to {output_file} ({len(df_combined)} rows).")
                return df_combined
                """, language="python")
            with sec2.expander("Model Code"):
                st.code("""
                    # Volume
                    X = df.drop(columns=['Next_Day_Volume'])  # Exclude actual volume and target
                    y = df['Next_Day_Volume']

                    # Split data into training and testing sets (80% train, 20% test)
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    # Initialize and train the XGBoost model
                    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=.1)
                    xgb_model.fit(X_train, y_train)

                    # Make predictions
                    y_pred = xgb_model.predict(X_test)
                """, language="python")
                sec2.markdown('#### Features \n'
                            '- Open - price at market open \n'   
                            '- High - highest price of the day \n'   
                            '- Low - lowest price of the day \n' 
                            '- Close - price at market close \n' 
                            '- Volume - total shares traded during the day \n' 
                            '- Day_of_Week - market day number \n' 
                            '- Week_of_Year - week number within the year \n'  
                            '- Month - month number  \n' 
                            '- Daily_Return - percent change in close price from previous day \n'
                            '- Price_Range - difference between high and low prices \n'
                            '- Close_to_Open_Change - percent change from open to close \n'
                            '- Average_Price - average of open, high, low, and close prices \n'
                            '- Lagged_Close - previous day\'s closing price \n'
                            '- Lagged_Volume - previous day\'s trading volume \n'
                            '- Volume_Ratio - current volume divided by 5-day average volume \n'
                            '- Lagged_Return - previous day\'s return \n'
                            '- Rolling_Volatility - recent price volatility / std dev of returns \n'
                            '- Close_to_High_Ratio - close price as a percentage of the daily high \n'
                            '#### Target \n'
                            '- Next_Day_Volatility - following day volatility \n')
        
        elif st.session_state.model_view == "Accuracy and Correlation":
            st.markdown("## Volume")

            df = pd.read_csv(VOLUME_FILE)
            y_pred = df['y_pred'].values
            residuals = df['residuals'].values

            fig, ax = plt.subplots(figsize=(15, 5))
            c.customize_graph(ax)
            ax.scatter(y_pred, residuals, color='white', edgecolors='k')
            ax.axhline(0, color='red', linestyle='--', linewidth=1)
            ax.set_title('Residual Plot: Predicted vs Residuals')
            ax.set_xlabel('Predicted Volatility')
            ax.set_ylabel('Residuals (Actual - Predicted)')
            ax.grid(True, linestyle='--', alpha=0.5)
            fig.tight_layout()
            st.pyplot(fig)
            col1, col2, col3 = st.columns(3)
            col1.markdown('#### RMSE: 11,477,552')
            col2.markdown('#### R² Score: 86.51%')
            col3.markdown('#### MAPE: 22.39%')
            st.markdown('---')

# -------------------------------------------------------------------------------------------
            st.markdown("## Volatility")

            df = pd.read_csv(VOLATILITY_FILE)
            y_pred = df['y_pred'].values
            residuals = df['residuals'].values

            fig, ax = plt.subplots(figsize=(15, 5))
            c.customize_graph(ax)
            ax.scatter(y_pred, residuals, color='white', edgecolors='k')
            ax.axhline(0, color='red', linestyle='--', linewidth=1)
            ax.set_title('Residual Plot: Predicted vs Residuals')
            ax.set_xlabel('Predicted Volatility')
            ax.set_ylabel('Residuals (Actual - Predicted)')
            ax.grid(True, linestyle='--', alpha=0.5)
            fig.tight_layout()
            st.pyplot(fig)
            col1, col2, col3 = st.columns(3)
            col1.markdown('#### RMSE: 0.37%')
            col2.markdown('#### R² Score: 77.5249%')
            col3.markdown('#### MAPE: 24.33%')
        
    elif st.session_state.deployment_step != "Select":
        st.write(f"Deployment step: {st.session_state.deployment_step}")

        if st.session_state.deployment_step == "Structure":
            # Code Here
            # Example: Show folder structure
            st.code("""
                SPYPROJECT
                |   docker_compose.yml
                    render.yaml
                |   README.md
                |   
                +---cleaned_data
                |       spy.csv
                |       spy_1min.csv
                |       spy_1min_current_day.csv
                |       spy_5min.csv
                |       spy_5min_current_day.csv
                |       spy_volatility.csv
                |       spy_volume.csv
                |       volatility_resids.csv
                |       volume_resids.csv
                |       
                +---data
                |       last_SPY_pull.csv
                |       spy_1min_data_2023_to_2024.csv
                |       spy_1min_data_2024_to_2025.csv
                |       SPY_1min_firstratedata.csv
                |       spy_1min_newdata.csv
                |       SPY_1_min.csv
                |       
                +---fastapi_app
                |   |   Dockerfile
                |   |   main.py
                |   |   requirements.txt
                |   |   
                |   +---data
                |   |       spy_1min_current.csv
                |   |       spy_1min_newdata.csv
                |   |       
                |   +---functions
                |   |   |   api_key.py
                |   |   |   color.py
                |   |   |   spy_updater.py
                |   |   |   
                |   |   \---__pycache__
                |   |           api_key.cpython-312.pyc
                |   |           color.cpython-312.pyc
                |   |           spy_updater.cpython-312.pyc
                |   |           
                |   +---model
                |   |       volatilitymodel.pkl
                |   |       volumemodel.pkl
                |   |       
                |   \---__pycache__
                |           main.cpython-312.pyc
                |           
                \---streamlitapp
                    |   Dockerfile
                    |   requirements.txt
                    |   streamapp.py
                    |   
                    +---functions
                    |   |   api_key.py
                    |   |   color.py
                    |   |   
                    |   \---__pycache__
                    |           color.cpython-312.pyc
                    |           
                    \---model
                            volatilitymodel.pkl
                            volumemodel.pkl
            """, language="bash")

        elif st.session_state.deployment_step == "Streamlit":
            # Code Here
            st.markdown("""
            - Main interface for users
            - Charts, predictions, toggles, and real-time updates
            - Interactive select boxes and visualizations
            """)

        elif st.session_state.deployment_step == "FastAPI":
            # Code Here
            st.markdown("""
            - Lightweight API for model prediction
            - Handles POST requests with JSON input
            - Used in backend to decouple UI from ML model
            """)

        elif st.session_state.deployment_step == "Docker":
            # Code Here
            st.markdown("""
            - Used Docker Compose to combine Streamlit + FastAPI
            - Easy to scale, deploy, and maintain
            - Ensures consistent environments across devices
            """)


    # view = st.selectbox("Select Volatility View", [ 
    #     "Current Day's Volatility",
    #     "2025 Daily Prediction Accuracy",
    #     "Monthly Average & Accuracy"
    # ])
# --- PAGE: ABOUT ---
elif page == "About":
    st.title("About This Project")

    col1, col2, col3 = st.columns(3)

    # Content descriptions
    col1.markdown("""
    ### Project
    - DDI Final Project
    - Goal: Devolop a Machine Learning App utilizing
        - Machine Learning
        - API requests and development
        - Streamlit UI
        - Docker Integration
    - Personal: Make a machine learning app that was fully autonomous
        - to make an app that any trader could use for the market
        - Make predictions with current data
        - display (semi) real-time stock data
    """)

    col2.markdown("""
    ### SPY Breakdown
    - SPY: an ETF that is built to follow the performance of the S&P 500 index.
        - S&P 500: a list that aims to track the performance of the 500 largest publicly traded U.S. companies
        - ETF(Exchange-Traded Fund) -a basket of stocks you can buy or sell, and it lets you invest in lots of companies at once instead of picking just one.
    - Importance: the model tracks volatility and volume for SPY Option Strategies
        - Options are contracts that let you bet on whether a stock will go up or down, without actually owning the stock
            - They’re high-risk, high-reward, and will expire worthless after a set time
        - Volatility Importance: Options prices are heavily influenced by implied volatility
            - Higher Volatility = More Expensive Options
        - Volume Importance: more liquidity, tighter spreads, and easier entry/exit for trades
            - High Volume = Quicker Trades
        
        
    """)

    col3.markdown("""
    ### Author
    Tyler Zenk
    - Enrolled in Data & Development Immersive (DDI) 
    - Pursuing a Data Science degree at UMGC
    - Working as an Intelligence Analyst for the U.S. Space Force
    ---
                  
    - Specializing in development and machine learning. 
    - Designing and deploying data solutions
    - Utilizing statistical models
    - Real-time analytics
    - Data focused decision making
    ---
    https://github.com/tylerzonk    """)

