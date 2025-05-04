import os
import pandas as pd
import numpy as np
from polygon.rest import RESTClient
from datetime import datetime, timedelta
import pytz
import yfinance as yf
from datetime import datetime, timedelta, timezone

# === CONFIG ===
POLYGON_API_KEY = "e5abVB0UydjFwCnLizrzyiHibeTtkCxD"
TICKER = "SPY"
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_DIR = os.path.join(BASE_DIR, "cleaned_data")
ONE_MIN_FILE = os.path.join(DATA_DIR, "spy_1min.csv")
FIVE_MIN_FILE = os.path.join(DATA_DIR, "spy_5min.csv")
TMP_FILE = "data/spy_1min_newdata.csv"
CURRENT_TMP_FILE = "data/spy_1min_current.csv"
CURRENT_DAY_FILE = os.path.join(DATA_DIR, "spy_1min_current_day.csv")
LAST_1FILE = os.path.join(DATA_DIR, "spy_1min_yesterday.csv")
LAST_5FILE = os.path.join(DATA_DIR, "spy_5min_yesterday.csv")


# === Clean and Format ===
def clean_and_format(file_path):
    df = pd.read_csv(file_path)
    df.drop_duplicates(subset=['timestamp'], inplace=True)

    # Drop the 'transactions' column if it exists
    if 'transactions' in df.columns:
        df.drop('transactions', axis=1, inplace=True)

    if 'otc' in df.columns:
        df.drop('otc', axis=1, inplace=True)

    # Convert to datetime, check if the timestamp is timezone-aware
    df['timestamp'] = pd.to_datetime(df['timestamp'], 
                                     unit='ms' if df['timestamp'].dtype == 'int64' else None)

    # Localize if timestamp is naive
    if df['timestamp'].dt.tz is None:
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
    else:
        df['timestamp'] = df['timestamp'].dt.tz_convert('US/Eastern')

    # Trading session flags
    df['pre_market'] = ((df['timestamp'].dt.time >= pd.to_datetime('04:00').time()) & 
                        (df['timestamp'].dt.time < pd.to_datetime('09:30').time())).astype(int)
    df['market'] = ((df['timestamp'].dt.time >= pd.to_datetime('09:30').time()) & 
                    (df['timestamp'].dt.time <= pd.to_datetime('16:00').time())).astype(int)
    df['after_market'] = ((df['timestamp'].dt.time > pd.to_datetime('16:00').time()) & 
                          (df['timestamp'].dt.time <= pd.to_datetime('20:00').time())).astype(int)

    df['date'] = df['timestamp'].dt.date
    df['time'] = df['timestamp'].dt.time
    df = df.sort_values(by=['timestamp'])
    df['time_diff'] = df['timestamp'].diff().dt.total_seconds() / 60

    # Fix FutureWarning by assigning the result to the column
    df['time_diff'] = df['time_diff'].fillna(0)

    df_time_idx = df.set_index('timestamp')
    df['open'] = df['open'].round(4)
    df['high'] = df['high'].round(4)
    df['low'] = df['low'].round(4)
    df['close'] = df['close'].round(4)
    df['vwap'] = (df['volume'] * df['close']).cumsum() / df['volume'].cumsum()
    # Aggregation dictionary for resampling
    aggregation_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'pre_market': 'last',
        'market': 'last',
        'after_market': 'last',
        'time_diff': 'sum'
    }

    df_5min = df_time_idx.resample('5min').agg(aggregation_dict).dropna()

    df_5min['timestamp'] = df_5min.index
    df_5min.reset_index(drop=True, inplace=True)
    df_5min['date'] = df_5min['timestamp'].dt.date
    df_5min['time'] = df_5min['timestamp'].dt.time

    # VWAP Calculation
    df_5min['vwap'] = (df_5min['volume'] * df_5min['close']).cumsum() / df_5min['volume'].cumsum()

    # Round open, high, low, close to 4 decimal places
    df_5min['open'] = df_5min['open'].round(4)
    df_5min['high'] = df_5min['high'].round(4)
    df_5min['low'] = df_5min['low'].round(4)
    df_5min['close'] = df_5min['close'].round(4)

    # Final columns to keep, excluding 'transactions'
    df_5min = df_5min[['open', 'high', 'low', 'close', 'volume', 'vwap', 'timestamp', 
                        'pre_market', 'market', 'after_market', 'date', 'time', 'time_diff']]

    return df, df_5min


# === FETCH HISTORICAL DATA FROM POLYGON ===
def fetch_polygon_data():
    eastern = pytz.timezone("US/Eastern")
    client = RESTClient(POLYGON_API_KEY)

    # Load the existing data from the 1-minute file
    df_existing = pd.read_csv(ONE_MIN_FILE, parse_dates=['timestamp'])
    
    last_timestamp = df_existing['timestamp'].max()

    if last_timestamp.tzinfo is None:
        last_timestamp = last_timestamp.tz_localize('UTC')
    else:
        if isinstance(last_timestamp.tzinfo, pytz.tzinfo.StaticTzInfo):
            if last_timestamp.tzinfo.zone != 'UTC':
                last_timestamp = last_timestamp.tz_convert('UTC')
        elif isinstance(last_timestamp.tzinfo, timezone):
            offset_minutes = last_timestamp.tzinfo.utcoffset(last_timestamp).total_seconds() / 60
            if offset_minutes != 0:
                last_timestamp = last_timestamp.astimezone(pytz.UTC)
        
    last_timestamp = last_timestamp.tz_convert('US/Eastern')

    today = datetime.now(eastern).date()
    yesterday = today - timedelta(days=1)

    if last_timestamp.date() >= yesterday:
        print(f"No need to fetch data. Data for {last_timestamp.date()} is already up-to-date.")
        return TMP_FILE, 1

    start_date = last_timestamp.date()
    end_date = yesterday
    
    print(f"Polygon fetch: {start_date} to {end_date}")

    aggs = []
    for a in client.list_aggs(TICKER, 1, "minute", str(start_date), str(end_date), adjusted=False, sort="asc", limit=50000):
        aggs.append(a)

    if not aggs:
        print(f"No data retrieved for the requested range: {start_date} to {end_date}")
        return None

    df = pd.DataFrame(aggs)
    df.to_csv(TMP_FILE, index=False)
    return TMP_FILE, 0


# === FETCH TODAY'S INTRADAY DATA FROM YAHOO FINANCE ===
def fetch_today_from_yf():
    data = yf.download('SPY', period="1d", interval="1m")
    data.columns = data.columns.droplevel(1)
    data.rename(columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    }, inplace=True)
    data.reset_index(inplace=True)
    data.rename(columns={'Datetime': 'timestamp'}, inplace=True)

    # Convert to UTC and then to Eastern Time
    # Ensure timestamp is converted to US/Eastern without tz_localize() errors
    if data['timestamp'].dt.tz is None:
        data['timestamp'] = data['timestamp'].dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
    else:
        data['timestamp'] = data['timestamp'].dt.tz_convert('US/Eastern')

    # Drop last row if volume is 0.0
    if data.iloc[-1]['volume'] == 0.0:
        print("Last row has 0.0 volume, dropping it.")
        data = data.iloc[:-1]

    # VWAP Calculation
    data['vwap'] = (data['volume'] * data['close']).cumsum() / data['volume'].cumsum()

    # Round OHLC to 4 decimal places
    for col in ['open', 'high', 'low', 'close']:
        data[col] = data[col].round(4)

    print(data.tail(3))

    os.makedirs(os.path.dirname(TMP_FILE), exist_ok=True)
    data.to_csv(TMP_FILE, index=False)
    return TMP_FILE


# === FETCH LAST 5 MINUTES FROM YAHOO FINANCE ===
def fetch_yf_current():
    data = yf.download('SPY', period="1d", interval="1m", auto_adjust=True, progress=False)

    if data.empty or len(data) < 2:
        print("Insufficient data pulled from Yahoo Finance.")
        return

    data.columns = data.columns.droplevel(1)
    data.rename(columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    }, inplace=True)
    data.reset_index(inplace=True)
    data.rename(columns={'Datetime': 'timestamp'}, inplace=True)

    if data.iloc[-1]['volume'] == 0.0:
        print("Last row has 0.0 volume, dropping it.")
        data = data.iloc[:-1]

    if data.empty:
        print("No usable data after dropping low-volume rows.")
        return

    # ✅ Drop unexpected columns (e.g. 'otc') if they sneak in
    expected_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
    data = data[[col for col in data.columns if col in expected_cols]]

    # Load previously saved current temp data (if exists)
    if os.path.exists(CURRENT_TMP_FILE):
        df_prev = pd.read_csv(CURRENT_TMP_FILE, parse_dates=['timestamp'])
        latest_time = df_prev['timestamp'].max()
        data = data[data['timestamp'] > latest_time]

    if data.empty:
        print("No new rows since last pull.")
        return

    # Save this batch for future comparison
    data.to_csv(CURRENT_TMP_FILE, index=False)

    # Clean and format only the expected structure
    df_1min, df_5min = clean_and_format(CURRENT_TMP_FILE)

    # Optional safety check before merging
    if df_1min.empty or df_5min.empty:
        print("Warning: Cleaned DataFrame is empty, skipping merge.")
        return

    merge_and_save(df_1min, df_5min)
    print(f"Appended {len(df_1min)} new rows @ {datetime.now().strftime('%H:%M:%S')}")


# === MERGE TO EXISTING DATA ===
def merge_and_save(df_1min_new, df_5min_new):
    eastern = pytz.timezone("US/Eastern")
    today = datetime.now(eastern).date()
    print(f"Today's date: {today}")

    # Read the previous day's 1-minute and 5-minute data
    df_1min_old = pd.read_csv(ONE_MIN_FILE, parse_dates=['timestamp', 'date'])
    df_5min_old = pd.read_csv(FIVE_MIN_FILE, parse_dates=['timestamp', 'date'])

    # Merge the old and new data (1-minute and 5-minute)
    df_1min_combined = pd.concat([df_1min_old, df_1min_new], ignore_index=True).drop_duplicates(subset=['timestamp']).sort_values(by='timestamp')
    df_5min_combined = pd.concat([df_5min_old, df_5min_new], ignore_index=True).drop_duplicates(subset=['timestamp']).sort_values(by='timestamp')

    # Save combined 1-min and 5-min data back to their respective files
    df_1min_combined.to_csv(ONE_MIN_FILE, index=False)
    df_5min_combined.to_csv(FIVE_MIN_FILE, index=False)

    # --- Handle CURRENT_DAY_FILE for today ---
    if os.path.exists(CURRENT_DAY_FILE):
        # If the current day file exists, read it
        df_existing = pd.read_csv(CURRENT_DAY_FILE, parse_dates=['timestamp', 'date'])

        # Ensure the 'date' column is in the correct datetime.date format
        df_existing['date'] = pd.to_datetime(df_existing['date']).dt.date

        # Filter to keep only today's data
        df_existing_today = df_existing[df_existing['date'] == today]

        # Concatenate today's data with the new 1-minute data
        df_today_combined = pd.concat([df_existing_today, df_1min_new], ignore_index=True)
        
        # Drop duplicates based on the timestamp to avoid any duplicates in the final file
        df_today_combined.drop_duplicates(subset=['timestamp'], inplace=True)
    else:
        # If the file doesn't exist, start fresh with the new 1-minute data
        df_today_combined = df_1min_new.copy()

    # Ensure 'date' column exists, create it if necessary
    if 'date' not in df_today_combined.columns:
        df_today_combined['timestamp'] = pd.to_datetime(df_today_combined['timestamp'])
        df_today_combined['date'] = df_today_combined['timestamp'].dt.date

    # Filter to keep only today's data (ensure that any old data isn't present)
    df_today_combined = df_today_combined[df_today_combined['date'] == today]

    # Save the updated current day's data to CURRENT_DAY_FILE
    df_today_combined.to_csv(CURRENT_DAY_FILE, index=False)
    print(f"Saved {len(df_today_combined)} rows of today's data to {CURRENT_DAY_FILE}.")

def update_daily_summary():
    print("Updating daily summary (spy.csv)...")
    daily_file = os.path.join(DATA_DIR, "spy.csv")

    # Load timestamp and make it timezone-aware
    df_1min = pd.read_csv(ONE_MIN_FILE)
    df_1min['timestamp'] = pd.to_datetime(df_1min['timestamp'], utc=True).dt.tz_convert('US/Eastern')

    # Create a tz-naive 'date' column by flooring then removing tz info
    df_1min['date'] = df_1min['timestamp'].dt.floor('D').dt.tz_localize(None)

    # Get today's date also as tz-naive
    today = pd.Timestamp.now(tz='US/Eastern').floor('D').tz_localize(None)

    # Filter out today's partial data
    df_1min = df_1min[df_1min['date'] < today]

    # Aggregate
    daily_agg = df_1min.groupby('date').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).reset_index()

    # Rename and add date parts
    daily_agg.rename(columns={'date': 'Date', 'open': 'Open', 'high': 'High',
                              'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)

    daily_agg['Day'] = daily_agg['Date'].dt.day
    daily_agg['Weekday'] = daily_agg['Date'].dt.weekday
    daily_agg['Week'] = daily_agg['Date'].dt.isocalendar().week
    daily_agg['Month'] = daily_agg['Date'].dt.month
    daily_agg['Year'] = daily_agg['Date'].dt.year

    # Read existing file and ensure its Date column is tz-naive too
    if os.path.exists(daily_file):
        df_existing = pd.read_csv(daily_file, parse_dates=['Date'])
        df_existing['Date'] = df_existing['Date'].dt.tz_localize(None)
        df = pd.concat([df_existing, daily_agg], ignore_index=True).drop_duplicates(subset=['Date']).sort_values('Date')
    else:
        df = daily_agg.sort_values('Date')

    df.to_csv(daily_file, index=False)
    print(f"Saved daily summary to {daily_file} ({len(df)} total rows).")

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

def create_prior_data():
    
    print("Creating prior's data...")

    # Read the source CSV
    df_1min = pd.read_csv(ONE_MIN_FILE, parse_dates=['timestamp'])
    df_1min['timestamp'] = pd.to_datetime(df_1min['timestamp'], errors='coerce')

    # Get the most recent date in the data
    last_date = df_1min['timestamp'].max().date()

    # Filter rows with that date
    df_last_day = df_1min[df_1min['timestamp'].dt.date == last_date]

    # Save to output file (overwrite)
    df_last_day.to_csv(LAST_1FILE, index=False)

    # Read the source CSV
    df_5min = pd.read_csv(FIVE_MIN_FILE, parse_dates=['timestamp'])
    df_5min['timestamp'] = pd.to_datetime(df_5min['timestamp'], errors='coerce')

    # Get the most recent date in the data
    last_date = df_5min['timestamp'].max().date()

    # Filter rows with that date
    df_last_day = df_5min[df_5min['timestamp'].dt.date == last_date]

    # Save to output file (overwrite)
    df_last_day.to_csv(LAST_5FILE, index=False)

    print(f"Saved data for {last_date} to {LAST_1FILE}")
    print(f"Saved data for {last_date} to {LAST_5FILE}")

# === MASTER RUNNER ===
def run_full_sync():
    print("Running SPY full sync...")

    # Backfill historical (up to yesterday)
    polygon_file, skip = fetch_polygon_data()
    if skip != 1:
        df1min_poly, df5min_poly = clean_and_format(polygon_file)
        merge_and_save(df1min_poly, df5min_poly)
    else:
        print("No new data to fetch from Polygon.")

    # Fetch today's data from Yahoo Finance
    yf_file_today = fetch_today_from_yf()
    df1min_yf, df5min_yf = clean_and_format(yf_file_today)
    merge_and_save(df1min_yf, df5min_yf)
    print("SPY data up-to-date.")

    update_daily_summary()
    create_prior_data()

    # Create volume features and save to CSV
    spy_csv_path = os.path.join(DATA_DIR, "spy.csv")
    create_volume_features(spy_csv_path)
    create_volatility_features(spy_csv_path)


if __name__ == "__main__":
    run_full_sync()
