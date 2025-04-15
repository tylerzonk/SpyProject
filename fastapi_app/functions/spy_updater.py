import os
import pandas as pd
from polygon.rest import RESTClient
from datetime import datetime, timedelta
import pytz
import yfinance as yf
from api_key import polygon_api
from datetime import datetime, timedelta, timezone

# === CONFIG ===
POLYGON_API_KEY = polygon_api
TICKER = "SPY"
DATA_DIR = "cleaned_data"
ONE_MIN_FILE = os.path.join(DATA_DIR, "spy_1min.csv")
FIVE_MIN_FILE = os.path.join(DATA_DIR, "spy_5min.csv")
TMP_FILE = "data/spy_1min_newdata.csv"
CURRENT_TMP_FILE = "data/spy_1min_current.csv"

# === Clean and Format ===
def clean_and_format(file_path):
    df = pd.read_csv(file_path)
    df.drop_duplicates(subset=['timestamp'], inplace=True)

    # Drop the 'transactions' column if it exists
    if 'transactions' in df.columns:
        df.drop('transactions', axis=1, inplace=True)

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
    data['timestamp'] = data['timestamp'].dt.tz_convert('UTC').dt.tz_convert('US/Eastern')

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
    data.to_csv(TMP_FILE, index=False)
    return TMP_FILE

# === FETCH LAST 5 MINUTES FROM YAHOO FINANCE ===
def fetch_yf_current():
    data = yf.download('SPY', period="1d", interval="1m", progress=False)

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

    # Convert timezone
    data['timestamp'] = data['timestamp'].dt.tz_convert('UTC').dt.tz_convert('US/Eastern')

    # Take last 5 minutes
    data = data.tail(5).copy()

    # Drop last row if volume is 0
    if data.iloc[-1]['volume'] == 0.0:
        print("Last row has 0.0 volume, dropping it.")
        data = data.iloc[:-1]

    if data.empty:
        print("No usable data after dropping low-volume rows.")
        return

    # Load previously saved current temp data (if exists)
    if os.path.exists(CURRENT_TMP_FILE):
        df_prev = pd.read_csv(CURRENT_TMP_FILE, parse_dates=['timestamp'])
        df_prev['timestamp'] = df_prev['timestamp'].dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
        latest_time = df_prev['timestamp'].max()
        data = data[data['timestamp'] > latest_time]

    if data.empty:
        print("No new rows since last pull.")
        return

    # Save this batch for future comparison
    data.to_csv(CURRENT_TMP_FILE, index=False)

    # VWAP & rounding
    data['vwap'] = (data['volume'] * data['close']).cumsum() / data['volume'].cumsum()
    for col in ['open', 'high', 'low', 'close']:
        data[col] = data[col].round(4)

    # Clean/format & merge
    df_1min, df_5min = clean_and_format_df(data)
    merge_and_save(df_1min, df_5min)

    print(f"Appended {len(df_1min)} new rows @ {datetime.now().strftime('%H:%M:%S')}")

# === MERGE TO EXISTING DATA ===
def merge_and_save(df_1min_new, df_5min_new):
    df_1min_old = pd.read_csv(ONE_MIN_FILE, parse_dates=['timestamp'])
    df_5min_old = pd.read_csv(FIVE_MIN_FILE, parse_dates=['timestamp'])

    df_1min_combined = pd.concat([df_1min_old, df_1min_new], ignore_index=True).drop_duplicates(subset=['timestamp']).sort_values(by='timestamp')
    df_5min_combined = pd.concat([df_5min_old, df_5min_new], ignore_index=True).drop_duplicates(subset=['timestamp']).sort_values(by='timestamp')

    df_1min_combined.to_csv(ONE_MIN_FILE, index=False)
    df_5min_combined.to_csv(FIVE_MIN_FILE, index=False)
    print("Merged and saved updated data.")


# === MASTER RUNNER ===
def run_full_sync():
    print("Running SPY full sync...")

    # Backfill historical (up to yesterday)
    polygon_file, skip = fetch_polygon_data()
    if skip != 1:
        df1min_poly, df5min_poly = clean_and_format(polygon_file)
        merge_and_save(df1min_poly, df5min_poly)

    # Fetch today's data from Yahoo Finance
    yf_file_today = fetch_today_from_yf()
    df1min_yf, df5min_yf = clean_and_format(yf_file_today)
    merge_and_save(df1min_yf, df5min_yf)
    print("SPY data up-to-date.")


if __name__ == "__main__":
    run_full_sync()
