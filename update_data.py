import os
import requests
import datetime
import io
import pandas as pd
import numpy as np

DATA_DIR = "data"
PARQUET_PATH = os.path.join(DATA_DIR, "nse_rolling_60d.parquet")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
}

def fetch_sec_bhavdata(target_date: datetime.date):
    """
    Downloads official NSE sec_bhavdata_full CSV containing OHLCV,
    official PREV_CLOSE, and delivery quantities for all equities.
    """
    date_str = target_date.strftime("%d%m%Y")
    url = f"https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{date_str}.csv"
    
    session = requests.Session()
    session.headers.update(HEADERS)
    
    try:
        r = session.get(url, timeout=15)
        if r.status_code != 200 or len(r.content) < 5000:
            return None, False
            
        df = pd.read_csv(io.StringIO(r.text))
        df.columns = [str(c).strip().upper() for c in df.columns]
        
        if "SERIES" not in df.columns or "SYMBOL" not in df.columns:
            return None, False
            
        df["SERIES"] = df["SERIES"].astype(str).str.strip().str.upper()
        df["SYMBOL"] = df["SYMBOL"].astype(str).str.strip().str.upper()
        
        # Keep standard equity segment
        df = df[df["SERIES"] == "EQ"].copy()
        
        col_map = {
            "CLOSE_PRICE": "CLOSE",
            "PREV_CLOSE": "PREV_CLOSE",
            "OPEN_PRICE": "OPEN",
            "HIGH_PRICE": "HIGH",
            "LOW_PRICE": "LOW",
            "TTL_TRD_QNTY": "TOTTRDQTY",
            "DELIV_QTY": "DELIV_QTY"
        }
        
        for k in col_map.keys():
            if k not in df.columns:
                return None, False
                
        df = df[["SYMBOL"] + list(col_map.keys())].rename(columns=col_map)
        
        for num in ["CLOSE", "PREV_CLOSE", "OPEN", "HIGH", "LOW", "TOTTRDQTY", "DELIV_QTY"]:
            df[num] = pd.to_numeric(df[num].astype(str).str.strip().str.replace("-", "0").str.replace(",", ""), errors="coerce").fillna(0.0)
            
        # Official price percentage change
        df["PCT_CHANGE"] = np.where(
            df["PREV_CLOSE"] > 0,
            ((df["CLOSE"] - df["PREV_CLOSE"]) / df["PREV_CLOSE"] * 100).round(2),
            np.where(df["OPEN"] > 0, ((df["CLOSE"] - df["OPEN"]) / df["OPEN"] * 100).round(2), 0.0)
        )
        
        df["DATE"] = pd.to_datetime(target_date)
        return df, True
        
    except Exception as e:
        print(f"Notice on date {target_date}: {e}")
        return None, False

def build_multi_day_database():
    os.makedirs(DATA_DIR, exist_ok=True)
    master_df = None
    
    if os.path.exists(PARQUET_PATH):
        try:
            master_df = pd.read_parquet(PARQUET_PATH)
            master_df["DATE"] = pd.to_datetime(master_df["DATE"])
            print(f"Loaded existing parquet: {len(master_df)} rows.")
        except Exception:
            pass

    today = datetime.date.today()
    collected_days = []
    
    # Ingest the last 5 trading days to establish multi-day rolling history
    target_sessions = 5
    lookback = 0
    
    while len(collected_days) < target_sessions and lookback < 15:
        candidate = today - datetime.timedelta(days=lookback)
        lookback += 1
        
        if candidate.weekday() >= 5: # Skip Saturday/Sunday
            continue
            
        print(f"Fetching session: {candidate}...")
        df_day, success = fetch_sec_bhavdata(candidate)
        if success and df_day is not None and not df_day.empty:
            print(f"-> Ingested {len(df_day)} stocks for {candidate}")
            collected_days.append(df_day)

    if not collected_days:
        raise RuntimeError("Could not download recent trading sessions from NSE archives.")

    new_data = pd.concat(collected_days, ignore_index=True)

    if master_df is not None:
        # Avoid duplicate date overlaps
        existing_dates = set(master_df["DATE"].dt.date.unique())
        new_data = new_data[~new_data["DATE"].dt.date.isin(existing_dates)]
        combined = pd.concat([master_df, new_data], ignore_index=True)
    else:
        combined = new_data

    # Retain the most recent 60 trading days
    unique_dates = sorted(combined["DATE"].drop_duplicates(), reverse=True)[:60]
    final_df = combined[combined["DATE"].isin(unique_dates)].sort_values(["SYMBOL", "DATE"])

    final_df.to_parquet(PARQUET_PATH, index=False)
    print(f"SUCCESS: Saved {len(final_df)} records across {len(unique_dates)} dates into {PARQUET_PATH}.")

if __name__ == "__main__":
    build_multi_day_database()
