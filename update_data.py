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
    Downloads official NSE sec_bhavdata_full file containing both
    OHLCV pricing, PREV_CLOSE, and delivery quantities in a single unzipped CSV.
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
        # Strip all leading/trailing whitespace from column names and values
        df.columns = [str(c).strip().upper() for c in df.columns]
        
        # Verify required columns exist
        if "SERIES" not in df.columns or "SYMBOL" not in df.columns:
            return None, False
            
        df["SERIES"] = df["SERIES"].astype(str).str.strip().str.upper()
        df["SYMBOL"] = df["SYMBOL"].astype(str).str.strip().str.upper()
        
        # Filter for standard equity market
        df = df[df["SERIES"] == "EQ"].copy()
        
        # Map official columns
        col_map = {
            "CLOSE_PRICE": "CLOSE",
            "PREV_CLOSE": "PREV_CLOSE",
            "HIGH_PRICE": "HIGH",
            "LOW_PRICE": "LOW",
            "TTL_TRD_QNTY": "TOTTRDQTY",
            "DELIV_QTY": "DELIV_QTY"
        }
        
        for k in col_map.keys():
            if k not in df.columns:
                print(f"Missing column: {k}")
                return None, False
                
        df = df[["SYMBOL"] + list(col_map.keys())].rename(columns=col_map)
        
        # Convert numeric fields cleanly
        for num in ["CLOSE", "PREV_CLOSE", "HIGH", "LOW", "TOTTRDQTY", "DELIV_QTY"]:
            df[num] = pd.to_numeric(df[num].astype(str).str.strip().str.replace("-", "0"), errors="coerce").fillna(0.0)
            
        # Official percentage change using verified PREV_CLOSE
        df["PCT_CHANGE"] = np.where(
            df["PREV_CLOSE"] > 0,
            ((df["CLOSE"] - df["PREV_CLOSE"]) / df["PREV_CLOSE"] * 100).round(2),
            0.0
        )
        
        df["DATE"] = pd.to_datetime(target_date)
        return df, True
        
    except Exception as e:
        print(f"Error reading session for {target_date}: {e}")
        return None, False

def update_rolling_history():
    os.makedirs(DATA_DIR, exist_ok=True)
    master_df = None
    
    if os.path.exists(PARQUET_PATH):
        try:
            master_df = pd.read_parquet(PARQUET_PATH)
            master_df["DATE"] = pd.to_datetime(master_df["DATE"])
            print(f"Loaded existing parquet: {len(master_df)} rows.")
        except Exception as e:
            print(f"Starting fresh parquet due to: {e}")

    today = datetime.date.today()
    found_data = None
    
    # Adaptive Search: check backwards for latest trading session
    for lookback in range(0, 10):
        candidate = today - datetime.timedelta(days=lookback)
        if candidate.weekday() >= 5: # Skip weekends
            continue
            
        print(f"Fetching official NSE session: {candidate}...")
        df_candidate, success = fetch_sec_bhavdata(candidate)
        if success and df_candidate is not None and not df_candidate.empty:
            print(f"Successfully loaded {len(df_candidate)} equities for {candidate}.")
            found_data = df_candidate
            break

    if found_data is None:
        raise RuntimeError("No active trading session file returned by NSE archives.")

    if master_df is not None:
        # Remove candidate date if already present to prevent duplicate rows
        master_df = master_df[master_df["DATE"] != found_data["DATE"].iloc[0]]
        combined = pd.concat([master_df, found_data], ignore_index=True)
    else:
        combined = found_data

    # Retain the latest 60 unique active dates
    unique_dates = sorted(combined["DATE"].drop_duplicates(), reverse=True)[:60]
    final_df = combined[combined["DATE"].isin(unique_dates)].sort_values(["SYMBOL", "DATE"])

    final_df.to_parquet(PARQUET_PATH, index=False)
    print(f"COMPLETE: Saved {len(final_df)} records across {len(unique_dates)} dates to {PARQUET_PATH}.")

if __name__ == "__main__":
    update_rolling_history()
