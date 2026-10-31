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
    Downloads official NSE sec_bhavdata_full CSV containing exact
    exchange CLOSE, PREV_CLOSE, and delivery volume.
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
        
        # Filter for equity market
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
            
        # Official percentage change directly from exchange PREV_CLOSE
        df["PCT_CHANGE"] = np.where(
            df["PREV_CLOSE"] > 0,
            ((df["CLOSE"] - df["PREV_CLOSE"]) / df["PREV_CLOSE"] * 100).round(2),
            0.0
        )
        
        df["DATE"] = pd.to_datetime(target_date)
        return df, True
        
    except Exception as e:
        print(f"Fetch notice for {target_date}: {e}")
        return None, False

def build_official_database():
    os.makedirs(DATA_DIR, exist_ok=True)
    today = datetime.date.today()
    collected_days = []
    
    # Download the last 5 trading days
    lookback = 0
    while len(collected_days) < 5 and lookback < 15:
        candidate = today - datetime.timedelta(days=lookback)
        lookback += 1
        
        if candidate.weekday() >= 5: # Skip weekends
            continue
            
        print(f"Fetching official NSE trading day: {candidate}...")
        df_day, success = fetch_sec_bhavdata(candidate)
        if success and df_day is not None and not df_day.empty:
            print(f"-> Ingested {len(df_day)} stocks for {candidate}")
            collected_days.append(df_day)

    if not collected_days:
        raise RuntimeError("NSE archives could not be reached. Check workflow network access.")

    # Overwrite the entire dataset with verified exchange records
    final_df = pd.concat(collected_days, ignore_index=True)
    final_df = final_df.sort_values(["SYMBOL", "DATE"])

    # Overwrite the old synthetic parquet directly
    final_df.to_parquet(PARQUET_PATH, index=False)
    print(f"SUCCESS: Replaced database with {len(final_df)} real NSE records at {PARQUET_PATH}.")

if __name__ == "__main__":
    build_official_database()
