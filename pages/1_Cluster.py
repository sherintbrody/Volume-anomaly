
import requests
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

# === CONFIG ===
API_KEY = st.secrets["API_KEY"]
ACCOUNT_ID = st.secrets["ACCOUNT_ID"]
BASE_URL = "https://api-fxpractice.oanda.com/v3"

INSTRUMENTS = {
    "XAU_USD": {
        "base_cluster_pct": 0.15  # 0.15% base for Gold
    },
    "NAS100_USD": {
        "base_cluster_pct": 0.20  # 0.20% base for NAS100 (adjust instrument symbol as needed)
    }
}

ADR_LOOKBACK_DAYS = 20

# === FUNCTIONS ===
def fetch_candles(instrument, count=ADR_LOOKBACK_DAYS+1):
    """
    Fetch daily candles for the given instrument from OANDA.
    Returns a DataFrame with open, high, low, close, time.
    """
    endpoint = f"{OANDA_API_URL}/instruments/{instrument}/candles"
    params = {
        "granularity": "D",
        "count": count,
        "price": "M"  # midpoint
    }
    headers = {
        "Authorization": f"Bearer {API_TOKEN}"
    }
    resp = requests.get(endpoint, headers=headers, params=params)
    resp.raise_for_status()
    j = resp.json()
    candles = j["candles"]
    # parse into DataFrame
    rows = []
    for c in candles:
        if not c["complete"]:
            continue
        pt = datetime.fromisoformat(c["time"].replace("Z",""))
        o = float(c["mid"]["o"])
        h = float(c["mid"]["h"])
        l = float(c["mid"]["l"])
        c_close = float(c["mid"]["c"])
        rows.append({"time": pt, "open": o, "high": h, "low": l, "close": c_close})
    df = pd.DataFrame(rows)
    df = df.sort_values("time")
    return df

def compute_adr(df):
    """
    Compute ADR as the average of (high - low) across the DataFrame rows.
    """
    df["range"] = df["high"] - df["low"]
    adr = df["range"].mean()
    return adr

def compute_cluster_range(current_price, adr, cluster_pct):
    """
    Given current price, ADR (in points), and cluster % (e.g., 0.20%),
    compute cluster range in points = ADR * cluster_pct
    and then compute upper/lower bounds.
    """
    cluster_points = adr * (cluster_pct / 100.0)
    upper = current_price + cluster_points
    lower = current_price - cluster_points
    return cluster_points, lower, upper

def fetch_current_price(instrument):
    """
    Fetch current mid-price for the instrument.
    """
    endpoint = f"{OANDA_API_URL}/accounts/{ACCOUNT_ID}/pricing"
    params = {
        "instruments": instrument
    }
    headers = {
        "Authorization": f"Bearer {API_TOKEN}"
    }
    resp = requests.get(endpoint, headers=headers, params=params)
    resp.raise_for_status()
    j = resp.json()
    price_info = j["prices"][0]
    # Use mid of bid+ask
    bid = float(price_info["bids"][0]["price"])
    ask = float(price_info["asks"][0]["price"])
    mid = (bid + ask) / 2.0
    return mid

# === MAIN ===
if __name__ == "__main__":
    for instr, meta in INSTRUMENTS.items():
        print(f"\nInstrument: {instr}")
        try:
            df = fetch_candles(instr)
            adr = compute_adr(df)
            print(f"  ADR (avg high-low) over last {ADR_LOOKBACK_DAYS} days = {adr:.4f} points")

            current_price = fetch_current_price(instr)
            print(f"  Current price mid = {current_price:.4f}")

            cluster_pct = meta["base_cluster_pct"]
            cluster_points, lower, upper = compute_cluster_range(current_price, adr, cluster_pct)
            print(f"  Using cluster % = {cluster_pct:.2f}% → cluster width = ±{cluster_points:.4f} points")
            print(f"  Cluster range: Lower = {lower:.4f}, Upper = {upper:.4f}")

        except Exception as e:
            print("  Error processing instrument:", e)
