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
        "base_cluster_pct": 0.20  # 0.20% base for NAS100
    }
}

ADR_LOOKBACK_DAYS = 20

# === FUNCTIONS ===
def fetch_candles(instrument, count=ADR_LOOKBACK_DAYS+1):
    """
    Fetch daily candles for the given instrument from OANDA.
    Returns a DataFrame with open, high, low, close, time.
    """
    endpoint = f"{BASE_URL}/instruments/{instrument}/candles"
    params = {
        "granularity": "D",
        "count": count,
        "price": "M"  # midpoint
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}"
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
    endpoint = f"{BASE_URL}/accounts/{ACCOUNT_ID}/pricing"
    params = {
        "instruments": instrument
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}"
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

# === STREAMLIT APP ===
def main():
    st.title("ADR & Cluster Range Calculator")
    st.write(f"Analyzing {ADR_LOOKBACK_DAYS}-day Average Daily Range")
    
    for instr, meta in INSTRUMENTS.items():
        st.subheader(f"📊 {instr}")
        
        try:
            with st.spinner(f"Fetching data for {instr}..."):
                # Fetch candles and compute ADR
                df = fetch_candles(instr)
                adr = compute_adr(df)
                
                # Get current price
                current_price = fetch_current_price(instr)
                
                # Compute cluster range
                cluster_pct = meta["base_cluster_pct"]
                cluster_points, lower, upper = compute_cluster_range(current_price, adr, cluster_pct)
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Current Price", f"{current_price:.4f}")
            
            with col2:
                st.metric("ADR (points)", f"{adr:.4f}")
            
            with col3:
                st.metric("Cluster %", f"{cluster_pct:.2f}%")
            
            st.write("**Cluster Range:**")
            col4, col5, col6 = st.columns(3)
            
            with col4:
                st.metric("Lower Bound", f"{lower:.4f}")
            
            with col5:
                st.metric("Cluster Width", f"±{cluster_points:.4f}")
            
            with col6:
                st.metric("Upper Bound", f"{upper:.4f}")
            
            # Show recent price data
            with st.expander(f"View Recent Price Data for {instr}"):
                st.dataframe(df.tail(10))
            
            st.divider()
            
        except Exception as e:
            st.error(f"Error processing {instr}: {str(e)}")

if __name__ == "__main__":
    main()
