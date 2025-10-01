import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import pytz

# -----------------------------
# CONFIGURATION
# -----------------------------
OANDA_API_URL = "https://api-fxpractice.oanda.com/v3/instruments/{}/candles"
OANDA_API_KEY = "YOUR_OANDA_API_KEY"   # <-- replace with your key

# -----------------------------
# HELPERS
# -----------------------------
def ist_to_utc_iso(dt_obj):
    """Convert IST datetime object to UTC ISO8601 string with Z suffix."""
    ist = pytz.timezone("Asia/Kolkata")
    dt_ist = ist.localize(dt_obj)
    dt_utc = dt_ist.astimezone(pytz.utc)
    return dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

def fetch_candles_range(instrument, granularity, date_from, date_to):
    """Fetch candles from OANDA REST API for a custom date/time range."""
    url = OANDA_API_URL.format(instrument)
    headers = {"Authorization": f"Bearer {OANDA_API_KEY}"}
    params = {
        "granularity": granularity,
        "price": "M",
        "from": date_from,
        "to": date_to
    }
    r = requests.get(url, headers=headers, params=params)
    r.raise_for_status()
    data = r.json()["candles"]
    
    records = []
    for c in data:
        records.append({
            "time": c["time"],
            "Open": float(c["mid"]["o"]),
            "High": float(c["mid"]["h"]),
            "Low": float(c["mid"]["l"]),
            "Close": float(c["mid"]["c"])
        })
    return pd.DataFrame(records)

def body_as_atr_multiple(df, atr_value):
    """Compute each candle's body size as a multiple of a manually supplied ATR."""
    body = (df['Close'] - df['Open']).abs()
    multiples = body / atr_value
    return multiples.round(2)

# -----------------------------
# STREAMLIT DASHBOARD
# -----------------------------
st.set_page_config(page_title="ATR Body Multiple Dashboard", layout="wide")

st.title("📊 ATR Body Multiple Dashboard")

# Sidebar controls
st.sidebar.header("Controls")

instrument = st.sidebar.selectbox(
    "Select Instrument",
    ["XAU_USD", "NAS100_USD", "US30_USD"]
)

timeframe = st.sidebar.radio(
    "Select Timeframe",
    ["H4", "D"],
    index=0
)

atr_val = st.sidebar.number_input(
    "Manual ATR (price units)",
    min_value=0.1,
    value=20.0,
    step=0.1
)

col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start Date (IST)")
    start_time = st.time_input("Start Time (IST)", value=datetime.now().time())
with col2:
    end_date = st.date_input("End Date (IST)")
    end_time = st.time_input("End Time (IST)", value=datetime.now().time())

# Combine into datetime objects
start_dt = datetime.combine(start_date, start_time)
end_dt = datetime.combine(end_date, end_time)

# Convert IST -> UTC ISO8601
date_from_utc = ist_to_utc_iso(start_dt)
date_to_utc = ist_to_utc_iso(end_dt)

# Fetch data
if st.sidebar.button("Fetch Data"):
    with st.spinner("Fetching candles from OANDA..."):
        df = fetch_candles_range(instrument, timeframe, date_from_utc, date_to_utc)
        if df.empty:
            st.warning("No candles returned for this range.")
        else:
            df["Body_x_ATR"] = body_as_atr_multiple(df, atr_val)
            df["Body_x_ATR_str"] = df["Body_x_ATR"].astype(str) + "x ATR"
            
            # Display table
            st.subheader(f"{instrument} ({timeframe}) Candle Bodies vs ATR")
            st.dataframe(
                df[["time","Open","Close","Body_x_ATR_str"]],
                use_container_width=True
            )
            
            # Chart
            st.subheader("Body Multiples Chart")
            st.bar_chart(df.set_index("time")["Body_x_ATR"])
