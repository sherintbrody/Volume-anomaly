import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import pytz

# -----------------------------
# CONFIGURATION
# -----------------------------
OANDA_API_URL = "https://api-fxpractice.oanda.com/v3/instruments/{}/candles"
API_KEY = st.secrets["API_KEY"]
ACCOUNT_ID = st.secrets["ACCOUNT_ID"]

# -----------------------------
# TIMEZONE CONVERSION
# -----------------------------
def ist_to_utc_iso(dt_obj):
    ist = pytz.timezone("Asia/Kolkata")
    dt_ist = ist.localize(dt_obj)
    dt_utc = dt_ist.astimezone(pytz.utc)
    return dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

# -----------------------------
# FETCH CANDLES
# -----------------------------
def fetch_candles_range(instrument, granularity, date_from, date_to):
    url = OANDA_API_URL.format(instrument)
    headers = {"Authorization": f"Bearer {API_KEY}"}
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

# -----------------------------
# BODY MULTIPLES vs ATR
# -----------------------------
def body_as_atr_multiple(df, atr_value):
    body = (df['Close'] - df['Open']).abs()
    multiples = body / atr_value
    signal = pd.cut(multiples, bins=[0, 0.7, 1.3, float('inf')],
                    labels=["Weak", "Neutral", "Strong"])
    return multiples.round(2), signal

# -----------------------------
# STREAMLIT DASHBOARD
# -----------------------------
st.set_page_config(page_title="ATR Body Dashboard", layout="wide")
st.title("📊 ATR Body Multiple Dashboard")

# Sidebar controls
st.sidebar.header("🔧 Controls")

instrument = st.sidebar.selectbox("Instrument", ["XAU_USD", "NAS100_USD", "US30_USD"])
timeframe = st.sidebar.radio("Timeframe", ["H4", "D"])
atr_val = st.sidebar.number_input("Manual ATR (price units)", min_value=0.1, value=20.0, step=0.1)

st.sidebar.markdown("### 📅 Select Time Range (IST)")
start_date = st.sidebar.date_input("Start Date", value=datetime.today())
start_time = st.sidebar.time_input("Start Time", value=datetime.strptime("09:00:00", "%H:%M:%S").time())
end_date = st.sidebar.date_input("End Date", value=datetime.today())
end_time = st.sidebar.time_input("End Time", value=datetime.strptime("21:00:00", "%H:%M:%S").time())

# Combine IST datetime
start_dt = datetime.combine(start_date, start_time)
end_dt = datetime.combine(end_date, end_time)

# Convert to UTC ISO8601
date_from_utc = ist_to_utc_iso(start_dt)
date_to_utc = ist_to_utc_iso(end_dt)

# Fetch and display
if st.sidebar.button("🚀 Fetch Candles"):
    with st.spinner("Fetching data from OANDA..."):
        df = fetch_candles_range(instrument, timeframe, date_from_utc, date_to_utc)
        if df.empty:
            st.warning("No candles returned for this range.")
        else:
            df["Body_x_ATR"], df["Signal"] = body_as_atr_multiple(df, atr_val)
            df["Body_x_ATR_str"] = df["Body_x_ATR"].astype(str) + "x ATR"
            
            st.subheader(f"{instrument} ({timeframe}) Candle Body Multiples")
            st.dataframe(df[["time", "Open", "Close", "Body_x_ATR_str", "Signal"]], use_container_width=True)
            
            st.subheader("📊 Body ÷ ATR Chart")
            st.bar_chart(df.set_index("time")["Body_x_ATR"])
            
            st.download_button("📥 Export to CSV", df.to_csv(index=False), file_name="atr_body_multiples.csv")
