import streamlit as st
from datetime import datetime, timezone
import requests
import csv
import os

# 🧭 Page Setup
st.set_page_config(page_title="Pivot Calculator", page_icon="📈")

# 🔐 OANDA Credentials from Secrets
try:
    API_KEY = st.secrets["API_KEY"]
    ACCOUNT_ID = st.secrets["ACCOUNT_ID"]
except Exception:
    st.error("🔐 API credentials not found in secrets. Please configure `API_KEY` and `ACCOUNT_ID`.")
    st.stop()

HEADERS = {"Authorization": f"Bearer {API_KEY}"}
BASE_URL = f"https://api-fxpractice.oanda.com/v3/instruments/{{}}/candles"
LOG_FILE = "pivot_log.csv"
# 📈 Instruments
INSTRUMENTS = {
    "GOLD": "XAU_USD",
    "NAS100": "NAS100_USD",
    "US30": "US30_USD"
}

# 🔍 Fetch OHLC
def fetch_yesterdays_ohlc(instrument):
    params = {"granularity": "D", "count": 2, "price": "M"}
    url = BASE_URL.format(instrument)
    r = requests.get(url, headers=HEADERS, params=params)
    candles = r.json().get('candles', [])
    if len(candles) < 2:
        raise ValueError("Not enough candles returned")
    yesterday = candles[-2]
    date = yesterday['time'][:10]
    ohlc = yesterday['mid']
    return float(ohlc['o']), float(ohlc['h']), float(ohlc['l']), float(ohlc['c']), date

# 📊 Pivot Logic
def calculate_pivots(high, low, close):
    pivot = (high + low + close) / 3
    r1 = 2 * pivot - low
    r2 = pivot + (high - low)
    r3 = high + 2 * (pivot - low)
    s1 = 2 * pivot - high
    s2 = pivot - (high - low)
    s3 = low - 2 * (high - pivot)
    return round(r3, 2), round(r2, 2), round(r1, 2), round(pivot, 2), round(s1, 2), round(s2, 2), round(s3, 2)

# 🧾 Log to CSV
def log_to_csv(name, date, o, h, l, c, pivots):
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, date, o, h, l, c] + list(pivots))

# 🚀 Run Pivot Calculation
def run_pivot():
    today = datetime.now(timezone.utc).date()
    st.subheader(f"📅 Pivot Levels for {today}")
    for name, symbol in INSTRUMENTS.items():
        try:
            o, h, l, c, candle_date = fetch_yesterdays_ohlc(symbol)
            pivots = calculate_pivots(h, l, c)
            log_to_csv(name, candle_date, o, h, l, c, pivots)
            r3, r2, r1, p, s1, s2, s3 = pivots
            st.markdown(f"### 📊 {name} ({symbol}) — Candle Date: {candle_date}")
            st.write(f"Open: {o:.2f}  High: {h:.2f}  Low: {l:.2f}  Close: {c:.2f}")
            st.write(f"R3: {r3:.2f}  R2: {r2:.2f}  R1: {r1:.2f}  Pivot: {p:.2f}")
            st.write(f"S1: {s1:.2f}  S2: {s2:.2f}  S3: {s3:.2f}")
            st.markdown("---")
        except Exception as e:
            st.error(f"{name}: Failed — {e}")

# 📂 View Logs
def view_logs():
    if not os.path.exists(LOG_FILE):
        st.warning("⚠️ No logs found.")
        return
    with open(LOG_FILE, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            name, date, o, h, l, c, r3, r2, r1, p, s1, s2, s3 = row
            st.markdown(f"### 📊 {name} — {date}")
            st.write(f"Open: {o}  High: {h}  Low: {l}  Close: {c}")
            st.write(f"R3: {r3}  R2: {r2}  R1: {r1}  Pivot: {p}")
            st.write(f"S1: {s1}  S2: {s2}  S3: {s3}")
            st.markdown("---")

# 🧭 UI Controls
st.title("📈 Pivot Point Calculator")

action = st.radio("Choose Action", ["Calculate Pivots", "View Logs"])
if action == "Calculate Pivots":
    run_pivot()
else:
    view_logs()
