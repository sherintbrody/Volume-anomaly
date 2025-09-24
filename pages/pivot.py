import streamlit as st
from datetime import datetime, timezone, timedelta
import requests
import csv
import os
import pandas as pd

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
BASE_URL = "https://api-fxpractice.oanda.com/v3/instruments/{}/candles"
LOG_FILE = "pivot_log.csv"

# 📈 Instruments
INSTRUMENTS = {
    "GOLD": "XAU_USD",
    "NAS100": "NAS100_USD",
    "US30": "US30_USD",
}

# Use Mid prices (matches your reference script)
PRICE_TYPE = "M"  # "M"=mid, "B"=bid, "A"=ask
OHLC_KEY = {"B": "bid", "M": "mid", "A": "ask"}[PRICE_TYPE]

def iso_midnight_utc(d):
    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")

def _request_candles(instrument, params):
    url = BASE_URL.format(instrument)
    r = requests.get(url, headers=HEADERS, params=params, timeout=20)
    r.raise_for_status()
    return r.json().get("candles", [])

def _extract_ohlc(candle):
    ohlc = candle[OHLC_KEY]
    return float(ohlc["o"]), float(ohlc["h"]), float(ohlc["l"]), float(ohlc["c"])

# 🔍 Last completed candle (same method as your working script)
def fetch_last_completed_candle(instrument, granularity="D"):
    params = {"granularity": granularity, "price": PRICE_TYPE, "count": 2}
    candles = _request_candles(instrument, params)
    if len(candles) < 2:
        raise ValueError("Not enough candles returned")
    c = candles[-2]  # last completed
    o, h, l, c_close = _extract_ohlc(c)
    return o, h, l, c_close, c["time"][:10]

# 🔁 Prior completed candle strictly before a selected date
def fetch_prior_candle_before_date(instrument, granularity, selected_date):
    if granularity == "D":
        params = {"granularity": "D", "price": PRICE_TYPE, "to": iso_midnight_utc(selected_date), "count": 1}
        candles = _request_candles(instrument, params)
        if candles:
            c = candles[-1]
            o, h, l, c_close = _extract_ohlc(c)
            return o, h, l, c_close, c["time"][:10]
        # Fallback if boundary yields nothing
        t = selected_date - timedelta(days=1)
        for _ in range(9):
            params["to"] = iso_midnight_utc(t)
            candles = _request_candles(instrument, params)
            if candles:
                c = candles[-1]
                o, h, l, c_close = _extract_ohlc(c)
                return o, h, l, c_close, c["time"][:10]
            t -= timedelta(days=1)
        raise ValueError(f"No prior daily candle found before {selected_date} for {instrument}")
    else:
        params = {"granularity": "W", "price": PRICE_TYPE, "to": iso_midnight_utc(selected_date), "count": 1}
        candles = _request_candles(instrument, params)
        if candles:
            c = candles[-1]
            o, h, l, c_close = _extract_ohlc(c)
            return o, h, l, c_close, c["time"][:10]
        # Fallback step back by weeks if needed
        t = selected_date - timedelta(days=7)
        for _ in range(5):
            params["to"] = iso_midnight_utc(t)
            candles = _request_candles(instrument, params)
            if candles:
                c = candles[-1]
                o, h, l, c_close = _extract_ohlc(c)
                return o, h, l, c_close, c["time"][:10]
            t -= timedelta(days=7)
        raise ValueError(f"No prior weekly candle found before {selected_date} for {instrument}")

# 📊 Pivot Logic (Classic)
def calculate_pivots(high, low, close):
    pivot = (high + low + close) / 3
    r1 = 2 * pivot - low
    r2 = pivot + (high - low)
    r3 = high + 2 * (pivot - low)
    s1 = 2 * pivot - high
    s2 = pivot - (high - low)
    s3 = low - 2 * (high - pivot)
    return round(r3, 4), round(r2, 4), round(r1, 4), round(pivot, 4), round(s1, 4), round(s2, 4), round(s3, 4)

# 🧾 Log to CSV
def log_to_csv(name, date, o, h, l, c, pivots):
    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Name","Date","Open","High","Low","Close","R3","R2","R1","Pivot","S1","S2","S3"])
        writer.writerow([name, date, o, h, l, c] + list(pivots))

def fmt4(v):
    try:
        return f"{float(v):.4f}"
    except Exception:
        return str(v)

# 🧰 Native, theme-aware pivot list with per-level copy
def render_pivot_levels_native(rows):
    h1, h2, h3 = st.columns([1.0, 1.6, 1.0])
    h1.markdown("**Level**")
    h2.markdown("**Value**")
    h3.markdown("**Copy**")
    for lvl, val in rows:
        val_str = fmt4(val)
        c1, c2, c3 = st.columns([1.0, 1.6, 1.0])
        c1.markdown(f"**{lvl}**")
        c2.markdown(f"`{val_str}`")
        c3.code(val_str, language="text")
    with st.expander("Copy all levels"):
        all_text = "\n".join(f"{lvl}: {fmt4(val)}" for lvl, val in rows)
        st.code(all_text, language="text")

# 🚀 Run Pivot Calculation
def run_pivot(granularity="D", custom_date=None):
    today = datetime.now(timezone.utc).date()
    label = "Daily" if granularity == "D" else "Weekly"
    pivot_date = custom_date if custom_date else today  # pivot date = selected date
    basis = "previous trading day (D candle)" if granularity == "D" else "previous week (W candle)"
    st.subheader(f"📅 {label} Pivot Levels for {pivot_date} — based on {basis}")

    for name, symbol in INSTRUMENTS.items():
        try:
            if custom_date:
                # Force prior period explicitly to avoid 'to' boundary including same period
                if granularity == "D":
                    query_date = custom_date - timedelta(days=1)
                else:
                    query_date = custom_date - timedelta(days=7)
                o, h, l, c, used_date = fetch_prior_candle_before_date(symbol, granularity, query_date)
            else:
                o, h, l, c, used_date = fetch_last_completed_candle(symbol, granularity)

            pivots = calculate_pivots(h, l, c)
            log_to_csv(name, used_date, o, h, l, c, pivots)
            r3, r2, r1, p, s1, s2, s3 = pivots

            st.markdown(f"### 📊 {name} — candle used: {used_date}")

            cols = st.columns(4)
            cols[0].metric("Open", f"{o:.2f}")
            cols[1].metric("High", f"{h:.2f}")
            cols[2].metric("Low", f"{l:.2f}")
            cols[3].metric("Close", f"{c:.2f}", delta=f"{(c - o):+.2f}")

            st.markdown("#### 📌 Pivot Levels")
            rows = [("R3", r3), ("R2", r2), ("R1", r1), ("Pivot", p), ("S1", s1), ("S2", s2), ("S3", s3)]
            render_pivot_levels_native(rows)

            st.divider()
        except Exception as e:
            st.error(f"{name}: Failed — {e}")

# 📂 View Logs
def view_logs():
    if not os.path.exists(LOG_FILE):
        st.warning("⚠️ No logs found.")
        return
    try:
        df = pd.read_csv(LOG_FILE)
        st.dataframe(df, use_container_width=True)
    except Exception as e:
        st.error(f"Failed to read logs: {e}")

# 🧭 Sidebar Controls
st.sidebar.title("📈 Pivot Dashboard")
action = st.sidebar.radio("Choose Action", ["Calculate Pivots", "View Logs"])
if action == "Calculate Pivots":
    timeframe = st.sidebar.radio("Select Timeframe", ["Daily", "Weekly"], horizontal=True)
    granularity = "D" if timeframe == "Daily" else "W"

    use_custom = st.sidebar.toggle("Use custom date", value=False)
    custom_date = None
    if use_custom:
        custom_date = st.sidebar.date_input("Select date", value=datetime.now(timezone.utc).date())

    run_pivot(granularity, custom_date=custom_date if use_custom else None)
else:
    view_logs()
