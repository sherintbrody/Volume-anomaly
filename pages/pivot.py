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

# 🔍 Latest previous completed candle (for current pivots)
def fetch_ohlc(instrument, granularity="D"):
    params = {"granularity": granularity, "count": 2, "price": "M"}
    url = BASE_URL.format(instrument)
    r = requests.get(url, headers=HEADERS, params=params, timeout=20)
    r.raise_for_status()
    candles = r.json().get("candles", [])
    if len(candles) < 2:
        raise ValueError("Not enough candles returned")
    prev = candles[-2]  # last completed
    date = prev["time"][:10]
    ohlc = prev["mid"]
    return float(ohlc["o"]), float(ohlc["h"]), float(ohlc["l"]), float(ohlc["c"]), date

def iso_z(d):
    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")

# 🔎 Fetch daily candle by exact calendar date (YYYY-MM-DD)
def fetch_daily_candle_by_date(instrument, candle_date):
    url = BASE_URL.format(instrument)
    # Small window to ensure inclusion
    params = {
        "granularity": "D",
        "price": "M",
        "from": iso_z(candle_date - timedelta(days=3)),
        "to": iso_z(candle_date + timedelta(days=1)),
    }
    r = requests.get(url, headers=HEADERS, params=params, timeout=20)
    r.raise_for_status()
    candles = r.json().get("candles", [])
    if not candles:
        raise ValueError(f"No candles returned for {instrument} in selected window")

    target = candle_date.strftime("%Y-%m-%d")
    for c in candles:
        if c["time"][:10] == target and c.get("complete", True):
            ohlc = c["mid"]
            return float(ohlc["o"]), float(ohlc["h"]), float(ohlc["l"]), float(ohlc["c"]), c["time"][:10]
    raise ValueError(f"Daily candle for {instrument} on {target} not found")

# 🗓️ Previous trading day helper (skip Saturday/Sunday)
def previous_trading_day(d):
    d = d - timedelta(days=1)
    while d.weekday() >= 5:  # 5=Saturday, 6=Sunday
        d = d - timedelta(days=1)
    return d

# 🔎 Weekly: fetch the weekly candle that contains a given date
def fetch_weekly_candle_containing(instrument, target_date):
    url = BASE_URL.format(instrument)
    params = {
        "granularity": "W",
        "price": "M",
        "from": iso_z(target_date - timedelta(days=90)),
        "to": iso_z(target_date + timedelta(days=14)),
    }
    r = requests.get(url, headers=HEADERS, params=params, timeout=20)
    r.raise_for_status()
    candles = r.json().get("candles", [])
    if not candles:
        raise ValueError(f"No weekly candles for {instrument} in window")

    def start_date(iso_str):
        return datetime.strptime(iso_str[:19], "%Y-%m-%dT%H:%M:%S").date()

    for c in candles:
        start = start_date(c["time"])
        end = start + timedelta(days=7)
        if start <= target_date < end:
            ohlc = c["mid"]
            return float(ohlc["o"]), float(ohlc["h"]), float(ohlc["l"]), float(ohlc["c"]), c["time"][:10]
    raise ValueError(f"No weekly candle containing {target_date} for {instrument}")

# 🔁 Prior-period fetcher respecting weekends for Daily, previous week for Weekly
def fetch_prior_period_candle(instrument, granularity, selected_date):
    if granularity == "D":
        # Always use prior trading day (skip Sat/Sun), and step back again if holiday/no data
        look = previous_trading_day(selected_date)
        for _ in range(10):  # fallback for holidays
            try:
                return fetch_daily_candle_by_date(instrument, look)
            except Exception:
                look = previous_trading_day(look)
        raise ValueError(f"Could not find a prior trading day candle before {selected_date}")
    else:
        # Use the previous week relative to the selected date
        prev_week_date = selected_date - timedelta(days=7)
        return fetch_weekly_candle_containing(instrument, prev_week_date)

# 📊 Pivot Logic (Classic)
def calculate_pivots(high, low, close):
    pivot = (high + low + close) / 3
    r1 = 2 * pivot - low
    r2 = pivot + (high - low)
    r3 = high + 2 * (pivot - low)
    s1 = 2 * pivot - high
    s2 = pivot - (high - low)
    s3 = low - 2 * (high - pivot)
    return (
        round(r3, 4),
        round(r2, 4),
        round(r1, 4),
        round(pivot, 4),
        round(s1, 4),
        round(s2, 4),
        round(s3, 4),
    )

# 🧾 Log to CSV
def log_to_csv(name, date, o, h, l, c, pivots):
    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                [
                    "Name",
                    "Date",
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "R3",
                    "R2",
                    "R1",
                    "Pivot",
                    "S1",
                    "S2",
                    "S3",
                ]
            )
        writer.writerow([name, date, o, h, l, c] + list(pivots))

def fmt4(v):
    try:
        return f"{float(v):.4f}"
    except Exception:
        return str(v)

# 🧰 Native, theme-aware pivot list with per-level copy
def render_pivot_levels_native(rows):
    # Header
    h1, h2, h3 = st.columns([1.0, 1.6, 1.0])
    h1.markdown("**Level**")
    h2.markdown("**Value**")
    h3.markdown("**Copy**")

    # Rows
    for lvl, val in rows:
        val_str = fmt4(val)
        c1, c2, c3 = st.columns([1.0, 1.6, 1.0])
        c1.markdown(f"**{lvl}**")
        c2.markdown(f"`{val_str}`")
        c3.code(val_str, language="text")

    # Copy all (label + value) if needed
    with st.expander("Copy all levels"):
        all_text = "\n".join(f"{lvl}: {fmt4(val)}" for lvl, val in rows)
        st.code(all_text, language="text")

# 🚀 Run Pivot Calculation
def run_pivot(granularity="D", custom_date=None):
    today = datetime.now(timezone.utc).date()
    label = "Daily" if granularity == "D" else "Weekly"
    hdr_date = custom_date if custom_date else today
    basis = "previous trading day" if granularity == "D" else "previous week"
    st.subheader(f"📅 {label} Pivot Levels for {hdr_date} (based on {basis})")

    for name, symbol in INSTRUMENTS.items():
        try:
            if custom_date:
                o, h, l, c, candle_date = fetch_prior_period_candle(symbol, granularity, custom_date)
            else:
                o, h, l, c, candle_date = fetch_ohlc(symbol, granularity)

            pivots = calculate_pivots(h, l, c)
            log_to_csv(name, candle_date, o, h, l, c, pivots)
            r3, r2, r1, p, s1, s2, s3 = pivots

            st.markdown(f"### 📊 {name} — candle used: {candle_date}")

            # Native metrics (theme-aware)
            cols = st.columns(4)
            cols[0].metric("Open", f"{o:.2f}")
            cols[1].metric("High", f"{h:.2f}")
            cols[2].metric("Low", f"{l:.2f}")
            cols[3].metric("Close", f"{c:.2f}", delta=f"{(c - o):+.2f}")

            st.markdown("#### 📌 Pivot Levels")
            rows = [
                ("R3", r3),
                ("R2", r2),
                ("R1", r1),
                ("Pivot", p),
                ("S1", s1),
                ("S2", s2),
                ("S3", s3),
            ]
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
