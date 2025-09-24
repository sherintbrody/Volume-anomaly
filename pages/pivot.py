import streamlit as st
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
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

NY_TZ = ZoneInfo("America/New_York")

def iso_z_from_dt(dt_utc: datetime) -> str:
    # Ensure UTC and RFC3339 Z
    dt_utc = dt_utc.astimezone(timezone.utc).replace(microsecond=0)
    return dt_utc.isoformat().replace("+00:00", "Z")

def parse_oanda_time_z(ts: str) -> datetime:
    # Parses "YYYY-MM-DDTHH:MM:SS.ssssssZ" safely to aware UTC datetime
    return datetime.strptime(ts[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)

# 🔍 Latest previous completed candle (for current pivots)
def fetch_ohlc(instrument, granularity="D"):
    params = {"granularity": granularity, "count": 2, "price": "M"}
    url = BASE_URL.format(instrument)
    r = requests.get(url, headers=HEADERS, params=params, timeout=20)
    r.raise_for_status()
    candles = r.json().get("candles", [])
    if len(candles) < 2:
        raise ValueError("Not enough candles returned")
    prev = candles[-2]  # last completed candle
    date = prev["time"][:10]
    ohlc = prev["mid"]
    return float(ohlc["o"]), float(ohlc["h"]), float(ohlc["l"]), float(ohlc["c"]), date

# 🔎 Custom DAILY: previous US trading day (aligned to New York midnight, skip Sat/Sun)
def fetch_prev_daily_candle_us(instrument, selected_date):
    """
    selected_date: date the user picked (US-based). We fetch the prior US trading day.
    Strategy: align daily candles to New York midnight and scan backwards to the last weekday.
    """
    url = BASE_URL.format(instrument)
    # Anchor at NY midnight of the selected date
    anchor_ny = datetime(selected_date.year, selected_date.month, selected_date.day, 0, 0, tzinfo=NY_TZ)
    anchor_utc = anchor_ny.astimezone(timezone.utc)

    params = {
        "granularity": "D",
        "price": "M",
        "alignmentTimezone": "America/New_York",
        "dailyAlignment": 0,  # align daily candles to 00:00 NY time
        "to": iso_z_from_dt(anchor_utc),
        "count": 7,  # get up to a week back to skip weekends/holidays
    }
    r = requests.get(url, headers=HEADERS, params=params, timeout=20)
    r.raise_for_status()
    candles = r.json().get("candles", [])
    if not candles:
        raise ValueError(f"No daily candles found before {selected_date} (US) for {instrument}")

    # Walk backward to find the most recent weekday strictly before selected_date
    for c in reversed(candles):
        t_utc = parse_oanda_time_z(c["time"])
        t_ny_date = t_utc.astimezone(NY_TZ).date()
        if t_ny_date < selected_date and t_ny_date.weekday() < 5:  # Mon-Fri
            ohlc = c["mid"]
            return float(ohlc["o"]), float(ohlc["h"]), float(ohlc["l"]), float(ohlc["c"]), t_ny_date.strftime("%Y-%m-%d")
    raise ValueError(f"No prior US trading day found before {selected_date} for {instrument}")

# 🔎 Custom WEEKLY: previous weekly candle before selected date (aligned to New York TZ)
def fetch_prev_weekly_candle_us(instrument, selected_date):
    url = BASE_URL.format(instrument)
    # Anchor at NY midnight of the selected date
    anchor_ny = datetime(selected_date.year, selected_date.month, selected_date.day, 0, 0, tzinfo=NY_TZ)
    anchor_utc = anchor_ny.astimezone(timezone.utc)

    params = {
        "granularity": "W",
        "price": "M",
        "alignmentTimezone": "America/New_York",
        "weeklyAlignment": "Friday",  # common weekly alignment for FX
        "to": iso_z_from_dt(anchor_utc),
        "count": 1,  # last completed weekly candle before anchor
    }
    r = requests.get(url, headers=HEADERS, params=params, timeout=20)
    r.raise_for_status()
    candles = r.json().get("candles", [])
    if not candles:
        raise ValueError(f"No weekly candles found before {selected_date} (US) for {instrument}")

    c = candles[-1]
    ohlc = c["mid"]
    # Convert start time to NY date label for clarity
    t_utc = parse_oanda_time_z(c["time"])
    ny_date = t_utc.astimezone(NY_TZ).date()
    return float(ohlc["o"]), float(ohlc["h"]), float(ohlc["l"]), float(ohlc["c"]), ny_date.strftime("%Y-%m-%d")

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
    basis = "previous US trading day" if granularity == "D" else "previous week (US TZ)"
    st.subheader(f"📅 {label} Pivot Levels for {hdr_date} — based on {basis}")

    for name, symbol in INSTRUMENTS.items():
        try:
            if custom_date:
                if granularity == "D":
                    o, h, l, c, used_date = fetch_prev_daily_candle_us(symbol, custom_date)
                else:
                    o, h, l, c, used_date = fetch_prev_weekly_candle_us(symbol, custom_date)
            else:
                o, h, l, c, used_date = fetch_ohlc(symbol, granularity)

            pivots = calculate_pivots(h, l, c)
            # Log using the displayed (NY-based) date for clarity
            log_to_csv(name, used_date, o, h, l, c, pivots)
            r3, r2, r1, p, s1, s2, s3 = pivots

            st.markdown(f"### 📊 {name} — candle used (NY date): {used_date}")

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
        custom_date = st.sidebar.date_input("Select date", value=datetime.now(NY_TZ).date())

    run_pivot(granularity, custom_date=custom_date if use_custom else None)
else:
    view_logs()
