
import streamlit as st
from datetime import datetime, timezone
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

# 🔍 Fetch OHLC (daily or weekly)
def fetch_ohlc(instrument, granularity="D"):
    params = {"granularity": granularity, "count": 2, "price": "M"}
    url = BASE_URL.format(instrument)
    r = requests.get(url, headers=HEADERS, params=params, timeout=20)
    r.raise_for_status()
    candles = r.json().get("candles", [])
    if len(candles) < 2:
        raise ValueError("Not enough candles returned")
    prev = candles[-2]
    date = prev["time"][:10]
    ohlc = prev["mid"]
    return float(ohlc["o"]), float(ohlc["h"]), float(ohlc["l"]), float(ohlc["c"]), date

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

# 🧰 Native, theme-aware UI with "copy per two pivot values"
def render_pivot_pairs_native(level_values):
    """
    level_values: dict like {"R3": 123.4567, "R2": ...}
    Shows pairs with one copy control per two values (via st.code).
    """
    pairs = [("R3", "R2"), ("R1", "Pivot"), ("S1", "S2"), ("S3", None)]

    # Header row
    h1, h2, h3 = st.columns([1.0, 1.0, 1.2])
    h1.markdown("**Left**")
    h2.markdown("**Right**")
    h3.markdown("**Copy Pair**")

    for left, right in pairs:
        c1, c2, c3 = st.columns([1.0, 1.0, 1.2])

        # Left value
        lv = level_values.get(left)
        lv_str = fmt4(lv) if lv is not None else ""
        c1.markdown(f"**{left}**: `{lv_str}`")

        # Right value
        if right is not None:
            rv = level_values.get(right)
            rv_str = fmt4(rv) if rv is not None else ""
            c2.markdown(f"**{right}**: `{rv_str}`")
            pair_text = f"{left}: {lv_str}\n{right}: {rv_str}"
        else:
            c2.write("")
            pair_text = f"{left}: {lv_str}"

        # Copy both with one native copy icon
        c3.code(pair_text, language="text")

    # Copy all at once
    with st.expander("Copy all levels"):
        all_text = "\n".join(f"{k}: {fmt4(v)}" for k, v in level_values.items())
        st.code(all_text, language="text")

# 🚀 Run Pivot Calculation
def run_pivot(granularity="D"):
    today = datetime.now(timezone.utc).date()
    label = "Daily" if granularity == "D" else "Weekly"
    st.subheader(f"📅 {label} Pivot Levels for {today}")

    for name, symbol in INSTRUMENTS.items():
        try:
            o, h, l, c, candle_date = fetch_ohlc(symbol, granularity)
            pivots = calculate_pivots(h, l, c)
            log_to_csv(name, candle_date, o, h, l, c, pivots)
            r3, r2, r1, p, s1, s2, s3 = pivots

            st.markdown(f"### 📊 {name}")

            # Native metrics (theme-aware)
            cols = st.columns(4)
            cols[0].metric("Open", f"{o:.2f}")
            cols[1].metric("High", f"{h:.2f}")
            cols[2].metric("Low", f"{l:.2f}")
            cols[3].metric("Close", f"{c:.2f}", delta=f"{(c - o):+.2f}")

            st.markdown("#### 📌 Pivot Levels")

            level_values = {
                "R3": r3,
                "R2": r2,
                "R1": r1,
                "Pivot": p,
                "S1": s1,
                "S2": s2,
                "S3": s3,
            }
            render_pivot_pairs_native(level_values)

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
    run_pivot(granularity)
else:
    view_logs()
