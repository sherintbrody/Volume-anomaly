import streamlit as st
from datetime import datetime, timezone, timedelta
import requests
import csv
import os
import pandas as pd

# 🧭 Page Setup (modern wide layout)
st.set_page_config(
    page_title="Pivot Calculator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

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

# Price source (Mid, to match your reference behavior)
PRICE_TYPE = "M"  # "M"=mid, "B"=bid, "A"=ask
OHLC_KEY = {"B": "bid", "M": "mid", "A": "ask"}[PRICE_TYPE]

# 💄 Simple badges for a modern header
BADGE_CSS = """
<style>
.badge {
  display:inline-block; padding:4px 10px; border-radius:999px; font-size:12px;
  background: rgba(2,132,199,.12); color:#0284c7; margin-right:6px; border:1px solid rgba(2,132,199,.25)
}
.badge.neutral {
  background: rgba(148,163,184,.12); color:#334155; border-color: rgba(148,163,184,.25);
}
</style>
"""
st.markdown(BADGE_CSS, unsafe_allow_html=True)

def iso_midnight_utc(d):
    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")

# ⚡️ Cached session + cached HTTP calls for snappy UX
@st.cache_resource
def get_session():
    s = requests.Session()
    s.headers.update(HEADERS)
    return s

@st.cache_data(show_spinner=False, ttl=60)
def _request_candles_cached(instrument, params_tuple):
    params = dict(params_tuple)
    url = BASE_URL.format(instrument)
    s = get_session()
    r = s.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json().get("candles", [])

def _request_candles(instrument, params):
    # Normalize params to a hashable tuple for caching
    params_tuple = tuple(sorted(params.items()))
    return _request_candles_cached(instrument, params_tuple)

def _extract_ohlc(candle):
    ohlc = candle[OHLC_KEY]
    return float(ohlc["o"]), float(ohlc["h"]), float(ohlc["l"]), float(ohlc["c"])

# 🔍 Last completed candle (native OANDA D/W candles)
def fetch_last_completed_candle(instrument, granularity="D"):
    params = {"granularity": granularity, "price": PRICE_TYPE, "count": 2}
    candles = _request_candles(instrument, params)
    if len(candles) < 2:
        raise ValueError("Not enough candles returned")
    c = candles[-2]  # last completed
    o, h, l, c_close = _extract_ohlc(c)
    return o, h, l, c_close, c["time"][:10]

# 🔁 Prior completed candle strictly before a selected date
# Uses to=selected_date 00:00Z and count=1 to get the previous D/W candle
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
        # Fallback: step back by weeks
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

def fmt_val(v, d=4):
    try:
        return f"{float(v):.{d}f}"
    except Exception:
        return str(v)

# 🧰 Native, theme-aware pivot list with per-level copy
def render_pivot_levels_native(rows, d=4):
    h1, h2, h3 = st.columns([1.0, 1.6, 1.0])
    h1.markdown("**Level**")
    h2.markdown("**Value**")
    h3.markdown("**Copy**")
    for lvl, val in rows:
        val_str = fmt_val(val, d)
        c1, c2, c3 = st.columns([1.0, 1.6, 1.0])
        c1.markdown(f"**{lvl}**")
        c2.markdown(f"`{val_str}`")
        c3.code(val_str, language="text")
    with st.expander("Copy all levels"):
        all_text = "\n".join(f"{lvl}: {fmt_val(val, d)}" for lvl, val in rows)
        st.code(all_text, language="text")

# 🚀 Run Pivot Calculation (modern UI)
def run_pivot(granularity="D", custom_date=None, selected_names=None, decimals=4):
    today = datetime.now(timezone.utc).date()
    label = "Daily" if granularity == "D" else "Weekly"
    pivot_date = custom_date if custom_date else today

    # Header + badges
    st.subheader("📈 Pivot Dashboard")
    st.markdown(
        f'<span class="badge">Pivot date: {pivot_date}</span>'
        f'<span class="badge neutral">Timeframe: {label}</span>'
        f'<span class="badge neutral">Price: {PRICE_TYPE}</span>'
        f'<span class="badge neutral">Decimals: {decimals}</span>',
        unsafe_allow_html=True,
    )

    names = selected_names if selected_names else list(INSTRUMENTS.keys())
    results = []

    for name in names:
        symbol = INSTRUMENTS[name]
        try:
            # Force prior period explicitly to avoid boundary including same period
            if custom_date:
                query_date = custom_date - timedelta(days=1) if granularity == "D" else custom_date - timedelta(days=7)
                o, h, l, c, used_date = fetch_prior_candle_before_date(symbol, granularity, query_date)
            else:
                o, h, l, c, used_date = fetch_last_completed_candle(symbol, granularity)

            r3, r2, r1, p, s1, s2, s3 = calculate_pivots(h, l, c)

            # Card-style section (bordered container)
            with st.container(border=True):
                st.markdown(f"### {name}")
                st.caption(f"Candle used: {used_date}")

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Open", fmt_val(o, decimals))
                m2.metric("High", fmt_val(h, decimals))
                m3.metric("Low",  fmt_val(l, decimals))
                delta = float(c - o)
                m4.metric("Close", fmt_val(c, decimals), delta=f"{delta:+.{decimals}f}")

                st.markdown("#### 📌 Pivot Levels")
                rows = [("R3", r3), ("R2", r2), ("R1", r1), ("Pivot", p), ("S1", s1), ("S2", s2), ("S3", s3)]
                render_pivot_levels_native(rows, d=decimals)

            # Collect for export and logging
            results.append({
                "Name": name, "CandleDate": used_date,
                "Open": o, "High": h, "Low": l, "Close": c,
                "R3": r3, "R2": r2, "R1": r1, "Pivot": p, "S1": s1, "S2": s2, "S3": s3
            })
            log_to_csv(name, used_date, o, h, l, c, (r3, r2, r1, p, s1, s2, s3))

        except Exception as e:
            st.error(f"{name}: Failed — {e}")

    # Export pivots (CSV)
    if results:
        df = pd.DataFrame(results)
        st.download_button(
            "⬇️ Download pivots (CSV)",
            data=df.to_csv(index=False),
            file_name=f"pivots_{label.lower()}_{pivot_date}.csv",
            mime="text/csv",
            use_container_width=True,
        )

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

# 🧭 Sidebar Controls (modern extras)
st.sidebar.title("📈 Pivot Dashboard")
action = st.sidebar.radio("Choose Action", ["Calculate Pivots", "View Logs"])

if action == "Calculate Pivots":
    timeframe = st.sidebar.radio("Select Timeframe", ["Daily", "Weekly"], horizontal=True)
    granularity = "D" if timeframe == "Daily" else "W"

    use_custom = st.sidebar.toggle("Use custom date", value=False)
    custom_date = st.sidebar.date_input("Select date", value=datetime.now(timezone.utc).date()) if use_custom else None

    # Instrument multiselect
    selected_names = st.sidebar.multiselect(
        "Instruments",
        options=list(INSTRUMENTS.keys()),
        default=list(INSTRUMENTS.keys()),
    )

    # Decimals control
    decimals = st.sidebar.slider("Decimals", 2, 6, 4, help="Number of decimals to display")

    run_pivot(granularity, custom_date=custom_date, selected_names=selected_names, decimals=decimals)
else:
    view_logs()
