import requests, json, os
import streamlit as st
from datetime import datetime, timedelta, time
import pytz
import pandas as pd
from collections import defaultdict
import wcwidth

# ====== CONFIG ======
API_KEY = st.secrets["API_KEY"]
ACCOUNT_ID = st.secrets["ACCOUNT_ID"]
BASE_URL = "https://api-fxpractice.oanda.com/v3"

INSTRUMENTS = {
    "XAUUSD": "XAU_USD",
    "NAS100": "NAS100_USD",
    "US30": "US30_USD"
}

IST = pytz.timezone("Asia/Kolkata")
UTC = pytz.utc
headers = {"Authorization": f"Bearer {API_KEY}"}

# ====== SIDEBAR CONFIG ======
st.sidebar.title("🔧 Backtest Settings")

st.sidebar.multiselect(
    "Select Instruments",
    options=list(INSTRUMENTS.keys()),
    default=list(INSTRUMENTS.keys()),
    key="selected_instruments"
)

st.sidebar.radio(
    "🕒 Time Bucket",
    ["15 min", "30 min", "1 hour"],
    index=2,
    key="bucket_choice"
)

st.sidebar.slider(
    "📈 Threshold Multiplier",
    min_value=1.0, max_value=3.0, step=0.1,
    value=1.4,
    key="threshold_multiplier"
)

today = datetime.now(IST).date()

start_date = st.sidebar.date_input("📅 Start Date", value=today - timedelta(days=30), max_value=today)
end_date = st.sidebar.date_input("📅 End Date", value=today, min_value=start_date, max_value=today)


# ====== UTILITIES ======
def get_time_bucket(dt_ist, bucket_size_minutes):
    bucket_start_minute = (dt_ist.minute // bucket_size_minutes) * bucket_size_minutes
    bucket_start = dt_ist.replace(minute=bucket_start_minute, second=0, microsecond=0)
    bucket_end = bucket_start + timedelta(minutes=bucket_size_minutes)
    return f"{bucket_start.strftime('%I:%M %p')}–{bucket_end.strftime('%I:%M %p')}"

def get_sentiment(candle):
    o = float(candle["mid"]["o"])
    c = float(candle["mid"]["c"])
    return "🟩" if c > o else "🟥" if c < o else "▪️"

def pad_display(s, width):
    pad_len = width - sum(wcwidth.wcwidth(ch) for ch in s)
    return s + " " * max(pad_len, 0)

def get_spike_bar(multiplier):
    if multiplier < 1.2:
        return pad_display("", 5)
    bars = int((multiplier - 1.2) * 5)
    bar_str = "┃" * max(1, min(bars, 5))
    return pad_display(bar_str, 5)

# ====== DATA FETCH ======
@st.cache_data(ttl=600)
def fetch_candles(instrument_code, from_time, to_time):
    now_utc = datetime.now(UTC)
    from_time = min(from_time, now_utc)
    to_time = min(to_time, now_utc)

    params = {
        "granularity": "M15",
        "price": "M",
        "from": from_time.isoformat(),
        "to": to_time.isoformat()
    }
    url = f"{BASE_URL}/accounts/{ACCOUNT_ID}/instruments/{instrument_code}/candles"
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=20)
    except Exception as e:
        st.error(f"❌ Network error for {instrument_code}: {e}")
        return []

    if resp.status_code != 200:
        st.error(f"❌ Failed to fetch {instrument_code} data: {resp.text}")
        return []
    return resp.json().get("candles", [])

def compute_bucket_averages(code, bucket_size_minutes, start_utc, end_utc):
    bucket_volumes = defaultdict(list)
    candles = fetch_candles(code, start_utc, end_utc)
    for c in candles:
        try:
            t_utc = datetime.strptime(c["time"], "%Y-%m-%dT%H:%M:%S.%f000Z")
        except ValueError:
            t_utc = datetime.strptime(c["time"], "%Y-%m-%dT%H:%M:%S.000Z")
        t_ist = t_utc.replace(tzinfo=UTC).astimezone(IST)
        bucket = get_time_bucket(t_ist, bucket_size_minutes)
        bucket_volumes[bucket].append(c["volume"])
    return {b: (sum(vs) / len(vs)) for b, vs in bucket_volumes.items() if vs}

# ====== TABLE RENDERING ======
def render_table_streamlit(name, rows, bucket_minutes):
    st.subheader(f"{name} — Detected Spikes")
    columns = [
        "Time (IST)", f"Time Bucket ({bucket_minutes} min)",
        "Open", "High", "Low", "Close",
        "Volume", "Spike Δ", "Strength", "Sentiment"
    ]
    df = pd.DataFrame(rows, columns=columns)
    st.dataframe(df, width="stretch")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Export CSV", csv, f"{name}_backtest.csv", "text/csv")

# ====== BACKTEST EXECUTION ======
def run_backtest():
    st.title("📊 Historical Volume Spike Backtest")
    bucket_minutes = {"15 min": 15, "30 min": 30, "1 hour": 60}[st.session_state.bucket_choice]
    threshold_multiplier = st.session_state.threshold_multiplier

    start_utc = IST.localize(datetime.combine(start_date, time(0, 0))).astimezone(UTC)
    end_utc = IST.localize(datetime.combine(end_date + timedelta(days=1), time(0, 0))).astimezone(UTC)

    for name in st.session_state.selected_instruments:
        code = INSTRUMENTS[name]
        all_rows = []

        current_day = start_utc
        while current_day < end_utc:
            target_start = current_day
            target_end = target_start + timedelta(days=1)
            current_day += timedelta(days=1)

            avg_start = target_start - timedelta(days=21)
            avg_end = target_start
            bucket_avg = compute_bucket_averages(code, bucket_minutes, avg_start, avg_end)

            candles = fetch_candles(code, target_start, target_end)
            for c in candles:
                try:
                    t_utc = datetime.strptime(c["time"], "%Y-%m-%dT%H:%M:%S.%f000Z")
                except ValueError:
                    t_utc = datetime.strptime(c["time"], "%Y-%m-%dT%H:%M:%S.000Z")
                t_ist = t_utc.replace(tzinfo=UTC).astimezone(IST)
                bucket = get_time_bucket(t_ist, bucket_minutes)
                vol = c["volume"]
                avg = bucket_avg.get(bucket, 0)
                threshold = avg * threshold_multiplier if avg else 0
                over = (threshold > 0 and vol > threshold)
                mult = (vol / threshold) if over and threshold > 0 else 0
                spike_diff = f"🔺{vol - int(threshold)}" if over else ""
                strength = get_spike_bar(mult) if over else pad_display("", 5)
                sentiment = get_sentiment(c)

                if over:
                    all_rows.append([
                        t_ist.strftime("%Y-%m-%d %I:%M %p"),
                        bucket,
                        f"{float(c['mid']['o']):.1f}",
                        f"{float(c['mid']['h']):.1f}",
                        f"{float(c['mid']['l']):.1f}",
                        f"{float(c['mid']['c']):.1f}",
                        vol,
                        spike_diff,
                        strength,
                        sentiment
                    ])

        if all_rows:
            render_table_streamlit(name, all_rows, bucket_minutes)
        else:
            st.info(f"ℹ️ No spikes detected for {name} in selected range.")

# ====== PAGE CONFIG ======
st.set_page_config(page_title="Volume Spike Backtest", layout="wide")

# ====== HEADER ======
st.markdown("""
<h1 style='text-align: center; color: #2E8B57;'>📊 Historical Volume Spike Detector</h1>
<hr style='border:1px solid #ccc;'>
""", unsafe_allow_html=True)

# ====== MAIN EXECUTION ======
run_backtest()
