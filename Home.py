import requests, json, os
import streamlit as st
from datetime import datetime, timedelta, time
import pytz
import pandas as pd
from collections import defaultdict
import wcwidth
from streamlit_autorefresh import st_autorefresh

# ====== PAGE CONFIG ======
st.set_page_config(
    page_title="Volume Spike Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ====== THEME/STYLE ======
BADGE_CSS = """
<style>
:root {
    --chip-bg: rgba(2,132,199,.12);
    --chip-fg: #0284c7;
    --chip-br: rgba(2,132,199,.25);
    --chip2-bg: rgba(148,163,184,.12);
    --chip2-fg: #334155;
    --chip2-br: rgba(148,163,184,.25);
    --ok-bg: rgba(16,185,129,.12);
    --ok-fg: #059669;
    --ok-br: rgba(16,185,129,.25);
    --warn-bg: rgba(234,179,8,.12);
    --warn-fg: #a16207;
    --warn-br: rgba(234,179,8,.25);
}
.badges { margin: 2px 0 12px 0; }
.badge {
    display:inline-block; padding:4px 10px; border-radius:999px; font-size:12px;
    background: var(--chip-bg); color: var(--chip-fg); margin-right:6px;
    display:inline-block;
    padding:4px 10px;
    border-radius:999px;
    font-size:12px;
    background: var(--chip-bg);
    color: var(--chip-fg);
    margin-right:6px;
    border:1px solid var(--chip-br)
}
.badge.neutral { background: var(--chip2-bg); color:var(--chip2-fg); border-color:var(--chip2-br); }
.badge.ok { background: var(--ok-bg); color:var(--ok-fg); border-color:var(--ok-br); }
.badge.warn { background: var(--warn-bg); color:var(--warn-fg); border-color:var(--warn-br); }
.badge.neutral {
    background: var(--chip2-bg);
    color:var(--chip2-fg);
    border-color:var(--chip2-br);
}
.badge.ok {
    background: var(--ok-bg);
    color:var(--ok-fg);
    border-color:var(--ok-br);
}
.badge.warn {
    background: var(--warn-bg);
    color:var(--warn-fg);
    border-color:var(--warn-br);
}
.section-title { margin: 0 0 4px 0; }
</style>
"""
st.markdown(BADGE_CSS, unsafe_allow_html=True)

# ====== CONFIG ======
API_KEY = "5a0f5c6147a2bd7c832d63a6252f0c01-041561ca55b1549327e8c00f3d645f13"
ACCOUNT_ID = "101-004-37091392-001"
BASE_URL = "https://api-fxpractice.oanda.com/v3"

INSTRUMENTS = {
    "XAUUSD": "XAU_USD",
    "NAS100": "NAS100_USD",
    "US30": "US30_USD"
}

IST = pytz.timezone("Asia/Kolkata")
UTC = pytz.utc
headers = {"Authorization": f"Bearer {API_KEY}"}

ALERT_STATE_FILE = "last_alert_state.json"
ALERT_DATE_FILE = "last_alert_date.txt"

# How many candles to display in the table
DISPLAY_ROWS = 13

# Number of trading days to use for averaging
TRADING_DAYS_FOR_AVERAGE = 21

# ====== ALERT MEMORY ======
def load_alerted_candles():
    if os.path.exists(ALERT_STATE_FILE):
        try:
            with open(ALERT_STATE_FILE, "r") as f:
                return set(json.load(f))
        except:
            return set()
    return set()

def save_alerted_candles(alerted_set):
    with open(ALERT_STATE_FILE, "w") as f:
        json.dump(list(alerted_set), f)

def reset_if_new_day():
    today = datetime.now(IST).date().isoformat()
    if os.path.exists(ALERT_DATE_FILE):
        with open(ALERT_DATE_FILE, "r") as f:
            last = f.read().strip()
        if last != today:
            with open(ALERT_STATE_FILE, "w") as f:
                f.write("[]")
    with open(ALERT_DATE_FILE, "w") as f:
        f.write(today)

# ====== SIDEBAR CONFIG ======
st.sidebar.title("🔧 Settings")

if "selected_instruments" not in st.session_state:
    st.session_state.selected_instruments = list(INSTRUMENTS.keys())
if "refresh_minutes" not in st.session_state:
    st.session_state.refresh_minutes = 5
if "bucket_choice" not in st.session_state:
    st.session_state.bucket_choice = "1 hour"
if "enable_telegram_alerts" not in st.session_state:
    st.session_state.enable_telegram_alerts = False
if "candle_size" not in st.session_state:
    st.session_state.candle_size = "15 min"  # NEW: 15 min or 4 hour

# Optional: provide Telegram secrets via Streamlit Cloud
TELEGRAM_BOT_TOKEN = st.secrets.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = st.secrets.get("TELEGRAM_CHAT_ID", "")

st.sidebar.multiselect(
    "Select Instruments to Monitor",
    options=list(INSTRUMENTS.keys()),
    default=st.session_state.selected_instruments,
    key="selected_instruments"
)

st.sidebar.slider(
    "Auto-refresh interval (minutes)",
    min_value=1, max_value=15,
    value=st.session_state.refresh_minutes,
    key="refresh_minutes"
)

# NEW: Candle Size toggle
st.sidebar.radio(
    "📏 Candle Size",
    ["15 min", "4 hour"],
    index=["15 min", "4 hour"].index(st.session_state.candle_size),
    key="candle_size"
)

# Show bucket selection only for 15m mode. In 4h mode bucket is position-based.
if st.session_state.candle_size == "15 min":
    st.sidebar.radio(
        "🕒 Select Time Bucket",
        ["15 min", "30 min", "1 hour"],
        index=["15 min", "30 min", "1 hour"].index(st.session_state.bucket_choice),
        key="bucket_choice"
    )
else:
    st.sidebar.caption("🕒 Comparison: By candle position (1st-6th of day)")

st.sidebar.toggle(
    "Enable Telegram Alerts",
    value=st.session_state.enable_telegram_alerts,
    key="enable_telegram_alerts"
)

st.sidebar.slider(
    "📈 Threshold Multiplier",
    min_value=1.0,
    max_value=3.0,
    step=0.1,
    value=1.618,
    key="threshold_multiplier"
)

# ====== AUTO-REFRESH ======
refresh_ms = st.session_state.refresh_minutes * 60 * 1000
refresh_count = st_autorefresh(interval=refresh_ms, limit=None, key="volume-refresh")

# ====== TELEGRAM ALERT ======
def send_telegram_alert(message):
    if not st.session_state.enable_telegram_alerts:
        return
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        st.warning("Telegram is ON but secrets missing. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        resp = requests.post(url, data=payload, timeout=10)
        if resp.status_code != 200:
            st.error(f"Telegram alert failed: {resp.text}")
    except Exception as e:
        st.error(f"Telegram alert exception: {e}")

# ====== OANDA DATA FETCH ======
@st.cache_resource
def get_session():
    s = requests.Session()
    s.headers.update(headers)
    return s

@st.cache_data(ttl=600, show_spinner=False)
def fetch_candles(instrument_code, from_time, to_time, granularity="M15"):
    now_utc = datetime.now(UTC)
    from_time = min(from_time, now_utc)
    to_time = min(to_time, now_utc)

    params = {
        "granularity": granularity,
        "price": "M",
        "from": from_time.isoformat(),
        "to": to_time.isoformat()
    }
    url = f"{BASE_URL}/accounts/{ACCOUNT_ID}/instruments/{instrument_code}/candles"
    try:
        s = get_session()
        resp = s.get(url, params=params, timeout=20)
    except Exception as e:
        st.error(f"❌ Network error for {instrument_code}: {e}")
        return []

    if resp.status_code != 200:
        st.error(f"❌ Failed to fetch {instrument_code} data: {resp.text}")
        return []
    return resp.json().get("candles", [])

# ====== UTILITIES ======
def get_time_bucket(dt_ist, bucket_size_minutes):
    """Calculate time bucket for 15-minute mode"""
    bucket_start_minute = (dt_ist.minute // bucket_size_minutes) * bucket_size_minutes
    bucket_start = dt_ist.replace(minute=bucket_start_minute, second=0, microsecond=0)
    bucket_end = bucket_start + timedelta(minutes=bucket_size_minutes)
    return f"{bucket_start.strftime('%I:%M %p')}–{bucket_end.strftime('%I:%M %p')}"

def get_4h_time_range(dt_ist):
    """Get the actual 4-hour time range starting from the candle's opening time"""
    # The candle's opening time is dt_ist
    # Add 4 hours to get the end time (or 3 hours for shortened candles)
    end_time = dt_ist + timedelta(hours=4)
    return f"{dt_ist.strftime('%I:%M %p')}–{end_time.strftime('%I:%M %p')}"

def get_candle_position_in_day(dt_ist):
    """Get the position of a 4H candle in the day (1st, 2nd, 3rd, etc.)"""
    # Get start of day in IST
    day_start = dt_ist.replace(hour=0, minute=0, second=0, microsecond=0)
    # Calculate hours since day start
    hours_since_start = (dt_ist - day_start).total_seconds() / 3600
    # Each 4H candle position (0-based, then add 1 for 1-based)
    position = int(hours_since_start // 4) + 1
    return f"Candle #{position}"

def format_bucket_label(minutes):
    if minutes == 240:  # 4-hour mode
        return "4 hour"
    elif minutes % 60 == 0:
        h = minutes // 60
        return f"{h} hour" if h == 1 else f"{h} hours"
    return f"{minutes} min"

def is_weekend(date):
    """Check if a date is Saturday (5) or Sunday (6)"""
    return date.weekday() in [5, 6]

@st.cache_data(ttl=600)
def compute_bucket_averages(code, bucket_size_minutes, granularity):
def compute_bucket_averages(code, bucket_size_minutes, granularity, skip_weekends=True):
    """Compute averages for comparison"""
    if granularity == "H4":
        # For 4H mode, use position-based averaging
        return compute_4h_position_averages(code)
        return compute_4h_position_averages(code, skip_weekends)
    else:
        # For 15-minute mode, use time bucket averaging
        return compute_15m_bucket_averages(code, bucket_size_minutes)
        return compute_15m_bucket_averages(code, bucket_size_minutes, skip_weekends)

def compute_15m_bucket_averages(code, bucket_size_minutes):
    """Time-bucket based averaging for 15-minute mode, using last 21 full *past* days"""
def compute_15m_bucket_averages(code, bucket_size_minutes, skip_weekends=True):
    """Time-bucket based averaging for 15-minute mode, collecting last N trading days"""
    bucket_volumes = defaultdict(list)
    today_ist = datetime.now(IST).date()
    now_utc = datetime.now(UTC)

    # ✅ Start from yesterday, go back 21 days
    for i in range(1, 22):  # 1 = yesterday, 21 = 21 days ago
        day_ist = today_ist - timedelta(days=i)
    trading_days_collected = 0
    days_back = 1
    max_lookback = 60  # Safety limit to prevent infinite loop
    
    while trading_days_collected < TRADING_DAYS_FOR_AVERAGE and days_back < max_lookback:
        day_ist = today_ist - timedelta(days=days_back)
        
        # Skip weekends if enabled
        if skip_weekends and is_weekend(day_ist):
            days_back += 1
            continue
            
        start_ist = IST.localize(datetime.combine(day_ist, time(0, 0)))
        end_ist = IST.localize(datetime.combine(day_ist + timedelta(days=1), time(0, 0)))

        start_utc = start_ist.astimezone(UTC)
        end_utc = min(end_ist.astimezone(UTC), now_utc)

        candles = fetch_candles(code, start_utc, end_utc, granularity="M15")
        for c in candles:
            if not c.get("complete", True):
                continue  # ⛔ skip the last forming 15m of any past day (rare, but safe)
            try:
                t_utc = datetime.strptime(c["time"], "%Y-%m-%dT%H:%M:%S.%f000Z")
            except ValueError:
                t_utc = datetime.strptime(c["time"], "%Y-%m-%dT%H:%M:%S.000Z")
            t_ist = t_utc.replace(tzinfo=UTC).astimezone(IST)
            bucket = get_time_bucket(t_ist, bucket_size_minutes)
            bucket_volumes[bucket].append(c["volume"])
        
        # Only count as a trading day if we got candles
        if candles:
            trading_days_collected += 1
            
            for c in candles:
                if not c.get("complete", True):
                    continue  # Skip incomplete candles
                try:
                    t_utc = datetime.strptime(c["time"], "%Y-%m-%dT%H:%M:%S.%f000Z")
                except ValueError:
                    t_utc = datetime.strptime(c["time"], "%Y-%m-%dT%H:%M:%S.000Z")
                t_ist = t_utc.replace(tzinfo=UTC).astimezone(IST)
                bucket = get_time_bucket(t_ist, bucket_size_minutes)
                bucket_volumes[bucket].append(c["volume"])
        
        days_back += 1

    # Return simple averages
    return {b: (sum(vs) / len(vs)) for b, vs in bucket_volumes.items() if vs}

def compute_4h_position_averages(code):
    """Position-based averaging for 4H mode, excluding today's candles"""
def compute_4h_position_averages(code, skip_weekends=True):
    """Position-based averaging for 4H mode, collecting last N trading days"""
    position_volumes = defaultdict(list)
    today_ist = datetime.now(IST).date()
    now_utc = datetime.now(UTC)

    for i in range(1, 22):  # start from yesterday
        day_ist = today_ist - timedelta(days=i)
        start_ist = IST.localize(datetime.combine(day_ist, time(0, 0)))
        end_ist = IST.localize(datetime.combine(day_ist + timedelta(days=1), time(0, 0)))
        
        start_utc = start_ist.astimezone(UTC)
        end_utc = min(end_ist.astimezone(UTC), now_utc)
        
        candles = fetch_candles(code, start_utc, end_utc, granularity="H4")
        for c in candles:
            if not c.get("complete", True):
                continue
            try:
                t_utc = datetime.strptime(c["time"], "%Y-%m-%dT%H:%M:%S.%f000Z")
            except ValueError:
                t_utc = datetime.strptime(c["time"], "%Y-%m-%dT%H:%M:%S.000Z")
            t_ist = t_utc.replace(tzinfo=UTC).astimezone(IST)
            time_range = get_4h_time_range(t_ist)
            position_volumes[time_range].append(c["volume"])
    
    return {p: (sum(vs) / len(vs)) for p, vs in position_volumes.items() if vs}

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

# ====== CORE PROCESS ======
def process_instrument(name, code, bucket_size_minutes, granularity, alerted_candles):
    bucket_avg = compute_bucket_averages(code, bucket_size_minutes, granularity)
    now_utc = datetime.now(UTC)
    is_4h_mode = (granularity == "H4")
    
    per_candle_minutes = 15 if granularity == "M15" else 240
    # Fetch more history to fill the table comfortably
    candles_needed = 40 if granularity == "M15" else 26
    from_time = now_utc - timedelta(minutes=per_candle_minutes * candles_needed)
    
    candles = fetch_candles(code, from_time, now_utc, granularity=granularity)
    if not candles:
        return [], [], {}
    
    rows = []
    spikes_last_two = []
    last_two_candles = candles[-2:] if len(candles) >= 2 else candles
    last_summary = {}
    
    for c in candles:
        try:
            t_utc = datetime.strptime(c["time"], "%Y-%m-%dT%H:%M:%S.%f000Z")
        except ValueError:
            t_utc = datetime.strptime(c["time"], "%Y-%m-%dT%H:%M:%S.000Z")
        t_ist = t_utc.replace(tzinfo=UTC).astimezone(IST)
        
        # Get bucket/time range based on mode
        if is_4h_mode:
            # For 4H mode, use the actual time range as the bucket
            bucket = get_4h_time_range(t_ist)
            display_bucket = bucket  # Display the same time range
        else:
            # For 15-minute mode, use time bucket
            bucket = get_time_bucket(t_ist, bucket_size_minutes)
            display_bucket = bucket
        
        vol = c["volume"]
        avg = bucket_avg.get(bucket, 0)
        threshold_multiplier = st.session_state.threshold_multiplier
        threshold = avg * threshold_multiplier if avg else 0
        over = (threshold > 0 and vol > threshold)
        mult = (vol / threshold) if over and threshold > 0 else (vol / avg if avg else 0)
        
        spike_diff = f"▲{vol - int(threshold)}" if over else ""
        sentiment = get_sentiment(c)
        
        rows.append([
            t_ist.strftime("%Y-%m-%d %I:%M %p"),
            display_bucket,
            f"{float(c['mid']['o']):.1f}",
            f"{float(c['mid']['h']):.1f}",
            f"{float(c['mid']['l']):.1f}",
            f"{float(c['mid']['c']):.1f}",
            vol,
            spike_diff,
            sentiment
        ])
        
        last_summary = {
            "time": t_ist.strftime("%Y-%m-%d %I:%M %p"),
            "bucket": bucket,
            "volume": vol,
            "avg": avg,
            "threshold": threshold,
            "multiplier": mult,
            "over": over,
            "sentiment": sentiment
        }
        
        if c in last_two_candles and over:
            candle_id = f"{name}_{c['time']}_{round(float(c['mid']['o']), 2)}"
            if candle_id not in alerted_candles:
                spikes_last_two.append(
                    f"{name} {t_ist.strftime('%I:%M %p')} — Vol {vol} ({spike_diff}) {sentiment}"
                )
                alerted_candles.add(candle_id)
    
    return rows, spikes_last_two, last_summary

# ====== TABLE RENDERING ======
def render_card(name, rows, bucket_minutes, summary, is_4h_mode=False):
    st.markdown(f"### {name}", help="Instrument")
    
    if is_4h_mode:
        bucket_lbl = "Time Range"
        comparison_label = "4 Hour Mode"
    else:
        bucket_lbl = format_bucket_label(bucket_minutes)
        comparison_label = f"Bucket: {bucket_lbl}"
    
    if summary:
        chips = [
            f'<span class="badge neutral">{comparison_label}</span>',
            f'<span class="badge neutral">Last: {summary["time"]}</span>',
            f'<span class="badge {"ok" if summary["over"] else "neutral"}">Spike: {"Yes" if summary["over"] else "No"}</span>',
        ]
        st.markdown(f'<div class="badges">{" ".join(chips)}</div>', unsafe_allow_html=True)
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Volume", f"{summary['volume']:,}")
        c2.metric(f"{'Range' if is_4h_mode else 'Bucket'} Avg", f"{summary['avg']:.0f}")
        c3.metric("Threshold", f"{summary['threshold']:.0f}")
        c4.metric("Multiplier", f"{summary['multiplier']:.2f}")
    
    if is_4h_mode:
        columns = [
            "Time (IST)",
            "Time Range (4H)",
            "Open", "High", "Low", "Close",
            "Volume", "Spike Δ", "Sentiment"
        ]
    else:
        columns = [
            "Time (IST)",
            f"Time Bucket ({bucket_lbl})",
            "Open", "High", "Low", "Close",
            "Volume", "Spike Δ", "Sentiment"
        ]
    
    trimmed_rows = rows[-DISPLAY_ROWS:] if len(rows) > DISPLAY_ROWS else rows
    df = pd.DataFrame(trimmed_rows, columns=columns)
    
    # <-- replaced deprecated use_container_width with width="stretch"
    st.dataframe(
        df,
        width="stretch",
        hide_index=True,
        height=520,
        column_config={
            "Open": st.column_config.NumberColumn(format="%.1f"),
            "High": st.column_config.NumberColumn(format="%.1f"),
            "Low": st.column_config.NumberColumn(format="%.1f"),
            "Close": st.column_config.NumberColumn(format="%.1f"),
            "Volume": st.column_config.NumberColumn(format="%.0f"),
            "Spike Δ": st.column_config.TextColumn(),
            "Sentiment": st.column_config.TextColumn(help="🟩 up, 🟥 down, ▪️ flat"),
        },
    )
    
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Export to CSV",
        data=csv,
        file_name=f"{name}_volume_spikes.csv",
        mime="text/csv",
        width="stretch",
    )

# ====== DASHBOARD EXECUTION ======
def run_volume_check():
    reset_if_new_day()
    alerted_candles = load_alerted_candles()
    all_spike_msgs = []
    
    if not st.session_state.selected_instruments:
        st.warning("⚠️ No instruments selected. Please choose at least one.")
        return
    
    # NEW: derive granularity + bucket from Candle Size toggle
    if st.session_state.candle_size == "4 hour":
        granularity = "H4"
        bucket_minutes = 240
        is_4h_mode = True
    else:
        granularity = "M15"
        bucket_minutes = {"15 min": 15, "30 min": 30, "1 hour": 60}[st.session_state.bucket_choice]
        is_4h_mode = False
    
    top_l, top_r = st.columns([3, 2])
    with top_l:
        st.subheader("📊 Volume Anomaly Detector")
        now_ist = datetime.now(IST).strftime("%Y-%m-%d %I:%M %p")
        tele_status = "ON" if st.session_state.enable_telegram_alerts else "OFF"
        comparison_type = "Time Range" if is_4h_mode else f"Bucket: {bucket_minutes}m"
        st.markdown(
            f'<div class="badges">'
            f'<span class="badge">IST: {now_ist}</span>'
            f'<span class="badge neutral">Candle: {"4h" if granularity=="H4" else "15m"}</span>'
            f'<span class="badge neutral">{comparison_type}</span>'
            f'<span class="badge neutral">Threshold × {st.session_state.threshold_multiplier:.2f}</span>'
            f'<span class="badge neutral">Auto-refresh: {st.session_state.refresh_minutes}m</span>'
            f'<span class="badge {"ok" if tele_status=="ON" else "warn"}">Telegram: {tele_status}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with top_r:
        st.info("Tip: Turn on Telegram alerts in the sidebar to receive spike notifications.")
    
    names = st.session_state.selected_instruments
    cols = st.columns(2) if len(names) > 1 else [st.container()]
    col_idx = 0
    
    all_rows_have_data = False
    trading_days_collected = 0
    days_back = 1
    max_lookback = 60  # Safety limit

    for name in names:
        code = INSTRUMENTS[name]
        with cols[col_idx]:
            with st.container(border=True):
                rows, spikes, summary = process_instrument(name, code, bucket_minutes, granularity, alerted_candles)
                if rows:
                    all_rows_have_data = True
                    render_card(name, rows, bucket_minutes, summary, is_4h_mode)
                else:
                    st.warning(f"No recent data for {name}")
    while trading_days_collected < TRADING_DAYS_FOR_AVERAGE and days_back < max_lookback:
        day_ist = today_ist - timedelta(days=days_back)

        col_idx = (col_idx + 1) % len(cols)
        
        if spikes:
            all_spike_msgs.extend(spikes)
    
    if all_spike_msgs:
        formatted_msgs = []
        for raw in all_spike_msgs:
            try:
                parts = raw.split(" — Vol ")
                instrument_time = parts[0].split()
                instrument = instrument_time[0]
                time_str = " ".join(instrument_time[1:])
                vol_part = parts[1]
                vol_val = vol_part.split(" ")[0]
                spike_delta = vol_part.split("(")[-1].split(")")[0]
                sentiment = vol_part.split()[-1]
                formatted_msgs.append(
                    f"🔍 Instrument: {instrument}\n"
                    f"🕒 Time: {time_str}\n"
                    f"📊 Volume: {vol_val} {spike_delta}\n"
                    f"📈 Sentiment: {sentiment}"
                )
            except:
                formatted_msgs.append(raw)
        
        comparison_label = "4H time range" if is_4h_mode else f"{format_bucket_label(bucket_minutes)} bucket"
        alert_msg = f"⚡ Volume Spike Alert — {comparison_label}\n\n" + "\n\n".join(formatted_msgs)
        print(alert_msg)
        send_telegram_alert(alert_msg)
    else:
        if all_rows_have_data:
            st.info("ℹ️ No spikes in the last two candles.")
    
    save_alerted_candles(alerted_candles)

# ====== MAIN ======
run_volume_check()
        # Skip weekends if enabled
        if skip_weekends and is_weekend(day_ist):
            days_back += 1
            continue
            
        start_ist = IST.localize(datetime.combine(day_ist, time(0, 0)))
        end_ist = IST.localize
