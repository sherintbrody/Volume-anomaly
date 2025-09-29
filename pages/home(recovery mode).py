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
    display:inline-block;
    padding:4px 10px;
    border-radius:999px;
    font-size:12px;
    background: var(--chip-bg);
    color: var(--chip-fg);
    margin-right:6px;
    border:1px solid var(--chip-br)
}
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
# OANDA Configuration
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

# Initialize all session state variables
if "selected_instruments" not in st.session_state:
    st.session_state.selected_instruments = list(INSTRUMENTS.keys())
if "refresh_minutes" not in st.session_state:
    st.session_state.refresh_minutes = 5
if "bucket_choice" not in st.session_state:
    st.session_state.bucket_choice = "1 hour"
if "enable_telegram_alerts" not in st.session_state:
    st.session_state.enable_telegram_alerts = False
if "candle_size" not in st.session_state:
    st.session_state.candle_size = "15 min"
if "skip_weekends" not in st.session_state:
    st.session_state.skip_weekends = True
if "telegram_bot_token" not in st.session_state:
    st.session_state.telegram_bot_token = ""
if "telegram_chat_id" not in st.session_state:
    st.session_state.telegram_chat_id = ""

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

st.sidebar.radio(
    "📏 Candle Size",
    ["15 min", "4 hour"],
    index=["15 min", "4 hour"].index(st.session_state.candle_size),
    key="candle_size"
)

if st.session_state.candle_size == "15 min":
    st.sidebar.radio(
        "🕒 Select Time Bucket",
        ["15 min", "30 min", "1 hour"],
        index=["15 min", "30 min", "1 hour"].index(st.session_state.bucket_choice),
        key="bucket_choice"
    )
else:
    st.sidebar.caption("🕒 Comparison: By candle position (1st-6th of day)")

# Telegram Settings Section
st.sidebar.markdown("### 📱 Telegram Alerts")

st.sidebar.toggle(
    "Enable Telegram Alerts",
    value=st.session_state.enable_telegram_alerts,
    key="enable_telegram_alerts"
)

if st.session_state.enable_telegram_alerts:
    st.sidebar.markdown("**Enter Telegram Credentials:**")
    
    bot_token = st.sidebar.text_input(
        "Bot Token", 
        value=st.session_state.telegram_bot_token,
        type="password",
        help="Get from @BotFather on Telegram"
    )
    chat_id = st.sidebar.text_input(
        "Chat ID", 
        value=st.session_state.telegram_chat_id,
        help="Get from @userinfobot"
    )
    
    # Update session state
    st.session_state.telegram_bot_token = bot_token
    st.session_state.telegram_chat_id = chat_id
    
    if st.sidebar.button("🧪 Send Test Alert"):
        if bot_token and chat_id:
            test_msg = (
                "*Test Alert Successful!*\n\n"
                f"Time: {datetime.now(IST).strftime('%Y-%m-%d %I:%M %p')} IST\n"
                f"Dashboard: Volume Spike Monitor\n"
                f"Status: Telegram integration working"
            )
            success = send_telegram_alert(test_msg, bot_token, chat_id)
            if success:
                st.sidebar.success("Test message sent!")
            else:
                st.sidebar.error("Test failed. Check credentials.")
        else:
            st.sidebar.warning("Please enter both Bot Token and Chat ID")

st.sidebar.slider(
    "📈 Threshold Multiplier",
    min_value=1.0,
    max_value=3.0,
    step=0.1,
    value=1.618,
    key="threshold_multiplier"
)

st.sidebar.toggle(
    "Skip Weekends in Average",
    value=st.session_state.skip_weekends,
    key="skip_weekends",
    help="When ON, uses only trading days (Mon-Fri) for volume averages"
)

# ====== AUTO-REFRESH ======
refresh_ms = st.session_state.refresh_minutes * 60 * 1000
refresh_count = st_autorefresh(interval=refresh_ms, limit=None, key="volume-refresh")

# ====== TELEGRAM ALERT FUNCTION ======
def send_telegram_alert(message, bot_token=None, chat_id=None):
    """Send Telegram alert"""
    
    # Use provided credentials or get from session state
    if not bot_token:
        bot_token = st.session_state.get("telegram_bot_token", "")
    if not chat_id:
        chat_id = st.session_state.get("telegram_chat_id", "")
    
    if not bot_token or not chat_id:
        return False
    
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }
    
    try:
        resp = requests.post(url, json=payload, timeout=10)
        return resp.status_code == 200
    except:
        return False

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
        st.error(f"Network error for {instrument_code}: {e}")
        return []
    
    if resp.status_code != 200:
        st.error(f"Failed to fetch {instrument_code} data: {resp.text}")
        return []
    return resp.json().get("candles", [])

# ====== UTILITIES ======
def get_time_bucket(dt_ist, bucket_size_minutes):
    """Calculate time bucket for 15-minute mode"""
    bucket_start_minute = (dt_ist.minute // bucket_size_minutes) * bucket_size_minutes
    bucket_start = dt_ist.replace(minute=bucket_start_minute, second=0, microsecond=0)
    bucket_end = bucket_start + timedelta(minutes=bucket_size_minutes)
    return f"{bucket_start.strftime('%I:%M %p')}-{bucket_end.strftime('%I:%M %p')}"

def get_4h_time_range(dt_ist):
    """Get the actual 4-hour time range starting from the candle's opening time"""
    end_time = dt_ist + timedelta(hours=4)
    return f"{dt_ist.strftime('%I:%M %p')}-{end_time.strftime('%I:%M %p')}"

def format_bucket_label(minutes):
    if minutes == 240:
        return "4 hour"
    elif minutes % 60 == 0:
        h = minutes // 60
        return f"{h} hour" if h == 1 else f"{h} hours"
    return f"{minutes} min"

def is_weekend(date):
    """Check if a date is Saturday (5) or Sunday (6)"""
    return date.weekday() in [5, 6]

def get_sentiment(candle):
    o = float(candle["mid"]["o"])
    c = float(candle["mid"]["c"])
    if c > o:
        return "UP"
    elif c < o:
        return "DOWN"
    else:
        return "FLAT"

def get_body_percentage(candle):
    """Calculate the body percentage of a candle (4H only)"""
    try:
        o = float(candle["mid"]["o"])
        h = float(candle["mid"]["h"])
        l = float(candle["mid"]["l"])
        c = float(candle["mid"]["c"])
        
        body = abs(c - o)
        total_range = h - l
        
        if total_range == 0:
            return "0.0%"
        
        body_pct = (body / total_range) * 100
        return f"{body_pct:.1f}%"
    except:
        return "-"

@st.cache_data(ttl=600)
def compute_bucket_averages(code, bucket_size_minutes, granularity, skip_weekends=True):
    """Compute averages for comparison"""
    if granularity == "H4":
        return compute_4h_position_averages(code, skip_weekends)
    else:
        return compute_15m_bucket_averages(code, bucket_size_minutes, skip_weekends)

def compute_15m_bucket_averages(code, bucket_size_minutes, skip_weekends=True):
    """Time-bucket based averaging for 15-minute mode"""
    bucket_volumes = defaultdict(list)
    today_ist = datetime.now(IST).date()
    now_utc = datetime.now(UTC)
    
    trading_days_collected = 0
    days_back = 1
    max_lookback = 60
    
    while trading_days_collected < TRADING_DAYS_FOR_AVERAGE and days_back < max_lookback:
        day_ist = today_ist - timedelta(days=days_back)
        
        if skip_weekends and is_weekend(day_ist):
            days_back += 1
            continue
            
        start_ist = IST.localize(datetime.combine(day_ist, time(0, 0)))
        end_ist = IST.localize(datetime.combine(day_ist + timedelta(days=1), time(0, 0)))
        
        start_utc = start_ist.astimezone(UTC)
        end_utc = min(end_ist.astimezone(UTC), now_utc)
        
        candles = fetch_candles(code, start_utc, end_utc, granularity="M15")
        
        if candles:
            trading_days_collected += 1
            
            for c in candles:
                if not c.get("complete", True):
                    continue
                try:
                    t_utc = datetime.strptime(c["time"], "%Y-%m-%dT%H:%M:%S.%f000Z")
                except ValueError:
                    t_utc = datetime.strptime(c["time"], "%Y-%m-%dT%H:%M:%S.000Z")
                t_ist = t_utc.replace(tzinfo=UTC).astimezone(IST)
                bucket = get_time_bucket(t_ist, bucket_size_minutes)
                bucket_volumes[bucket].append(c["volume"])
        
        days_back += 1
    
    return {b: (sum(vs) / len(vs)) for b, vs in bucket_volumes.items() if vs}

def compute_4h_position_averages(code, skip_weekends=True):
    """Position-based averaging for 4H mode"""
    position_volumes = defaultdict(list)
    today_ist = datetime.now(IST).date()
    now_utc = datetime.now(UTC)
    
    trading_days_collected = 0
    days_back = 1
    max_lookback = 60
    
    while trading_days_collected < TRADING_DAYS_FOR_AVERAGE and days_back < max_lookback:
        day_ist = today_ist - timedelta(days=days_back)
        
        if skip_weekends and is_weekend(day_ist):
            days_back += 1
            continue
            
        start_ist = IST.localize(datetime.combine(day_ist, time(0, 0)))
        end_ist = IST.localize(datetime.combine(day_ist + timedelta(days=1), time(0, 0)))
        
        start_utc = start_ist.astimezone(UTC)
        end_utc = min(end_ist.astimezone(UTC), now_utc)
        
        candles = fetch_candles(code, start_utc, end_utc, granularity="H4")
        
        if candles:
            trading_days_collected += 1
            
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
        
        days_back += 1
    
    return {p: (sum(vs) / len(vs)) for p, vs in position_volumes.items() if vs}

# ====== CORE PROCESS ======
def process_instrument(name, code, bucket_size_minutes, granularity, alerted_candles):
    bucket_avg = compute_bucket_averages(code, bucket_size_minutes, granularity, skip_weekends=st.session_state.skip_weekends)
    now_utc = datetime.now(UTC)
    is_4h_mode = (granularity == "H4")
    
    per_candle_minutes = 15 if granularity == "M15" else 240
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
        
        if is_4h_mode:
            bucket = get_4h_time_range(t_ist)
            display_bucket = bucket
        else:
            bucket = get_time_bucket(t_ist, bucket_size_minutes)
            display_bucket = bucket
        
        vol = c["volume"]
        avg = bucket_avg.get(bucket, 0)
        threshold_multiplier = st.session_state.threshold_multiplier
        threshold = avg * threshold_multiplier if avg else 0
        over = (threshold > 0 and vol > threshold)
        mult = (vol / threshold) if over and threshold > 0 else (vol / avg if avg else 0)
        
        spike_diff = f"+{vol - int(threshold)}" if over else ""
        sentiment = get_sentiment(c)
        
        if is_4h_mode:
            body_pct = get_body_percentage(c)
            rows.append([
                t_ist.strftime("%Y-%m-%d %I:%M %p"),
                display_bucket,
                f"{float(c['mid']['o']):.1f}",
                f"{float(c['mid']['h']):.1f}",
                f"{float(c['mid']['l']):.1f}",
                f"{float(c['mid']['c']):.1f}",
                vol,
                spike_diff,
                sentiment,
                body_pct
            ])
        else:
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
                    f"{name} {t_ist.strftime('%I:%M %p')} - Vol {vol} ({spike_diff}) {sentiment}"
                )
                alerted_candles.add(candle_id)
    
    return rows, spikes_last_two, last_summary

# ====== TABLE RENDERING ======
def render_card(name, rows, bucket_minutes, summary, is_4h_mode=False):
    st.markdown(f"### {name}")
    
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
            "Volume", "Spike", "Sentiment", "Body %"
        ]
    else:
        columns = [
            "Time (IST)",
            f"Time Bucket ({bucket_lbl})",
            "Open", "High", "Low", "Close",
            "Volume", "Spike", "Sentiment"
        ]
    
    trimmed_rows = rows[-DISPLAY_ROWS:] if len(rows) > DISPLAY_ROWS else rows
    df = pd.DataFrame(trimmed_rows, columns=columns)
    
    st.dataframe(df, use_container_width=True, hide_index=True, height=520)

# ====== DASHBOARD EXECUTION ======
def run_volume_check():
    reset_if_new_day()
    alerted_candles = load_alerted_candles()
    all_spike_msgs = []
    
    if not st.session_state.selected_instruments:
        st.warning("No instruments selected. Please choose at least one.")
        return
    
    if st.session_state.candle_size == "4 hour":
        granularity = "H4"
        bucket_minutes = 240
        is_4h_mode = True
    else:
        granularity = "M15"
        bucket_minutes = {"15 min": 15, "30 min": 30, "1 hour": 60}[st.session_state.bucket_choice]
        is_4h_mode = False
    
    st.subheader("Volume Anomaly Detector")
    
    names = st.session_state.selected_instruments
    cols = st.columns(2) if len(names) > 1 else [st.container()]
    col_idx = 0
    
    for name in names:
        code = INSTRUMENTS[name]
        with cols[col_idx]:
            with st.container(border=True):
                rows, spikes, summary = process_instrument(name, code, bucket_minutes, granularity, alerted_candles)
                if rows:
                    render_card(name, rows, bucket_minutes, summary, is_4h_mode)
                else:
                    st.warning(f"No recent data for {name}")
        
        col_idx = (col_idx + 1) % len(cols)
        
        if spikes:
            all_spike_msgs.extend(spikes)
    
    if all_spike_msgs and st.session_state.enable_telegram_alerts:
        alert_msg = "*Volume Spike Alert*\n\n" + "\n".join(all_spike_msgs)
        send_telegram_alert(alert_msg)
    
    save_alerted_candles(alerted_candles)

# ====== MAIN ======
run_volume_check()
