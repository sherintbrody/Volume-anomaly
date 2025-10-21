
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

ALERT_STATE_FILE = "last_alert_state.json"
ALERT_DATE_FILE = "last_alert_date.txt"

DISPLAY_ROWS = 13
TRADING_DAYS_FOR_AVERAGE = 21

# Skip weekends is always ON
SKIP_WEEKENDS = True

TELEGRAM_BOT_TOKEN = st.secrets.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = st.secrets.get("TELEGRAM_CHAT_ID", "")

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

# Initialize session state
if "selected_instruments" not in st.session_state:
    st.session_state.selected_instruments = list(INSTRUMENTS.keys())
if "refresh_minutes" not in st.session_state:
    st.session_state.refresh_minutes = 5
if "bucket_choice" not in st.session_state:
    st.session_state.bucket_choice = "15 min"
if "enable_telegram_alerts" not in st.session_state:
    st.session_state.enable_telegram_alerts = False
if "candle_size" not in st.session_state:
    st.session_state.candle_size = "15 min"
if "alert_multiplier" not in st.session_state:
    st.session_state.alert_multiplier = 2.0

# Instrument Selection
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

st.sidebar.markdown("---")

st.sidebar.toggle(
    "Enable Telegram Alerts",
    value=st.session_state.enable_telegram_alerts,
    key="enable_telegram_alerts"
)

st.sidebar.slider(
    "📢 Alert Multiplier (Volume/Avg)",
    min_value=1.0,
    max_value=5.0,
    step=0.1,
    value=st.session_state.alert_multiplier,
    key="alert_multiplier",
    help="Send alert when Volume ÷ Average ≥ this multiplier"
)

# ====== AUTO-REFRESH ======
refresh_ms = st.session_state.refresh_minutes * 60 * 1000
refresh_count = st_autorefresh(interval=refresh_ms, limit=None, key="volume-refresh")

# ====== TELEGRAM ALERT ======
def send_telegram_alert(message):
    """Send Telegram alert with improved error handling"""
    if not st.session_state.enable_telegram_alerts:
        return False
    
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        st.warning("⚠️ Telegram is ON but secrets missing. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.")
        return False
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True
    }
    
    try:
        resp = requests.post(url, data=payload, timeout=10)
        if resp.status_code == 200:
            return True
        else:
            st.error(f"❌ Telegram alert failed: {resp.text}")
            return False
    except Exception as e:
        st.error(f"❌ Telegram exception: {e}")
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
    end_time = dt_ist + timedelta(hours=4)
    return f"{dt_ist.strftime('%I:%M %p')}–{end_time.strftime('%I:%M %p')}"

def get_candle_position_in_day(dt_ist):
    """Get the position of a 4H candle in the day (1st, 2nd, 3rd, etc.)"""
    day_start = dt_ist.replace(hour=0, minute=0, second=0, microsecond=0)
    hours_since_start = (dt_ist - day_start).total_seconds() / 3600
    position = int(hours_since_start // 4) + 1
    return f"Candle #{position}"

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
    return "🟩" if c > o else "🟥" if c < o else "▪️"

def get_body_percentage(candle):
    """Calculate the body percentage of a candle"""
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
        return "—"

def pad_display(s, width):
    pad_len = width - sum(wcwidth.wcwidth(ch) for ch in s)
    return s + " " * max(pad_len, 0)

def get_spike_bar(multiplier):
    if multiplier < 1.2:
        return pad_display("", 5)
    bars = int((multiplier - 1.2) * 5)
    bar_str = "┃" * max(1, min(bars, 5))
    return pad_display(bar_str, 5)

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
    """Process instrument and detect volume spikes - ALERT BASED ON MULTIPLIER"""
    bucket_avg = compute_bucket_averages(code, bucket_size_minutes, granularity, skip_weekends=SKIP_WEEKENDS)
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
    last_summary = {}
    
    # Get indices of last two COMPLETE candles
    complete_candle_indices = [i for i, c in enumerate(candles) if c.get("complete", True)]
    last_two_indices = set(complete_candle_indices[-2:]) if len(complete_candle_indices) >= 2 else set(complete_candle_indices)
    
    for idx, c in enumerate(candles):
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
        
        # Calculate multiplier
        mult = (vol / avg) if avg > 0 else 0
        
        # Alert trigger (based on alert_multiplier)
        alert_trigger = mult >= st.session_state.alert_multiplier
        
        # Visual spike indication (using alert_multiplier as threshold)
        threshold = avg * st.session_state.alert_multiplier if avg else 0
        visual_over = alert_trigger
        
        spike_diff = f"▲{vol - int(threshold)}" if visual_over else ""
        sentiment = get_sentiment(c)
        
        # Calculate body percentage for both modes
        body_pct = get_body_percentage(c)
        
        # Build row based on mode - now both include body_pct
        rows.append([
            t_ist.strftime("%Y-%m-%d %I:%M %p"),
            display_bucket,
            f"{float(c['mid']['o']):.1f}",
            vol,
            spike_diff,
            sentiment,
            body_pct
        ])
        
        last_summary = {
            "time": t_ist.strftime("%Y-%m-%d %I:%M %p"),
            "bucket": bucket,
            "volume": vol,
            "avg": avg,
            "threshold": threshold,
            "multiplier": mult,
            "over": visual_over,
            "sentiment": sentiment
        }
        
        # Alert based on multiplier (vol/avg >= alert_multiplier)
        if idx in last_two_indices and alert_trigger and c.get("complete", True):
            candle_id = f"{name}_{c['time']}"
            if candle_id not in alerted_candles:
                spike_msg = (
                    f"*{name}* @ {t_ist.strftime('%I:%M %p')}\n"
                    f"📊 Volume: `{vol:,}` (Avg: `{int(avg):,}`)\n"
                    f"📈 Multiplier: `{mult:.2f}x` (Alert Trigger: `≥{st.session_state.alert_multiplier:.2f}x`)\n"
                    f"💹 Sentiment: {sentiment}"
                )
                spikes_last_two.append(spike_msg)
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
        c4.metric("Multiplier", f"{summary['multiplier']:.2f}x")
    
    # Both modes now have the same columns including Body %
    if is_4h_mode:
        columns = [
            "Time (IST)",
            "Time Range (4H)",
            "Open",
            "Volume", 
            "Spike Δ", 
            "Sentiment", 
            "Body %"
        ]
    else:
        columns = [
            "Time (IST)",
            f"Time Bucket ({bucket_lbl})",
            "Open",
            "Volume", 
            "Spike Δ", 
            "Sentiment",
            "Body %"
        ]
    
    trimmed_rows = rows[-DISPLAY_ROWS:] if len(rows) > DISPLAY_ROWS else rows
    df = pd.DataFrame(trimmed_rows, columns=columns)
    
    # Both modes now include Body % in column_config
    column_config = {
        "Open": st.column_config.NumberColumn(format="%.1f"),
        "Volume": st.column_config.NumberColumn(format="%.0f"),
        "Spike Δ": st.column_config.TextColumn(),
        "Sentiment": st.column_config.TextColumn(help="🟩 up, 🟥 down, ▪️ flat"),
        "Body %": st.column_config.TextColumn(
            help="Body as % of total range. Higher % = stronger directional move, Lower % = indecision/doji"
        )
    }
    
    st.dataframe(
        df,
        width="stretch",
        hide_index=True,
        height=520,
        column_config=column_config,
    )
    
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Export to CSV",
        data=csv,
        file_name=f"{name}_volume_spikes.csv",
        mime="text/csv"
    )

# ====== DASHBOARD EXECUTION ======
def run_volume_check():
    reset_if_new_day()
    alerted_candles = load_alerted_candles()
    all_spike_msgs = []
    
    if not st.session_state.selected_instruments:
        st.warning("⚠️ No instruments selected. Please choose at least one.")
        return
    
    if st.session_state.candle_size == "4 hour":
        granularity = "H4"
        bucket_minutes = 240
        is_4h_mode = True
    else:
        granularity = "M15"
        bucket_minutes = {"15 min": 15, "30 min": 30, "1 hour": 60}[st.session_state.bucket_choice]
        is_4h_mode = False
    
    top_l, top_r = st.columns([3, 1])
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
            f'<span class="badge neutral">Alert Trigger ≥ {st.session_state.alert_multiplier:.2f}x</span>'
            f'<span class="badge neutral">Auto-refresh: {st.session_state.refresh_minutes}m</span>'
            f'<span class="badge {"ok" if tele_status=="ON" else "warn"}">Telegram: {tele_status}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with top_r:
        if st.button("🔄 Refresh Now", use_container_width=True, type="primary"):
            st.cache_data.clear()
            st.rerun()
    
    
    
    names = st.session_state.selected_instruments
    cols = st.columns(2) if len(names) > 1 else [st.container()]
    col_idx = 0
    
    all_rows_have_data = False
    
    for name in names:
        code = INSTRUMENTS.get(name)
        if not code:
            st.error(f"❌ Instrument code not found for {name}")
            continue
            
        with cols[col_idx]:
            with st.container(border=True):
                rows, spikes, summary = process_instrument(name, code, bucket_minutes, granularity, alerted_candles)
                if rows:
                    all_rows_have_data = True
                    render_card(name, rows, bucket_minutes, summary, is_4h_mode)
                else:
                    st.warning(f"No recent data for {name}")
        
        col_idx = (col_idx + 1) % len(cols)
        
        if spikes:
            all_spike_msgs.extend(spikes)
    
    # Send Telegram alerts
    if all_spike_msgs:
        comparison_label = "4H time range" if is_4h_mode else f"{format_bucket_label(bucket_minutes)} bucket"
        alert_msg = f"⚡ *Volume Spike Alert* — {comparison_label}\n\n" + "\n\n".join(all_spike_msgs)
        
        success = send_telegram_alert(alert_msg)
        
        if success:
            st.success(f"✅ Telegram alert sent! {len(all_spike_msgs)} spike(s) detected.")
        elif st.session_state.enable_telegram_alerts:
            st.warning("⚠️ Telegram alert failed to send. Check logs above.")
        
        print("="*50)
        print("VOLUME SPIKE DETECTED:")
        print(alert_msg)
        print("="*50)
    else:
        if all_rows_have_data:
            st.info("ℹ️ No spikes in the last two candles.")
    
    save_alerted_candles(alerted_candles)

# ====== MAIN ======
run_volume_check()

