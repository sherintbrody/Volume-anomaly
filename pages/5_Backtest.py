import requests, json, os
import streamlit as st
from datetime import datetime, timedelta, time
import pytz
import pandas as pd
from collections import defaultdict
import wcwidth

# ====== PAGE CONFIG ======
st.set_page_config(
    page_title="Volume Spike Backtesting",
    page_icon="📈",
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

# How many candles to display in the table
DISPLAY_ROWS = 13

# Number of trading days to use for averaging
TRADING_DAYS_FOR_AVERAGE = 21

# Skip weekends is always ON
SKIP_WEEKENDS = True

# ====== SIDEBAR CONFIG ======
st.sidebar.title("🔧 Backtest Settings")

# Initialize all session state variables
if "selected_instruments" not in st.session_state:
    st.session_state.selected_instruments = list(INSTRUMENTS.keys())
if "bucket_choice" not in st.session_state:
    st.session_state.bucket_choice = "1 hour"
if "candle_size" not in st.session_state:
    st.session_state.candle_size = "15 min"
if "backtest_date" not in st.session_state:
    st.session_state.backtest_date = datetime.now(IST).date() - timedelta(days=1)
if "threshold_multiplier" not in st.session_state:
    st.session_state.threshold_multiplier = 2.0

# Date Picker for Backtesting
st.sidebar.date_input(
    "📅 Select Date to Backtest",
    value=st.session_state.backtest_date,
    max_value=datetime.now(IST).date(),
    key="backtest_date",
    help="Choose a date to analyze volume spikes using the previous 21 trading days"
)

st.sidebar.multiselect(
    "Select Instruments to Analyze",
    options=list(INSTRUMENTS.keys()),
    default=st.session_state.selected_instruments,
    key="selected_instruments"
)

st.sidebar.radio(
    "🕐 Candle Size",
    ["15 min", "2 hour", "4 hour"],
    index=["15 min", "2 hour", "4 hour"].index(st.session_state.candle_size),
    key="candle_size"
)

if st.session_state.candle_size == "15 min":
    st.sidebar.radio(
        "🕒 Select Time Bucket",
        ["15 min", "30 min", "1 hour"],
        index=["15 min", "30 min", "1 hour"].index(st.session_state.bucket_choice),
        key="bucket_choice"
    )
elif st.session_state.candle_size == "2 hour":
    st.sidebar.caption("🕒 Comparison: By candle position (1st-12th of day)")
else:  # 4 hour
    st.sidebar.caption("🕒 Comparison: By candle position (1st-6th of day)")

# THRESHOLD MULTIPLIER SLIDER
threshold_value = st.sidebar.slider(
    "📈 Threshold Multiplier",
    min_value=1.0,
    max_value=5.0,
    step=0.1,
    value=st.session_state.threshold_multiplier,
    key="threshold_multiplier",
    help="Spike detected when: Volume > (21-Day Avg × Threshold)"
)

# Run Backtest Button
run_backtest = st.sidebar.button("🔍 Run Backtest", type="primary", use_container_width=True)

# Clear cache button for debugging
if st.sidebar.button("🔄 Clear Cache & Rerun"):
    st.cache_data.clear()
    st.rerun()

# ====== OANDA DATA FETCH ======
@st.cache_resource
def get_session():
    s = requests.Session()
    s.headers.update(headers)
    return s

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_candles(instrument_code, from_time, to_time, granularity="M15"):
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

def get_2h_time_range(dt_ist):
    """Get the actual 2-hour time range starting from the candle's opening time"""
    end_time = dt_ist + timedelta(hours=2)
    return f"{dt_ist.strftime('%I:%M %p')}–{end_time.strftime('%I:%M %p')}"

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
    elif minutes == 120:
        return "2 hour"
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
    """Calculate the body percentage of a candle (2H/4H only)"""
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
        return "–"

def pad_display(s, width):
    pad_len = width - sum(wcwidth.wcwidth(ch) for ch in s)
    return s + " " * max(pad_len, 0)

def get_spike_bar(multiplier):
    if multiplier < 1.2:
        return pad_display("", 5)
    bars = int((multiplier - 1.2) * 5)
    bar_str = "▃" * max(1, min(bars, 5))
    return pad_display(bar_str, 5)

@st.cache_data(ttl=3600)
def compute_bucket_averages(code, bucket_size_minutes, granularity, selected_date):
    """Compute averages for comparison"""
    if granularity == "H4":
        return compute_4h_position_averages(code, selected_date)
    elif granularity == "H2":
        return compute_2h_position_averages(code, selected_date)
    else:
        return compute_15m_bucket_averages(code, bucket_size_minutes, selected_date)

def compute_15m_bucket_averages(code, bucket_size_minutes, selected_date):
    """Time-bucket based averaging for 15-minute mode"""
    bucket_volumes = defaultdict(list)
    
    trading_days_collected = 0
    days_back = 1
    max_lookback = 60
    
    while trading_days_collected < TRADING_DAYS_FOR_AVERAGE and days_back < max_lookback:
        day_ist = selected_date - timedelta(days=days_back)
        
        if is_weekend(day_ist):
            days_back += 1
            continue
            
        start_ist = IST.localize(datetime.combine(day_ist, time(0, 0)))
        end_ist = IST.localize(datetime.combine(day_ist + timedelta(days=1), time(0, 0)))
        
        start_utc = start_ist.astimezone(UTC)
        end_utc = end_ist.astimezone(UTC)
        
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

def compute_2h_position_averages(code, selected_date):
    """Position-based averaging for 2H mode"""
    position_volumes = defaultdict(list)
    
    trading_days_collected = 0
    days_back = 1
    max_lookback = 60
    
    while trading_days_collected < TRADING_DAYS_FOR_AVERAGE and days_back < max_lookback:
        day_ist = selected_date - timedelta(days=days_back)
        
        if is_weekend(day_ist):
            days_back += 1
            continue
            
        start_ist = IST.localize(datetime.combine(day_ist, time(0, 0)))
        end_ist = IST.localize(datetime.combine(day_ist + timedelta(days=1), time(0, 0)))
        
        start_utc = start_ist.astimezone(UTC)
        end_utc = end_ist.astimezone(UTC)
        
        candles = fetch_candles(code, start_utc, end_utc, granularity="H2")
        
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
                time_range = get_2h_time_range(t_ist)
                position_volumes[time_range].append(c["volume"])
        
        days_back += 1
    
    return {p: (sum(vs) / len(vs)) for p, vs in position_volumes.items() if vs}

def compute_4h_position_averages(code, selected_date):
    """Position-based averaging for 4H mode"""
    position_volumes = defaultdict(list)
    
    trading_days_collected = 0
    days_back = 1
    max_lookback = 60
    
    while trading_days_collected < TRADING_DAYS_FOR_AVERAGE and days_back < max_lookback:
        day_ist = selected_date - timedelta(days=days_back)
        
        if is_weekend(day_ist):
            days_back += 1
            continue
            
        start_ist = IST.localize(datetime.combine(day_ist, time(0, 0)))
        end_ist = IST.localize(datetime.combine(day_ist + timedelta(days=1), time(0, 0)))
        
        start_utc = start_ist.astimezone(UTC)
        end_utc = end_ist.astimezone(UTC)
        
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
def process_instrument(name, code, bucket_size_minutes, granularity, selected_date, threshold_multiplier):
    """Process instrument for the selected backtest date"""
    bucket_avg = compute_bucket_averages(code, bucket_size_minutes, granularity, selected_date)
    
    is_multi_hour_mode = (granularity in ["H2", "H4"])
    
    # Fetch data for the selected date
    start_ist = IST.localize(datetime.combine(selected_date, time(0, 0)))
    end_ist = IST.localize(datetime.combine(selected_date + timedelta(days=1), time(0, 0)))
    
    start_utc = start_ist.astimezone(UTC)
    end_utc = end_ist.astimezone(UTC)
    
    candles = fetch_candles(code, start_utc, end_utc, granularity=granularity)
    if not candles:
        return [], []
    
    rows = []
    spikes_found = []
    
    for c in candles:
        try:
            t_utc = datetime.strptime(c["time"], "%Y-%m-%dT%H:%M:%S.%f000Z")
        except ValueError:
            t_utc = datetime.strptime(c["time"], "%Y-%m-%dT%H:%M:%S.000Z")
        t_ist = t_utc.replace(tzinfo=UTC).astimezone(IST)
        
        if granularity == "H2":
            bucket = get_2h_time_range(t_ist)
            display_bucket = bucket
        elif granularity == "H4":
            bucket = get_4h_time_range(t_ist)
            display_bucket = bucket
        else:
            bucket = get_time_bucket(t_ist, bucket_size_minutes)
            display_bucket = bucket
        
        vol = c["volume"]
        avg = bucket_avg.get(bucket, 0)
        
        # Threshold = Average × threshold_multiplier
        threshold = avg * threshold_multiplier if avg else 0
        
        # Spike detection: Volume must be GREATER than threshold
        over = (threshold > 0 and vol > threshold)
        
        # Actual multiplier: Volume / Average
        actual_multiplier = (vol / avg) if avg > 0 else 0
        
        spike_diff = f"▲{vol - int(threshold)}" if over else ""
        sentiment = get_sentiment(c)
        
        # Build row based on mode
        if is_multi_hour_mode:
            body_pct = get_body_percentage(c)
            rows.append([
                t_ist.strftime("%Y-%m-%d %I:%M %p"),
                display_bucket,
                vol,
                int(avg) if avg > 0 else 0,
                int(threshold) if threshold > 0 else 0,
                f"{actual_multiplier:.2f}x",
                spike_diff,
                sentiment,
                body_pct
            ])
        else:
            rows.append([
                t_ist.strftime("%Y-%m-%d %I:%M %p"),
                display_bucket,
                vol,
                int(avg) if avg > 0 else 0,
                int(threshold) if threshold > 0 else 0,
                f"{actual_multiplier:.2f}x",
                spike_diff,
                sentiment
            ])
        
        # Collect spikes (not displayed separately anymore, but kept for potential future use)
        if over:
            spikes_found.append({
                "instrument": name,
                "time": t_ist.strftime('%I:%M %p'),
                "volume": vol,
                "avg": int(avg),
                "threshold": int(threshold),
                "spike_diff": spike_diff,
                "sentiment": sentiment,
                "actual_multiplier": actual_multiplier
            })
    
    return rows, spikes_found

# ====== TABLE RENDERING ======
def render_card(name, rows, bucket_minutes, granularity="M15"):
    st.markdown(f"### {name}", help="Instrument")
    
    is_multi_hour_mode = (granularity in ["H2", "H4"])
    
    if granularity == "H2":
        bucket_lbl = "Time Range"
        comparison_label = "2 Hour Mode"
    elif granularity == "H4":
        bucket_lbl = "Time Range"
        comparison_label = "4 Hour Mode"
    else:
        bucket_lbl = format_bucket_label(bucket_minutes)
        comparison_label = f"Bucket: {bucket_lbl}"
    
    # Simple badge display without metrics
    chips = [
        f'<span class="badge neutral">{comparison_label}</span>',
    ]
    st.markdown(f'<div class="badges">{" ".join(chips)}</div>', unsafe_allow_html=True)
    
    # Define columns
    if is_multi_hour_mode:
        time_label = "Time Range (2H)" if granularity == "H2" else "Time Range (4H)"
        columns = [
            "Time (IST)",
            time_label,
            "Volume", "21-Day Avg", "Threshold", "Actual Mult", "Spike Δ", "Sentiment", "Body %"
        ]
    else:
        columns = [
            "Time (IST)",
            f"Time Bucket ({bucket_lbl})",
            "Volume", "21-Day Avg", "Threshold", "Actual Mult", "Spike Δ", "Sentiment"
        ]
    
    df = pd.DataFrame(rows, columns=columns)
    
    # Configure column display
    column_config = {
        "Volume": st.column_config.NumberColumn(format="%d", help="Actual volume for this candle"),
        "21-Day Avg": st.column_config.NumberColumn(format="%d", help="Average volume from previous 21 trading days"),
        "Threshold": st.column_config.NumberColumn(format="%d", help=f"21-Day Avg × {st.session_state.threshold_multiplier} = Spike cutoff"),
        "Actual Mult": st.column_config.TextColumn(help="Volume ÷ 21-Day Avg (shows true ratio)"),
        "Spike Δ": st.column_config.TextColumn(help="Volume - Threshold (shown only if spike detected)"),
        "Sentiment": st.column_config.TextColumn(help="🟩 up, 🟥 down, ▪️ flat"),
    }
    
    if is_multi_hour_mode:
        column_config["Body %"] = st.column_config.TextColumn(
            help="Body as % of total range. Higher % = stronger directional move, Lower % = indecision/doji"
        )
    
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        height=520,
        column_config=column_config,
    )
    
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Export to CSV",
        data=csv,
        file_name=f"{name}_volume_spikes_{st.session_state.backtest_date}.csv",
        mime="text/csv"
    )

# ====== BACKTEST EXECUTION ======
def run_backtest_analysis():
    selected_date = st.session_state.backtest_date
    threshold_multiplier = st.session_state.threshold_multiplier
    
    if not st.session_state.selected_instruments:
        st.warning("⚠️ No instruments selected. Please choose at least one.")
        return
    
    if st.session_state.candle_size == "4 hour":
        granularity = "H4"
        bucket_minutes = 240
    elif st.session_state.candle_size == "2 hour":
        granularity = "H2"
        bucket_minutes = 120
    else:
        granularity = "M15"
        bucket_minutes = {"15 min": 15, "30 min": 30, "1 hour": 60}[st.session_state.bucket_choice]
    
    st.subheader("📈 Volume Spike Backtesting")
    date_str = selected_date.strftime("%Y-%m-%d")
    
    # Build info badges
    if granularity == "H4":
        candle_label = "4h"
        bucket_label = "4h"
    elif granularity == "H2":
        candle_label = "2h"
        bucket_label = "2h"
    else:
        candle_label = "15m"
        bucket_label = st.session_state.bucket_choice
    
    info_html = f"""
    <div class="badges">
        <span class="badge neutral">Date: {date_str}</span>
        <span class="badge">Candle: {candle_label}</span>
        <span class="badge">Bucket: {bucket_label}</span>
        <span class="badge warn">Threshold × {threshold_multiplier}</span>
        <span class="badge neutral">21 Trading Days Avg</span>
    </div>
    """
    st.markdown(info_html, unsafe_allow_html=True)
    
    st.divider()
    
    # Process each instrument
    for name in st.session_state.selected_instruments:
        code = INSTRUMENTS[name]
        
        with st.spinner(f"📊 Analyzing {name}..."):
            rows, spikes = process_instrument(
                name, code, bucket_minutes, granularity, 
                selected_date, threshold_multiplier
            )
        
        if not rows:
            st.warning(f"⚠️ No data available for {name} on {date_str}")
            continue
        
        render_card(name, rows, bucket_minutes, granularity)
        st.divider()

# ====== MAIN ======
if run_backtest:
    run_backtest_analysis()
else:
    st.markdown("""
    ### 📊 How It Works
    
    1. **Select a historical date** to analyze
    2. **Choose instruments** (XAUUSD, NAS100, US30)
    3. **Pick candle size** (15 min, 2 hour, or 4 hour)
    4. **Set threshold multiplier** (default: 2.0)
    5. **Run backtest** to see volume spikes using 21-day historical averages
    
    **Spike Detection:**
    - Compares each candle's volume to the 21-day average for that time bucket
    - Flags spikes when: `Volume > (21-Day Avg × Threshold Multiplier)`
    - Shows actual multiplier (Volume/Avg) for all candles
    - Weekends are automatically excluded from averages for cleaner comparisons
    
    **Timeframe Options:**
    - **15 min**: Time bucket comparison (15m, 30m, 1h buckets)
    - **2 hour**: Position-based comparison (up to 12 candles per day)
    - **4 hour**: Position-based comparison (up to 6 candles per day)
    """)
