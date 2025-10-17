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

# Number of trading days to use for averaging (SAME AS ORIGINAL)
TRADING_DAYS_FOR_AVERAGE = 21

# ====== SIDEBAR CONFIG ======
st.sidebar.title("🔧 Backtest Settings")

# Initialize all session state variables
if "selected_instruments" not in st.session_state:
    st.session_state.selected_instruments = list(INSTRUMENTS.keys())
if "bucket_choice" not in st.session_state:
    st.session_state.bucket_choice = "1 hour"
if "candle_size" not in st.session_state:
    st.session_state.candle_size = "15 min"
if "skip_weekends" not in st.session_state:
    st.session_state.skip_weekends = True
if "backtest_date" not in st.session_state:
    st.session_state.backtest_date = datetime.now(IST).date() - timedelta(days=1)
if "threshold_multiplier" not in st.session_state:
    st.session_state.threshold_multiplier = 1.618

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

# THRESHOLD MULTIPLIER SLIDER
threshold_value = st.sidebar.slider(
    "📈 Threshold Multiplier",
    min_value=1.0,
    max_value=3.0,
    step=0.1,
    value=st.session_state.threshold_multiplier,
    key="threshold_multiplier",
    help="Spike detected when: Volume > (21-Day Avg × Threshold)"
)

# Show example calculation


st.sidebar.toggle(
    "Skip Weekends in Average",
    value=st.session_state.skip_weekends,
    key="skip_weekends",
    help="When ON, uses only trading days (Mon-Fri) for volume averages"
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
def compute_bucket_averages(code, bucket_size_minutes, granularity, selected_date, skip_weekends=True):
    """Compute averages for comparison - EXACT SAME LOGIC AS ORIGINAL"""
    if granularity == "H4":
        return compute_4h_position_averages(code, selected_date, skip_weekends)
    else:
        return compute_15m_bucket_averages(code, bucket_size_minutes, selected_date, skip_weekends)

def compute_15m_bucket_averages(code, bucket_size_minutes, selected_date, skip_weekends=True):
    """Time-bucket based averaging for 15-minute mode, collecting last N trading days BEFORE selected date"""
    bucket_volumes = defaultdict(list)
    
    trading_days_collected = 0
    days_back = 1
    max_lookback = 60
    
    while trading_days_collected < TRADING_DAYS_FOR_AVERAGE and days_back < max_lookback:
        day_ist = selected_date - timedelta(days=days_back)
        
        # Skip weekends if enabled (EXACT SAME LOGIC)
        if skip_weekends and is_weekend(day_ist):
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

def compute_4h_position_averages(code, selected_date, skip_weekends=True):
    """Position-based averaging for 4H mode, collecting last N trading days BEFORE selected date"""
    position_volumes = defaultdict(list)
    
    trading_days_collected = 0
    days_back = 1
    max_lookback = 60
    
    while trading_days_collected < TRADING_DAYS_FOR_AVERAGE and days_back < max_lookback:
        day_ist = selected_date - timedelta(days=days_back)
        
        # Skip weekends if enabled (EXACT SAME LOGIC)
        if skip_weekends and is_weekend(day_ist):
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
    """Process instrument for the selected backtest date - EXACT SAME LOGIC AS ORIGINAL"""
    bucket_avg = compute_bucket_averages(code, bucket_size_minutes, granularity, selected_date, skip_weekends=st.session_state.skip_weekends)
    
    is_4h_mode = (granularity == "H4")
    
    # Fetch data for the selected date
    start_ist = IST.localize(datetime.combine(selected_date, time(0, 0)))
    end_ist = IST.localize(datetime.combine(selected_date + timedelta(days=1), time(0, 0)))
    
    start_utc = start_ist.astimezone(UTC)
    end_utc = end_ist.astimezone(UTC)
    
    candles = fetch_candles(code, start_utc, end_utc, granularity=granularity)
    if not candles:
        return [], [], {}
    
    rows = []
    spikes_found = []
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
        
        # EXACT CALCULATION: Threshold = Average × threshold_multiplier
        # Example: If avg = 10 and multiplier = 1.618, then threshold = 16.18
        threshold = avg * threshold_multiplier if avg else 0
        
        # SPIKE DETECTION: Volume must be GREATER than threshold
        # Example: Spike if vol > 16.18
        over = (threshold > 0 and vol > threshold)
        
        # ACTUAL MULTIPLIER: Volume / Average (not Volume / Threshold)
        # This shows the true ratio regardless of spike detection
        actual_multiplier = (vol / avg) if avg > 0 else 0
        
        spike_diff = f"▲{vol - int(threshold)}" if over else ""
        sentiment = get_sentiment(c)
        
        # Build row based on mode - INCLUDING AVG, THRESHOLD, AND ACTUAL MULTIPLIER
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
                int(avg) if avg > 0 else 0,  # Show 21-day average
                int(threshold) if threshold > 0 else 0,  # Show threshold (avg × multiplier)
                f"{actual_multiplier:.2f}x",  # Actual multiplier (Vol/Avg)
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
                int(avg) if avg > 0 else 0,  # Show 21-day average
                int(threshold) if threshold > 0 else 0,  # Show threshold (avg × multiplier)
                f"{actual_multiplier:.2f}x",  # Actual multiplier (Vol/Avg)
                spike_diff,
                sentiment
            ])
        
        last_summary = {
            "time": t_ist.strftime("%Y-%m-%d %I:%M %p"),
            "bucket": bucket,
            "volume": vol,
            "avg": avg,
            "threshold": threshold,
            "actual_multiplier": actual_multiplier,  # Changed to actual multiplier
            "over": over,
            "sentiment": sentiment
        }
        
        # Collect all spikes for the day
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
    
    return rows, spikes_found, last_summary

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
        c2.metric(f"21-Day Avg", f"{summary['avg']:.0f}")
        c3.metric(f"Threshold ({st.session_state.threshold_multiplier}×)", f"{summary['threshold']:.0f}")
        c4.metric("Actual Multiplier (Vol/Avg)", f"{summary['actual_multiplier']:.2f}×", 
                 delta="SPIKE ✓" if summary['over'] else "No Spike",
                 delta_color="normal" if summary['over'] else "off")
    
    # Define columns based on mode - INCLUDING AVG, THRESHOLD, AND ACTUAL MULTIPLIER
    if is_4h_mode:
        columns = [
            "Time (IST)",
            "Time Range (4H)",
            "Open", "High", "Low", "Close",
            "Volume", "21-Day Avg", "Threshold", "Actual Mult", "Spike Δ", "Sentiment", "Body %"
        ]
    else:
        columns = [
            "Time (IST)",
            f"Time Bucket ({bucket_lbl})",
            "Open", "High", "Low", "Close",
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
    
    if is_4h_mode:
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
    
    top_l, top_r = st.columns([3, 2])
    with top_l:
        st.subheader("📈 Volume Spike Backtesting")
        date_str = selected_date.strftime("%Y-%m-%d")
        comparison_type = "Time Range" if is_4h_mode else f"Bucket: {bucket_minutes}m"
        weekends_status = "OFF" if st.session_state.skip_weekends else "ON"
        st.markdown(
            f'<div class="badges">'
            f'<span class="badge">Date: {date_str}</span>'
            f'<span class="badge neutral">Candle: {"4h" if granularity=="H4" else "15m"}</span>'
            f'<span class="badge neutral">{comparison_type}</span>'
            f'<span class="badge ok">Threshold × {threshold_multiplier}</span>'
            f'<span class="badge neutral">21 Trading Days Avg</span>'
            f'<span class="badge {"ok" if st.session_state.skip_weekends else "warn"}">Weekends: {weekends_status}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with top_r:
        st.info(f"💡 **Spike Condition:**\n\nVolume > (21-Day Avg × {threshold_multiplier})\n\n**Actual Multiplier** = Volume ÷ Avg\n(shown regardless of spike)")
    
    names = st.session_state.selected_instruments
    cols = st.columns(2) if len(names) > 1 else [st.container()]
    col_idx = 0
    
    all_rows_have_data = False
    
    with st.spinner("Running backtest analysis..."):
        for name in names:
            code = INSTRUMENTS[name]
            with cols[col_idx]:
                with st.container(border=True):
                    rows, spikes, summary = process_instrument(
                        name, code, bucket_minutes, granularity, selected_date, threshold_multiplier
                    )
                    if rows:
                        all_rows_have_data = True
                        render_card(name, rows, bucket_minutes, summary, is_4h_mode)
                    else:
                        st.warning(f"No data available for {name} on {date_str}")
            
            col_idx = (col_idx + 1) % len(cols)
            
            if spikes:
                all_spike_msgs.extend(spikes)
    
    # Display spike summary
    if all_spike_msgs:
        st.success(f"✅ Found {len(all_spike_msgs)} volume spike(s) on {date_str}")
        with st.expander("📋 View All Spikes", expanded=True):
            for spike in all_spike_msgs:
                st.markdown(
                    f"🔔 **{spike['instrument']}** @ {spike['time']}\n\n"
                    f"- Volume: **{spike['volume']:,}**\n"
                    f"- 21-Day Avg: {spike['avg']:,}\n"
                    f"- Threshold: {spike['threshold']:,} (= {spike['avg']:,} × {threshold_multiplier})\n"
                    f"- **Actual Multiplier: {spike['actual_multiplier']:.2f}×** (Vol ÷ Avg)\n"
                    f"- Spike Delta: **{spike['spike_diff']}**\n"
                    f"- Sentiment: {spike['sentiment']}\n"
                    f"---"
                )
    else:
        if all_rows_have_data:
            st.info(f"ℹ️ No volume spikes detected on {date_str}\n\nNo candles had Volume > (21-Day Avg × {threshold_multiplier})")

# ====== MAIN ======
st.title("📊 Volume Spike Backtesting Tool")
st.caption("Test volume spike detection on historical dates using the same logic as the live dashboard")

if run_backtest or "backtest_run" in st.session_state:
    st.session_state.backtest_run = True
    run_backtest_analysis()
else:
    st.info("👈 Configure your backtest parameters in the sidebar and click 'Run Backtest' to begin analysis")
    
    # Show instructions
    with st.expander("📖 How to Use", expanded=True):
        st.markdown(f"""
        ### Backtesting Instructions
        
        1. **Select Date**: Choose a historical date to analyze
        2. **Choose Instruments**: Select one or more instruments (XAUUSD, NAS100, US30)
        3. **Configure Settings**:
           - **Candle Size**: 15 min or 4 hour
           - **Time Bucket**: 15 min, 30 min, or 1 hour (for 15 min candles)
           - **Threshold Multiplier**: Configurable (1.0 to 3.0, default 1.618)
           - **Skip Weekends**: Use only trading days for calculations
        4. **Run Backtest**: Click the button to analyze the selected date
        
        ### Spike Detection Formula
        ```
        21-Day Avg = Sum of volumes / 21 trading days
        Threshold = 21-Day Avg × Threshold Multiplier
        Actual Multiplier = Volume ÷ 21-Day Avg
        
        SPIKE DETECTED IF: Volume > Threshold
        ```
        
        ### Example Calculation
        - 21-Day Average = 100
        - Threshold Multiplier = 1.618 (your setting)
        - Threshold = 100 × 1.618 = **161.8**
        - Current Volume = 200
        - **Actual Multiplier = 200 ÷ 100 = 2.0×**
        - Is 200 > 161.8? **YES → SPIKE!** ✅
        - Spike Delta = 200 - 161.8 = 38.2
        
        ### Table Columns Explained
        - **Volume**: Actual volume for that candle
        - **21-Day Avg**: Average from previous 21 trading days
        - **Threshold**: The cutoff (Avg × Your Multiplier Setting)
        - **Actual Mult**: Volume ÷ Avg (shows true ratio)
        - **Spike Δ**: How much above threshold (if spike)
        
        **Note:** Actual Multiplier is always calculated (Vol ÷ Avg) to show you the real ratio, independent of whether it's a spike or not.
        """)
