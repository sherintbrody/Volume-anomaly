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

TRADING_DAYS_FOR_AVERAGE = 21
SKIP_WEEKENDS = True

# ====== INITIALIZE SESSION STATE ======
if "selected_instruments" not in st.session_state:
    st.session_state.selected_instruments = list(INSTRUMENTS.keys())
if "candle_size" not in st.session_state:
    st.session_state.candle_size = "1 hour"
if "start_date" not in st.session_state:
    st.session_state.start_date = datetime.now(IST).date() - timedelta(days=30)
if "end_date" not in st.session_state:
    st.session_state.end_date = datetime.now(IST).date() - timedelta(days=1)
if "threshold_multiplier" not in st.session_state:
    st.session_state.threshold_multiplier = 2.0

# ====== SIDEBAR CONFIG ======
st.sidebar.title("🔧 Backtest Settings")

st.sidebar.date_input(
    "📅 Start Date",
    max_value=datetime.now(IST).date(),
    key="start_date"
)

st.sidebar.date_input(
    "📅 End Date",
    max_value=datetime.now(IST).date(),
    key="end_date"
)

selected_instruments = st.sidebar.multiselect(
    "Select Instruments to Analyze",
    options=list(INSTRUMENTS.keys()),
    default=list(INSTRUMENTS.keys()),
    key="selected_instruments"
)

# Candle Size — 1 hour, 4 hour, Daily only
candle_size = st.sidebar.radio(
    "🕐 Candle Size",
    ["1 hour", "4 hour", "Daily"],
    key="candle_size"
)

if st.session_state.candle_size == "1 hour":
    st.sidebar.caption("🕒 Comparison: By candle time range (up to 24 per day)")
elif st.session_state.candle_size == "4 hour":
    st.sidebar.caption("🕒 Comparison: By candle time range (up to 6 per day)")
else:  # Daily
    st.sidebar.caption("📅 Comparison: Each daily candle vs 21-day avg daily volume")

threshold_multiplier = st.sidebar.slider(
    "📈 Threshold Multiplier",
    min_value=1.0,
    max_value=5.0,
    step=0.1,
    value=2.0,
    key="threshold_multiplier",
    help="Spike detected when: Volume > (21-Day Avg × Threshold)"
)

run_backtest = st.sidebar.button("🔍 Run Backtest", type="primary", use_container_width=True)

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
def fetch_candles(instrument_code, from_time, to_time, granularity="H1"):
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
def parse_candle_time(time_str):
    try:
        return datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S.%f000Z")
    except ValueError:
        return datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S.000Z")

def get_1h_time_range(dt_ist):
    end_time = dt_ist + timedelta(hours=1)
    return f"{dt_ist.strftime('%I:%M %p')}–{end_time.strftime('%I:%M %p')}"

def get_4h_time_range(dt_ist):
    end_time = dt_ist + timedelta(hours=4)
    return f"{dt_ist.strftime('%I:%M %p')}–{end_time.strftime('%I:%M %p')}"

def is_weekend(date):
    return date.weekday() in [5, 6]

def get_sentiment(candle):
    o = float(candle["mid"]["o"])
    c = float(candle["mid"]["c"])
    return "🟩" if c > o else "🟥" if c < o else "▪️"

def get_body_percentage(candle):
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

# ====== AVERAGE COMPUTATION ======
@st.cache_data(ttl=3600)
def compute_bucket_averages(code, granularity, reference_date):
    """
    reference_date : the day we want to analyse.
    Averages are built from the 21 trading days BEFORE reference_date.
    Daily mode returns a single scalar keyed as 'DAILY_AVG'.
    """
    if granularity == "H4":
        return compute_4h_position_averages(code, reference_date)
    elif granularity == "H1":
        return compute_1h_position_averages(code, reference_date)
    else:  # Daily
        return compute_daily_averages(code, reference_date)

def compute_1h_position_averages(code, reference_date):
    position_volumes = defaultdict(list)
    trading_days_collected = 0
    days_back = 1
    max_lookback = 60

    while trading_days_collected < TRADING_DAYS_FOR_AVERAGE and days_back < max_lookback:
        day_ist = reference_date - timedelta(days=days_back)
        if is_weekend(day_ist):
            days_back += 1
            continue

        start_utc = IST.localize(
            datetime.combine(day_ist, time(0, 0))
        ).astimezone(UTC)
        end_utc = IST.localize(
            datetime.combine(day_ist + timedelta(days=1), time(0, 0))
        ).astimezone(UTC)

        candles = fetch_candles(code, start_utc, end_utc, granularity="H1")
        if candles:
            trading_days_collected += 1
            for c in candles:
                if not c.get("complete", True):
                    continue
                t_ist = parse_candle_time(c["time"]).replace(tzinfo=UTC).astimezone(IST)
                position_volumes[get_1h_time_range(t_ist)].append(c["volume"])
        days_back += 1

    return {p: sum(vs) / len(vs) for p, vs in position_volumes.items() if vs}

def compute_4h_position_averages(code, reference_date):
    position_volumes = defaultdict(list)
    trading_days_collected = 0
    days_back = 1
    max_lookback = 60

    while trading_days_collected < TRADING_DAYS_FOR_AVERAGE and days_back < max_lookback:
        day_ist = reference_date - timedelta(days=days_back)
        if is_weekend(day_ist):
            days_back += 1
            continue

        start_utc = IST.localize(
            datetime.combine(day_ist, time(0, 0))
        ).astimezone(UTC)
        end_utc = IST.localize(
            datetime.combine(day_ist + timedelta(days=1), time(0, 0))
        ).astimezone(UTC)

        candles = fetch_candles(code, start_utc, end_utc, granularity="H4")
        if candles:
            trading_days_collected += 1
            for c in candles:
                if not c.get("complete", True):
                    continue
                t_ist = parse_candle_time(c["time"]).replace(tzinfo=UTC).astimezone(IST)
                position_volumes[get_4h_time_range(t_ist)].append(c["volume"])
        days_back += 1

    return {p: sum(vs) / len(vs) for p, vs in position_volumes.items() if vs}

def compute_daily_averages(code, reference_date):
    """
    Collect the daily candle volume for each of the 21 trading days
    BEFORE reference_date, then return a single average keyed as 'DAILY_AVG'.

    IMPORTANT: We match strictly on t_utc.date() because OANDA daily candles
    are anchored to NY close (UTC) — never use IST date here, or the same
    physical candle could be matched to two different calendar days.
    """
    daily_volumes = []
    trading_days_collected = 0
    days_back = 1
    max_lookback = 60

    while trading_days_collected < TRADING_DAYS_FOR_AVERAGE and days_back < max_lookback:
        day_ist = reference_date - timedelta(days=days_back)
        if is_weekend(day_ist):
            days_back += 1
            continue

        # Fetch a 2-day window to ensure OANDA returns the daily candle
        start_utc = IST.localize(
            datetime.combine(day_ist, time(0, 0))
        ).astimezone(UTC)
        end_utc = IST.localize(
            datetime.combine(day_ist + timedelta(days=2), time(0, 0))
        ).astimezone(UTC)

        candles = fetch_candles(code, start_utc, end_utc, granularity="D")
        for c in candles:
            t_utc = parse_candle_time(c["time"])
            # Strict UTC-date match only — and break after first match
            if t_utc.date() == day_ist and c.get("complete", True):
                daily_volumes.append(c["volume"])
                trading_days_collected += 1
                break

        days_back += 1

    if not daily_volumes:
        return {"DAILY_AVG": 0}
    return {"DAILY_AVG": sum(daily_volumes) / len(daily_volumes)}

# ====== CORE PROCESS — single day ======
def process_instrument(name, code, granularity, selected_date, threshold_multiplier):
    """
    Process one calendar day.
    Averages are recomputed relative to selected_date so the lookback window
    is always the 21 trading days that precede the candle being examined.
    """
    bucket_avg = compute_bucket_averages(code, granularity, selected_date)
    rows = []
    spikes_found = []

    # ── Daily mode ─────────────────────────────────────────────────────────────
    if granularity == "D":
        start_utc = IST.localize(
            datetime.combine(selected_date, time(0, 0))
        ).astimezone(UTC)
        end_utc = IST.localize(
            datetime.combine(selected_date + timedelta(days=2), time(0, 0))
        ).astimezone(UTC)

        candles = fetch_candles(code, start_utc, end_utc, granularity="D")
        if not candles:
            return [], []

        avg = bucket_avg.get("DAILY_AVG", 0)
        threshold = avg * threshold_multiplier if avg else 0

        # Strict UTC-date match — find THE ONE candle for selected_date
        matched_candle = None
        for c in candles:
            t_utc = parse_candle_time(c["time"])
            if t_utc.date() == selected_date and c.get("complete", True):
                matched_candle = c
                break

        if matched_candle is None:
            return [], []

        c = matched_candle
        vol = c["volume"]
        over = threshold > 0 and vol > threshold
        actual_multiplier = (vol / avg) if avg > 0 else 0
        spike_diff = f"▲{vol - int(threshold)}" if over else ""
        sentiment = get_sentiment(c)
        body_pct = get_body_percentage(c)

        rows.append([
            selected_date.strftime("%Y-%m-%d"),  # always use selected_date for consistency
            "Full Day",
            vol,
            int(avg) if avg > 0 else 0,
            int(threshold) if threshold > 0 else 0,
            f"{actual_multiplier:.2f}x",
            spike_diff,
            sentiment,
            body_pct,
        ])

        if over:
            spikes_found.append({
                "instrument": name,
                "time": selected_date.strftime("%Y-%m-%d"),
                "volume": vol,
                "avg": int(avg),
                "threshold": int(threshold),
                "spike_diff": spike_diff,
                "sentiment": sentiment,
                "actual_multiplier": actual_multiplier,
            })

        return rows, spikes_found

    # ── Intraday modes (H1, H4) ────────────────────────────────────────────────
    start_utc = IST.localize(
        datetime.combine(selected_date, time(0, 0))
    ).astimezone(UTC)
    end_utc = IST.localize(
        datetime.combine(selected_date + timedelta(days=1), time(0, 0))
    ).astimezone(UTC)

    candles = fetch_candles(code, start_utc, end_utc, granularity=granularity)
    if not candles:
        return [], []

    for c in candles:
        t_ist = parse_candle_time(c["time"]).replace(tzinfo=UTC).astimezone(IST)
        bucket = get_1h_time_range(t_ist) if granularity == "H1" else get_4h_time_range(t_ist)

        vol = c["volume"]
        avg = bucket_avg.get(bucket, 0)
        threshold = avg * threshold_multiplier if avg else 0
        over = threshold > 0 and vol > threshold
        actual_multiplier = (vol / avg) if avg > 0 else 0
        spike_diff = f"▲{vol - int(threshold)}" if over else ""
        sentiment = get_sentiment(c)
        body_pct = get_body_percentage(c)

        rows.append([
            t_ist.strftime("%Y-%m-%d %I:%M %p"),
            bucket,
            vol,
            int(avg) if avg > 0 else 0,
            int(threshold) if threshold > 0 else 0,
            f"{actual_multiplier:.2f}x",
            spike_diff,
            sentiment,
            body_pct,
        ])

        if over:
            spikes_found.append({
                "instrument": name,
                "time": t_ist.strftime("%I:%M %p"),
                "volume": vol,
                "avg": int(avg),
                "threshold": int(threshold),
                "spike_diff": spike_diff,
                "sentiment": sentiment,
                "actual_multiplier": actual_multiplier,
            })

    return rows, spikes_found

# ====== CORE PROCESS — date range ======
def process_date_range(name, code, granularity, start_date, end_date, threshold_multiplier):
    all_rows = []
    all_spikes = []
    current_day = start_date

    while current_day <= end_date:
        if is_weekend(current_day):
            current_day += timedelta(days=1)
            continue

        rows, spikes = process_instrument(
            name, code, granularity, current_day, threshold_multiplier
        )
        all_rows.extend(rows)
        all_spikes.extend(spikes)
        current_day += timedelta(days=1)

    return all_rows, all_spikes

# ====== TABLE RENDERING ======
def render_card(name, rows, granularity, start_date=None, end_date=None):
    st.markdown(f"### {name}", help="Instrument")

    if granularity == "H1":
        comparison_label = "1 Hour Mode"
        date_col         = "Time (IST)"
        time_col_label   = "Time Range (1H)"
    elif granularity == "H4":
        comparison_label = "4 Hour Mode"
        date_col         = "Time (IST)"
        time_col_label   = "Time Range (4H)"
    else:  # Daily
        comparison_label = "Daily Mode"
        date_col         = "Date"
        time_col_label   = "Session"   # ← must differ from date_col to avoid duplicate column error

    st.markdown(
        f'<div class="badges"><span class="badge neutral">{comparison_label}</span></div>',
        unsafe_allow_html=True
    )

    columns = [
        date_col,
        time_col_label,
        "Volume",
        "21-Day Avg",
        "Threshold",
        "Actual Mult",
        "Spike Δ",
        "Sentiment",
        "Body %",
    ]

    df = pd.DataFrame(rows, columns=columns)

    column_config = {
        "Volume": st.column_config.NumberColumn(
            format="%d",
            help="Actual volume for this candle"
        ),
        "21-Day Avg": st.column_config.NumberColumn(
            format="%d",
            help="Average volume from previous 21 trading days"
        ),
        "Threshold": st.column_config.NumberColumn(
            format="%d",
            help=f"21-Day Avg × {st.session_state.threshold_multiplier} = Spike cutoff"
        ),
        "Actual Mult": st.column_config.TextColumn(
            help="Volume ÷ 21-Day Avg (true ratio)"
        ),
        "Spike Δ": st.column_config.TextColumn(
            help="Volume − Threshold (shown only when spike detected)"
        ),
        "Sentiment": st.column_config.TextColumn(
            help="🟩 up  🟥 down  ▪️ flat"
        ),
        "Body %": st.column_config.TextColumn(
            help="Body as % of total candle range. Higher = stronger directional move."
        ),
    }

    # Daily: dynamic height based on row count; intraday: fixed 520px
    table_height = max(120, min(len(df) * 40 + 60, 600)) if granularity == "D" else 520

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        height=table_height,
        column_config=column_config,
    )

    date_label = (
        f"{start_date}_{end_date}"
        if (start_date and end_date)
        else datetime.now(IST).strftime("%Y-%m-%d")
    )
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Export to CSV",
        data=csv,
        file_name=f"{name}_volume_spikes_{date_label}.csv",
        mime="text/csv",
        key=f"download_{name}_{date_label}_{granularity}",
    )

# ====== BACKTEST EXECUTION ======
def run_backtest_analysis():
    start_date           = st.session_state.start_date
    end_date             = st.session_state.end_date
    threshold_multiplier = st.session_state.threshold_multiplier

    if start_date > end_date:
        st.error("❌ Start date must be before or equal to End date.")
        return

    if not st.session_state.selected_instruments:
        st.warning("⚠️ No instruments selected. Please choose at least one.")
        return

    if st.session_state.candle_size == "4 hour":
        granularity  = "H4"
        candle_label = "4h"
    elif st.session_state.candle_size == "1 hour":
        granularity  = "H1"
        candle_label = "1h"
    else:  # Daily
        granularity  = "D"
        candle_label = "Daily"

    st.subheader("📈 Volume Spike Backtesting")
    date_str = f"{start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')}"

    info_html = f"""
    <div class="badges">
        <span class="badge neutral">Range: {date_str}</span>
        <span class="badge">Candle: {candle_label}</span>
        <span class="badge warn">Threshold × {threshold_multiplier}</span>
        <span class="badge neutral">21 Trading Days Avg</span>
    </div>
    """
    st.markdown(info_html, unsafe_allow_html=True)
    st.divider()

    for name in st.session_state.selected_instruments:
        code = INSTRUMENTS[name]

        with st.spinner(f"📊 Analyzing {name}…"):
            rows, spikes = process_date_range(
                name, code, granularity,
                start_date, end_date, threshold_multiplier
            )

        if not rows:
            st.warning(f"⚠️ No data available for {name} in the selected range.")
            continue

        render_card(name, rows, granularity, start_date, end_date)
        st.divider()

# ====== MAIN ======
if run_backtest:
    run_backtest_analysis()
else:
    st.markdown("""
    ### 📊 How It Works

    1. **Select a date range** to analyze
    2. **Choose instruments** (XAUUSD, NAS100, US30)
    3. **Pick candle size** (1 hour, 4 hour, or Daily)
    4. **Set threshold multiplier** (default: 2.0)
    5. **Run backtest** to see volume spikes using 21-day historical averages

    **Spike Detection:**
    - Compares each candle's volume to the 21-day average for that time slot
    - Flags spikes when: `Volume > (21-Day Avg × Threshold Multiplier)`
    - Shows actual multiplier (Volume ÷ Avg) for every candle
    - Weekends are automatically excluded from all averages

    **Timeframe Options:**
    - **1 hour**  — up to 24 candles per day, compared by time-of-day slot
    - **4 hour**  — up to 6 candles per day, compared by time-of-day slot
    - **Daily**   — one candle per day, compared against the 21-day avg daily volume
    """)
