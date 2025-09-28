import streamlit as st
import requests
import pandas as pd
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import pytz
import plotly.graph_objects as go

# --- Timezone Setup ---
IST = pytz.timezone('Asia/Kolkata')
UTC = pytz.UTC

# --- Twelve Data API ---
API_KEY = st.secrets["TWELVE_DATA"]["API_KEY"]
BASE_URL = "https://api.twelvedata.com/time_series"

def fetch_ohlc(symbol, start, end, interval="4h"):
    """Fetch 4H OHLC data from Twelve Data within UTC range."""
    params = {
        "symbol": symbol,
        "interval": interval,
        "start_date": start.strftime("%Y-%m-%d %H:%M:%S"),
        "end_date": end.strftime("%Y-%m-%d %H:%M:%S"),
        "apikey": API_KEY,
        "timezone": "UTC",
        "format": "JSON"
    }
    response = requests.get(BASE_URL, params=params).json()
    values = response.get("values", [])
    df = pd.DataFrame(values)
    
    if not df.empty:
        # Convert to IST
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df["datetime_ist"] = df["datetime"].dt.tz_convert(IST)
        df = df.sort_values("datetime").reset_index(drop=True)
        df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)
    
    return df

# --- Candle Structure ---
@dataclass
class Candle:
    name: int
    datetime: datetime
    datetime_ist: datetime
    open: float
    high: float
    low: float
    close: float

# --- New Pattern Rules ---
pattern_rules = {
    "base": {
        1:  {"max_range_atr": 0.8},
        2:  {"max_range_atr": 1.0, "max_close_span_atr": 0.25},
        3:  {"max_range_atr": 1.1, "max_highlow_span_atr": 0.2},
        4:  {"max_range_atr": 1.2, "max_close_span_atr": 0.3},
        5:  {"max_range_atr": 1.3, "max_midpoint_span_atr": 0.4},
        6:  {"max_range_atr": 1.4, "max_close_span_atr": 0.5},
        "additional": {
            "max_range_diff_atr": 0.2,
            "no_new_extreme_atr": 0.2,
            "midpoints_within_atr": 0.3
        }
    },
    "rally": {
        1:  {"min_range_atr": 1.2, "min_net_move_atr": 1.0, "final_close_pct": 0.75},
        2:  {"min_range_atr": 1.0, "min_net_move_atr": 1.2},
        3:  {"min_range_atr": 1.2, "min_net_move_atr": 1.5},
        4:  {"min_range_atr": 1.2, "min_net_move_atr": 2.0},
        5:  {"min_range_atr": 1.1, "min_net_move_atr": 2.5, "final_close_pct": 0.70},
        6:  {"min_range_atr": 1.1, "min_net_move_atr": 3.0},
        "additional": {
            "no_inside_final": True,
            "final_close_upper_pct": 0.30,
            "max_retrace_atr": 0.2
        }
    },
    "drop": {
        1:  {"min_range_atr": 1.2, "min_net_move_atr": 1.0, "final_close_pct": 0.25},
        2:  {"min_range_atr": 1.0, "min_net_move_atr": 1.2},
        3:  {"min_range_atr": 1.2, "min_net_move_atr": 1.5},
        4:  {"min_range_atr": 1.2, "min_net_move_atr": 2.0},
        5:  {"min_range_atr": 1.1, "min_net_move_atr": 2.5, "final_close_pct": 0.30},
        6:  {"min_range_atr": 1.1, "min_net_move_atr": 3.0},
        "additional": {
            "no_inside_final": True,
            "final_close_lower_pct": 0.30,
            "max_retrace_atr": 0.2
        }
    }
}

# --- Helper Functions ---
def range_atr(candle, atr):
    return (candle["high"] - candle["low"]) / atr

def net_move_atr(candles, atr):
    return abs(candles[-1]["close"] - candles[0]["open"]) / atr

def close_span_atr(candles, atr):
    closes = [c["close"] for c in candles]
    return (max(closes) - min(closes)) / atr

def midpoint(candle):
    return (candle["high"] + candle["low"]) / 2

def midpoint_span_atr(candles, atr):
    mps = [midpoint(c) for c in candles]
    return (max(mps) - min(mps)) / atr

def highlow_span_atr(candles, atr):
    highs = [c["high"] for c in candles]
    lows  = [c["low"]  for c in candles]
    return (max(highs) - min(lows)) / atr

def no_new_extreme(candles, atr):
    first = candles[0]
    highs = [c["high"] for c in candles]
    lows  = [c["low"]  for c in candles]
    return (max(highs) - first["high"]) / atr <= pattern_rules["base"]["additional"]["no_new_extreme_atr"] \
       and (first["low"] - min(lows)) / atr <= pattern_rules["base"]["additional"]["no_new_extreme_atr"]

def is_inside_bar(bar, prev_bar):
    return bar["high"] < prev_bar["high"] and bar["low"] > prev_bar["low"]

def monotonic_closes(candles, direction="up", max_retrace_atr=0.2, atr=None):
    closes = [c["close"] for c in candles]
    if direction == "up":
        if any(closes[i] <= closes[i-1] for i in range(1, len(closes))):
            return False
    else:
        if any(closes[i] >= closes[i-1] for i in range(1, len(closes))):
            return False
    if atr:
        extremes = (max(closes) - min(closes)) / atr
        return extremes <= max_retrace_atr
    return True

# --- Core Validation ---
def validate_pattern(candles, atr, pattern):
    n = len(candles)
    rules = pattern_rules[pattern].get(n)
    if not rules:
        return False, f"No rules for {pattern} with {n} candles"
    
    results = []
    if pattern == "base":
        results.append(all(range_atr(c, atr) <= rules["max_range_atr"] for c in candles))
        if "max_close_span_atr" in rules:
            results.append(close_span_atr(candles, atr) <= rules["max_close_span_atr"])
        if "max_highlow_span_atr" in rules:
            results.append(highlow_span_atr(candles, atr) <= rules["max_highlow_span_atr"])
        if "max_midpoint_span_atr" in rules:
            results.append(midpoint_span_atr(candles, atr) <= rules["max_midpoint_span_atr"])
        add = pattern_rules["base"]["additional"]
        results.append(highlow_span_atr(candles, atr) - range_atr(candles[0], atr) <= add["max_range_diff_atr"])
        results.append(no_new_extreme(candles, atr))
        results.append(midpoint_span_atr(candles, atr) <= add["midpoints_within_atr"])
    else:
        results.append(all(range_atr(c, atr) >= rules["min_range_atr"] for c in candles))
        results.append(net_move_atr(candles, atr) >= rules["min_net_move_atr"])
        highs = [c["high"] for c in candles]
        lows  = [c["low"]  for c in candles]
        if pattern == "rally":
            results.append(all(highs[i] > highs[i-1] and lows[i] > lows[i-1] for i in range(1, n)))
            if "final_close_pct" in rules:
                rng = candles[-1]["high"] - candles[-1]["low"]
                results.append((candles[-1]["close"] - candles[-1]["low"]) / rng >= rules["final_close_pct"])
        else:
            results.append(all(highs[i] < highs[i-1] and lows[i] < lows[i-1] for i in range(1, n)))
            if "final_close_pct" in rules:
                rng = candles[-1]["high"] - candles[-1]["low"]
                results.append((candles[-1]["high"] - candles[-1]["close"]) / rng >= rules["final_close_pct"])
        add = pattern_rules[pattern]["additional"]
        if add.get("no_inside_final"):
            results.append(not is_inside_bar(candles[-1], candles[-2]))
        if add.get("final_close_upper_pct") and pattern=="rally":
            rng = candles[-1]["high"] - candles[-1]["low"]
            pct = (candles[-1]["high"] - candles[-1]["close"]) / rng
            results.append(pct <= add["final_close_upper_pct"])
        if add.get("final_close_lower_pct") and pattern=="drop":
            rng = candles[-1]["high"] - candles[-1]["low"]
            pct = (candles[-1]["close"] - candles[-1]["low"]) / rng
            results.append(pct <= add["final_close_lower_pct"])
        results.append(monotonic_closes(candles,
                                       direction="up" if pattern=="rally" else "down",
                                       max_retrace_atr=add["max_retrace_atr"],
                                       atr=atr))
    
    return all(results), "passed" if all(results) else "failed"

# --- Plot function (same as before) ---
def plot_zone_chart(df, selected_candles_df=None):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df['datetime_ist'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Price'
    ))
    if selected_candles_df is not None and not selected_candles_df.empty:
        fig.add_trace(go.Scatter(
            x=selected_candles_df['datetime_ist'],
            y=selected_candles_df['high'] * 1.001,
            mode='markers',
            marker=dict(symbol='triangle-down', size=12, color='orange'),
            name='Selected Candles'
        ))
    fig.update_layout(
        title="Pattern Validation (IST)",
        yaxis_title="Price",
        xaxis_title="Time (IST)",
        height=600,
        xaxis_rangeslider_visible=False
    )
    return fig

# --- Streamlit UI ---
st.title("🎯 Pattern Validator (Rally / Drop / Base)")

current_time_ist = datetime.now(IST)
st.caption(f"Current Time (IST): {current_time_ist.strftime('%Y-%m-%d %H:%M:%S')}")

st.sidebar.header("Configuration")
symbol = st.sidebar.text_input("Symbol", value="XAU/USD")
mode = st.sidebar.radio("Selection Mode",
    ["Automatic (Last N Candles)", "Manual Time Range", "Custom Candle Selection"])
pattern = st.sidebar.selectbox("Pattern to Validate", ["rally", "drop", "base"])
current_atr = st.sidebar.number_input("Current ATR", min_value=0.01, value=0.75, step=0.01)

df = None
selected_candles = []

col1, col2 = st.columns([2, 1])

with col1:
    if mode == "Automatic (Last N Candles)":
        st.subheader("📊 Automatic Mode")
        n_candles = st.slider("Number of candles", 1, 6, 3)
        if st.button("Analyze Last Candles"):
            end_utc = datetime.now(UTC)
            start_utc = end_utc - timedelta(days=5)
            with st.spinner("Fetching data..."):
                df = fetch_ohlc(symbol, start_utc, end_utc)
            if not df.empty:
                candles = [
                    dict(open=row.open, high=row.high, low=row.low, close=row.close)
                    for _, row in df.tail(n_candles).iterrows()
                ]
                ok, message = validate_pattern(candles, current_atr, pattern)
                st.success(f"✅ {pattern.title()} {message}" if ok else f"❌ {pattern.title()} {message}")

    elif mode == "Manual Time Range":
        st.subheader("🕐 Manual Time Range")
        st.info("⏰ Enter time in IST")
        col_start, col_end = st.columns(2)
        with col_start:
            start_date = st.date_input("Start Date", value=datetime.now(IST).date() - timedelta(days=3))
            start_time = st.time_input("Start Time (IST)", value=datetime.now(IST).time())
        with col_end:
            end_date = st.date_input("End Date", value=datetime.now(IST).date())
            end_time = st.time_input("End Time (IST)", value=datetime.now(IST).time())
        if st.button("Validate Range"):
            start_ist = IST.localize(datetime.combine(start_date, start_time))
            end_ist = IST.localize(datetime.combine(end_date, end_time))
            start_utc = start_ist.astimezone(UTC)
            end_utc = end_ist.astimezone(UTC)
            with st.spinner("Fetching data..."):
                df = fetch_ohlc(symbol, start_utc, end_utc)
            if not df.empty:
                sel = df[(df['datetime_ist'] >= start_ist) & (df['datetime_ist'] <= end_ist)].copy()
                if not sel.empty:
                    candles = [
                        dict(open=row.open, high=row.high, low=row.low, close=row.close)
                        for _, row in sel.iterrows()
                    ]
                    ok, message = validate_pattern(candles, current_atr, pattern)
                    st.success(f"✅ {pattern.title()} {message}" if ok else f"❌ {pattern.title()} {message}")
                    selected_candles = sel

    else:
        st.subheader("🎯 Custom Candle Selection")
        if st.button("Load Recent Data"):
            end_utc = datetime.now(UTC)
            start_utc = end_utc - timedelta(days=10)
            with st.spinner("Loading data..."):
                df = fetch_ohlc(symbol, start_utc, end_utc)
            if not df.empty:
                st.session_state['df'] = df
                st.success(f"Loaded {len(df)} candles")
        if 'df' in st.session_state:
            df = st.session_state['df']
            df_display = df[['datetime_ist','open','high','low','close']].copy()
            df_display['datetime_ist'] = df_display['datetime_ist'].dt.strftime('%Y-%m-%d %H:%M IST')
            st.write("Select candles (1–6):")
            indices = st.multiselect("Indices", options=df_display.index.tolist(),
                format_func=lambda x: f"{x}: {df_display.loc[x,'datetime_ist']} O:{df_display.loc[x,'open']:.2f} C:{df_display.loc[x,'close']:.2f}")
            if indices and st.button("Validate Selected"):
                indices = sorted(indices)
                candles = [
                    dict(open=df.loc[idx,'open'], high=df.loc[idx,'high'], low=df.loc[idx,'low'], close=df.loc[idx,'close'])
                    for idx in indices
                ]
                ok, message = validate_pattern(candles, current_atr, pattern)
                st.success(f"✅ {pattern.title()} {message}" if ok else f"❌ {pattern.title()} {message}")
                selected_candles = df.iloc[indices]

# --- Chart Display ---
if df is not None and not df.empty:
    st.subheader("📈 Chart")
    sel_df = selected_candles if isinstance(selected_candles, pd.DataFrame) else None
    fig = plot_zone_chart(df, sel_df)
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("📊 Recent Data")
    display_df = df[['datetime_ist','open','high','low','close']].tail(20).copy()
    display_df['datetime_ist'] = display_df['datetime_ist'].dt.strftime('%Y-%m-%d %H:%M IST')
    display_df.columns = ['Time (IST)','Open','High','Low','Close']
    st.dataframe(display_df, use_container_width=True, hide_index=True)
