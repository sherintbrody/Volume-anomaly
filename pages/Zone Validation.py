import streamlit as st
import requests
import pandas as pd
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import pytz
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# --- Page Config ---
st.set_page_config(
    page_title="Pattern Validator",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #1557a4;
        transform: translateY(-2px);
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# --- Timezone Setup ---
IST = pytz.timezone('Asia/Kolkata')
UTC = pytz.UTC

# --- Initialize Session State ---
if 'df' not in st.session_state:
    st.session_state.df = None
if 'last_fetch' not in st.session_state:
    st.session_state.last_fetch = None
if 'atr_values' not in st.session_state:
    st.session_state.atr_values = None

# --- Twelve Data API ---
API_KEY = st.secrets["TWELVE_DATA"]["API_KEY"]
BASE_URL = "https://api.twelvedata.com/time_series"

@st.cache_data(ttl=300)  # Cache for 5 minutes
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

# --- ATR Calculation ---
def calculate_atr(df, period=21):
    """Calculate Average True Range"""
    high = df['high']
    low = df['low']
    close = df['close']
    
    # True Range calculation
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # ATR calculation
    atr = tr.rolling(window=period).mean()
    
    return atr

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

# --- Pattern Rules (same as before) ---
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

# --- Helper Functions (same as before) ---
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

# --- Enhanced Plot function with ATR ---
def plot_combined_chart(df, selected_candles_df=None, atr_values=None):
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=('Price Action', 'ATR (21 Period)')
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df['datetime_ist'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ),
        row=1, col=1
    )
    
    # Selected candles markers
    if selected_candles_df is not None and not selected_candles_df.empty:
        fig.add_trace(
            go.Scatter(
                x=selected_candles_df['datetime_ist'],
                y=selected_candles_df['high'] * 1.001,
                mode='markers',
                marker=dict(symbol='triangle-down', size=15, color='#ff9800'),
                name='Selected',
                showlegend=True
            ),
            row=1, col=1
        )
    
    # ATR chart
    if atr_values is not None:
        fig.add_trace(
            go.
