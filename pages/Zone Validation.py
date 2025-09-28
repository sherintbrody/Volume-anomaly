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
    page_title="Pattern Validator Pro",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .success-box {
        padding: 10px;
        border-radius: 5px;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 10px 0;
    }
    .error-box {
        padding: 10px;
        border-radius: 5px;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        margin: 10px 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .warning-box {
        padding: 10px;
        border-radius: 5px;
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Timezone Setup ---
IST = pytz.timezone('Asia/Kolkata')
UTC = pytz.UTC

# --- Twelve Data API ---
API_KEY = st.secrets["TWELVE_DATA"]["API_KEY"]
BASE_URL = "https://api.twelvedata.com/time_series"

# --- Helper function to check if candle is complete ---
def is_candle_complete(candle_time, interval_hours=4):
    """Check if a candle is complete based on current time"""
    current_time = datetime.now(UTC)
    candle_end_time = candle_time + timedelta(hours=interval_hours)
    return current_time >= candle_end_time

# --- Caching for API calls ---
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
        
        # Check if last candle is complete
        if len(df) > 0:
            last_candle_time = df.iloc[-1]['datetime']
            df['is_complete'] = True
            if not is_candle_complete(last_candle_time):
                df.loc[df.index[-1], 'is_complete'] = False
        
        # Calculate ATR (excluding incomplete candles)
        df = calculate_atr(df, period=21)
    
    return df

# --- ATR Calculation ---
def calculate_atr(df, period=21):
    """Calculate Average True Range excluding incomplete candles"""
    df_copy = df.copy()
    
    # Mark incomplete candle for reference
    incomplete_candle_exists = False
    if len(df_copy) > 0 and 'is_complete' in df_copy.columns:
        incomplete_candle_exists = not df_copy.iloc[-1]['is_complete']
    
    # Calculate True Range for all candles
    df_copy['prev_close'] = df_copy['close'].shift(1)
    df_copy['tr1'] = df_copy['high'] - df_copy['low']
    df_copy['tr2'] = abs(df_copy['high'] - df_copy['prev_close'])
    df_copy['tr3'] = abs(df_copy['low'] - df_copy['prev_close'])
    df_copy['true_range'] = df_copy[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Calculate ATR excluding the last candle if it's incomplete
    if incomplete_candle_exists:
        # Calculate ATR up to the second-to-last candle
        df_copy['atr'] = df_copy['true_range'].iloc[:-1].rolling(window=period).mean()
        # Forward fill the last ATR value to the incomplete candle
        df_copy.loc[df_copy.index[-1], 'atr'] = df_copy['atr'].iloc[-2] if len(df_copy) > 1 else np.nan
        # Add a flag to indicate this ATR is carried forward
        df_copy['atr_projected'] = False
        df_copy.loc[df_copy.index[-1], 'atr_projected'] = True
    else:
        # Normal ATR calculation if all candles are complete
        df_copy['atr'] = df_copy['true_range'].rolling(window=period).mean()
        df_copy['atr_projected'] = False
    
    # Clean up temporary columns
    df_copy.drop(['prev_close', 'tr1', 'tr2', 'tr3'], axis=1, inplace=True)
    
    return df_copy

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

# --- NEW Pattern Rules Based on Your Specifications ---
pattern_rules = {
    "base": {
        1: {
            "max_range_atr": 1.0
        },
        2: {
            "max_range_atr": 1.1,
            "max_close_diff_atr": 0.25,
            "no_new_extreme_atr": 0.2
        },
        3: {
            "max_range_atr": 1.2,
            "max_extremes_atr": 0.4,
            "max_close_span_atr": 0.3
        },
        4: {
            "max_range_atr": 1.3,
            "max_extremes_atr": 0.5,
            "max_net_move_atr": 0.2
        },
        5: {
            "max_range_atr": 1.4,
            "max_extremes_atr": 0.6,
            "max_net_move_atr": 0.3
        },
        6: {
            "max_range_atr": 1.5,
            "max_extremes_atr": 0.7,
            "max_net_move_atr": 0.3
        }
    },
    "rally": {
        1: {
            "min_range_atr": 1.0,
            "close_upper_pct": 0.30  # Close in upper 30%
        },
        2: {
            "min_range_atr": 1.0,
            "higher_high_low": True,
            "min_net_move_atr": 0.8
        },
        3: {
            "min_bars_range_atr": {"count": 2, "min": 1.0},
            "hh_hl_sequence": True,
            "min_net_move_atr": 1.2
        },
        4: {
            "min_bars_range_atr": {"count": 3, "min": 1.0},
            "hh_hl_sequence": True,
            "min_net_move_atr": 1.5,
            "no_bearish_engulfing": True
        },
        5: {
            "min_bars_range_atr": {"count": 4, "min": 1.0},
            "hh_hl_sequence": True,
            "min_net_move_atr": 2.0,
            "final_close_upper_pct": 0.40
        },
        6: {
            "min_bars_range_atr": {"count": 5, "min": 1.0},
            "hh_hl_sequence": True,
            "min_net_move_atr": 2.5,
            "monotonic_closes": True
        }
    },
    "drop": {
        1: {
            "min_range_atr": 1.0,
            "close_lower_pct": 0.30  # Close in lower 30%
        },
        2: {
            "min_range_atr": 1.0,
            "lower_high_low": True,
            "min_net_move_atr": 0.8
        },
        3: {
            "min_bars_range_atr": {"count": 2, "min": 1.0},
            "lh_ll_sequence": True,
            "min_net_move_atr": 1.2
        },
        4: {
            "min_bars_range_atr": {"count": 3, "min": 1.0},
            "lh_ll_sequence": True,
            "min_net_move_atr": 1.5,
            "no_bullish_engulfing": True
        },
        5: {
            "min_bars_range_atr": {"count": 4, "min": 1.0},
            "lh_ll_sequence": True,
            "min_net_move_atr": 2.0,
            "final_close_lower_pct": 0.40
        },
        6: {
            "min_bars_range_atr": {"count": 5, "min": 1.0},
            "lh_ll_sequence": True,
            "min_net_move_atr": 2.5,
            "monotonic_closes": True
        }
    }
}

# --- Helper Functions ---
def range_atr(high, low, atr):
    return (high - low) / atr if atr > 0 else 0

def net_move_atr(candles, atr):
    return abs(candles[-1]["close"] - candles[0]["open"]) / atr if atr > 0 else 0

def close_span_atr(candles, atr):
    closes = [c["close"] for c in candles]
    return (max(closes) - min(closes)) / atr if atr > 0 else 0

def extremes_atr(candles, atr):
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]
    return (max(highs) - min(lows)) / atr if atr > 0 else 0

def check_hh_hl_sequence(candles):
    """Check for higher highs and higher lows sequence"""
    for i in range(1, len(candles)):
        if candles[i]["high"] <= candles[i-1]["high"] or candles[i]["low"] <= candles[i-1]["low"]:
            return False
    return True

def check_lh_ll_sequence(candles):
    """Check for lower highs and lower lows sequence"""
    for i in range(1, len(candles)):
        if candles[i]["high"] >= candles[i-1]["high"] or candles[i]["low"] >= candles[i-1]["low"]:
            return False
    return True

def check_monotonic_closes(candles, direction="up"):
    """Check for monotonic rising or falling closes"""
    closes = [c["close"] for c in candles]
    if direction == "up":
        return all(closes[i] > closes[i-1] for i in range(1, len(closes)))
    else:
        return all(closes[i] < closes[i-1] for i in range(1, len(closes)))

def check_bearish_engulfing(candles):
    """Check if any candle is bearish engulfing"""
    for i in range(1, len(candles)):
        if (candles[i]["open"] > candles[i-1]["close"] and 
            candles[i]["close"] < candles[i-1]["open"]):
            return True
    return False

def check_bullish_engulfing(candles):
    """Check if any candle is bullish engulfing"""
    for i in range(1, len(candles)):
        if (candles[i]["open"] < candles[i-1]["close"] and 
            candles[i]["close"] > candles[i-1]["open"]):
            return True
    return False

# --- Core Validation with New Logic ---
def validate_pattern_detailed(candles, atr, pattern):
    n = len(candles)
    if n < 1 or n > 6:
        return False, f"Invalid number of candles: {n}. Must be between 1 and 6.", {}
    
    rules = pattern_rules[pattern].get(n)
    if not rules:
        return False, f"No rules for {pattern} with {n} candles", {}
    
    results = {}
    
    if pattern == "base":
        # Base pattern validation
        if "max_range_atr" in rules:
            results['range_check'] = all(range_atr(c["high"], c["low"], atr) <= rules["max_range_atr"] for c in candles)
        
        if n == 2:
            # Closes within ±0.25 ATR
            close_diff = abs(candles[1]["close"] - candles[0]["close"]) / atr
            results['close_diff'] = close_diff <= rules["max_close_diff_atr"]
            
            # No new high/low beyond first bar ±0.2 ATR
            new_high = (candles[1]["high"] - candles[0]["high"]) / atr
            new_low = (candles[0]["low"] - candles[1]["low"]) / atr
            results['no_new_extreme'] = (new_high <= rules["no_new_extreme_atr"] and 
                                        new_low <= rules["no_new_extreme_atr"])
        
        if n >= 3:
            if "max_extremes_atr" in rules:
                results['extremes'] = extremes_atr(candles, atr) <= rules["max_extremes_atr"]
            if "max_close_span_atr" in rules:
                results['close_span'] = close_span_atr(candles, atr) <= rules["max_close_span_atr"]
            if "max_net_move_atr" in rules:
                results['net_move'] = net_move_atr(candles, atr) <= rules["max_net_move_atr"]
    
    elif pattern == "rally":
        # Rally pattern validation
        if n == 1:
            # Range check
            results['range_check'] = range_atr(candles[0]["high"], candles[0]["low"], atr) >= rules["min_range_atr"]
            # Close in upper 30%
            candle_range = candles[0]["high"] - candles[0]["low"]
            if candle_range > 0:
                close_position = (candles[0]["close"] - candles[0]["low"]) / candle_range
                results['close_position'] = close_position >= (1 - rules["close_upper_pct"])
        
        elif n == 2:
            # Each range >= 1.0 ATR
            results['range_check'] = all(range_atr(c["high"], c["low"], atr) >= rules["min_range_atr"] for c in candles)
            # Higher high & higher low
            results['hh_hl'] = (candles[1]["high"] > candles[0]["high"] and 
                               candles[1]["low"] > candles[0]["low"])
            # Net move
            results['net_move'] = net_move_atr(candles, atr) >= rules["min_net_move_atr"]
        
        else:  # n >= 3
            # Minimum bars with range >= threshold
            if "min_bars_range_atr" in rules:
                bars_meeting = sum(1 for c in candles if range_atr(c["high"], c["low"], atr) >= rules["min_bars_range_atr"]["min"])
                results['min_bars_range'] = bars_meeting >= rules["min_bars_range_atr"]["count"]
            
            # HH & HL sequence
            if "hh_hl_sequence" in rules:
                results['hh_hl_sequence'] = check_hh_hl_sequence(candles)
            
            # Net move
            results['net_move'] = net_move_atr(candles, atr) >= rules["min_net_move_atr"]
            
            # No bearish engulfing (n=4)
            if "no_bearish_engulfing" in rules:
                results['no_bearish_engulfing'] = not check_bearish_engulfing(candles)
            
            # Final close position (n=5)
            if "final_close_upper_pct" in rules:
                final_range = candles[-1]["high"] - candles[-1]["low"]
                if final_range > 0:
                    close_position = (candles[-1]["close"] - candles[-1]["low"]) / final_range
                    results['final_close_position'] = close_position >= (1 - rules["final_close_upper_pct"])
            
            # Monotonic closes (n=6)
            if "monotonic_closes" in rules:
                results['monotonic_closes'] = check_monotonic_closes(candles, direction="up")
    
    elif pattern == "drop":
        # Drop pattern validation
        if n == 1:
            # Range check
            results['range_check'] = range_atr(candles[0]["high"], candles[0]["low"], atr) >= rules["min_range_atr"]
            # Close in lower 30%
            candle_range = candles[0]["high"] - candles[0]["low"]
            if candle_range > 0:
                close_position = (candles[0]["close"] - candles[0]["low"]) / candle_range
                results['close_position'] = close_position <= rules["close_lower_pct"]
        
        elif n == 2:
            # Each range >= 1.0 ATR
            results['range_check'] = all(range_atr(c["high"], c["low"], atr) >= rules["min_range_atr"] for c in candles)
            # Lower high & lower low
            results['lh_ll'] = (candles[1]["high"] < candles[0]["high"] and 
                               candles[1]["low"] < candles[0]["low"])
            # Net move (for drop: first open - last close)
            net_move = abs(candles[0]["open"] - candles[-1]["close"]) / atr
            results['net_move'] = net_move >= rules["min_net_move_atr"]
        
        else:  # n >= 3
            # Minimum bars with range >= threshold
            if "min_bars_range_atr" in rules:
                bars_meeting = sum(1 for c in candles if range_atr(c["high"], c["low"], atr) >= rules["min_bars_range_atr"]["min"])
                results['min_bars_range'] = bars_meeting >= rules["min_bars_range_atr"]["count"]
            
            # LH & LL sequence
            if "lh_ll_sequence" in rules:
                results['lh_ll_sequence'] = check_lh_ll_sequence(candles)
            
            # Net move (for drop: first open - last close)
            net_move = abs(candles[0]["open"] - candles[-1]["close"]) / atr
            results['net_move'] = net_move >= rules["min_net_move_atr"]
            
            # No bullish engulfing (n=4)
            if "no_bullish_engulfing" in rules:
                results['no_bullish_engulfing'] = not check_bullish_engulfing(candles)
            
            # Final close position (n=5)
            if "final_close_lower_pct" in rules:
                final_range = candles[-1]["high"] - candles[-1]["low"]
                if final_range > 0:
                    close_position = (candles[-1]["close"] - candles[-1]["low"]) / final_range
                    results['final_close_position'] = close_position <= rules["final_close_lower_pct"]
            
            # Monotonic closes (n=6)
            if "monotonic_closes" in rules:
                results['monotonic_closes'] = check_monotonic_closes(candles, direction="down")
    
    overall = all(results.values())
    return overall, "Pattern validation passed" if overall else "Pattern validation failed", results

# --- Enhanced Plot function ---
def plot_combined_chart(df, selected_candles_df=None, show_atr=True):
    """Create combined chart with price and ATR"""
    # Show last 7 days of data for ATR graph
    last_7_days = df.tail(42)  # Approximately 7 days of 4H candles
    
    rows = 2 if show_atr else 1
    row_heights = [0.65, 0.35] if show_atr else [1.0]
    
    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=row_heights,
        subplot_titles=('Pattern Analysis Dashboard', 'ATR (21 Period) - Last 7 Days') if show_atr else ('Pattern Analysis Dashboard',)
    )
    
    # Price chart
    fig.add_trace(go.Candlestick(
        x=df['datetime_ist'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Price',
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350'
    ), row=1, col=1)
    
    # Mark incomplete candles
    if 'is_complete' in df.columns:
        incomplete_df = df[~df['is_complete']]
        if not incomplete_df.empty:
            fig.add_trace(go.Scatter(
                x=incomplete_df['datetime_ist'],
                y=incomplete_df['high'] * 1.005,
                mode='markers+text',
                marker=dict(symbol='x', size=12, color='red'),
                text='Forming',
                textposition="top center",
                name='Incomplete',
                showlegend=True
            ), row=1, col=1)
    
    # Selected candles markers
    if selected_candles_df is not None and not selected_candles_df.empty:
        fig.add_trace(go.Scatter(
            x=selected_candles_df['datetime_ist'],
            y=selected_candles_df['high'] * 1.002,
            mode='markers',
            marker=dict(symbol='triangle-down', size=15, color='#ff9800'),
            name='Selected',
            showlegend=True
        ), row=1, col=1)
    
    # ATR chart - Last 7 days only
    if show_atr and 'atr' in last_7_days.columns:
        # Main ATR line
        fig.add_trace(go.Scatter(
            x=last_7_days['datetime_ist'],
            y=last_7_days['atr'],
            mode='lines',
            name='ATR',
            line=dict(color='#2196f3', width=2),
            showlegend=False
        ), row=2, col=1)
        
        # Mark projected ATR points
        if 'atr_projected' in last_7_days.columns:
            projected_df = last_7_days[last_7_days['atr_projected']]
            if not projected_df.empty:
                fig.add_trace(go.Scatter(
                    x=projected_df['datetime_ist'],
                    y=projected_df['atr'],
                    mode='markers',
                    marker=dict(symbol='circle-open', size=8, color='orange'),
                    name='Projected',
                    showlegend=True
                ), row=2, col=1)
        
        # Add current ATR line
        if not last_7_days['atr'].isna().all():
            current_atr = last_7_days['atr'].iloc[-1] if not last_7_days['atr_projected'].iloc[-1] else last_7_days['atr'].iloc[-2]
            fig.add_hline(
                y=current_atr, 
                line_dash="dash", 
                line_color="#ff5722",
                annotation_text=f"Current: {current_atr:.4f}",
                annotation_position="right",
                row=2, col=1
            )
    
    # Update layout
    fig.update_layout(
        height=650,
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        hovermode='x unified',
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    fig.update_xaxes(title_text="Time (IST)", row=rows, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    if show_atr:
        fig.update_yaxes(title_text="ATR", row=2, col=1)
    
    return fig

# --- Result Display Component ---
def display_validation_results(is_valid, message, pattern, details=None):
    """Display validation results with details"""
    pattern_emoji = {"rally": "📈", "drop": "📉", "base": "🔄"}
    pattern_name = {"rally": "Rally (Bullish Impulse)", "drop": "Drop (Bearish Impulse)", "base": "Base (Consolidation Zone)"}
    
    if is_valid:
        st.markdown(f"""
        <div class="success-box">
            <h3>{pattern_emoji[pattern]} {pattern_name[pattern]}: VALID ✅</h3>
            <p>{message}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="error-box">
            <h3>{pattern_emoji[pattern]} {pattern_name[pattern]}: INVALID ❌</h3>
            <p>{message}</p>
        </div>
        """, unsafe_allow_html=True)
    
    if details:
        with st.expander("📋 Detailed Validation Results"):
            cols = st.columns(2)
            for i, (check, result) in enumerate(details.items()):
                col = cols[i % 2]
                icon = "✅" if result else "❌"
                check_name = check.replace('_', ' ').title()
                col.markdown(f"{icon} **{check_name}**: {'Passed' if result else 'Failed'}")

# --- Metrics Display ---
def display_pattern_metrics(df, selected_candles, atr, incomplete_warning=False):
    """Display key metrics for pattern analysis"""
    st.markdown("### 📊 Pattern Metrics")
    
    if incomplete_warning:
        st.markdown("""
        <div class="warning-box">
            ⚠️ <b>Note:</b> The current candle is still forming. ATR calculation excludes incomplete candles.
        </div>
        """, unsafe_allow_html=True)
    
    if selected_candles:
        cols = st.columns(4)
        
        # Calculate metrics
        ranges = [(c["high"] - c["low"]) / atr for c in selected_candles]
        avg_range = np.mean(ranges)
        net_move = net_move_atr(selected_candles, atr)
        n_candles = len(selected_candles)
        
        cols[0].metric("Candles", n_candles)
        cols[1].metric("Avg Range (ATR)", f"{avg_range:.2f}")
        cols[2].metric("Net Move (ATR)", f"{net_move:.2f}")
        cols[3].metric("Current ATR", f"{atr:.4f}")

# --- Streamlit UI ---
st.title("🎯 Advanced Pattern Validator Pro")
st.markdown("### Rally / Drop / Base Pattern Analysis with ATR")

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    symbol = st.text_input("📈 Symbol", value="XAU/USD", help="Enter trading symbol")
    
    pattern = st.selectbox(
        "🎨 Pattern Type", 
        ["rally", "drop", "base"],
        format_func=lambda x: {"rally": "📈 Rally", "drop": "📉 Drop", "base": "🔄 Base"}[x]
    )
    
    st.markdown("---")
    
    mode = st.radio(
        "🎯 Selection Mode",
        ["Automatic (Last N Candles)", "Manual Time Range", "Custom Candle Selection"],
        help="Choose how to select candles for analysis"
    )
    
    st.markdown("---")
    
    # ATR Settings
    st.markdown("### 📊 ATR Settings")
    use_auto_atr = st.checkbox("Auto-detect ATR", value=True)
    if not use_auto_atr:
        current_atr = st.number_input("Manual ATR Value", min_value=0.01, value=0.75, step=0.01)

# Main content
df = None
selected_candles = []

# Create tabs for different views
tab1, tab2, tab3 = st.tabs(["📊 Analysis", "📈 Chart", "📋 Data"])

with tab1:
    if mode == "Automatic (Last N Candles)":
        st.markdown("### 🔄 Automatic Analysis")
        col1, col2 = st.columns([3, 1])
        with col1:
            n_candles = st.slider("Number of candles to analyze", 1, 6, 3, help="Select the last N candles (max 6)")
        with col2:
            analyze_btn = st.button("🚀 Analyze", type="primary", use_container_width=True)
        
        if analyze_btn:
            end_utc = datetime.now(UTC)
            start_utc = end_utc - timedelta(days=10)
            
            with st.spinner("🔄 Fetching data and calculating ATR..."):
                df = fetch_ohlc(symbol, start_utc, end_utc)
            
            if not df.empty:
                st.session_state['df'] = df
                
                # Check for incomplete candle
                has_incomplete = 'is_complete' in df.columns and not df.iloc[-1]['is_complete']
                
                # Get ATR
                if use_auto_atr:
                    if has_incomplete and len(df) > 1:
                        # Use ATR from last complete candle
                        current_atr = df['atr'].iloc[-2]
                        st.info(f"📊 Auto-detected ATR: {current_atr:.4f} (from last complete candle)")
                    else:
                        current_atr = df['atr'].iloc[-1] if not df['atr'].isna().all() else 0.75
                        st.info(f"📊 Auto-detected ATR: {current_atr:.4f}")
                
                # Select candles (only complete ones for validation)
                if has_incomplete:
                    # Exclude the incomplete candle from analysis
                    analysis_df = df[df['is_complete']].tail(n_candles)
                else:
                    analysis_df = df.tail(n_candles)
                
                candles = [
                    dict(open=row.open, high=row.high, low=row.low, close=row.close)
                    for _, row in analysis_df.iterrows()
                ]
                selected_candles = candles
                st.session_state['selected_candles'] = analysis_df
                
                # Validate
                ok, message, details = validate_pattern_detailed(candles, current_atr, pattern)
                display_validation_results(ok, message, pattern, details)
                display_pattern_metrics(df, candles, current_atr, incomplete_warning=has_incomplete)

    elif mode == "Manual Time Range":
        st.markdown("### 🕐 Manual Time Range Selection")
        st.info("⏰ Select time range in IST")
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=datetime.now(IST).date() - timedelta(days=3))
            start_time = st.time_input("Start Time (IST)", value=datetime.now(IST).time())
        with col2:
            end_date = st.date_input("End Date", value=datetime.now(IST).date())
            end_time = st.time_input("End Time (IST)", value=datetime.now(IST).time())
        
        if st.button("🔍 Validate Range", type="primary"):
            start_ist = IST.localize(datetime.combine(start_date, start_time))
            end_ist = IST.localize(datetime.combine(end_date, end_time))
            start_utc = start_ist.astimezone(UTC) - timedelta(days=5)  # Extra data for ATR
            end_utc = end_ist.astimezone(UTC)
            
            with st.spinner("🔄 Fetching data and calculating ATR..."):
                df = fetch_ohlc(symbol, start_utc, end_utc)
            
            if not df.empty:
                st.session_state['df'] = df
                
                # Check for incomplete candle
                has_incomplete = 'is_complete' in df.columns and not df.iloc[-1]['is_complete']
                
                # Get ATR
                if use_auto_atr:
                    if has_incomplete and len(df) > 1:
                        current_atr = df['atr'].iloc[-2]
                        st.info(f"📊 Auto-detected ATR: {current_atr:.4f} (from last complete candle)")
                    else:
                        current_atr = df['atr'].iloc[-1] if not df['atr'].isna().all() else 0.75
                        st.info(f"📊 Auto-detected ATR: {current_atr:.4f}")
                
                # Select candles in range (only complete ones)
                sel = df[(df['datetime_ist'] >= start_ist) & (df['datetime_ist'] <= end_ist) & df['is_complete']].copy()
                
                if not sel.empty and len(sel) <= 6:
                    candles = [
                        dict(open=row.open, high=row.high, low=row.low, close=row.close)
                        for _, row in sel.iterrows()
                    ]
                    selected_candles = candles
                    st.session_state['selected_candles'] = sel
                    
                    # Validate
                    ok, message, details = validate_pattern_detailed(candles, current_atr, pattern)
                    display_validation_results(ok, message, pattern, details)
                    display_pattern_metrics(df, candles, current_atr, incomplete_warning=has_incomplete)
                elif len(sel) > 6:
                    st.warning(f"Selected range contains {len(sel)} candles. Maximum allowed is 6 candles.")
                else:
                    st.warning("No complete candles found in selected range")

    else:  # Custom selection
        st.markdown("### 🎯 Custom Candle Selection")
        
        if st.button("📥 Load Recent Data", type="primary"):
            end_utc = datetime.now(UTC)
            start_utc = end_utc - timedelta(days=15)
            
            with st.spinner("🔄 Loading data..."):
                df = fetch_ohlc(symbol, start_utc, end_utc)
            
            if not df.empty:
                st.session_state['df'] = df
                complete_count = df['is_complete'].sum() if 'is_complete' in df.columns else len(df)
                st.success(f"✅ Loaded {len(df)} candles ({complete_count} complete)")
        
        if 'df' in st.session_state:
            df = st.session_state['df']
            
            # Check for incomplete candle
            has_incomplete = 'is_complete' in df.columns and not df.iloc[-1]['is_complete']
            
            # Get ATR
            if use_auto_atr:
                if has_incomplete and len(df) > 1:
                    current_atr = df['atr'].iloc[-2]
                    st.info(f"📊 Auto-detected ATR: {current_atr:.4f} (from last complete candle)")
                else:
                    current_atr = df['atr'].iloc[-1] if not df['atr'].isna().all() else 0.75
                    st.info(f"📊 Auto-detected ATR: {current_atr:.4f}")
            
            # Create selection dataframe (only complete candles)
            df_complete = df[df['is_complete']] if 'is_complete' in df.columns else df
            df_display = df_complete[['datetime_ist','open','high','low','close']].copy()
            df_display['datetime_ist'] = df_display['datetime_ist'].dt.strftime('%Y-%m-%d %H:%M IST')
            
            st.markdown("#### Select candles (1-6 complete candles only)")
            indices = st.multiselect(
                "Choose candles by index",
                options=df_display.index.tolist(),
                format_func=lambda x: f"{x}: {df_display.loc[x,'datetime_ist']} | O:{df_display.loc[x,'open']:.2f} H:{df_display.loc[x,'high']:.2f} L:{df_display.loc[x,'low']:.2f} C:{df_display.loc[x,'close']:.2f}",
                max_selections=6
            )
            
            if indices and st.button("✅ Validate Selected", type="primary"):
                indices = sorted(indices)
                candles = [
                    dict(open=df.loc[idx,'open'], high=df.loc[idx,'high'], 
                         low=df.loc[idx,'low'], close=df.loc[idx,'close'])
                    for idx in indices
                ]
                selected_candles = candles
                st.session_state['selected_candles'] = df.iloc[indices]
                
                # Validate
                ok, message, details = validate_pattern_detailed(candles, current_atr, pattern)
                display_validation_results(ok, message, pattern, details)
                display_pattern_metrics(df, candles, current_atr, incomplete_warning=has_incomplete)

with tab2:
    st.markdown("### 📈 Interactive Chart")
    
    if 'df' in st.session_state and st.session_state['df'] is not None:
        df = st.session_state['df']
        sel_df = st.session_state.get('selected_candles', None)
        
        # Chart options
        col1, col2 = st.columns([1, 4])
        with col1:
            show_atr = st.checkbox("Show ATR", value=True)
        
        fig = plot_combined_chart(df, sel_df, show_atr=show_atr)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("📊 Load data to view chart")

with tab3:
    st.markdown("### 📋 Data View")
    
    if 'df' in st.session_state and st.session_state['df'] is not None:
        df = st.session_state['df']
        
        # Display options
        col1, col2, col3 = st.columns(3)
        with col1:
            show_last_n = st.number_input("Show last N candles", min_value=5, max_value=50, value=20)
        with col2:
            show_atr_col = st.checkbox("Show ATR column", value=True)
        with col3:
            show_complete_col = st.checkbox("Show Complete Status", value=True)
        
        # Prepare display dataframe
        display_df = df.tail(show_last_n).copy()
        display_df['datetime_ist'] = display_df['datetime_ist'].dt.strftime('%Y-%m-%d %H:%M IST')
        
        cols_to_show = ['datetime_ist','open','high','low','close']
        if show_atr_col and 'atr' in display_df.columns:
            cols_to_show.append('atr')
            display_df['atr'] = display_df['atr'].round(4)
        if show_complete_col and 'is_complete' in display_df.columns:
            cols_to_show.append('is_complete')
            display_df['is_complete'] = display_df['is_complete'].map({True: '✅ Complete', False: '🔄 Forming'})
        
        display_df = display_df[cols_to_show]
        display_df.columns = [col.replace('_', ' ').title() for col in cols_to_show]
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            height=400
        )
        
        # Download button
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"{symbol}_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("📊 Load data to view table")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <small>Pattern Validator Pro v3.0 | Exact Pattern Rules Implementation</small>
    </div>
    """,
    unsafe_allow_html=True
)
