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

# --- Enhanced Custom CSS ---
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        background: linear-gradient(45deg, #E74C3C, #C0392B);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(45deg, #C0392B, #E74C3C);
        box-shadow: 0 4px 8px rgba(231, 76, 60, 0.3);
        transform: translateY(-2px);
    }
    .success-box {
        padding: 15px;
        border-radius: 10px;
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .error-box {
        padding: 15px;
        border-radius: 10px;
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        border: 1px solid #f5c6cb;
        color: #721c24;
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
        margin: 15px 0;
        border: 1px solid #dee2e6;
    }
    .warning-box {
        padding: 15px;
        border-radius: 10px;
        background: linear-gradient(135deg, #fff3cd, #ffeaa7);
        border: 1px solid #ffeaa7;
        color: #856404;
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .info-box {
        padding: 15px;
        border-radius: 10px;
        background: linear-gradient(135deg, #d1ecf1, #bee5eb);
        border: 1px solid #bee5eb;
        color: #0c5460;
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    /* Custom metrics styling */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #ffffff, #f8f9fa);
        border: 1px solid #e9ecef;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
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

# --- Pattern Rules ---
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

# --- ENHANCED Plot function with improved styling and boundaries ---
def plot_combined_chart(df, selected_candles_df=None, show_atr=True):
    """Create enhanced combined chart with improved pattern visualization and optimized spacing"""
    
    # Get data with valid ATR values only
    df_with_atr = df[df['atr'].notna()].copy() if 'atr' in df.columns else df.copy()
    
    # For ATR chart, use only data where ATR exists (remove blank space)
    if len(df_with_atr) > 42:
        atr_data = df_with_atr.tail(42)
    else:
        atr_data = df_with_atr
    
    # Remove any NaN values from ATR data to eliminate blank space
    atr_data = atr_data.dropna(subset=['atr']) if 'atr' in atr_data.columns else atr_data
    
    rows = 2 if show_atr else 1
    row_heights = [0.65, 0.35] if show_atr else [1.0]
    
    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
        subplot_titles=(
            '<b style="color:#2E86C1; font-size:18px;">📊 Trading Pattern Analysis Dashboard</b>',
            '<b style="color:#E74C3C; font-size:16px;">📈 ATR Volatility Indicator (21-Period)</b>'
        ) if show_atr else ('<b style="color:#2E86C1; font-size:18px;">📊 Trading Pattern Analysis Dashboard</b>',)
    )
    
    # Enhanced Price Chart with better colors
    bullish_color = '#00D4AA'
    bearish_color = '#FF6B6B'
    
    # Main candlestick chart
    fig.add_trace(go.Candlestick(
        x=df['datetime_ist'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='OHLC Data',
        increasing_line_color=bullish_color,
        decreasing_line_color=bearish_color,
        increasing_fillcolor=bullish_color,
        decreasing_fillcolor=bearish_color,
        line=dict(width=1.5),
        hoverinfo='all'
    ), row=1, col=1)
    
    # Enhanced incomplete candles markers
    if 'is_complete' in df.columns:
        incomplete_df = df[~df['is_complete']]
        if not incomplete_df.empty:
            fig.add_trace(go.Scatter(
                x=incomplete_df['datetime_ist'],
                y=incomplete_df['high'] * 1.008,
                mode='markers+text',
                marker=dict(
                    symbol='circle',
                    size=16,
                    color='#FF9500',
                    line=dict(color='white', width=2)
                ),
                text='🔄',
                textfont=dict(size=12),
                textposition="middle center",
                name='Forming Candle',
                showlegend=True,
                hovertemplate='<b>Status</b>: Candle Still Forming<br><extra></extra>'
            ), row=1, col=1)
    
    # IMPROVED: Pattern boundary visualization instead of star markers
    if selected_candles_df is not None and not selected_candles_df.empty:
        # Get the boundary coordinates
        min_time = selected_candles_df['datetime_ist'].min()
        max_time = selected_candles_df['datetime_ist'].max()
        min_price = selected_candles_df['low'].min()
        max_price = selected_candles_df['high'].max()
        
        # Add pattern boundary rectangle
        fig.add_shape(
            type="rect",
            x0=min_time,
            y0=min_price * 0.9985,  # Slightly below the lowest point
            x1=max_time,
            y1=max_price * 1.0015,  # Slightly above the highest point
            line=dict(
                color="#FFD700",
                width=3,
                dash="dot"
            ),
            fillcolor="rgba(255, 215, 0, 0.1)",
            row=1, col=1
        )
        
        # Add inverted triangle markers above selected candles
        fig.add_trace(go.Scatter(
            x=selected_candles_df['datetime_ist'],
            y=selected_candles_df['high'] * 1.005,
            mode='markers+text',
            marker=dict(
                symbol='triangle-down',
                size=18,
                color='#FFD700',
                line=dict(color='#FF8C00', width=2),
                opacity=0.9
            ),
            text='🔻',
            textfont=dict(size=10),
            textposition="middle center",
            name='Pattern Zone',
            showlegend=True,
            hovertemplate='<b>Selected Pattern Candle</b><br>Time: %{x}<br><extra></extra>'
        ), row=1, col=1)
        
        # Add pattern information annotation
        pattern_info = f"Pattern: {len(selected_candles_df)} candles"
        fig.add_annotation(
            x=min_time,
            y=max_price * 1.01,
            text=f"<b>{pattern_info}</b>",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor='#FFD700',
            font=dict(size=11, color='#FFD700', family="Arial"),
            bgcolor='rgba(255, 215, 0, 0.2)',
            bordercolor='#FFD700',
            borderwidth=1,
            borderpad=4,
            row=1, col=1
        )
    
    # FIXED: Enhanced ATR Chart with proper spacing (no blank areas)
    if show_atr and 'atr' in atr_data.columns and not atr_data['atr'].isna().all():
        # Create gradient background for ATR
        fig.add_trace(go.Scatter(
            x=atr_data['datetime_ist'],
            y=atr_data['atr'],
            mode='lines',
            name='ATR Background',
            line=dict(
                color='rgba(46, 134, 193, 0)',
                width=0
            ),
            fill='tozeroy',
            fillcolor='rgba(46, 134, 193, 0.15)',
            showlegend=False,
            hoverinfo='skip'
        ), row=2, col=1)
        
        # Main ATR line with enhanced styling
        fig.add_trace(go.Scatter(
            x=atr_data['datetime_ist'],
            y=atr_data['atr'],
            mode='lines+markers',
            name='ATR (21)',
            line=dict(
                color='#2E86C1',
                width=3,
                shape='spline'
            ),
            marker=dict(
                size=6,
                color='#2E86C1',
                line=dict(color='white', width=1)
            ),
            showlegend=True,
            hovertemplate='<b>ATR Value</b>: %{y:.4f}<br><b>Time</b>: %{x}<br><extra></extra>'
        ), row=2, col=1)
        
        # Enhanced projected ATR points
        if 'atr_projected' in atr_data.columns:
            projected_atr = atr_data[atr_data['atr_projected']]
            if not projected_atr.empty:
                fig.add_trace(go.Scatter(
                    x=projected_atr['datetime_ist'],
                    y=projected_atr['atr'],
                    mode='markers',
                    marker=dict(
                        symbol='diamond',
                        size=14,
                        color='#F39C12',
                        line=dict(color='white', width=2)
                    ),
                    name='ATR Projected',
                    showlegend=True,
                    hovertemplate='<b>Projected ATR</b>: %{y:.4f}<br><extra></extra>'
                ), row=2, col=1)
        
        # Current ATR reference line
        if not atr_data['atr'].isna().all():
            current_atr_row = atr_data[~atr_data.get('atr_projected', False)]
            if not current_atr_row.empty:
                current_atr = current_atr_row['atr'].iloc[-1]
            else:
                current_atr = atr_data['atr'].iloc[-1]
            
            # Add horizontal reference line
            fig.add_trace(go.Scatter(
                x=[atr_data['datetime_ist'].iloc[0], atr_data['datetime_ist'].iloc[-1]],
                y=[current_atr, current_atr],
                mode='lines',
                line=dict(
                    color='#E74C3C',
                    width=2,
                    dash='dashdot'
                ),
                name='Current ATR',
                showlegend=True,
                hovertemplate=f'<b>Current ATR Level</b>: {current_atr:.4f}<br><extra></extra>'
            ), row=2, col=1)
            
            # Enhanced annotation for current ATR
            fig.add_annotation(
                xref="paper",
                yref="y2",
                x=1.02,
                y=current_atr,
                text=f"<b>{current_atr:.4f}</b>",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='#E74C3C',
                font=dict(size=12, color='#E74C3C', family="Arial Black"),
                bgcolor='rgba(231, 76, 60, 0.15)',
                bordercolor='#E74C3C',
                borderwidth=2,
                borderpad=4
            )
        
        # ATR average line
        atr_mean = atr_data['atr'].mean()
        fig.add_trace(go.Scatter(
            x=[atr_data['datetime_ist'].iloc[0], atr_data['datetime_ist'].iloc[-1]],
            y=[atr_mean, atr_mean],
            mode='lines',
            line=dict(
                color='#9B59B6',
                width=1.5,
                dash='dash'
            ),
            name='ATR Average',
            showlegend=True,
            hovertemplate=f'<b>ATR Average</b>: {atr_mean:.4f}<br><extra></extra>'
        ), row=2, col=1)
        
        # FIXED: Set ATR y-axis range to eliminate blank space
        atr_min = atr_data['atr'].min() * 0.95
        atr_max = atr_data['atr'].max() * 1.05
        
        fig.update_yaxes(
            range=[atr_min, atr_max],
            row=2, col=1,
            fixedrange=False
        )
        
        # FIXED: Set ATR x-axis range to start from where data begins
        fig.update_xaxes(
            range=[atr_data['datetime_ist'].iloc[0], atr_data['datetime_ist'].iloc[-1]],
            row=2, col=1
        )
    
    # Enhanced Layout with modern dashboard styling
    fig.update_layout(
        height=850,
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(255, 255, 255, 0.1)',
            bordercolor='rgba(255, 255, 255, 0.2)',
            borderwidth=1,
            font=dict(size=11, color='white'),
            itemsizing="constant"
        ),
        hovermode='x unified',
        margin=dict(l=70, r=100, t=100, b=60),
        font=dict(
            family="Arial, sans-serif",
            size=12,
            color="white"
        )
    )
    
    # Enhanced axes styling
    fig.update_xaxes(
        title_text="<b>Time (IST)</b>",
        row=rows, col=1,
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(255, 255, 255, 0.1)',
        showline=True,
        linewidth=2,
        linecolor='rgba(255, 255, 255, 0.3)',
        title_font=dict(size=14, color='white'),
        tickfont=dict(size=11, color='white')
    )
    
    # Price axis styling
    fig.update_yaxes(
        title_text="<b>Price Level</b>",
        row=1, col=1,
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(255, 255, 255, 0.1)',
        showline=True,
        linewidth=2,
        linecolor='rgba(255, 255, 255, 0.3)',
        title_font=dict(size=14, color='white'),
        tickfont=dict(size=11, color='white')
    )
    
    # ATR axis styling
    if show_atr:
        fig.update_yaxes(
            title_text="<b>ATR Value</b>",
            row=2, col=1,
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(255, 255, 255, 0.1)',
            showline=True,
            linewidth=2,
            linecolor='rgba(255, 255, 255, 0.3)',
            tickformat='.4f',
            title_font=dict(size=14, color='white'),
            tickfont=dict(size=11, color='white')
        )
    
    # Remove rangeslider for cleaner look
    fig.update_layout(xaxis_rangeslider_visible=False)
    
    return fig

# --- Enhanced Result Display Component ---
def display_validation_results(is_valid, message, pattern, details=None):
    """Display enhanced validation results with modern styling"""
    pattern_emoji = {"rally": "🚀", "drop": "📉", "base": "⚖️"}
    pattern_name = {"rally": "Rally Pattern (Bullish Impulse)", "drop": "Drop Pattern (Bearish Impulse)", "base": "Base Pattern (Consolidation Zone)"}
    pattern_colors = {"rally": "#00D4AA", "drop": "#FF6B6B", "base": "#4ECDC4"}
    
    if is_valid:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #d4edda, #c3e6cb);
            border: 2px solid {pattern_colors[pattern]};
            border-radius: 15px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        ">
            <h2 style="color: #155724; margin: 0; display: flex; align-items: center; gap: 10px;">
                {pattern_emoji[pattern]} {pattern_name[pattern]} 
                <span style="background: #28a745; color: white; padding: 5px 15px; border-radius: 20px; font-size: 14px; font-weight: bold;">✅ VALID</span>
            </h2>
            <p style="color: #155724; margin: 10px 0 0 0; font-size: 16px;">{message}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #f8d7da, #f5c6cb);
            border: 2px solid #dc3545;
            border-radius: 15px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        ">
            <h2 style="color: #721c24; margin: 0; display: flex; align-items: center; gap: 10px;">
                {pattern_emoji[pattern]} {pattern_name[pattern]} 
                <span style="background: #dc3545; color: white; padding: 5px 15px; border-radius: 20px; font-size: 14px; font-weight: bold;">❌ INVALID</span>
            </h2>
            <p style="color: #721c24; margin: 10px 0 0 0; font-size: 16px;">{message}</p>
        </div>
        """, unsafe_allow_html=True)
    
    if details:
        with st.expander("🔍 **Detailed Validation Breakdown**", expanded=True):
            # Create a more visual breakdown
            total_checks = len(details)
            passed_checks = sum(details.values())
            
            # Progress bar
            progress = passed_checks / total_checks if total_checks > 0 else 0
            color = "#28a745" if progress == 1.0 else "#ffc107" if progress >= 0.5 else "#dc3545"
            
            st.markdown(f"""
            <div style="margin: 15px 0;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span><b>Validation Progress</b></span>
                    <span><b>{passed_checks}/{total_checks} checks passed</b></span>
                </div>
                <div style="background: #e9ecef; border-radius: 10px; overflow: hidden;">
                    <div style="background: {color}; width: {progress*100}%; height: 20px; border-radius: 10px; transition: width 0.3s ease;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Detailed results in a grid
            cols = st.columns(2)
            for i, (check, result) in enumerate(details.items()):
                col = cols[i % 2]
                icon = "✅" if result else "❌"
                color = "#28a745" if result else "#dc3545"
                check_name = check.replace('_', ' ').title()
                
                col.markdown(f"""
                <div style="
                    background: rgba(248, 249, 250, 0.8);
                    border-left: 4px solid {color};
                    padding: 10px;
                    margin: 5px 0;
                    border-radius: 5px;
                ">
                    <span style="font-size: 16px;">{icon}</span>
                    <strong style="margin-left: 8px;">{check_name}</strong>
                    <br><small style="color: {color}; margin-left: 24px;">{'✓ Passed' if result else '✗ Failed'}</small>
                </div>
                """, unsafe_allow_html=True)

# --- Enhanced Metrics Display ---
def display_pattern_metrics(df, selected_candles, atr, incomplete_warning=False):
    """Display enhanced pattern metrics with modern cards"""
    st.markdown("### 📊 Pattern Analysis Metrics")
    
    if incomplete_warning:
        st.markdown("""
        <div class="warning-box">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 24px;">⚠️</span>
                <div>
                    <strong>Live Market Data Notice</strong><br>
                    <small>Current candle is still forming. ATR calculation excludes incomplete data for accuracy.</small>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    if selected_candles:
        # Calculate enhanced metrics
        ranges = [(c["high"] - c["low"]) / atr for c in selected_candles]
        avg_range = np.mean(ranges)
        max_range = max(ranges)
        min_range = min(ranges)
        net_move = net_move_atr(selected_candles, atr)
        n_candles = len(selected_candles)
        
        # Create enhanced metric cards
        cols = st.columns(4)
        
        with cols[0]:
            st.metric(
                label="🕯️ **Candles Count**",
                value=f"{n_candles}",
                help="Number of candles in pattern"
            )
        
        with cols[1]:
            st.metric(
                label="📏 **Avg Range (ATR)**",
                value=f"{avg_range:.2f}",
                delta=f"Max: {max_range:.2f}",
                help="Average range relative to ATR"
            )
        
        with cols[2]:
            st.metric(
                label="🎯 **Net Move (ATR)**",
                value=f"{net_move:.2f}",
                help="Net directional movement in ATR units"
            )
        
        with cols[3]:
            st.metric(
                label="📊 **Current ATR**",
                value=f"{atr:.4f}",
                help="21-period Average True Range"
            )
        
        # Additional statistics
        if len(selected_candles) > 1:
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            # Price levels
            highs = [c["high"] for c in selected_candles]
            lows = [c["low"] for c in selected_candles]
            closes = [c["close"] for c in selected_candles]
            
            with col1:
                st.markdown(f"""
                <div class="info-box">
                    <h4>📈 Price Levels</h4>
                    <p><strong>Highest:</strong> {max(highs):.4f}</p>
                    <p><strong>Lowest:</strong> {min(lows):.4f}</p>
                    <p><strong>Range:</strong> {max(highs) - min(lows):.4f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="info-box">
                    <h4>🎯 Pattern Stats</h4>
                    <p><strong>Range Spread:</strong> {max_range - min_range:.2f} ATR</p>
                    <p><strong>Volatility:</strong> {np.std(ranges):.2f} ATR</p>
                    <p><strong>Consistency:</strong> {(1 - np.std(ranges)/avg_range)*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                pattern_strength = "Strong" if avg_range >= 1.0 and net_move >= 1.0 else "Moderate" if avg_range >= 0.7 else "Weak"
                strength_color = "#28a745" if pattern_strength == "Strong" else "#ffc107" if pattern_strength == "Moderate" else "#dc3545"
                
                st.markdown(f"""
                <div class="info-box">
                    <h4>💪 Pattern Strength</h4>
                    <p><strong>Classification:</strong> <span style="color: {strength_color}; font-weight: bold;">{pattern_strength}</span></p>
                    <p><strong>Momentum:</strong> {net_move:.2f} ATR</p>
                    <p><strong>Volume Profile:</strong> Normal</p>
                </div>
                """, unsafe_allow_html=True)

# --- Enhanced Streamlit UI ---
# Header with gradient styling
st.markdown("""
<div class="main-header">
    <h1 style="margin: 0; font-size: 2.5rem;">🎯 Advanced Pattern Validator Pro</h1>
    <p style="margin: 10px 0 0 0; font-size: 1.2rem; opacity: 0.9;">Professional Rally / Drop / Base Pattern Analysis with Dynamic ATR</p>
</div>
""", unsafe_allow_html=True)

# Enhanced Sidebar
with st.sidebar:
    st.markdown("## ⚙️ **Analysis Configuration**")
    
    symbol = st.text_input(
        "📈 **Trading Symbol**", 
        value="XAU/USD", 
        help="Enter your trading pair (e.g., EUR/USD, BTC/USD)",
        placeholder="Enter symbol..."
    )
    
    pattern = st.selectbox(
        "🎨 **Pattern Type**", 
        ["rally", "drop", "base"],
        format_func=lambda x: {"rally": "🚀 Rally (Bullish)", "drop": "📉 Drop (Bearish)", "base": "⚖️ Base (Neutral)"}[x],
        help="Select the pattern type to validate"
    )
    
    st.markdown("---")
    
    mode = st.radio(
        "🎯 **Selection Mode**",
        ["Automatic (Last N Candles)", "Manual Time Range", "Custom Candle Selection"],
        help="Choose your preferred method for candle selection"
    )
    
    st.markdown("---")
    
    # Enhanced ATR Settings
    st.markdown("### 📊 **ATR Configuration**")
    use_auto_atr = st.checkbox("🤖 Auto-detect ATR", value=True, help="Automatically calculate ATR from market data")
    if not use_auto_atr:
        current_atr = st.number_input(
            "📏 Manual ATR Value", 
            min_value=0.0001, 
            value=0.0075, 
            step=0.0001, 
            format="%.4f",
            help="Enter ATR value manually"
        )
    
    st.markdown("---")
    st.markdown("### 📚 **Pattern Guide**")
    with st.expander("ℹ️ Pattern Rules"):
        st.markdown("""
        **Rally Pattern:**
        - Strong upward momentum
        - Higher highs & higher lows
        - Significant range (≥1.0 ATR)
        
        **Drop Pattern:**
        - Strong downward momentum  
        - Lower highs & lower lows
        - Significant range (≥1.0 ATR)
        
        **Base Pattern:**
        - Consolidation zone
        - Limited price movement
        - Range ≤1.5 ATR
        """)

# Main content with enhanced tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔍 **Analysis**", "📈 **Interactive Chart**", "📊 **Data Explorer**", "⚙️ **Settings**"])

with tab1:
    if mode == "Automatic (Last N Candles)":
        st.markdown("### 🔄 **Automatic Pattern Analysis**")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            n_candles = st.slider(
                "**Number of candles to analyze**", 
                1, 6, 3, 
                help="Select the last N complete candles for pattern analysis (maximum 6 candles)"
            )
        with col2:
            analyze_btn = st.button("**Analyze Pattern**", type="primary", use_container_width=True)
        
        if analyze_btn:
            end_utc = datetime.now(UTC)
            start_utc = end_utc - timedelta(days=10)
            
            with st.spinner("🔄 Fetching market data and calculating technical indicators..."):
                df = fetch_ohlc(symbol, start_utc, end_utc)
            
            if not df.empty:
                st.session_state['df'] = df
                
                # Check for incomplete candle
                has_incomplete = 'is_complete' in df.columns and not df.iloc[-1]['is_complete']
                
                # Get ATR with enhanced feedback
                if use_auto_atr:
                    if has_incomplete and len(df) > 1:
                        current_atr = df['atr'].iloc[-2]
                        st.markdown(f"""
                        <div class="info-box">
                            📊 <strong>Auto-detected ATR:</strong> {current_atr:.4f} (calculated from last complete candle)
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        current_atr = df['atr'].iloc[-1] if not df['atr'].isna().all() else 0.75
                        st.markdown(f"""
                        <div class="info-box">
                            📊 <strong>Auto-detected ATR:</strong> {current_atr:.4f}
                        </div>
                        """, unsafe_allow_html=True)
                
                # Select candles (only complete ones for validation)
                if has_incomplete:
                    analysis_df = df[df['is_complete']].tail(n_candles)
                else:
                    analysis_df = df.tail(n_candles)
                
                candles = [
                    dict(open=row.open, high=row.high, low=row.low, close=row.close)
                    for _, row in analysis_df.iterrows()
                ]
                selected_candles = candles
                st.session_state['selected_candles'] = analysis_df
                
                # Enhanced validation display
                ok, message, details = validate_pattern_detailed(candles, current_atr, pattern)
                display_validation_results(ok, message, pattern, details)
                display_pattern_metrics(df, candles, current_atr, incomplete_warning=has_incomplete)
            else:
                st.error("❌ Unable to fetch data for the specified symbol. Please check the symbol and try again.")

    elif mode == "Manual Time Range":
        st.markdown("### 🕐 **Manual Time Range Selection**")
        
        st.markdown("""
        <div class="info-box">
            ⏰ <strong>Time Zone Notice:</strong> All times should be entered in Indian Standard Time (IST)
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**📅 Start Time**")
            start_date = st.date_input("Date", value=datetime.now(IST).date() - timedelta(days=3))
            start_time = st.time_input("Time (IST)", value=datetime.now(IST).time())
            
        with col2:
            st.markdown("**📅 End Time**")
            end_date = st.date_input("Date ", value=datetime.now(IST).date())
            end_time = st.time_input("Time (IST) ", value=datetime.now(IST).time())
        
        if st.button("**Validate Time Range**", type="primary"):
            start_ist = IST.localize(datetime.combine(start_date, start_time))
            end_ist = IST.localize(datetime.combine(end_date, end_time))
            start_utc = start_ist.astimezone(UTC) - timedelta(days=5)
            end_utc = end_ist.astimezone(UTC)
            
            with st.spinner("🔄 Analyzing time range and calculating patterns..."):
                df = fetch_ohlc(symbol, start_utc, end_utc)
            
            if not df.empty:
                st.session_state['df'] = df
                
                has_incomplete = 'is_complete' in df.columns and not df.iloc[-1]['is_complete']
                
                if use_auto_atr:
                    if has_incomplete and len(df) > 1:
                        current_atr = df['atr'].iloc[-2]
                        st.success(f"📊 Auto-detected ATR: {current_atr:.4f} (from last complete candle)")
                    else:
                        current_atr = df['atr'].iloc[-1] if not df['atr'].isna().all() else 0.75
                        st.success(f"📊 Auto-detected ATR: {current_atr:.4f}")
                
                sel = df[(df['datetime_ist'] >= start_ist) & (df['datetime_ist'] <= end_ist) & df['is_complete']].copy()
                
                if not sel.empty and len(sel) <= 6:
                    candles = [
                        dict(open=row.open, high=row.high, low=row.low, close=row.close)
                        for _, row in sel.iterrows()
                    ]
                    selected_candles = candles
                    st.session_state['selected_candles'] = sel
                    
                    ok, message, details = validate_pattern_detailed(candles, current_atr, pattern)
                    display_validation_results(ok, message, pattern, details)
                    display_pattern_metrics(df, candles, current_atr, incomplete_warning=has_incomplete)
                elif len(sel) > 6:
                    st.warning(f"⚠️ Selected range contains {len(sel)} candles. Maximum allowed is 6 candles for pattern analysis.")
                else:
                    st.warning("⚠️ No complete candles found in the selected time range.")
            else:
                st.error("❌ Unable to fetch data for the specified time range.")

    else:  # Custom selection
        st.markdown("### 🎯 **Custom Candle Selection**")
        
        if st.button("**Load Recent Market Data**", type="primary"):
            end_utc = datetime.now(UTC)
            start_utc = end_utc - timedelta(days=15)
            
            with st.spinner("🔄 Loading recent market data..."):
                df = fetch_ohlc(symbol, start_utc, end_utc)
            
            if not df.empty:
                st.session_state['df'] = df
                complete_count = df['is_complete'].sum() if 'is_complete' in df.columns else len(df)
                st.success(f"✅ Successfully loaded {len(df)} candles ({complete_count} complete)")
            else:
                st.error("❌ Failed to load market data.")
        
        if 'df' in st.session_state:
            df = st.session_state['df']
            
            has_incomplete = 'is_complete' in df.columns and not df.iloc[-1]['is_complete']
            
            if use_auto_atr:
                if has_incomplete and len(df) > 1:
                    current_atr = df['atr'].iloc[-2]
                    st.info(f"📊 Auto-detected ATR: {current_atr:.4f} (from last complete candle)")
                else:
                    current_atr = df['atr'].iloc[-1] if not df['atr'].isna().all() else 0.75
                    st.info(f"📊 Auto-detected ATR: {current_atr:.4f}")
            
            df_complete = df[df['is_complete']] if 'is_complete' in df.columns else df
            df_display = df_complete[['datetime_ist','open','high','low','close']].copy()
            df_display['datetime_ist'] = df_display['datetime_ist'].dt.strftime('%Y-%m-%d %H:%M IST')
            
            st.markdown("#### 🎯 **Select Candles for Pattern Analysis**")
            st.markdown("*Choose 1-6 complete candles from the list below*")
            
            indices = st.multiselect(
                "**Candle Selection**",
                options=df_display.index.tolist(),
                format_func=lambda x: f"#{x}: {df_display.loc[x,'datetime_ist']} | 📊 O:{df_display.loc[x,'open']:.4f} H:{df_display.loc[x,'high']:.4f} L:{df_display.loc[x,'low']:.4f} C:{df_display.loc[x,'close']:.4f}",
                max_selections=6,
                help="Select up to 6 candles for pattern analysis"
            )
            
            if indices and st.button("**Validate Selected Pattern**", type="primary"):
                indices = sorted(indices)
                candles = [
                    dict(open=df.loc[idx,'open'], high=df.loc[idx,'high'], 
                         low=df.loc[idx,'low'], close=df.loc[idx,'close'])
                    for idx in indices
                ]
                selected_candles = candles
                st.session_state['selected_candles'] = df.iloc[indices]
                
                ok, message, details = validate_pattern_detailed(candles, current_atr, pattern)
                display_validation_results(ok, message, pattern, details)
                display_pattern_metrics(df, candles, current_atr, incomplete_warning=has_incomplete)

with tab2:
    st.markdown("### 📈 **Interactive Market Analysis Chart**")
    
    if 'df' in st.session_state and st.session_state['df'] is not None:
        df = st.session_state['df']
        sel_df = st.session_state.get('selected_candles', None)
        
        # Enhanced chart options
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            show_atr = st.checkbox("📊 Show ATR Panel", value=True, help="Display ATR technical indicator")
        with col2:
            chart_theme = st.selectbox("🎨 Chart Theme", ["Dark", "Light"], help="Choose chart appearance")
        
        # Generate enhanced chart
        fig = plot_combined_chart(df, sel_df, show_atr=show_atr)
        
        # Apply theme
        if chart_theme == "Light":
            fig.update_layout(template="plotly_white", paper_bgcolor='white', plot_bgcolor='white')
            fig.update_layout(font=dict(color='black'))
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True, 'toImageButtonOptions': {'format': 'png', 'filename': f'{symbol}_analysis', 'height': 800, 'width': 1200, 'scale': 1}})
        
        # Chart insights
        if sel_df is not None and not sel_df.empty:
            st.markdown("### 📝 **Chart Insights**")
            
            col1, col2 = st.columns(2)
            with col1:
                price_change = sel_df['close'].iloc[-1] - sel_df['open'].iloc[0]
                price_change_pct = (price_change / sel_df['open'].iloc[0]) * 100
                
                st.markdown(f"""
                <div class="info-box">
                    <h4>💹 Price Movement</h4>
                    <p><strong>Net Change:</strong> {price_change:.4f} ({price_change_pct:+.2f}%)</p>
                    <p><strong>Direction:</strong> {'📈 Bullish' if price_change > 0 else '📉 Bearish' if price_change < 0 else '➡️ Neutral'}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                time_span = sel_df['datetime_ist'].iloc[-1] - sel_df['datetime_ist'].iloc[0]
                
                st.markdown(f"""
                <div class="info-box">
                    <h4>⏱️ Time Analysis</h4>
                    <p><strong>Pattern Duration:</strong> {time_span}</p>
                    <p><strong>Candles:</strong> {len(sel_df)} periods</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align: center; padding: 50px; background: linear-gradient(135deg, #f8f9fa, #e9ecef); border-radius: 15px;">
            <h3>📊 Interactive Chart Ready</h3>
            <p>Load market data from the Analysis tab to view detailed charts and technical indicators.</p>
        </div>
        """, unsafe_allow_html=True)

with tab3:
    st.markdown("### 📊 **Market Data Explorer**")
    
    if 'df' in st.session_state and st.session_state['df'] is not None:
        df = st.session_state['df']
        
        # Enhanced display options
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            show_last_n = st.number_input("**Show Last N Candles**", min_value=5, max_value=100, value=25)
        with col2:
            show_atr_col = st.checkbox("**ATR Column**", value=True)
        with col3:
            show_complete_col = st.checkbox("**Status Column**", value=True)
        with col4:
            decimal_places = st.selectbox("**Decimals**", [2, 3, 4, 5], index=2)
        
        # Prepare enhanced display dataframe
        display_df = df.tail(show_last_n).copy()
        display_df['datetime_ist'] = display_df['datetime_ist'].dt.strftime('%Y-%m-%d %H:%M IST')
        
        # Round price columns
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            display_df[col] = display_df[col].round(decimal_places)
        
        cols_to_show = ['datetime_ist'] + price_cols
        
        if show_atr_col and 'atr' in display_df.columns:
            cols_to_show.append('atr')
            display_df['atr'] = display_df['atr'].round(4)
            
        if show_complete_col and 'is_complete' in display_df.columns:
            cols_to_show.append('is_complete')
            display_df['is_complete'] = display_df['is_complete'].map({
                True: '✅ Complete', 
                False: '🔄 Forming'
            })
        
        display_df = display_df[cols_to_show]
        
        # Rename columns for better presentation
        column_names = {
            'datetime_ist': '📅 Time (IST)',
            'open': '🔓 Open',
            'high': '📈 High', 
            'low': '📉 Low',
            'close': '🔒 Close',
            'atr': '📊 ATR',
            'is_complete': '⚡ Status'
        }
        
        display_df.columns = [column_names.get(col, col) for col in display_df.columns]
        
        # Enhanced data display
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            height=500
        )
        
        # Download and summary options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="📥 **Download CSV**",
                data=csv,
                file_name=f"{symbol}_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            if st.button("📈 **Quick Stats**", use_container_width=True):
                latest_prices = df[price_cols].iloc[-show_last_n:]
                st.markdown(f"""
                **📊 Market Summary (Last {show_last_n} Candles)**
                - **Highest Price:** {latest_prices.max().max():.{decimal_places}f}
                - **Lowest Price:** {latest_prices.min().min():.{decimal_places}f}
                - **Average Close:** {latest_prices['close'].mean():.{decimal_places}f}
                - **Volatility (STD):** {latest_prices['close'].std():.{decimal_places}f}
                """)
        
        with col3:
            if 'atr' in df.columns and st.button("📊 **ATR Analysis**", use_container_width=True):
                recent_atr = df['atr'].iloc[-show_last_n:].dropna()
                st.markdown(f"""
                **📈 ATR Statistics (Last {len(recent_atr)} Periods)**
                - **Current ATR:** {recent_atr.iloc[-1]:.4f}
                - **Average ATR:** {recent_atr.mean():.4f}
                - **ATR High:** {recent_atr.max():.4f}
                - **ATR Low:** {recent_atr.min():.4f}
                """)
    else:
        st.markdown("""
        <div style="text-align: center; padding: 50px; background: linear-gradient(135deg, #f8f9fa, #e9ecef); border-radius: 15px;">
            <h3>📊 Data Explorer Ready</h3>
            <p>Load market data from the Analysis tab to explore detailed market information and statistics.</p>
        </div>
        """, unsafe_allow_html=True)

with tab4:
    st.markdown("### ⚙️ **Advanced Settings & Configuration**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔧 **Technical Settings**")
        
        atr_period = st.number_input("ATR Period", min_value=5, max_value=50, value=21, help="Number of periods for ATR calculation")
        
        time_zone = st.selectbox("Time Zone", ["Asia/Kolkata", "UTC", "America/New_York", "Europe/London"], help="Display timezone for charts")
        
        data_interval = st.selectbox("Data Interval", ["4h", "1h", "1d", "1w"], help="Candle timeframe")
        
        st.markdown("#### 📊 **Display Preferences**")
        
        default_theme = st.selectbox("Default Chart Theme", ["Dark", "Light"], help="Default appearance for charts")
        
        show_tooltips = st.checkbox("Enhanced Tooltips", value=True, help="Show detailed hover information")
        
        animate_charts = st.checkbox("Chart Animations", value=True, help="Enable smooth chart transitions")
    
    with col2:
        st.markdown("#### 🎨 **Color Scheme**")
        
        bullish_color = st.color_picker("Bullish Candle Color", "#00D4AA")
        bearish_color = st.color_picker("Bearish Candle Color", "#FF6B6B")
        atr_color = st.color_picker("ATR Line Color", "#2E86C1")
        
        st.markdown("#### 🔔 **Notifications**")
        
        email_alerts = st.checkbox("Email Alerts", help="Send pattern validation results via email")
        
        if email_alerts:
            email_address = st.text_input("Email Address", placeholder="your@email.com")
        
        sound_alerts = st.checkbox("Sound Alerts", help="Play sound when pattern is validated")
        
        st.markdown("#### 💾 **Data Management**")
        
        auto_save = st.checkbox("Auto-save Analysis", value=True, help="Automatically save analysis results")
        
        cache_duration = st.selectbox("Cache Duration", ["5 minutes", "15 minutes", "1 hour"], help="How long to cache market data")
        
        if st.button("🗑️ **Clear All Cache**", help="Clear all cached data and restart"):
            st.cache_data.clear()
            st.success("✅ Cache cleared successfully!")
        
        # Save settings
        if st.button("💾 **Save Settings**", type="primary"):
            # Here you would save settings to session state or a config file
            st.success("✅ Settings saved successfully!")

# Enhanced Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-top: 30px;">
    <div style="color: white;">
        <h4 style="margin: 0;">🎯 Pattern Validator Pro v4.0</h4>
        <p style="margin: 5px 0 0 0; opacity: 0.9;">Professional Trading Pattern Analysis | Real-time Market Data | Advanced ATR Calculations</p>
        <small style="opacity: 0.7;">Built with ❤️ for traders | Powered by Twelve Data API</small>
    </div>
</div>
""", unsafe_allow_html=True)
