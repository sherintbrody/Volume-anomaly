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
</style>
""", unsafe_allow_html=True)

# --- Timezone Setup ---
IST = pytz.timezone('Asia/Kolkata')
UTC = pytz.UTC

# --- Twelve Data API ---
API_KEY = st.secrets["TWELVE_DATA"]["API_KEY"]
BASE_URL = "https://api.twelvedata.com/time_series"

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
        
        # Calculate ATR
        df = calculate_atr(df, period=21)
    
    return df

# --- ATR Calculation ---
def calculate_atr(df, period=21):
    """Calculate Average True Range"""
    df['prev_close'] = df['close'].shift(1)
    
    # True Range calculation
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['prev_close'])
    df['tr3'] = abs(df['low'] - df['prev_close'])
    df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # ATR calculation
    df['atr'] = df['true_range'].rolling(window=period).mean()
    
    # Clean up temporary columns
    df.drop(['prev_close', 'tr1', 'tr2', 'tr3'], axis=1, inplace=True)
    
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

# --- Optimized Helper Functions ---
@st.cache_data
def range_atr(high, low, atr):
    return (high - low) / atr

@st.cache_data
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

# --- Core Validation (with detailed results) ---
def validate_pattern_detailed(candles, atr, pattern):
    n = len(candles)
    rules = pattern_rules[pattern].get(n)
    if not rules:
        return False, f"No rules for {pattern} with {n} candles", {}
    
    results = {}
    
    if pattern == "base":
        results['range_check'] = all(range_atr(c["high"], c["low"], atr) <= rules["max_range_atr"] for c in candles)
        if "max_close_span_atr" in rules:
            results['close_span'] = close_span_atr(candles, atr) <= rules["max_close_span_atr"]
        if "max_highlow_span_atr" in rules:
            results['highlow_span'] = highlow_span_atr(candles, atr) <= rules["max_highlow_span_atr"]
        if "max_midpoint_span_atr" in rules:
            results['midpoint_span'] = midpoint_span_atr(candles, atr) <= rules["max_midpoint_span_atr"]
        add = pattern_rules["base"]["additional"]
        results['range_diff'] = highlow_span_atr(candles, atr) - range_atr(candles[0]["high"], candles[0]["low"], atr) <= add["max_range_diff_atr"]
        results['no_extreme'] = no_new_extreme(candles, atr)
        results['midpoints_within'] = midpoint_span_atr(candles, atr) <= add["midpoints_within_atr"]
    else:
        results['range_check'] = all(range_atr(c["high"], c["low"], atr) >= rules["min_range_atr"] for c in candles)
        results['net_move'] = net_move_atr(candles, atr) >= rules["min_net_move_atr"]
        highs = [c["high"] for c in candles]
        lows  = [c["low"]  for c in candles]
        if pattern == "rally":
            results['structure'] = all(highs[i] > highs[i-1] and lows[i] > lows[i-1] for i in range(1, n))
            if "final_close_pct" in rules:
                rng = candles[-1]["high"] - candles[-1]["low"]
                results['final_close'] = (candles[-1]["close"] - candles[-1]["low"]) / rng >= rules["final_close_pct"]
        else:
            results['structure'] = all(highs[i] < highs[i-1] and lows[i] < lows[i-1] for i in range(1, n))
            if "final_close_pct" in rules:
                rng = candles[-1]["high"] - candles[-1]["low"]
                results['final_close'] = (candles[-1]["high"] - candles[-1]["close"]) / rng >= rules["final_close_pct"]
        add = pattern_rules[pattern]["additional"]
        if add.get("no_inside_final"):
            results['no_inside'] = not is_inside_bar(candles[-1], candles[-2])
        if add.get("final_close_upper_pct") and pattern=="rally":
            rng = candles[-1]["high"] - candles[-1]["low"]
            pct = (candles[-1]["high"] - candles[-1]["close"]) / rng
            results['close_upper'] = pct <= add["final_close_upper_pct"]
        if add.get("final_close_lower_pct") and pattern=="drop":
            rng = candles[-1]["high"] - candles[-1]["low"]
            pct = (candles[-1]["close"] - candles[-1]["low"]) / rng
            results['close_lower'] = pct <= add["final_close_lower_pct"]
        results['monotonic'] = monotonic_closes(candles,
                                       direction="up" if pattern=="rally" else "down",
                                       max_retrace_atr=add["max_retrace_atr"],
                                       atr=atr)
    
    overall = all(results.values())
    return overall, "passed" if overall else "failed", results

# --- Enhanced Plot function ---
def plot_combined_chart(df, selected_candles_df=None, show_atr=True):
    """Create combined chart with price and ATR"""
    rows = 2 if show_atr else 1
    row_heights = [0.7, 0.3] if show_atr else [1.0]
    
    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
        subplot_titles=('Price Action', 'ATR (21 Period)') if show_atr else ('Price Action',)
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
    
    # ATR chart
    if show_atr and 'atr' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['datetime_ist'],
            y=df['atr'],
            mode='lines',
            name='ATR',
            line=dict(color='#2196f3', width=2),
            showlegend=False
        ), row=2, col=1)
        
        # Add current ATR line
        if not df['atr'].isna().all():
            current_atr = df['atr'].iloc[-1]
            fig.add_hline(y=current_atr, line_dash="dash", line_color="#ff5722", 
                         annotation_text=f"Current: {current_atr:.2f}", row=2, col=1)
    
    # Update layout
    fig.update_layout(
        title=dict(text="Pattern Analysis Dashboard", font=dict(size=20)),
        height=700,
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    fig.update_xaxes(title_text="Time (IST)", row=rows, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    if show_atr:
        fig.update_yaxes(title_text="ATR Value", row=2, col=1)
    
    return fig

# --- Result Display Component ---
def display_validation_results(is_valid, message, pattern, details=None):
    """Display validation results with details"""
    if is_valid:
        st.markdown(f"""
        <div class="success-box">
            <h3>✅ {pattern.title()} Pattern: VALID</h3>
            <p>{message}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="error-box">
            <h3>❌ {pattern.title()} Pattern: INVALID</h3>
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
def display_pattern_metrics(df, selected_candles, atr):
    """Display key metrics for pattern analysis"""
    st.markdown("### 📊 Pattern Metrics")
    
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
        cols[3].metric("Current ATR", f"{atr:.2f}")

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
    
    # Display current time
    current_time_ist = datetime.now(IST)
    st.markdown(f"""
    <div class="metric-card">
        <small>Current Time (IST)</small><br>
        <strong>{current_time_ist.strftime('%Y-%m-%d %H:%M:%S')}</strong>
    </div>
    """, unsafe_allow_html=True)

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
            n_candles = st.slider("Number of candles to analyze", 1, 6, 3, help="Select the last N candles")
        with col2:
            analyze_btn = st.button("🚀 Analyze", type="primary", use_container_width=True)
        
        if analyze_btn:
            end_utc = datetime.now(UTC)
            start_utc = end_utc - timedelta(days=10)
            
            with st.spinner("🔄 Fetching data and calculating ATR..."):
                df = fetch_ohlc(symbol, start_utc, end_utc)
            
            if not df.empty:
                st.session_state['df'] = df
                
                # Get ATR
                if use_auto_atr:
                    current_atr = df['atr'].iloc[-1] if not df['atr'].isna().all() else 0.75
                    st.info(f"📊 Auto-detected ATR: {current_atr:.2f}")
                
                # Select candles
                candles = [
                    dict(open=row.open, high=row.high, low=row.low, close=row.close)
                    for _, row in df.tail(n_candles).iterrows()
                ]
                selected_candles = candles
                st.session_state['selected_candles'] = df.tail(n_candles)
                
                # Validate
                ok, message, details = validate_pattern_detailed(candles, current_atr, pattern)
                display_validation_results(ok, message, pattern, details)
                display_pattern_metrics(df, candles, current_atr)

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
                
                # Get ATR
                if use_auto_atr:
                    current_atr = df['atr'].iloc[-1] if not df['atr'].isna().all() else 0.75
                    st.info(f"📊 Auto-detected ATR: {current_atr:.2f}")
                
                # Select candles in range
                sel = df[(df['datetime_ist'] >= start_ist) & (df['datetime_ist'] <= end_ist)].copy()
                
                if not sel.empty:
                    candles = [
                        dict(open=row.open, high=row.high, low=row.low, close=row.close)
                        for _, row in sel.iterrows()
                    ]
                    selected_candles = candles
                    st.session_state['selected_candles'] = sel
                    
                    # Validate
                    ok, message, details = validate_pattern_detailed(candles, current_atr, pattern)
                    display_validation_results(ok, message, pattern, details)
                    display_pattern_metrics(df, candles, current_atr)
                else:
                    st.warning("No candles found in selected range")

    else:  # Custom selection
        st.markdown("### 🎯 Custom Candle Selection")
        
        if st.button("📥 Load Recent Data", type="primary"):
            end_utc = datetime.now(UTC)
            start_utc = end_utc - timedelta(days=15)
            
            with st.spinner("🔄 Loading data..."):
                df = fetch_ohlc(symbol, start_utc, end_utc)
            
            if not df.empty:
                st.session_state['df'] = df
                st.success(f"✅ Loaded {len(df)} candles")
        
        if 'df' in st.session_state:
            df = st.session_state['df']
            
            # Get ATR
            if use_auto_atr:
                current_atr = df['atr'].iloc[-1] if not df['atr'].isna().all() else 0.75
                st.info(f"📊 Auto-detected ATR: {current_atr:.2f}")
            
            # Create selection dataframe
            df_display = df[['datetime_ist','open','high','low','close']].copy()
            df_display['datetime_ist'] = df_display['datetime_ist'].dt.strftime('%Y-%m-%d %H:%M IST')
            
            st.markdown("#### Select candles (1-6 candles)")
            indices = st.multiselect(
                "Choose candles by index",
                options=df_display.index.tolist(),
                format_func=lambda x: f"{x}: {df_display.loc[x,'datetime_ist']} | O:{df_display.loc[x,'open']:.2f} H:{df_display.loc[x,'high']:.2f} L:{df_display.loc[x,'low']:.2f} C:{df_display.loc[x,'close']:.2f}"
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
                display_pattern_metrics(df, candles, current_atr)

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
        col1, col2 = st.columns(2)
        with col1:
            show_last_n = st.number_input("Show last N candles", min_value=5, max_value=50, value=20)
        with col2:
            show_atr_col = st.checkbox("Show ATR column", value=True)
        
        # Prepare display dataframe
        display_df = df.tail(show_last_n).copy()
        display_df['datetime_ist'] = display_df['datetime_ist'].dt.strftime('%Y-%m-%d %H:%M IST')
        
        cols_to_show = ['datetime_ist','open','high','low','close']
        if show_atr_col and 'atr' in display_df.columns:
            cols_to_show.append('atr')
            display_df['atr'] = display_df['atr'].round(2)
        
        display_df = display_df[cols_to_show]
        display_df.columns = ['Time (IST)','Open','High','Low','Close'] + (['ATR'] if 'atr' in cols_to_show else [])
        display_df = display_df.round(2)
        
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
        <small>Pattern Validator Pro v2.0 | Real-time market analysis with ATR</small>
    </div>
    """,
    unsafe_allow_html=True
)
