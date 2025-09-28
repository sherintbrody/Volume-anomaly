import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import pytz
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page Config
st.set_page_config(
    page_title="Pattern Validator Pro",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Timezone Setup
IST = pytz.timezone('Asia/Kolkata')
UTC = pytz.UTC

# OANDA API Configuration
API_KEY = st.secrets["API_KEY"]
ACCOUNT_ID = st.secrets["ACCOUNT_ID"]
BASE_URL = "https://api-fxpractice.oanda.com/v3"

# Symbol Mapping
SYMBOL_MAPPING = {
    "NAS100": "NAS100_USD", "US30": "US30_USD", "XAU/USD": "XAU_USD",
    "SPX500": "SPX500_USD", "XAG/USD": "XAG_USD", "EUR/USD": "EUR_USD",
    "GBP/USD": "GBP_USD", "USD/JPY": "USD_JPY", "USD/CHF": "USD_CHF",
    "AUD/USD": "AUD_USD", "USD/CAD": "USD_CAD", "NZD/USD": "NZD_USD"
}

# Pattern Rules
pattern_rules = {
    "base": {
        1: {"max_range_atr": 1.0},
        2: {"max_range_atr": 1.1, "max_close_diff_atr": 0.25, "no_new_extreme_atr": 0.2},
        3: {"max_range_atr": 1.2, "max_extremes_atr": 0.4, "max_close_span_atr": 0.3},
        4: {"max_range_atr": 1.3, "max_extremes_atr": 0.5, "max_net_move_atr": 0.2},
        5: {"max_range_atr": 1.4, "max_extremes_atr": 0.6, "max_net_move_atr": 0.3},
        6: {"max_range_atr": 1.5, "max_extremes_atr": 0.7, "max_net_move_atr": 0.3}
    },
    "rally": {
        1: {"min_range_atr": 1.0, "close_upper_pct": 0.30},
        2: {"min_range_atr": 1.0, "higher_high_low": True, "min_net_move_atr": 0.8},
        3: {"min_bars_range_atr": {"count": 2, "min": 1.0}, "hh_hl_sequence": True, "min_net_move_atr": 1.2},
        4: {"min_bars_range_atr": {"count": 3, "min": 1.0}, "hh_hl_sequence": True, "min_net_move_atr": 1.5, "no_bearish_engulfing": True},
        5: {"min_bars_range_atr": {"count": 4, "min": 1.0}, "hh_hl_sequence": True, "min_net_move_atr": 2.0, "final_close_upper_pct": 0.40},
        6: {"min_bars_range_atr": {"count": 5, "min": 1.0}, "hh_hl_sequence": True, "min_net_move_atr": 2.5, "monotonic_closes": True}
    },
    "drop": {
        1: {"min_range_atr": 1.0, "close_lower_pct": 0.30},
        2: {"min_range_atr": 1.0, "lower_high_low": True, "min_net_move_atr": 0.8},
        3: {"min_bars_range_atr": {"count": 2, "min": 1.0}, "lh_ll_sequence": True, "min_net_move_atr": 1.2},
        4: {"min_bars_range_atr": {"count": 3, "min": 1.0}, "lh_ll_sequence": True, "min_net_move_atr": 1.5, "no_bullish_engulfing": True},
        5: {"min_bars_range_atr": {"count": 4, "min": 1.0}, "lh_ll_sequence": True, "min_net_move_atr": 2.0, "final_close_lower_pct": 0.40},
        6: {"min_bars_range_atr": {"count": 5, "min": 1.0}, "lh_ll_sequence": True, "min_net_move_atr": 2.5, "monotonic_closes": True}
    }
}

# Helper Functions
def convert_symbol_to_oanda(symbol):
    return SYMBOL_MAPPING.get(symbol, symbol.replace("/", "_"))

def is_candle_complete(candle_time, interval_hours=4):
    current_time = datetime.now(UTC)
    candle_end_time = candle_time + timedelta(hours=interval_hours)
    return current_time >= candle_end_time

@st.cache_data(ttl=300)
def fetch_ohlc(symbol, start, end, interval="4h"):
    instrument = convert_symbol_to_oanda(symbol)
    granularity = {"1h": "H1", "4h": "H4", "1d": "D", "1w": "W"}.get(interval, "H4")
    
    url = f"{BASE_URL}/instruments/{instrument}/candles"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    params = {
        "granularity": granularity,
        "from": start.strftime("%Y-%m-%dT%H:%M:%S.000000Z"),
        "to": end.strftime("%Y-%m-%dT%H:%M:%S.000000Z"),
        "price": "M"
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        
        candles = data.get("candles", [])
        if not candles:
            return pd.DataFrame()
        
        df_data = []
        for candle in candles:
            if candle.get("complete", True):
                mid = candle.get("mid", {})
                df_data.append({
                    "datetime": candle["time"],
                    "open": float(mid.get("o", 0)),
                    "high": float(mid.get("h", 0)),
                    "low": float(mid.get("l", 0)),
                    "close": float(mid.get("c", 0)),
                    "complete": candle.get("complete", True)
                })
        
        df = pd.DataFrame(df_data)
        if not df.empty:
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
            df["datetime_ist"] = df["datetime"].dt.tz_convert(IST)
            df = df.sort_values("datetime").reset_index(drop=True)
            
            if len(df) > 0:
                last_candle_time = df.iloc[-1]['datetime']
                df['is_complete'] = df['complete']
                if not is_candle_complete(last_candle_time, 4):
                    df.loc[df.index[-1], 'is_complete'] = False
            
            df = calculate_atr(df, period=21)
        
        return df
    except Exception as e:
        st.error(f"API Error: {e}")
        return pd.DataFrame()

def calculate_atr(df, period=21):
    df_copy = df.copy()
    
    incomplete_candle_exists = len(df_copy) > 0 and 'is_complete' in df_copy.columns and not df_copy.iloc[-1]['is_complete']
    
    df_copy['prev_close'] = df_copy['close'].shift(1)
    df_copy['tr1'] = df_copy['high'] - df_copy['low']
    df_copy['tr2'] = abs(df_copy['high'] - df_copy['prev_close'])
    df_copy['tr3'] = abs(df_copy['low'] - df_copy['prev_close'])
    df_copy['true_range'] = df_copy[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    if incomplete_candle_exists:
        df_copy['atr'] = df_copy['true_range'].iloc[:-1].rolling(window=period).mean()
        if len(df_copy) > 1:
            df_copy.loc[df_copy.index[-1], 'atr'] = df_copy['atr'].iloc[-2] if not pd.isna(df_copy['atr'].iloc[-2]) else np.nan
        df_copy['atr_projected'] = False
        df_copy.loc[df_copy.index[-1], 'atr_projected'] = True
    else:
        df_copy['atr'] = df_copy['true_range'].rolling(window=period).mean()
        df_copy['atr_projected'] = False
    
    df_copy.drop(['prev_close', 'tr1', 'tr2', 'tr3'], axis=1, inplace=True, errors='ignore')
    return df_copy

# Pattern Validation Functions
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
    for i in range(1, len(candles)):
        if candles[i]["high"] <= candles[i-1]["high"] or candles[i]["low"] <= candles[i-1]["low"]:
            return False
    return True

def check_lh_ll_sequence(candles):
    for i in range(1, len(candles)):
        if candles[i]["high"] >= candles[i-1]["high"] or candles[i]["low"] >= candles[i-1]["low"]:
            return False
    return True

def check_monotonic_closes(candles, direction="up"):
    closes = [c["close"] for c in candles]
    if direction == "up":
        return all(closes[i] > closes[i-1] for i in range(1, len(closes)))
    else:
        return all(closes[i] < closes[i-1] for i in range(1, len(closes)))

def check_bearish_engulfing(candles):
    for i in range(1, len(candles)):
        if candles[i]["open"] > candles[i-1]["close"] and candles[i]["close"] < candles[i-1]["open"]:
            return True
    return False

def check_bullish_engulfing(candles):
    for i in range(1, len(candles)):
        if candles[i]["open"] < candles[i-1]["close"] and candles[i]["close"] > candles[i-1]["open"]:
            return True
    return False

def validate_pattern_detailed(candles, atr, pattern):
    n = len(candles)
    if n < 1 or n > 6:
        return False, f"Invalid number of candles: {n}", {}
    
    rules = pattern_rules[pattern].get(n)
    if not rules:
        return False, f"No rules for {pattern} with {n} candles", {}
    
    results = {}
    
    if pattern == "base":
        if n == 1:
            if "max_range_atr" in rules:
                results['range_check'] = all(range_atr(c["high"], c["low"], atr) <= rules["max_range_atr"] for c in candles)
            overall = all(results.values())
        elif n == 2:
            core_criteria = {}
            if "max_range_atr" in rules:
                core_criteria['range_check'] = all(range_atr(c["high"], c["low"], atr) <= rules["max_range_atr"] for c in candles)
                results['range_check'] = core_criteria['range_check']
            close_diff = abs(candles[1]["close"] - candles[0]["close"]) / atr
            core_criteria['close_diff'] = close_diff <= rules["max_close_diff_atr"]
            results['close_diff'] = core_criteria['close_diff']
            new_high = (candles[1]["high"] - candles[0]["high"]) / atr
            new_low = (candles[0]["low"] - candles[1]["low"]) / atr
            core_criteria['no_new_extreme'] = (new_high <= rules["no_new_extreme_atr"] and new_low <= rules["no_new_extreme_atr"])
            results['no_new_extreme'] = core_criteria['no_new_extreme']
            satisfied_core = sum(1 for result in core_criteria.values() if result)
            overall = satisfied_core >= 2
        else:
            core_criteria = {}
            if "max_range_atr" in rules:
                core_criteria['range_check'] = all(range_atr(c["high"], c["low"], atr) <= rules["max_range_atr"] for c in candles)
                results['range_check'] = core_criteria['range_check']
            if "max_extremes_atr" in rules:
                core_criteria['extremes'] = extremes_atr(candles, atr) <= rules["max_extremes_atr"]
                results['extremes'] = core_criteria['extremes']
            if "max_net_move_atr" in rules:
                core_criteria['net_move'] = net_move_atr(candles, atr) <= rules["max_net_move_atr"]
                results['net_move'] = core_criteria['net_move']
            if "max_close_span_atr" in rules:
                results['close_span'] = close_span_atr(candles, atr) <= rules["max_close_span_atr"]
            satisfied_core = sum(1 for result in core_criteria.values() if result)
            overall = satisfied_core >= 2
    
    elif pattern == "rally":
        if n == 1:
            results['range_check'] = range_atr(candles[0]["high"], candles[0]["low"], atr) >= rules["min_range_atr"]
            candle_range = candles[0]["high"] - candles[0]["low"]
            if candle_range > 0:
                close_position = (candles[0]["close"] - candles[0]["low"]) / candle_range
                results['close_position'] = close_position >= (1 - rules["close_upper_pct"])
            overall = all(results.values())
        elif n == 2:
            results['range_check'] = all(range_atr(c["high"], c["low"], atr) >= rules["min_range_atr"] for c in candles)
            results['hh_hl'] = (candles[1]["high"] > candles[0]["high"] and candles[1]["low"] > candles[0]["low"])
            results['net_move'] = net_move_atr(candles, atr) >= rules["min_net_move_atr"]
            overall = all(results.values())
        else:
            core_criteria = {}
            if "min_bars_range_atr" in rules:
                bars_meeting = sum(1 for c in candles if range_atr(c["high"], c["low"], atr) >= rules["min_bars_range_atr"]["min"])
                core_criteria['min_bars_range'] = bars_meeting >= rules["min_bars_range_atr"]["count"]
                results['min_bars_range'] = core_criteria['min_bars_range']
            if "hh_hl_sequence" in rules:
                core_criteria['hh_hl_sequence'] = check_hh_hl_sequence(candles)
                results['hh_hl_sequence'] = core_criteria['hh_hl_sequence']
            core_criteria['net_move'] = net_move_atr(candles, atr) >= rules["min_net_move_atr"]
            results['net_move'] = core_criteria['net_move']
            if "no_bearish_engulfing" in rules:
                results['no_bearish_engulfing'] = not check_bearish_engulfing(candles)
            if "final_close_upper_pct" in rules:
                final_range = candles[-1]["high"] - candles[-1]["low"]
                if final_range > 0:
                    close_position = (candles[-1]["close"] - candles[-1]["low"]) / final_range
                    results['final_close_position'] = close_position >= (1 - rules["final_close_upper_pct"])
            if "monotonic_closes" in rules:
                results['monotonic_closes'] = check_monotonic_closes(candles, direction="up")
            satisfied_core = sum(1 for result in core_criteria.values() if result)
            overall = satisfied_core >= 2
    
    elif pattern == "drop":
        if n == 1:
            results['range_check'] = range_atr(candles[0]["high"], candles[0]["low"], atr) >= rules["min_range_atr"]
            candle_range = candles[0]["high"] - candles[0]["low"]
            if candle_range > 0:
                close_position = (candles[0]["close"] - candles[0]["low"]) / candle_range
                results['close_position'] = close_position <= rules["close_lower_pct"]
            overall = all(results.values())
        elif n == 2:
            results['range_check'] = all(range_atr(c["high"], c["low"], atr) >= rules["min_range_atr"] for c in candles)
            results['lh_ll'] = (candles[1]["high"] < candles[0]["high"] and candles[1]["low"] < candles[0]["low"])
            net_move = abs(candles[0]["open"] - candles[-1]["close"]) / atr
            results['net_move'] = net_move >= rules["min_net_move_atr"]
            overall = all(results.values())
        else:
            core_criteria = {}
            if "min_bars_range_atr" in rules:
                bars_meeting = sum(1 for c in candles if range_atr(c["high"], c["low"], atr) >= rules["min_bars_range_atr"]["min"])
                core_criteria['min_bars_range'] = bars_meeting >= rules["min_bars_range_atr"]["count"]
                results['min_bars_range'] = core_criteria['min_bars_range']
            if "lh_ll_sequence" in rules:
                core_criteria['lh_ll_sequence'] = check_lh_ll_sequence(candles)
                results['lh_ll_sequence'] = core_criteria['lh_ll_sequence']
            net_move = abs(candles[0]["open"] - candles[-1]["close"]) / atr
            core_criteria['net_move'] = net_move >= rules["min_net_move_atr"]
            results['net_move'] = core_criteria['net_move']
            if "no_bullish_engulfing" in rules:
                results['no_bullish_engulfing'] = not check_bullish_engulfing(candles)
            if "final_close_lower_pct" in rules:
                final_range = candles[-1]["high"] - candles[-1]["low"]
                if final_range > 0:
                    close_position = (candles[-1]["close"] - candles[-1]["low"]) / final_range
                    results['final_close_position'] = close_position <= rules["final_close_lower_pct"]
            if "monotonic_closes" in rules:
                results['monotonic_closes'] = check_monotonic_closes(candles, direction="down")
            satisfied_core = sum(1 for result in core_criteria.values() if result)
            overall = satisfied_core >= 2
    
    return overall, "Pattern validation passed" if overall else "Pattern validation failed", results

def plot_chart(df, selected_candles_df=None, show_atr=True):
    df_with_atr = df[df['atr'].notna()].copy() if 'atr' in df.columns else df.copy()
    atr_data = df_with_atr.tail(42) if len(df_with_atr) > 42 else df_with_atr
    atr_data = atr_data.dropna(subset=['atr']) if 'atr' in atr_data.columns else atr_data
    
    rows = 2 if show_atr else 1
    row_heights = [0.65, 0.35] if show_atr else [1.0]
    
    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
        subplot_titles=(None, 'ATR (21-Period)') if show_atr else (None,)
    )
    
    fig.add_trace(go.Candlestick(
        x=df['datetime_ist'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='OHLC'
    ), row=1, col=1)
    
    if selected_candles_df is not None and not selected_candles_df.empty:
        min_time = selected_candles_df['datetime_ist'].min()
        max_time = selected_candles_df['datetime_ist'].max()
        min_price = selected_candles_df['low'].min()
        max_price = selected_candles_df['high'].max()
        
        fig.add_shape(
            type="rect",
            x0=min_time, y0=min_price * 0.9985,
            x1=max_time, y1=max_price * 1.0015,
            line=dict(color="#FFD700", width=3, dash="dot"),
            fillcolor="rgba(255, 215, 0, 0.1)",
            row=1, col=1
        )
    
    if show_atr and 'atr' in atr_data.columns and not atr_data['atr'].isna().all():
        fig.add_trace(go.Scatter(
            x=atr_data['datetime_ist'],
            y=atr_data['atr'],
            mode='lines',
            name='ATR',
            line=dict(color='#2E86C1', width=2)
        ), row=2, col=1)
    
    fig.update_layout(
        height=700,
        showlegend=True,
        hovermode='x unified',
        xaxis_rangeslider_visible=False
    )
    
    return fig

# Main UI
st.title("🎯 Pattern Validator Pro")
st.caption("Professional Pattern Analysis with OANDA Real-Time Data")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    
    symbol = st.selectbox("Symbol", list(SYMBOL_MAPPING.keys()), index=2)
    pattern = st.selectbox("Pattern Type", ["rally", "drop", "base"])
    mode = st.radio("Selection Mode", ["Automatic (Last N Candles)", "Manual Time Range"])
    
    st.divider()
    
    use_auto_atr = st.checkbox("Auto-detect ATR", value=True)
    if not use_auto_atr:
        current_atr = st.number_input("Manual ATR", min_value=0.0001, value=0.0075, step=0.0001, format="%.4f")

# Main Tabs
tab1, tab2, tab3 = st.tabs(["Analysis", "Chart", "Data"])

with tab1:
    if mode == "Automatic (Last N Candles)":
        col1, col2 = st.columns([3, 1])
        with col1:
            n_candles = st.slider("Number of candles", 1, 6, 3)
        with col2:
            analyze_btn = st.button("Analyze", type="primary")
        
        if analyze_btn:
            end_utc = datetime.now(UTC)
            start_utc = end_utc - timedelta(days=10)
            
            with st.spinner("Fetching data..."):
                df = fetch_ohlc(symbol, start_utc, end_utc)
            
            if not df.empty:
                st.session_state['df'] = df
                
                has_incomplete = 'is_complete' in df.columns and not df.iloc[-1]['is_complete']
                
                if use_auto_atr:
                    current_atr = df['atr'].iloc[-2] if has_incomplete and len(df) > 1 else df['atr'].iloc[-1]
                    st.info(f"Auto-detected ATR: {current_atr:.4f}")
                
                analysis_df = df[df['is_complete']].tail(n_candles) if has_incomplete else df.tail(n_candles)
                
                candles = [dict(open=row.open, high=row.high, low=row.low, close=row.close) 
                          for _, row in analysis_df.iterrows()]
                
                st.session_state['selected_candles'] = analysis_df
                
                ok, message, details = validate_pattern_detailed(candles, current_atr, pattern)
                
                if ok:
                    st.success(f"✅ {pattern.upper()} pattern VALID - {message}")
                else:
                    st.error(f"❌ {pattern.upper()} pattern INVALID - {message}")
                
                if details:
                    with st.expander("Validation Details"):
                        for check, result in details.items():
                            st.write(f"{'✅' if result else '❌'} {check.replace('_', ' ').title()}")
                
                # Metrics
                if candles:
                    ranges = [(c["high"] - c["low"]) / current_atr for c in candles]
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Candles", len(candles))
                    col2.metric("Avg Range (ATR)", f"{np.mean(ranges):.2f}")
                    col3.metric("Net Move (ATR)", f"{net_move_atr(candles, current_atr):.2f}")
                    col4.metric("Current ATR", f"{current_atr:.4f}")
    
    else:  # Manual Time Range
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date")
            start_time = st.time_input("Start Time")
        with col2:
            end_date = st.date_input("End Date")
            end_time = st.time_input("End Time")
        
        if st.button("Validate Range", type="primary"):
            try:
                start_ist = IST.localize(datetime.combine(start_date, start_time))
                end_ist = IST.localize(datetime.combine(end_date, end_time))
                
                if end_ist <= start_ist:
                    st.error("End time must be after start time!")
                else:
                    start_utc = start_ist.astimezone(UTC) - timedelta(days=5)
                    end_utc = end_ist.astimezone(UTC)
                    
                    with st.spinner("Fetching data..."):
                        df = fetch_ohlc(symbol, start_utc, end_utc)
                    
                    if not df.empty:
                        st.session_state['df'] = df
                        
                        if use_auto_atr:
                            current_atr = df['atr'].iloc[-1] if not df['atr'].isna().all() else 0.0075
                            st.info(f"Auto-detected ATR: {current_atr:.4f}")
                        
                        sel = df[(df['datetime_ist'] >= start_ist) & (df['datetime_ist'] <= end_ist) & df['is_complete']].copy()
                        
                        if not sel.empty and len(sel) <= 6:
                            candles = [dict(open=row.open, high=row.high, low=row.low, close=row.close) 
                                      for _, row in sel.iterrows()]
                            st.session_state['selected_candles'] = sel
                            
                            ok, message, details = validate_pattern_detailed(candles, current_atr, pattern)
                            
                            if ok:
                                st.success(f"✅ {pattern.upper()} pattern VALID")
                            else:
                                st.error(f"❌ {pattern.upper()} pattern INVALID")
                            
                            if details:
                                with st.expander("Validation Details"):
                                    for check, result in details.items():
                                        st.write(f"{'✅' if result else '❌'} {check.replace('_', ' ').title()}")
                        elif len(sel) > 6:
                            st.warning(f"Selected range has {len(sel)} candles. Max allowed is 6.")
                        else:
                            st.warning("No complete candles in selected range.")
            except Exception as e:
                st.error(f"Error: {e}")

with tab2:
    if 'df' in st.session_state:
        df = st.session_state['df']
        sel_df = st.session_state.get('selected_candles', None)
        
        show_atr = st.checkbox("Show ATR", value=True)
        fig = plot_chart(df, sel_df, show_atr=show_atr)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Load data from Analysis tab to view chart")

with tab3:
    if 'df' in st.session_state:
        df = st.session_state['df']
        show_last = st.number_input("Show last N candles", min_value=5, max_value=100, value=25)
        display_df = df.tail(show_last)[['datetime_ist', 'open', 'high', 'low', 'close', 'atr']]
        st.dataframe(display_df, use_container_width=True)
    else:
        st.info("Load data from Analysis tab to view data")
