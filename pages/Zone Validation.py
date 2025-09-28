import streamlit as st
import requests
import pandas as pd
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import plotly.graph_objects as go

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
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime").reset_index(drop=True)
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)
    return df

# --- Zone Structure ---
@dataclass
class Zone:
    type: str  # "rally" or "drop"
    start_time: datetime
    end_time: datetime
    zone_high: float  # Upper boundary of zone
    zone_low: float   # Lower boundary of zone
    origin_high: float  # Highest point of the move
    origin_low: float   # Lowest point of the move
    candle_count: int
    strength: str  # "strong", "moderate", "weak"

# --- Candle Structure ---
@dataclass
class Candle:
    name: int
    datetime: datetime
    open: float
    high: float
    low: float
    close: float

# --- Validation Rules ---
rules = {
    1: dict(valid=1, inside=0, atr_exp=0, scale=1.0),
    2: dict(valid=2, inside=0, atr_exp=1, scale=0.8),
    3: dict(valid=2, inside=1, atr_exp=1, scale=0.8),
    4: dict(valid=3, inside=1, atr_exp=1, scale=0.75),
    5: dict(valid=4, inside=1, atr_exp=2, scale=0.7),
    6: dict(valid=5, inside=1, atr_exp=2, scale=0.7),
}

def is_inside(prev, curr):
    return curr.high < prev.high and curr.low > prev.low

def calculate_zone_boundaries(candles, zone_type):
    """Calculate the exact zone boundaries based on candle structure."""
    if not candles:
        return None
    
    # For rally: zone is from the lowest low to the open of first candle
    # For drop: zone is from the open of first candle to the highest high
    
    all_highs = [c.high for c in candles]
    all_lows = [c.low for c in candles]
    first_open = candles[0].open
    
    if zone_type == "rally":
        zone_low = min(all_lows)
        zone_high = first_open
        origin_low = zone_low
        origin_high = max(all_highs)
    else:  # drop
        zone_low = first_open
        zone_high = max(all_highs)
        origin_low = min(all_lows)
        origin_high = zone_high
    
    return Zone(
        type=zone_type,
        start_time=candles[0].datetime,
        end_time=candles[-1].datetime,
        zone_low=zone_low,
        zone_high=zone_high,
        origin_low=origin_low,
        origin_high=origin_high,
        candle_count=len(candles),
        strength="strong" if len(candles) <= 2 else "moderate" if len(candles) <= 4 else "weak"
    )

def validate_rally_drop(candles, atr_series, current_atr):
    """Validate rally/drop zone using scaled ATR logic and inside candle rules."""
    n = len(candles)
    if n < 1 or n > 6:
        return False, "Candle count out of range", None

    rule = rules[n]
    valid = 0
    inside_cnt = 0
    atr_exp = 0
    direction = None
    min_body = current_atr * rule["scale"]

    for i in range(n):
        c = candles[i]
        prev = candles[i-1] if i > 0 else None
        body = abs(c.close - c.open)
        full = c.high - c.low
        atr = atr_series.get(c.name, 0)

        cond = (body >= 0.6 * full) or (body >= min_body)
        if cond and atr >= current_atr:
            valid += 1
            if body >= atr:
                atr_exp += 1
            dir_curr = c.close > c.open
            if direction is None:
                direction = dir_curr
            elif direction != dir_curr:
                return False, "Mixed directional bias", None
        else:
            if prev and is_inside(prev, c):
                inside_cnt += 1
            else:
                return False, "Non-valid non-inside candle", None

    ok = (
        valid >= rule["valid"] and
        inside_cnt <= rule["inside"] and
        atr_exp >= rule["atr_exp"]
    )
    
    zone_type = "rally" if direction else "drop"
    zone = calculate_zone_boundaries(candles, zone_type) if ok else None
    
    return ok, f"{n}-candle {zone_type} validation: {ok}", zone

def plot_zone_chart(df, zone, selected_candles_df=None):
    """Create a candlestick chart with zone visualization."""
    fig = go.Figure()
    
    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=df['datetime'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Price'
    ))
    
    # Add zone rectangle if valid
    if zone:
        # Zone area (tradeable area)
        fig.add_shape(
            type="rect",
            x0=zone.start_time,
            x1=df['datetime'].iloc[-1],  # Extend to current
            y0=zone.zone_low,
            y1=zone.zone_high,
            fillcolor="green" if zone.type == "rally" else "red",
            opacity=0.2,
            line=dict(width=2, color="green" if zone.type == "rally" else "red"),
            name="Zone Area"
        )
        
        # Origin move visualization
        fig.add_shape(
            type="rect",
            x0=zone.start_time,
            x1=zone.end_time,
            y0=zone.origin_low,
            y1=zone.origin_high,
            fillcolor="blue",
            opacity=0.1,
            line=dict(width=1, color="blue", dash="dot"),
            name="Origin Move"
        )
        
        # Add zone labels
        fig.add_annotation(
            x=zone.start_time,
            y=zone.zone_high if zone.type == "rally" else zone.zone_low,
            text=f"{zone.type.upper()} ZONE ({zone.strength})",
            showarrow=True,
            arrowhead=2,
            arrowcolor="green" if zone.type == "rally" else "red",
            bgcolor="white",
            bordercolor="green" if zone.type == "rally" else "red"
        )
    
    # Highlight selected candles if manual mode
    if selected_candles_df is not None and not selected_candles_df.empty:
        fig.add_trace(go.Scatter(
            x=selected_candles_df['datetime'],
            y=selected_candles_df['high'] * 1.001,  # Slightly above high
            mode='markers',
            marker=dict(symbol='triangle-down', size=12, color='orange'),
            name='Selected Candles'
        ))
    
    fig.update_layout(
        title="Rally/Drop Zone Analysis",
        yaxis_title="Price",
        xaxis_title="Time",
        height=600,
        xaxis_rangeslider_visible=False
    )
    
    return fig

# --- Streamlit UI ---
st.title("🎯 Rally/Drop Zone Validator")

# Sidebar configuration
st.sidebar.header("Configuration")

# Symbol selection
symbol = st.sidebar.text_input("Symbol", value="XAU/USD")

# Mode selection
mode = st.sidebar.radio(
    "Selection Mode",
    ["Automatic (Last N Candles)", "Manual Time Range", "Custom Candle Selection"]
)

# ATR input
current_atr = st.sidebar.number_input(
    "Current ATR", 
    min_value=0.01, 
    value=0.75, 
    step=0.01,
    help="Average True Range for validation"
)

# Initialize variables
df = None
zone = None
selected_candles = []

# Fetch data based on mode
col1, col2 = st.columns([2, 1])

with col1:
    if mode == "Automatic (Last N Candles)":
        st.subheader("📊 Automatic Mode")
        
        # Number of candles to analyze
        n_candles = st.slider("Number of candles to analyze", 1, 6, 3)
        
        if st.button("Analyze Last Candles"):
            end = datetime.now(timezone.utc)
            start = end - timedelta(days=5)
            
            with st.spinner("Fetching data..."):
                df = fetch_ohlc(symbol, start, end)
                
            if not df.empty:
                df["atr"] = (df["high"] - df["low"]).abs()
                atr_series = df["atr"].to_dict()
                
                # Get last N candles
                candles = [
                    Candle(
                        name=i, 
                        datetime=row.datetime,
                        open=row.open, 
                        high=row.high, 
                        low=row.low, 
                        close=row.close
                    )
                    for i, row in df.tail(n_candles).iterrows()
                ]
                
                result, message, zone = validate_rally_drop(candles, atr_series, current_atr)
                
                if result:
                    st.success(f"✅ {message}")
                else:
                    st.error(f"❌ {message}")
    
    elif mode == "Manual Time Range":
        st.subheader("🕐 Manual Time Range")
        
        col_start, col_end = st.columns(2)
        with col_start:
            start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=3))
            start_time = st.time_input("Start Time (UTC)", value=datetime.now().time())
        
        with col_end:
            end_date = st.date_input("End Date", value=datetime.now())
            end_time = st.time_input("End Time (UTC)", value=datetime.now().time())
        
        if st.button("Validate Time Range"):
            start = datetime.combine(start_date, start_time).replace(tzinfo=timezone.utc)
            end = datetime.combine(end_date, end_time).replace(tzinfo=timezone.utc)
            
            # Fetch wider range for context
            context_start = start - timedelta(days=2)
            context_end = end + timedelta(days=1)
            
            with st.spinner("Fetching data..."):
                df = fetch_ohlc(symbol, context_start, context_end)
            
            if not df.empty:
                # Filter candles within selected range
                mask = (df['datetime'] >= start) & (df['datetime'] <= end)
                selected_df = df[mask].copy()
                
                if not selected_df.empty:
                    df["atr"] = (df["high"] - df["low"]).abs()
                    atr_series = df["atr"].to_dict()
                    
                    candles = [
                        Candle(
                            name=i,
                            datetime=row.datetime,
                            open=row.open,
                            high=row.high,
                            low=row.low,
                            close=row.close
                        )
                        for i, row in selected_df.iterrows()
                    ]
                    
                    st.info(f"Found {len(candles)} candles in selected range")
                    
                    if 1 <= len(candles) <= 6:
                        result, message, zone = validate_rally_drop(candles, atr_series, current_atr)
                        
                        if result:
                            st.success(f"✅ {message}")
                        else:
                            st.error(f"❌ {message}")
                    else:
                        st.warning(f"⚠️ Selected range contains {len(candles)} candles. Please select 1-6 candles.")
                        zone = None
                else:
                    st.warning("No candles found in selected range")
    
    else:  # Custom Candle Selection
        st.subheader("🎯 Custom Candle Selection")
        
        # Fetch recent data for selection
        if st.button("Load Recent Data"):
            end = datetime.now(timezone.utc)
            start = end - timedelta(days=10)
            
            with st.spinner("Loading data..."):
                df = fetch_ohlc(symbol, start, end)
            
            if not df.empty:
                st.session_state['df'] = df
                st.success(f"Loaded {len(df)} candles")
        
        # Display data table for selection
        if 'df' in st.session_state:
            df = st.session_state['df']
            
            # Create selectable dataframe
            df_display = df[['datetime', 'open', 'high', 'low', 'close']].copy()
            df_display['datetime'] = df_display['datetime'].dt.strftime('%Y-%m-%d %H:%M')
            df_display.index = range(len(df_display))
            
            st.write("Select candles to validate (max 6):")
            selected_indices = st.multiselect(
                "Select candle indices:",
                options=df_display.index.tolist(),
                format_func=lambda x: f"{x}: {df_display.loc[x, 'datetime']} | O:{df_display.loc[x, 'open']:.2f} C:{df_display.loc[x, 'close']:.2f}"
            )
            
            if selected_indices and st.button("Validate Selected Candles"):
                df["atr"] = (df["high"] - df["low"]).abs()
                atr_series = df["atr"].to_dict()
                
                # Sort indices to maintain chronological order
                selected_indices = sorted(selected_indices)
                
                candles = [
                    Candle(
                        name=idx,
                        datetime=df.loc[idx, 'datetime'],
                        open=df.loc[idx, 'open'],
                        high=df.loc[idx, 'high'],
                        low=df.loc[idx, 'low'],
                        close=df.loc[idx, 'close']
                    )
                    for idx in selected_indices
                ]
                
                if 1 <= len(candles) <= 6:
                    result, message, zone = validate_rally_drop(candles, atr_series, current_atr)
                    
                    if result:
                        st.success(f"✅ {message}")
                    else:
                        st.error(f"❌ {message}")
                    
                    # Store selected candles for visualization
                    selected_candles = df.iloc[selected_indices]
                else:
                    st.warning(f"Please select between 1 and 6 candles (currently: {len(candles)})")

# Display zone details in sidebar
with col2:
    if zone:
        st.subheader("📍 Zone Details")
        st.metric("Type", zone.type.upper())
        st.metric("Strength", zone.strength.upper())
        st.metric("Zone High", f"{zone.zone_high:.2f}")
        st.metric("Zone Low", f"{zone.zone_low:.2f}")
        st.metric("Zone Height", f"{zone.zone_high - zone.zone_low:.2f}")
        st.metric("Candles", zone.candle_count)
        
        # Trading suggestions
        st.subheader("💡 Trading Ideas")
        if zone.type == "rally":
            st.info(f"🟢 **Buy Zone**: {zone.zone_low:.2f} - {zone.zone_high:.2f}")
            st.caption("Look for bullish reactions in this zone")
        else:
            st.info(f"🔴 **Sell Zone**: {zone.zone_low:.2f} - {zone.zone_high:.2f}")
            st.caption("Look for bearish reactions in this zone")

# Display chart
if df is not None and not df.empty:
    st.subheader("📈 Zone Visualization")
    
    # Determine what to highlight
    selected_df = None
    if mode == "Manual Time Range" and zone:
        selected_df = df[(df['datetime'] >= zone.start_time) & (df['datetime'] <= zone.end_time)]
    elif mode == "Custom Candle Selection" and 'selected_candles' in locals():
        selected_df = selected_candles
    
    fig = plot_zone_chart(df, zone, selected_df)
    st.plotly_chart(fig, use_container_width=True)
    
    # Display zone summary
    if zone:
        st.subheader("📋 Zone Summary")
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.info(f"**Zone Type:** {zone.type.upper()}")
        with col_b:
            st.info(f"**Zone Range:** {zone.zone_high - zone.zone_low:.2f}")
        with col_c:
            st.info(f"**Origin Move:** {zone.origin_high - zone.origin_low:.2f}")
