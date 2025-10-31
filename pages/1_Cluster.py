import requests
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import numpy as np

# === CONFIG ===
API_KEY = st.secrets["API_KEY"]
ACCOUNT_ID = st.secrets["ACCOUNT_ID"]
BASE_URL = "https://api-fxpractice.oanda.com/v3"

INSTRUMENTS = {
    "XAU_USD": {
        "base_cluster_pct": 0.15,  # Base/minimum cluster %
        "max_cluster_pct": 0.35,   # Maximum cluster % in high volatility
        "display_name": "Gold (XAU/USD)"
    },
    "NAS100_USD": {
        "base_cluster_pct": 0.20,
        "max_cluster_pct": 0.45,
        "display_name": "NASDAQ 100"
    },
    "US30_USD": {
        "base_cluster_pct": 0.18,
        "max_cluster_pct": 0.40,
        "display_name": "Dow Jones 30"
    }
}

ADR_LOOKBACK_DAYS = 20
VOLATILITY_LOOKBACK = 5  # Days to measure recent volatility

# === FUNCTIONS ===
def fetch_candles(instrument, count=ADR_LOOKBACK_DAYS+1):
    """
    Fetch daily candles for the given instrument from OANDA.
    Returns a DataFrame with open, high, low, close, time.
    """
    endpoint = f"{BASE_URL}/instruments/{instrument}/candles"
    params = {
        "granularity": "D",
        "count": count,
        "price": "M"  # midpoint
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}"
    }
    resp = requests.get(endpoint, headers=headers, params=params)
    resp.raise_for_status()
    j = resp.json()
    candles = j["candles"]
    
    # parse into DataFrame
    rows = []
    for c in candles:
        if not c["complete"]:
            continue
        pt = datetime.fromisoformat(c["time"].replace("Z",""))
        o = float(c["mid"]["o"])
        h = float(c["mid"]["h"])
        l = float(c["mid"]["l"])
        c_close = float(c["mid"]["c"])
        rows.append({"time": pt, "open": o, "high": h, "low": l, "close": c_close})
    
    df = pd.DataFrame(rows)
    df = df.sort_values("time")
    return df

def compute_adr(df):
    """
    Compute ADR as the average of (high - low) across the DataFrame rows.
    """
    df["range"] = df["high"] - df["low"]
    adr = df["range"].mean()
    return adr

def calculate_dynamic_cluster_pct(df, base_pct, max_pct):
    """
    Calculate dynamic cluster percentage based on recent volatility.
    
    Methods:
    1. Volatility Ratio: Recent volatility vs historical average
    2. ATR expansion/contraction
    3. Standard deviation of returns
    """
    # Calculate daily ranges
    df["range"] = df["high"] - df["low"]
    df["range_pct"] = (df["range"] / df["close"]) * 100
    
    # Method 1: Recent vs Historical Volatility Ratio
    recent_avg = df["range_pct"].tail(VOLATILITY_LOOKBACK).mean()
    historical_avg = df["range_pct"].mean()
    volatility_ratio = recent_avg / historical_avg if historical_avg > 0 else 1
    
    # Method 2: Standard deviation of returns (volatility measure)
    df["returns"] = df["close"].pct_change()
    recent_std = df["returns"].tail(VOLATILITY_LOOKBACK).std()
    historical_std = df["returns"].std()
    std_ratio = recent_std / historical_std if historical_std > 0 else 1
    
    # Combined volatility score (average of both methods)
    volatility_score = (volatility_ratio + std_ratio) / 2
    
    # Scale cluster percentage based on volatility score
    # If volatility_score = 1 (normal), use base_pct
    # If volatility_score > 1 (high vol), increase cluster %
    # If volatility_score < 1 (low vol), keep close to base_pct
    
    if volatility_score > 1:
        # High volatility: scale up
        cluster_pct = base_pct + (max_pct - base_pct) * min((volatility_score - 1), 1)
    else:
        # Low volatility: slightly reduce but not below base
        cluster_pct = base_pct * max(volatility_score, 0.9)
    
    # Ensure within bounds
    cluster_pct = max(base_pct, min(cluster_pct, max_pct))
    
    return cluster_pct, volatility_score, recent_avg, historical_avg

def compute_cluster_range(current_price, adr, cluster_pct):
    """
    Given current price, ADR (in points), and cluster % (e.g., 0.20%),
    compute cluster range in points = ADR * cluster_pct
    and then compute upper/lower bounds.
    """
    cluster_points = adr * (cluster_pct / 100.0)
    upper = current_price + cluster_points
    lower = current_price - cluster_points
    return cluster_points, lower, upper

def fetch_current_price(instrument):
    """
    Fetch current mid-price for the instrument.
    """
    endpoint = f"{BASE_URL}/accounts/{ACCOUNT_ID}/pricing"
    params = {
        "instruments": instrument
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}"
    }
    resp = requests.get(endpoint, headers=headers, params=params)
    resp.raise_for_status()
    j = resp.json()
    price_info = j["prices"][0]
    
    # Use mid of bid+ask
    bid = float(price_info["bids"][0]["price"])
    ask = float(price_info["asks"][0]["price"])
    mid = (bid + ask) / 2.0
    return mid

def get_instrument_emoji(instrument):
    """Return appropriate emoji for each instrument"""
    emojis = {
        "XAU_USD": "🥇",
        "NAS100_USD": "💹",
        "US30_USD": "📈"
    }
    return emojis.get(instrument, "📊")

def get_volatility_indicator(volatility_score):
    """Return emoji and text for volatility level"""
    if volatility_score < 0.8:
        return "🟢", "Low"
    elif volatility_score < 1.2:
        return "🟡", "Normal"
    else:
        return "🔴", "High"

# === STREAMLIT APP ===
def main():
    st.set_page_config(page_title="ADR & Dynamic Cluster Calculator", page_icon="📊", layout="wide")
    
    st.title("📊 ADR & Dynamic Cluster Range Calculator")
    st.write(f"Analyzing {ADR_LOOKBACK_DAYS}-day Average Daily Range with Dynamic Cluster Adjustment")
    
    # Add explanation
    with st.expander("ℹ️ How Dynamic Clusters Work"):
        st.write("""
        **Dynamic Cluster Percentage** adjusts daily based on:
        - **Recent Volatility**: Last 5 days vs 20-day average
        - **Standard Deviation**: Recent price movement patterns
        - **Volatility Score**: Combined measure of market conditions
        
        When volatility is:
        - 🟢 **Low**: Cluster % stays near base value
        - 🟡 **Normal**: Cluster % uses base value
        - 🔴 **High**: Cluster % increases up to maximum value
        """)
    
    # Add refresh button
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("🔄 Refresh Data"):
            st.rerun()
    
    st.divider()
    
    # Create tabs for different instruments
    tab_names = [f"{get_instrument_emoji(instr)} {meta['display_name']}" 
                 for instr, meta in INSTRUMENTS.items()]
    tabs = st.tabs(tab_names)
    
    for tab, (instr, meta) in zip(tabs, INSTRUMENTS.items()):
        with tab:
            try:
                with st.spinner(f"Fetching data for {meta['display_name']}..."):
                    # Fetch candles and compute ADR
                    df = fetch_candles(instr)
                    adr = compute_adr(df)
                    
                    # Get current price
                    current_price = fetch_current_price(instr)
                    
                    # Calculate dynamic cluster percentage
                    base_pct = meta["base_cluster_pct"]
                    max_pct = meta["max_cluster_pct"]
                    cluster_pct, volatility_score, recent_vol, hist_vol = calculate_dynamic_cluster_pct(
                        df, base_pct, max_pct
                    )
                    
                    # Compute cluster range with dynamic percentage
                    cluster_points, lower, upper = compute_cluster_range(current_price, adr, cluster_pct)
                    
                    # Get volatility indicator
                    vol_emoji, vol_level = get_volatility_indicator(volatility_score)
                
                # Display main metrics
                st.subheader(f"{meta['display_name']} ({instr})")
                
                # Volatility status
                st.write(f"**Market Volatility**: {vol_emoji} {vol_level} (Score: {volatility_score:.2f})")
                
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("Current Price", f"{current_price:,.2f}")
                
                with col2:
                    st.metric("ADR (points)", f"{adr:,.2f}")
                
                with col3:
                    # Show dynamic cluster % with change from base
                    cluster_change = cluster_pct - base_pct
                    st.metric("Dynamic Cluster %", 
                             f"{cluster_pct:.3f}%",
                             delta=f"{cluster_change:+.3f}%" if abs(cluster_change) > 0.001 else "base",
                             delta_color="normal" if cluster_change > 0 else "inverse")
                
                with col4:
                    st.metric("Cluster Width", f"±{cluster_points:,.2f}")
                
                with col5:
                    st.metric("Volatility Score", f"{volatility_score:.2f}")
                
                # Display cluster range
                st.write("### 🎯 Dynamic Cluster Range")
                col6, col7, col8 = st.columns(3)
                
                with col6:
                    delta_lower = current_price - lower
                    st.metric("Lower Bound", f"{lower:,.2f}", 
                             delta=f"-{delta_lower:,.2f}", delta_color="inverse")
                
                with col7:
                    st.metric("Current Price", f"{current_price:,.2f}")
                
                with col8:
                    delta_upper = upper - current_price
                    st.metric("Upper Bound", f"{upper:,.2f}", 
                             delta=f"+{delta_upper:,.2f}", delta_color="inverse")
                
                # Progress bar showing position within cluster
                progress = (current_price - lower) / (upper - lower)
                st.progress(progress, text=f"Price position within cluster: {progress*100:.1f}%")
                
                # Volatility Analysis
                with st.expander(f"📈 Volatility Analysis"):
                    col9, col10, col11 = st.columns(3)
                    
                    with col9:
                        st.write("**Volatility Metrics**")
                        st.write(f"Recent Avg Range: {recent_vol:.3f}%")
                        st.write(f"Historical Avg: {hist_vol:.3f}%")
                        st.write(f"Ratio: {volatility_score:.2f}x")
                    
                    with col10:
                        st.write("**Cluster Adjustment**")
                        st.write(f"Base Cluster: {base_pct:.3f}%")
                        st.write(f"Current Cluster: {cluster_pct:.3f}%")
                        st.write(f"Max Cluster: {max_pct:.3f}%")
                    
                    with col11:
                        st.write("**Adjustment Range**")
                        adjustment_pct = ((cluster_pct - base_pct) / base_pct) * 100
                        st.write(f"Adjustment: {adjustment_pct:+.1f}%")
                        utilization = ((cluster_pct - base_pct) / (max_pct - base_pct)) * 100
                        st.write(f"Range Used: {utilization:.1f}%")
                
                # Show recent price data with volatility
                with st.expander(f"📋 View Recent Price Data"):
                    display_df = df.tail(10).copy()
                    display_df['daily_range'] = display_df['high'] - display_df['low']
                    display_df['range_pct'] = (display_df['daily_range'] / display_df['close']) * 100
                    display_df = display_df[['time', 'open', 'high', 'low', 'close', 'daily_range', 'range_pct']]
                    display_df['time'] = display_df['time'].dt.strftime('%Y-%m-%d')
                    
                    st.dataframe(
                        display_df.style.format({
                            'open': '{:,.2f}',
                            'high': '{:,.2f}',
                            'low': '{:,.2f}',
                            'close': '{:,.2f}',
                            'daily_range': '{:,.2f}',
                            'range_pct': '{:.3f}%'
                        }),
                        use_container_width=True
                    )
                
            except Exception as e:
                st.error(f"❌ Error processing {meta['display_name']}: {str(e)}")
                st.info("Please check your API credentials and instrument availability.")

    # Add footer with timestamp
    st.divider()
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.caption("Note: Cluster percentages adjust dynamically based on market volatility")

if __name__ == "__main__":
    main()
