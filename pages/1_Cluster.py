import requests
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

# === CONFIG ===
API_KEY = st.secrets["API_KEY"]
ACCOUNT_ID = st.secrets["ACCOUNT_ID"]
BASE_URL = "https://api-fxpractice.oanda.com/v3"

INSTRUMENTS = {
    "XAU_USD": {
        "base_cluster_pct": 0.15,  # 0.15% base for Gold
        "display_name": "Gold (XAU/USD)"
    },
    "NAS100_USD": {
        "base_cluster_pct": 0.20,  # 0.20% base for NAS100
        "display_name": "NASDAQ 100"
    },
    "US30_USD": {
        "base_cluster_pct": 0.18,  # 0.18% base for US30 (Dow Jones)
        "display_name": "Dow Jones 30"
    }
}

ADR_LOOKBACK_DAYS = 20

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

# === STREAMLIT APP ===
def main():
    st.set_page_config(page_title="ADR & Cluster Calculator", page_icon="📊", layout="wide")
    
    st.title("📊 ADR & Cluster Range Calculator")
    st.write(f"Analyzing {ADR_LOOKBACK_DAYS}-day Average Daily Range")
    
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
                    
                    # Compute cluster range
                    cluster_pct = meta["base_cluster_pct"]
                    cluster_points, lower, upper = compute_cluster_range(current_price, adr, cluster_pct)
                
                # Display main metrics
                st.subheader(f"{meta['display_name']} ({instr})")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Current Price", f"{current_price:,.2f}")
                
                with col2:
                    st.metric("ADR (points)", f"{adr:,.2f}")
                
                with col3:
                    st.metric("Cluster %", f"{cluster_pct:.2f}%")
                
                with col4:
                    st.metric("Cluster Width", f"±{cluster_points:,.2f}")
                
                # Display cluster range
                st.write("### 🎯 Cluster Range")
                col5, col6, col7 = st.columns(3)
                
                with col5:
                    delta_lower = current_price - lower
                    st.metric("Lower Bound", f"{lower:,.2f}", 
                             delta=f"-{delta_lower:,.2f}", delta_color="inverse")
                
                with col6:
                    st.metric("Current Price", f"{current_price:,.2f}")
                
                with col7:
                    delta_upper = upper - current_price
                    st.metric("Upper Bound", f"{upper:,.2f}", 
                             delta=f"+{delta_upper:,.2f}", delta_color="inverse")
                
                # Progress bar showing position within cluster
                progress = (current_price - lower) / (upper - lower)
                st.progress(progress, text=f"Price position within cluster: {progress*100:.1f}%")
                
                # Show recent price data
                with st.expander(f"📋 View Recent Price Data"):
                    # Add daily range column
                    display_df = df.tail(10).copy()
                    display_df['daily_range'] = display_df['high'] - display_df['low']
                    display_df = display_df[['time', 'open', 'high', 'low', 'close', 'daily_range']]
                    display_df['time'] = display_df['time'].dt.strftime('%Y-%m-%d')
                    
                    st.dataframe(
                        display_df.style.format({
                            'open': '{:,.2f}',
                            'high': '{:,.2f}',
                            'low': '{:,.2f}',
                            'close': '{:,.2f}',
                            'daily_range': '{:,.2f}'
                        }),
                        use_container_width=True
                    )
                
                # Add statistics
                with st.expander(f"📊 Statistics"):
                    col8, col9, col10 = st.columns(3)
                    
                    with col8:
                        st.write("**Daily Range Stats**")
                        st.write(f"Min: {df['range'].min():,.2f}")
                        st.write(f"Max: {df['range'].max():,.2f}")
                        st.write(f"Std Dev: {df['range'].std():,.2f}")
                    
                    with col9:
                        st.write("**ADR as % of Price**")
                        adr_pct = (adr / current_price) * 100
                        st.write(f"ADR/Price: {adr_pct:.2f}%")
                        cluster_pct_of_price = (cluster_points / current_price) * 100
                        st.write(f"Cluster/Price: {cluster_pct_of_price:.3f}%")
                    
                    with col10:
                        st.write("**Recent Performance**")
                        last_close = df.iloc[-1]['close']
                        prev_close = df.iloc[-2]['close'] if len(df) > 1 else last_close
                        change = last_close - prev_close
                        change_pct = (change / prev_close) * 100
                        st.write(f"Last Change: {change:+,.2f}")
                        st.write(f"Last Change %: {change_pct:+.2f}%")
                
            except Exception as e:
                st.error(f"❌ Error processing {meta['display_name']}: {str(e)}")
                st.info("Please check your API credentials and instrument availability.")

    # Add footer with timestamp
    st.divider()
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
