import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import pytz

# -----------------------------
# CONFIG & STYLING
# -----------------------------
st.set_page_config(
    page_title="ATR Body Dashboard", 
    page_icon="📊", 
    layout="wide"
)

# Enhanced CSS styling for better table readability
st.markdown("""
<style>
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin: 0.5rem 0;
}
.signal-strong { 
    background: linear-gradient(135deg, #4ade80, #22c55e);
    color: white; 
    padding: 6px 12px; 
    border-radius: 15px; 
    font-weight: 600;
    display: inline-block;
    box-shadow: 0 2px 8px rgba(34, 197, 94, 0.3);
}
.signal-neutral { 
    background: linear-gradient(135deg, #fbbf24, #f59e0b);
    color: white; 
    padding: 6px 12px; 
    border-radius: 15px; 
    font-weight: 600;
    display: inline-block;
    box-shadow: 0 2px 8px rgba(245, 158, 11, 0.3);
}
.signal-weak { 
    background: linear-gradient(135deg, #f87171, #ef4444);
    color: white; 
    padding: 6px 12px; 
    border-radius: 15px; 
    font-weight: 600;
    display: inline-block;
    box-shadow: 0 2px 8px rgba(239, 68, 68, 0.3);
}

/* Custom table styling */
.dataframe {
    font-family: 'Arial', sans-serif;
    border-radius: 15px;
    overflow: hidden;
    box-shadow: 0 8px 25px rgba(0,0,0,0.1);
}

.dataframe th {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    font-weight: 600;
    text-align: center;
    padding: 12px 8px;
    border: none;
}

.dataframe td {
    text-align: center;
    padding: 16px 8px;
    line-height: 1.6;    
    border-bottom: 1px solid #f0f0f0;
    min-height: 40px;    
}

.dataframe tr:hover {
    background-color: #f8fafc;
}
</style>
""", unsafe_allow_html=True)

OANDA_API_URL = "https://api-fxpractice.oanda.com/v3/instruments/{}/candles"
API_KEY = st.secrets["API_KEY"]
ACCOUNT_ID = st.secrets["ACCOUNT_ID"]

# -----------------------------
# CORE FUNCTIONS
# -----------------------------
@st.cache_data(ttl=300)
def fetch_oanda_data(instrument, timeframe, start_utc, end_utc):
    """Fetch candles from OANDA API"""
    headers = {"Authorization": f"Bearer {API_KEY}"}
    params = {
        "granularity": timeframe,
        "price": "M",
        "from": start_utc,
        "to": end_utc
    }
    
    response = requests.get(
        OANDA_API_URL.format(instrument), 
        headers=headers, 
        params=params,
        timeout=10
    )
    response.raise_for_status()
    return response.json()["candles"]

def process_candle_data(candles_data, atr_value):
    """Process raw candle data into structured DataFrame"""
    if not candles_data:
        return pd.DataFrame()
    
    # Convert to DataFrame
    records = []
    for candle in candles_data:
        # Parse UTC time and convert to IST
        utc_time = datetime.strptime(candle["time"][:19], "%Y-%m-%dT%H:%M:%S")
        utc_time = pytz.utc.localize(utc_time)
        ist_time = utc_time.astimezone(pytz.timezone("Asia/Kolkata"))
        
        open_price = float(candle["mid"]["o"])
        high_price = float(candle["mid"]["h"])
        low_price = float(candle["mid"]["l"])
        close_price = float(candle["mid"]["c"])
        
        # Calculate body and range
        body = abs(close_price - open_price)
        total_range = high_price - low_price
        body_percentage = (body / total_range * 100) if total_range > 0 else 0
        
        # Determine candle direction
        direction = "🟢" if close_price > open_price else "🔴" if close_price < open_price else "➖"
        
        records.append({
            "Time": ist_time.strftime("%m/%d %I:%M %p"),  # Shorter time format
            "Direction": direction,
            "Open": open_price,
            "High": high_price,
            "Low": low_price,
            "Close": close_price,
            "Body": body,
            "Body_Percentage": body_percentage,
            "Timestamp": ist_time  # For sorting
        })
    
    df = pd.DataFrame(records)
    
    # Calculate ATR multiples
    df["Body_ATR_Multiple"] = (df["Body"] / atr_value).round(2)
    
    # Signal classification
    df["Signal"] = pd.cut(
        df["Body_ATR_Multiple"],
        bins=[0, 0.7, 1.0, float('inf')],
        labels=["Weak", "Neutral", "Strong"]
    )
    
    # Sort by time (chronological order)
    df = df.sort_values("Timestamp").reset_index(drop=True)
    
    return df.drop("Timestamp", axis=1)  # Remove helper column

def convert_ist_to_utc(dt_ist):
    """Convert IST datetime to UTC ISO format"""
    ist_tz = pytz.timezone("Asia/Kolkata")
    utc_tz = pytz.utc
    dt_localized = ist_tz.localize(dt_ist)
    dt_utc = dt_localized.astimezone(utc_tz)
    return dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("📊 ATR Body Multiple Dashboard")
st.markdown("*Analyze candle body strength relative to Average True Range*")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Instrument selection
    instrument = st.selectbox(
        "📈 Instrument", 
        ["XAU_USD", "NAS100_USD", "US30_USD"],
        help="Select trading instrument"
    )
    
    # Timeframe
    timeframe = st.radio(
        "⏰ Timeframe", 
        ["H4", "D"],
        horizontal=True
    )
    
    # ATR value
    atr_value = st.number_input(
        "📊 ATR Value", 
        min_value=0.1, 
        value=20.0, 
        step=0.1,
        help="Manual ATR value in price units"
    )
    
    st.markdown("---")
    st.markdown("### 📅 Time Range (IST)")
    
    # Date inputs
    start_date = st.date_input("Start Date", value=datetime.today())
    start_time = st.time_input("Start Time", value=datetime.strptime("09:00", "%H:%M").time())
    
    end_date = st.date_input("End Date", value=datetime.today())
    end_time = st.time_input("End Time", value=datetime.strptime("21:00", "%H:%M").time())
    
    # Fetch button
    fetch_data = st.button("🚀 Fetch Data", type="primary", use_container_width=True)

# Main content
if fetch_data:
    # Combine datetime
    start_dt = datetime.combine(start_date, start_time)
    end_dt = datetime.combine(end_date, end_time)
    
    # Convert to UTC
    start_utc = convert_ist_to_utc(start_dt)
    end_utc = convert_ist_to_utc(end_dt)
    
    try:
        with st.spinner("🔄 Fetching data from OANDA..."):
            # Fetch and process data
            raw_candles = fetch_oanda_data(instrument, timeframe, start_utc, end_utc)
            df = process_candle_data(raw_candles, atr_value)
            
            if df.empty:
                st.warning("⚠️ No data found for the selected time range.")
            else:
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("📊 Total Candles", len(df))
                with col2:
                    strong_count = len(df[df["Signal"] == "Strong"])
                    st.metric("💪 Strong Signals", strong_count)
                with col3:
                    avg_multiple = df["Body_ATR_Multiple"].mean()
                    st.metric("📈 Avg Multiple", f"{avg_multiple:.2f}x")
                with col4:
                    avg_body_pct = df["Body_Percentage"].mean()
                    st.metric("📏 Avg Body %", f"{avg_body_pct:.1f}%")
                
                st.markdown("---")
                
                # Enhanced Data table
                st.subheader(f"📋 {instrument} ({timeframe}) - Detailed Candle Analysis")
                
                # Prepare display DataFrame with better formatting
                display_df = df.copy()
                
                # Format display columns
                display_df = display_df[[
                    "Time", "Direction", "Open", 
                    "Body_Percentage", "Body_ATR_Multiple", "Signal"
                ]]
                
                # Configure columns for better display
                column_config = {
                    "Time": st.column_config.TextColumn(
                        "🕒 Time (IST)",
                        help="Candle opening time in IST",
                        width="small",
                
                    ),
                    "Direction": st.column_config.TextColumn(
                        "📊 Dir",
                        help="🟢 Bullish, 🔴 Bearish, ➖ Doji",
                        width="small"
                    ),
                    "Open": st.column_config.NumberColumn(
                        "📈 Open",
                        format="%.2f",
                        width="small"
                    ),
                    
                    "Low": st.column_config.NumberColumn(
                        "⬇️ Low",
                        format="%.2f",
                        width="small"
                    ),
                    
                    "Body_Percentage": st.column_config.NumberColumn(
                        "📏 Body %",
                        help="Body as percentage of total candle range",
                        format="%.1f%%",
                        width="small"
                    ),
                    "Body_ATR_Multiple": st.column_config.NumberColumn(
                        " ATR",
                        help="Body size relative to ATR",
                        format="%.2fx",
                        width="small"
                    ),
                    "Signal": st.column_config.TextColumn(
                        "🎯 Signal",
                        help="Strength classification based on ATR multiple",
                        width="medium"
                    ),
                }
                
                # Display the enhanced dataframe
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config=column_config,
                    height=600  # Fixed height for better scrolling
                )
                
                # Additional insights
                st.markdown("### 📊 Quick Insights")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    high_body_pct = len(df[df["Body_Percentage"] > 70])
                    st.metric("🔥 High Body % (>70%)", high_body_pct, help="Candles with strong directional moves")
                
                with col2:
                    doji_count = len(df[df["Body_Percentage"] < 20])
                    st.metric("🕯️ Doji-like (<20%)", doji_count, help="Candles with small bodies indicating indecision")
                
                with col3:
                    max_body_pct = df["Body_Percentage"].max()
                    st.metric("📈 Max Body %", f"{max_body_pct:.1f}%", help="Strongest directional candle")
                
                # Download button
                st.markdown("---")
                csv_data = df.to_csv(index=False)
                st.download_button(
                    "📥 Download Complete Analysis (CSV)",
                    csv_data,
                    file_name=f"{instrument}_{timeframe}_atr_analysis.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
    except requests.exceptions.RequestException as e:
        st.error(f"❌ API Error: {str(e)}")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

else:
    # Welcome message
    st.info(" Configure your settings in the sidebar and click 'Fetch Data' to begin analysis.")
