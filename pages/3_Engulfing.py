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

# Modern CSS styling
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
.signal-strong { background-color: #22c55e; color: white; padding: 4px 8px; border-radius: 20px; }
.signal-neutral { background-color: #f59e0b; color: white; padding: 4px 8px; border-radius: 20px; }
.signal-weak { background-color: #ef4444; color: white; padding: 4px 8px; border-radius: 20px; }
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
        
        records.append({
            "Time": ist_time.strftime("%Y-%m-%d %I:%M %p"),
            "Open": float(candle["mid"]["o"]),
            "High": float(candle["mid"]["h"]),
            "Low": float(candle["mid"]["l"]),
            "Close": float(candle["mid"]["c"]),
            "Timestamp": ist_time  # For sorting
        })
    
    df = pd.DataFrame(records)
    
    # Calculate body multiples
    df["Body"] = abs(df["Close"] - df["Open"])
    df["Body_ATR_Multiple"] = (df["Body"] / atr_value).round(2)
    
    # Signal classification
    df["Signal"] = pd.cut(
        df["Body_ATR_Multiple"],
        bins=[0, 0.7, 1.3, float('inf')],
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
                    max_multiple = df["Body_ATR_Multiple"].max()
                    st.metric("🔥 Max Multiple", f"{max_multiple:.2f}x")
                
                st.markdown("---")
                
                # Data table
                st.subheader(f"📋 {instrument} ({timeframe}) - Candle Analysis")
                
                # Format display DataFrame
                display_df = df.copy()
                display_df["Body_ATR_Multiple"] = display_df["Body_ATR_Multiple"].apply(lambda x: f"{x:.2f}x")
                
                # Style the signal column
                def style_signal(val):
                    if val == "Strong":
                        return "background-color: #22c55e; color: white; border-radius: 10px; text-align: center"
                    elif val == "Neutral":
                        return "background-color: #f59e0b; color: white; border-radius: 10px; text-align: center"
                    else:
                        return "background-color: #ef4444; color: white; border-radius: 10px; text-align: center"
                
                # Display styled dataframe
                styled_df = display_df[["Time", "Open", "High", "Low", "Close", "Body_ATR_Multiple", "Signal"]].style.applymap(
                    style_signal, subset=["Signal"]
                ).format({
                    "Open": "{:.1f}",
                    "High": "{:.1f}", 
                    "Low": "{:.1f}",
                    "Close": "{:.1f}"
                })
                
                st.dataframe(styled_df, use_container_width=True, hide_index=True)
                
                # Download button
                csv_data = df.to_csv(index=False)
                st.download_button(
                    "📥 Download CSV",
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
    st.info("👆 Configure your settings in the sidebar and click 'Fetch Data' to begin analysis.")
