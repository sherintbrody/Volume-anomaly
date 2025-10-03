import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import pytz

# -----------------------------
# CONFIG & MODERN STYLING
# -----------------------------
st.set_page_config(
    page_title="ATR Body Dashboard", 
    page_icon="📊", 
    layout="wide"
)

# Enhanced Modern CSS styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        color: white;
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        font-size: 2.5rem;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-family: 'Poppins', sans-serif;
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
    }
    
    .metric-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        box-shadow: 0 8px 25px rgba(240, 147, 251, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        margin: 0.5rem 0;
    }
    
    .metric-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(240, 147, 251, 0.4);
    }
    
    .metric-box h3 {
        margin: 0;
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        font-size: 1.8rem;
    }
    
    .metric-box p {
        margin: 0.5rem 0 0 0;
        font-family: 'Poppins', sans-serif;
        font-weight: 400;
        opacity: 0.9;
    }
    
    .section-header {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1rem 2rem;
        border-radius: 15px;
        margin: 2rem 0 1rem 0;
        box-shadow: 0 6px 20px rgba(17, 153, 142, 0.3);
    }
    
    .section-header h2 {
        color: white;
        margin: 0;
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        font-size: 1.5rem;
    }
    
    .data-container {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
    }
    
    .config-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
    }
    
    .config-box h3 {
        margin: 0 0 1rem 0;
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        font-size: 1.4rem;
    }
    
    .signal-strong { 
        background: linear-gradient(135deg, #4ade80, #22c55e);
        color: white; 
        padding: 6px 12px; 
        border-radius: 20px; 
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(34, 197, 94, 0.3);
    }
    .signal-neutral { 
        background: linear-gradient(135deg, #fbbf24, #f59e0b);
        color: white; 
        padding: 6px 12px; 
        border-radius: 20px; 
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(245, 158, 11, 0.3);
    }
    .signal-weak { 
        background: linear-gradient(135deg, #f87171, #ef4444);
        color: white; 
        padding: 6px 12px; 
        border-radius: 20px; 
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(239, 68, 68, 0.3);
    }
    
    .chart-container {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
    }
    
    .welcome-box {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 8px 25px rgba(252, 182, 159, 0.3);
    }
    
    .welcome-box h3 {
        color: #8b4513;
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        margin: 0 0 1rem 0;
    }
    
    .welcome-box p {
        color: #8b4513;
        font-family: 'Poppins', sans-serif;
        margin: 0;
    }
    
    .help-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        color: white;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
    }
    
    .stSelectbox > div > div {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    .stNumberInput > div > div {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 25px;
        border: none;
        padding: 0.8rem 2rem;
        font-weight: 600;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# CONSTANTS
# -----------------------------
OANDA_API_URL = "https://api-fxpractice.oanda.com/v3/instruments/{}/candles"
API_KEY = st.secrets["API_KEY"]

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

# Main Header
st.markdown("""
<div class="main-header">
    <h1>📊 ATR Body Multiple Dashboard</h1>
    <p>Analyze candle body strength relative to Average True Range with advanced insights</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.markdown("""
    <div class="config-box">
        <h2>⚙️ Settings</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Instrument selection
    instrument = st.selectbox(
        "📈 Select Instrument", 
        ["US30_USD", "XAU_USD", "NAS100_USD"],
        help="Choose your preferred trading instrument"
    )
    
    # Timeframe
    timeframe = st.radio(
        "⏰ Timeframe Selection", 
        ["D", "H4"],
        horizontal=True,
        help="H4 = 4-hour candles, D = Daily candles"
    )
    
    # ATR value
    atr_value = st.number_input(
        "📊 ATR Value", 
        min_value=0.1, 
        value=20.0, 
        step=0.1,
        help="Enter your calculated ATR value in price units"
    )
    
    st.markdown("---")
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                padding: 1rem; border-radius: 15px; margin: 1rem 0; color: white;">
        <h4 style="margin: 0; font-family: 'Poppins', sans-serif;">📅 Time Range (IST)</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Date inputs
    st.markdown("**Start Date & Time**")
    col1, col2 = st.columns([1.5, 1])
    with col1:
        start_date = st.date_input("Date", value=datetime.today(), label_visibility="collapsed")
    with col2:
        start_time = st.time_input("Time", value=datetime.strptime("09:00", "%H:%M").time(), label_visibility="collapsed")

    st.markdown("**End Date & Time**")
    col3, col4 = st.columns([1.5, 1])
    with col3:
        end_date = st.date_input("Date", value=datetime.today(), label_visibility="collapsed", key="end_date")
    with col4:
        end_time = st.time_input("Time", value=datetime.strptime("21:00", "%H:%M").time(), label_visibility="collapsed", key="end_time")
    
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
        with st.spinner("🔄 Fetching data from OANDA API..."):
            # Fetch and process data
            raw_candles = fetch_oanda_data(instrument, timeframe, start_utc, end_utc)
            df = process_candle_data(raw_candles, atr_value)
            
            if df.empty:
                st.warning("⚠️ No data found for the selected time range.")
            else:
                # Summary metrics in beautiful boxes
                st.markdown("""
                <div class="section-header">
                    <h2>📊 Analysis Summary</h2>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-box">
                        <h3>{len(df)}</h3>
                        <p>📋 Total Candles</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    strong_count = len(df[df["Signal"] == "Strong"])
                    st.markdown(f"""
                    <div class="metric-box">
                        <h3>{strong_count}</h3>
                        <p>💪 Strong Signals</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    avg_multiple = df["Body_ATR_Multiple"].mean()
                    st.markdown(f"""
                    <div class="metric-box">
                        <h3>{avg_multiple:.2f}x</h3>
                        <p>📈 Average Multiple</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    max_multiple = df["Body_ATR_Multiple"].max()
                    st.markdown(f"""
                    <div class="metric-box">
                        <h3>{max_multiple:.2f}x</h3>
                        <p>🔥 Maximum Multiple</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Data table section
                st.markdown(f"""
                <div class="section-header">
                    <h2>📋 {instrument} ({timeframe}) - Detailed Analysis</h2>
                </div>
                """, unsafe_allow_html=True)
                
                with st.container():
                    st.markdown('<div class="data-container">', unsafe_allow_html=True)
                    
                    # Format display DataFrame
                    display_df = df.copy()
                    display_df["Body_ATR_Multiple"] = display_df["Body_ATR_Multiple"].apply(lambda x: f"{x:.2f}x")
                    
                    # Display dataframe with custom styling
                    st.dataframe(
                        display_df[["Time", "Open", "High", "Low", "Close", "Body_ATR_Multiple", "Signal"]],
                        width='stretch', 
                        hide_index=True,
                        
                        column_config={
                            "Time": st.column_config.TextColumn("🕒 Time (IST)", width="large"),
                            "Open": st.column_config.NumberColumn("📈 Open", format="%.1f"),
                            "High": st.column_config.NumberColumn("⬆️ High", format="%.1f"),
                            "Low": st.column_config.NumberColumn("⬇️ Low", format="%.1f"),
                            "Close": st.column_config.NumberColumn("📉 Close", format="%.1f"),
                            "Body_ATR_Multiple": st.column_config.TextColumn("📊 Body/ATR"),
                            "Signal": st.column_config.TextColumn("🎯 Signal"),
                        }
                    )
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Chart section
                st.markdown("""
                <div class="section-header">
                    <h2>📊 Interactive Chart Visualization</h2>
                </div>
                """, unsafe_allow_html=True)
                
                with st.container():
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    
                    chart_df = df.set_index("Time")["Body_ATR_Multiple"]
                    st.bar_chart(chart_df, height=450, use_container_width=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Download section
                st.markdown("""
                <div class="section-header">
                    <h2>💾 Export Data</h2>
                </div>
                """, unsafe_allow_html=True)
                
                csv_data = df.to_csv(index=False)
                st.download_button(
                    "📥 Download Complete Analysis (CSV)",
                    csv_data,
                    file_name=f"{instrument}_{timeframe}_atr_analysis.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
    except requests.exceptions.RequestException as e:
        st.error(f"❌ API Connection Error: {str(e)}")
    except Exception as e:
        st.error(f"❌ Processing Error: {str(e)}")

else:
    # Welcome section
    st.markdown("""
    <div class="welcome-box">
        <h3>🎯 Welcome to Advanced ATR Analysis</h3>
        <p>Configure your settings in the sidebar and click 'Fetch Data' to begin comprehensive candle analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced help section
    with st.expander("📚 Complete User Guide", expanded=False):
        st.markdown("""
        <div class="help-section">
            <h3>🎓 How to Master This Dashboard</h3>
            
            <h4>📋 Quick Start Guide:</h4>
            <ul>
                <li><strong>Select Instrument:</strong> Choose from XAU_USD (Gold), NAS100_USD (NASDAQ), or US30_USD (Dow Jones)</li>
                <li><strong>Pick Timeframe:</strong> H4 for 4-hour candles or D for daily analysis</li>
                <li><strong>Set ATR Value:</strong> Enter your pre-calculated Average True Range</li>
                <li><strong>Choose Time Range:</strong> Select start and end times in Indian Standard Time</li>
                <li><strong>Analyze Results:</strong> Review the comprehensive analysis and charts</li>
            </ul>
            
            <h4>🎯 Signal Interpretation:</h4>
            <ul>
                <li><strong>🟢 Strong (>1.3x ATR):</strong> Large candle bodies indicate strong momentum and trending conditions</li>
                <li><strong>🟡 Neutral (0.7x-1.3x ATR):</strong> Average-sized bodies suggest normal market activity</li>
                <li><strong>🔴 Weak (<0.7x ATR):</strong> Small bodies indicate low momentum, possible consolidation</li>
            </ul>
            
            <h4>💡 Pro Trading Tips:</h4>
            <ul>
                <li>Strong signals often precede continuation moves</li>
                <li>Multiple weak signals may indicate range-bound markets</li>
                <li>Use in conjunction with support/resistance levels</li>
                <li>Higher timeframes provide more reliable signals</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
