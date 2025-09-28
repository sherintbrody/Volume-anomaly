
import requests
import json
import os
import streamlit as st
from datetime import datetime, timedelta, time
import pytz
import pandas as pd
from collections import defaultdict
import wcwidth
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from concurrent.futures import ThreadPoolExecutor, as_completed
import time as time_module
import base64
from functools import lru_cache, wraps
import hashlib

# ====== PAGE CONFIG ======
st.set_page_config(
    page_title="Volume Spike Dashboard Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ====== MODERN THEME & ANIMATIONS ======
MODERN_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --primary-color: #6366f1;
    --success-color: #10b981;
    --warning-color: #f59e0b;
    --danger-color: #ef4444;
    --dark-bg: #0f172a;
    --card-bg: #1e293b;
    --border-color: #334155;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --gradient-1: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    --gradient-2: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    --gradient-3: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    --shadow-lg: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
}

/* Modern Card Design */
.stApp {
    font-family: 'Inter', sans-serif;
}

div[data-testid="stHorizontalBlock"] {
    gap: 1rem;
}

div[data-testid="column"] {
    background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
    padding: 1.5rem;
    border-radius: 16px;
    border: 1px solid rgba(148, 163, 184, 0.1);
    box-shadow: var(--shadow-lg);
    transition: all 0.3s ease;
}

div[data-testid="column"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 25px 30px -5px rgba(0, 0, 0, 0.2);
    border-color: rgba(99, 102, 241, 0.3);
}

/* Animated Badges */
.badge {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin-right: 8px;
    margin-bottom: 8px;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    animation: slideIn 0.5s ease;
    position: relative;
    overflow: hidden;
}

.badge::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
    transition: left 0.5s;
}

.badge:hover::before {
    left: 100%;
}

.badge.primary {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    box-shadow: 0 4px 6px rgba(99, 102, 241, 0.3);
}

.badge.success {
    background: linear-gradient(135deg, #10b981, #34d399);
    color: white;
    box-shadow: 0 4px 6px rgba(16, 185, 129, 0.3);
}

.badge.warning {
    background: linear-gradient(135deg, #f59e0b, #fbbf24);
    color: white;
    box-shadow: 0 4px 6px rgba(245, 158, 11, 0.3);
}

.badge.danger {
    background: linear-gradient(135deg, #ef4444, #f87171);
    color: white;
    box-shadow: 0 4px 6px rgba(239, 68, 68, 0.3);
}

.badge.info {
    background: linear-gradient(135deg, #3b82f6, #60a5fa);
    color: white;
    box-shadow: 0 4px 6px rgba(59, 130, 246, 0.3);
}

/* Pulse Animation for Alerts */
.pulse {
    animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

@keyframes slideIn {
    from {
        opacity: 0;
        transform: translateY(-10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Modern Metrics */
div[data-testid="metric-container"] {
    background: linear-gradient(145deg, #1e293b, #334155);
    padding: 1rem;
    border-radius: 12px;
    border: 1px solid rgba(148, 163, 184, 0.1);
    transition: all 0.3s ease;
}

div[data-testid="metric-container"]:hover {
    transform: scale(1.02);
    border-color: rgba(99, 102, 241, 0.5);
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
}

/* Glowing Effect for Spike Alerts */
.spike-alert {
    animation: glow 1.5s ease-in-out infinite alternate;
}

@keyframes glow {
    from { box-shadow: 0 0 10px #ef4444; }
    to { box-shadow: 0 0 20px #ef4444, 0 0 30px #ef4444; }
}

/* Modern Table Styling */
div[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

/* Loading Animation */
.loading-bar {
    height: 4px;
    background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 50%, #6366f1 100%);
    background-size: 200% 100%;
    animation: loading 1.5s linear infinite;
    border-radius: 2px;
}

@keyframes loading {
    0% { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}

/* Notification Style */
.notification {
    position: fixed;
    top: 20px;
    right: 20px;
    padding: 16px 24px;
    background: linear-gradient(135deg, #ef4444, #f87171);
    color: white;
    border-radius: 12px;
    box-shadow: 0 10px 25px rgba(239, 68, 68, 0.3);
    animation: slideInRight 0.5s ease;
    z-index: 1000;
}

@keyframes slideInRight {
    from {
        transform: translateX(100%);
        opacity: 0;
    }
    to {
        transform: translateX(0);
        opacity: 1;
    }
}
</style>
"""

st.markdown(MODERN_CSS, unsafe_allow_html=True)

# ====== SOUND ALERT SETUP ======
def create_sound_alert():
    """Generate base64 encoded sound alert"""
    sound_html = """
    <audio id="spike-alert-sound" style="display: none;">
        <source src="data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhByx9v+7unEcIC1Kw5e+lVA0MSG336r9sIAY+l9n1uHkoBSd+v+3ijkwKFmC47OWlVA0BSLP06rVnJwUugM/yz3UqBi9+veTenEwNGWS46+OeTQ0NWLHm7qpWFwhBm9/yvWwhBjtOy/fNdykELYLL7+CQSQ0XZrTu66hVFApGn+DyvmwhByx7v+7rnUYIC1Kw5e+lVA0MSG336bttJAU7nNPyvHYqBCh+v+3ijkwKFmC45u2pVhUITbLn77BdIQYnhtDwyHkpBDSHz+/bkEUJDlux5+2sVg0HVbPm77BdIARGnt3yvmwhBSuBzvLYiTcIHmjG98KWWCQJaK3oy4pMDAReoOXuvWogBz2Vz+2+eB8HM4fH8N2WRwUYaL3r56pWFABVrOfvp1YOD166+OurVRQKRp/g8r5sIAUthM/x1YQ+Cxxqy+7Ngy4GNpLN7tiJNwgZaLvt559NEAxPqOPwtmMcBjiP1/PMeS0GI3fH8N+RQAoUXrTp66hVFApGnt/yvmwhByx9v+7unEcIC1Kw5e+lVA0MSG336r9sIAY+l9n1uHkoBSd+v+3ijkwKFmC48OWlVA0BSLP06rVnJwUugM/yz3UqBi9+veTenEwNGWS46+OeTQ0NWLHm7qpWFwhBm9/yvWwhBjtOy/fNdykELYLL7+CQSQ0XZrTu66hVFApGn+DyvmwhByx7v+7rnUYIC1Kw5e+lVA0MSG336bttJAU7nNPyvHYqBCh+v+3ijkwKFmC45u2pVhUITbLn77BdIQYnhtDwyHkpBDSHz+/bkEUJDlux5+2sVg0HVbPm77BdIARGnt3yvmwhBSuBzvLYiTcIHmjG98KWWCQJaK3oy4pMDAReoOXuvWogBz2Vz+2+eB8HM4fH8N2WRwUYaL3r56pWFABVrOfvp1YOD166+OurVRQKRp/g8r5sIAUthM/x1YQ+Cxxqy+7Ngy4GNpLN7tiJNwgZaLvt559NEAxPqOPwtmMcBjiP1/PMeS0GI3fH8N+RQAoUXrTp66hVFApGnt/yvmwhByx9v+7unEcIC1Kw5e+lVA0MSG336r9sIAY+l9n1uHkoBSd+v+3ijkwKFmC48OWlVA0BSLP06rVnJwUugM/yz3UqBi9+veTenEwNGWS46+OeTQ0NWLHm7qpWFwhBm9/yvWwhBjtOy/fNdykELYLL7+CQSQ0XZrTu66hVFApGn+DyvmwhByx7v+7rnUYIC1Kw5e+lVA0MSG336bttJAU7nNPyvHYqBCh+v+3ijkwKFmC45u2pVhUITbLn77BdIQYnhtDwyHkpBDSHz+/bkEUJDlux5+2sVg0HVbPm77BdIARGnt3yvmwhBSuBzvLYiTcIHmjG98KWWCQJaK3oy4pMDAReoOXuvWogBz2Vz+2+eB8HM4fH8N2WRwUYaL3r56pWFABVrOfvp1YOD166+OurVRQKRp/g8r5sIAUthM/x1YQ+Cxxqy+7Ngy4GNpLN7tiJNwgZaLvt559NEAxPqOPwtmMcBjiP1/PMeS0GI3fH8N+RQAoUXrTp66hVFApGnt/yvmwhByx9v+7unEcIC1Kw5e+lVA0MSG336r9sIAY+l9n1uHkoBSd+v+3ijkwKFmC48OWlVA0BSLP06rVnJwUugM/yz3UqBi9+veTenEwNGWS46+OeTQ0NWLHm7qpWFwhBm9/yvWwhBjtOy/fNdykELYLL7+CQSQ0XZrTu66hVFApGn+DyvmwhByx7v+7rnUYIC1Kw5e+lVA0MSG336bttJAU7nNPyvHYqBCh+v+3ijkwKFmC45u2pVhUITbLn77BdIQYnhtDwyHkpBDSHz+/bkEUJDlux5+2sVg0HVbPm77BdIARGnt3yvmwhBSuBzvLYiTcIHmjG98KWWCQJaK3oy4pMDAReoOXuvWogBz2Vz+2+eB8HM4fH8N2WRwUYaL3r56pWFABVrOfvp1YOD166+OurVRQKRp/g8r5sIAUthM/x1YQ+Cxxqy+7Ngy4GNpLN7tiJNwgZaLvt55c=" type="audio/wav">
    </audio>
    <script>
        function playAlert() {
            var audio = document.getElementById('spike-alert-sound');
            audio.play();
        }
    </script>
    """
    return sound_html
    
# --- Telegram secrets (open keys) ---
TELEGRAM_BOT_TOKEN = st.secrets["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = st.secrets["TELEGRAM_CHAT_ID"]

def send_telegram_alert(message: str):
    """Send a formatted alert message to Telegram."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        r = requests.post(url, json=payload, timeout=10)
        r.raise_for_status()  # will raise if not 200
    except Exception as e:
        st.error(f"Telegram send failed: {e}")


        
# ====== SECURE CONFIG ======
# Check if we have secrets (Streamlit Cloud) or use fallback
try:
    # --- Twelve Data API ---
    API_KEY = st.secrets["API_KEY"]
    ACCOUNT_ID = st.secrets["ACCOUNT_ID"]
    BASE_URL = "https://api-fxpractice.oanda.com/v3"
    TELEGRAM_BOT_TOKEN = st.secrets.get("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID = st.secrets.get("TELEGRAM_CHAT_ID", "")
except:
    # Fallback for local development - REMOVE THESE IN PRODUCTION
    API_KEY = os.environ.get("API_KEY", "")
    ACCOUNT_ID = os.environ.get("ACCOUNT_ID", "")
    TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

if not API_KEY or not ACCOUNT_ID:
    st.error("⚠️ Please configure OANDA_API_KEY and OANDA_ACCOUNT_ID in .streamlit/secrets.toml")
    st.info("""
    Create a file `.streamlit/secrets.toml` with:
    ```
    OANDA_API_KEY = "your-api-key"
    OANDA_ACCOUNT_ID = "your-account-id"
    TELEGRAM_BOT_TOKEN = "optional-bot-token"
    TELEGRAM_CHAT_ID = "optional-chat-id"
    ```
    """)
    st.stop()

BASE_URL = "https://api-fxpractice.oanda.com/v3"

INSTRUMENTS = {
    "XAUUSD": "XAU_USD",
    "NAS100": "NAS100_USD",
    "US30": "US30_USD"
}

IST = pytz.timezone("Asia/Kolkata")
UTC = pytz.utc
headers = {"Authorization": f"Bearer {API_KEY}"}

ALERT_STATE_FILE = "last_alert_state.json"
ALERT_DATE_FILE = "last_alert_date.txt"
DISPLAY_ROWS = 13
TRADING_DAYS_FOR_AVERAGE = 21

# ====== PERFORMANCE OPTIMIZATION ======
class RateLimiter:
    """Rate limiter for API calls"""
    def __init__(self, calls_per_second=2):
        self.calls_per_second = calls_per_second
        self.last_call = 0
        self.min_interval = 1.0 / calls_per_second
    
    def wait_if_needed(self):
        elapsed = time_module.time() - self.last_call
        if elapsed < self.min_interval:
            time_module.sleep(self.min_interval - elapsed)
        self.last_call = time_module.time()

rate_limiter = RateLimiter(calls_per_second=2)

# ====== ENHANCED CACHING ======
def cache_key_generator(*args, **kwargs):
    """Generate cache key for complex arguments"""
    key_str = str(args) + str(sorted(kwargs.items()))
    return hashlib.md5(key_str.encode()).hexdigest()

def smart_cache(ttl_seconds=300):
    """Smart caching decorator with TTL"""
    def decorator(func):
        cache = {}
        cache_time = {}
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = cache_key_generator(*args, **kwargs)
            now = time_module.time()
            
            if key in cache and (now - cache_time[key]) < ttl_seconds:
                return cache[key]
            
            result = func(*args, **kwargs)
            cache[key] = result
            cache_time[key] = now
            return result
        
        return wrapper
    return decorator

# ====== BATCH API CALLS ======
def batch_fetch_candles(instruments, from_time, to_time, granularity="M15"):
    """Fetch candles for multiple instruments in parallel"""
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {}
        for name, code in instruments.items():
            future = executor.submit(
                fetch_candles_optimized, 
                code, from_time, to_time, granularity
            )
            futures[future] = name
        
        results = {}
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception as e:
                st.error(f"Error fetching {name}: {e}")
                results[name] = []
        
        return results

@st.cache_data(ttl=300, show_spinner=False)
def fetch_candles_optimized(instrument_code, from_time, to_time, granularity="M15"):
    """Optimized candle fetching with rate limiting and retry"""
    rate_limiter.wait_if_needed()
    
    # Ensure times are UTC
    now_utc = datetime.now(UTC)
    from_time = min(from_time, now_utc)
    to_time = min(to_time, now_utc)
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            params = {
                "granularity": granularity,
                "price": "M",
                "from": from_time.isoformat(),
                "to": to_time.isoformat()
            }
            url = f"{BASE_URL}/accounts/{ACCOUNT_ID}/instruments/{instrument_code}/candles"
            
            resp = requests.get(url, headers=headers, params=params, timeout=10)
            
            if resp.status_code == 200:
                return resp.json().get("candles", [])
            elif attempt < max_retries - 1:
                time_module.sleep(2 ** attempt)
            else:
                st.error(f"API Error: {resp.status_code} - {resp.text}")
                return []
                
        except Exception as e:
            if attempt < max_retries - 1:
                time_module.sleep(2 ** attempt)
            else:
                st.error(f"Network error: {e}")
                return []
    
    return []

# ====== ALERT MEMORY ======
def load_alerted_candles():
    if os.path.exists(ALERT_STATE_FILE):
        try:
            with open(ALERT_STATE_FILE, "r") as f:
                return set(json.load(f))
        except:
            return set()
    return set()

def save_alerted_candles(alerted_set):
    with open(ALERT_STATE_FILE, "w") as f:
        json.dump(list(alerted_set), f)

def reset_if_new_day():
    today = datetime.now(IST).date().isoformat()
    if os.path.exists(ALERT_DATE_FILE):
        with open(ALERT_DATE_FILE, "r") as f:
            last = f.read().strip()
        if last != today:
            with open(ALERT_STATE_FILE, "w") as f:
                f.write("[]")
    with open(ALERT_DATE_FILE, "w") as f:
        f.write(today)

# ====== SIDEBAR CONFIG ======
st.sidebar.markdown("## ⚙️ **Control Panel**")

# Initialize session state
if "selected_instruments" not in st.session_state:
    st.session_state.selected_instruments = list(INSTRUMENTS.keys())
if "refresh_minutes" not in st.session_state:
    st.session_state.refresh_minutes = 5
if "bucket_choice" not in st.session_state:
    st.session_state.bucket_choice = "1 hour"
if "enable_telegram_alerts" not in st.session_state:
    st.session_state.enable_telegram_alerts = False
if "enable_sound_alerts" not in st.session_state:
    st.session_state.enable_sound_alerts = True
if "candle_size" not in st.session_state:
    st.session_state.candle_size = "15 min"
if "skip_weekends" not in st.session_state:
    st.session_state.skip_weekends = True
if "show_charts" not in st.session_state:
    st.session_state.show_charts = True

with st.sidebar.expander("📊 **Instruments**", expanded=True):
    st.multiselect(
        "Select to Monitor",
        options=list(INSTRUMENTS.keys()),
        default=st.session_state.selected_instruments,
        key="selected_instruments"
    )


with st.sidebar.expander("⏱️ **Timing Settings**", expanded=True):
    st.slider(
        "Auto-refresh (minutes)",
        min_value=1, max_value=15,
        value=st.session_state.refresh_minutes,
        key="refresh_minutes"
    )
    
    st.radio(
        "Candle Size",
        ["15 min", "4 hour"],
        index=["15 min", "4 hour"].index(st.session_state.candle_size),
        key="candle_size"
    )
    
    if st.session_state.candle_size == "15 min":
        st.radio(
            "Time Bucket",
            ["15 min", "30 min", "1 hour"],
            index=["15 min", "30 min", "1 hour"].index(st.session_state.bucket_choice),
            key="bucket_choice"
        )

with st.sidebar.expander("🔔 **Alerts**", expanded=True):
    st.toggle(
        "🔊 Sound Alerts",
        value=st.session_state.enable_sound_alerts,
        key="enable_sound_alerts"
    )
    
    st.toggle(
        "📱 Telegram Alerts",
        value=st.session_state.enable_telegram_alerts,
        key="enable_telegram_alerts"
    )
    
    st.slider(
        "Threshold Multiplier",
        min_value=1.0,
        max_value=3.0,
        step=0.1,
        value=1.618,
        key="threshold_multiplier"
    )

with st.sidebar.expander("📈 **Display Options**", expanded=False):
    st.toggle(
        "Show Charts",
        value=st.session_state.show_charts,
        key="show_charts"
    )
    
    st.toggle(
        "Skip Weekends",
        value=st.session_state.skip_weekends,
        key="skip_weekends"
    )

with st.sidebar.expander("🧪 Test Telegram", expanded=False):
    if st.button("Send Test Alert"):
        resp = send_telegram_alert("✅ Test message from Streamlit dashboard")
        st.write("Telegram response:", resp)

# ====== AUTO-REFRESH ======
refresh_ms = st.session_state.refresh_minutes * 60 * 1000
refresh_count = st_autorefresh(interval=refresh_ms, limit=None, key="volume-refresh")

# ====== TELEGRAM ALERT ======
def send_telegram_alert(message):
    if not st.session_state.enable_telegram_alerts:
        return
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        resp = requests.post(url, data=payload, timeout=10)
    except:
        pass

# ====== DATA PROCESSING ======
def get_market_session(dt_ist):
    """Identify current market session"""
    hour = dt_ist.hour
    if 5 <= hour < 13:
        return "🌏 Asian", "info"
    elif 13 <= hour < 17:
        return "🌍 European", "primary"
    elif 17 <= hour < 22:
        return "🌎 US", "warning"
    else:
        return "🌙 After Hours", "secondary"

def get_time_bucket(dt_ist, bucket_size_minutes):
    bucket_start_minute = (dt_ist.minute // bucket_size_minutes) * bucket_size_minutes
    bucket_start = dt_ist.replace(minute=bucket_start_minute, second=0, microsecond=0)
    bucket_end = bucket_start + timedelta(minutes=bucket_size_minutes)
    return f"{bucket_start.strftime('%I:%M %p')}–{bucket_end.strftime('%I:%M %p')}"

def get_4h_time_range(dt_ist):
    end_time = dt_ist + timedelta(hours=4)
    return f"{dt_ist.strftime('%I:%M %p')}–{end_time.strftime('%I:%M %p')}"

def get_sentiment(candle):
    o = float(candle["mid"]["o"])
    c = float(candle["mid"]["c"])
    return "🟩" if c > o else "🟥" if c < o else "▪️"

def get_body_percentage(candle):
    try:
        o = float(candle["mid"]["o"])
        h = float(candle["mid"]["h"])
        l = float(candle["mid"]["l"])
        c = float(candle["mid"]["c"])
        body = abs(c - o)
        total_range = h - l
        if total_range == 0:
            return "0.0%"
        body_pct = (body / total_range) * 100
        return f"{body_pct:.1f}%"
    except:
        return "—"

def is_weekend(date):
    return date.weekday() in [5, 6]

@st.cache_data(ttl=600)
def compute_lazy_averages(code, bucket_size_minutes, granularity, skip_weekends=True):
    """Lazy load and compute averages"""
    if granularity == "H4":
        return compute_4h_position_averages(code, skip_weekends)
    else:
        return compute_15m_bucket_averages(code, bucket_size_minutes, skip_weekends)

def compute_15m_bucket_averages(code, bucket_size_minutes, skip_weekends=True):
    bucket_volumes = defaultdict(list)
    today_ist = datetime.now(IST).date()
    now_utc = datetime.now(UTC)
    
    trading_days_collected = 0
    days_back = 1
    max_lookback = 60
    
    while trading_days_collected < TRADING_DAYS_FOR_AVERAGE and days_back < max_lookback:
        day_ist = today_ist - timedelta(days=days_back)
        
        if skip_weekends and is_weekend(day_ist):
            days_back += 1
            continue
            
        start_ist = IST.localize(datetime.combine(day_ist, time(0, 0)))
        end_ist = IST.localize(datetime.combine(day_ist + timedelta(days=1), time(0, 0)))
        
        start_utc = start_ist.astimezone(UTC)
        end_utc = min(end_ist.astimezone(UTC), now_utc)
        
        candles = fetch_candles_optimized(code, start_utc, end_utc, granularity="M15")
        
        if candles:
            trading_days_collected += 1
            
            for c in candles:
                if not c.get("complete", True):
                    continue
                try:
                    t_utc = datetime.strptime(c["time"], "%Y-%m-%dT%H:%M:%S.%f000Z")
                except ValueError:
                    t_utc = datetime.strptime(c["time"], "%Y-%m-%dT%H:%M:%S.000Z")
                t_ist = t_utc.replace(tzinfo=UTC).astimezone(IST)
                bucket = get_time_bucket(t_ist, bucket_size_minutes)
                bucket_volumes[bucket].append(c["volume"])
        
        days_back += 1
    
    return {b: (sum(vs) / len(vs)) for b, vs in bucket_volumes.items() if vs}

def compute_4h_position_averages(code, skip_weekends=True):
    position_volumes = defaultdict(list)
    today_ist = datetime.now(IST).date()
    now_utc = datetime.now(UTC)
    
    trading_days_collected = 0
    days_back = 1
    max_lookback = 60
    
    while trading_days_collected < TRADING_DAYS_FOR_AVERAGE and days_back < max_lookback:
        day_ist = today_ist - timedelta(days=days_back)
        
        if skip_weekends and is_weekend(day_ist):
            days_back += 1
            continue
            
        start_ist = IST.localize(datetime.combine(day_ist, time(0, 0)))
        end_ist = IST.localize(datetime.combine(day_ist + timedelta(days=1), time(0, 0)))
        
        start_utc = start_ist.astimezone(UTC)
        end_utc = min(end_ist.astimezone(UTC), now_utc)
        
        candles = fetch_candles_optimized(code, start_utc, end_utc, granularity="H4")
        
        if candles:
            trading_days_collected += 1
            
            for c in candles:
                if not c.get("complete", True):
                    continue
                try:
                    t_utc = datetime.strptime(c["time"], "%Y-%m-%dT%H:%M:%S.%f000Z")
                except ValueError:
                    t_utc = datetime.strptime(c["time"], "%Y-%m-%dT%H:%M:%S.000Z")
                t_ist = t_utc.replace(tzinfo=UTC).astimezone(IST)
                time_range = get_4h_time_range(t_ist)
                position_volumes[time_range].append(c["volume"])
        
        days_back += 1
    
    return {p: (sum(vs) / len(vs)) for p, vs in position_volumes.items() if vs}

# ====== CHART CREATION ======
def create_volume_chart(name, rows, threshold, is_spike):
    """Create modern interactive volume chart"""
    if not rows or not st.session_state.show_charts:
        return None
    
    # Convert rows to DataFrame
    df_data = []
    for row in rows[-20:]:
        df_data.append({
            "Time": row[0],
            "Open": float(row[2]),
            "High": float(row[3]),
            "Low": float(row[4]),
            "Close": float(row[5]),
            "Volume": row[6],
            "Spike": row[7],
            "Sentiment": row[8]
        })
    
    df = pd.DataFrame(df_data)
    
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.03,
        subplot_titles=(f"{name} Price", "Volume Analysis")
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df["Time"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
            increasing_line_color='#10b981',
            decreasing_line_color='#ef4444'
        ),
        row=1, col=1
    )
    
    # Volume bars with spike highlighting
    colors = ['#ef4444' if spike else '#6366f1' for spike in df["Spike"]]
    
    fig.add_trace(
        go.Bar(
            x=df["Time"],
            y=df["Volume"],
            name="Volume",
            marker_color=colors,
            hovertemplate='Volume: %{y}<br>Time: %{x}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Add threshold line
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="orange",
        annotation_text=f"Threshold: {threshold:.0f}",
        row=2, col=1
    )
    
    # Update layout for modern look
    fig.update_layout(
        template="plotly_dark",
        height=500,
        showlegend=False,
        hovermode='x unified',
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor='rgba(30, 41, 59, 0.5)',
        plot_bgcolor='rgba(30, 41, 59, 0.5)',
        font=dict(color='#94a3b8')
    )
    
    # Hide x-axis labels and ticks completely
    fig.update_xaxes(
        showticklabels=False,
        showgrid=False,
        zeroline=False
    )
    
    # Keep y-axis grid styling
    fig.update_yaxes(gridcolor='rgba(51, 65, 85, 0.3)')
    
    return fig

# ====== MAIN PROCESSING ======
def process_instrument_optimized(name, code, bucket_size_minutes, granularity, alerted_candles):
    """Optimized instrument processing with lazy loading"""
    bucket_avg = compute_lazy_averages(code, bucket_size_minutes, granularity, skip_weekends=st.session_state.skip_weekends)
    now_utc = datetime.now(UTC)
    is_4h_mode = (granularity == "H4")
    
    per_candle_minutes = 15 if granularity == "M15" else 240
    candles_needed = 40 if granularity == "M15" else 26
    from_time = now_utc - timedelta(minutes=per_candle_minutes * candles_needed)
    
    candles = fetch_candles_optimized(code, from_time, now_utc, granularity=granularity)
    if not candles:
        return [], [], {}, 0
    
    rows = []
    spikes_last_two = []
    last_two_candles = candles[-2:] if len(candles) >= 2 else candles
    last_summary = {}
    threshold_value = 0
    
    for c in candles:
        try:
            t_utc = datetime.strptime(c["time"], "%Y-%m-%dT%H:%M:%S.%f000Z")
        except ValueError:
            t_utc = datetime.strptime(c["time"], "%Y-%m-%dT%H:%M:%S.000Z")
        t_ist = t_utc.replace(tzinfo=UTC).astimezone(IST)
        
        if is_4h_mode:
            bucket = get_4h_time_range(t_ist)
        else:
            bucket = get_time_bucket(t_ist, bucket_size_minutes)
        
        vol = c["volume"]
        avg = bucket_avg.get(bucket, 0)
        threshold_multiplier = st.session_state.threshold_multiplier
        threshold = avg * threshold_multiplier if avg else 0
        threshold_value = threshold
        over = (threshold > 0 and vol > threshold)
        mult = (vol / threshold) if over and threshold > 0 else (vol / avg if avg else 0)
        
        spike_diff = f"▲{vol - int(threshold)}" if over else ""
        sentiment = get_sentiment(c)
        
        rows.append([
            t_ist.strftime("%Y-%m-%d %I:%M %p"),
            bucket,
            f"{float(c['mid']['o']):.1f}",
            f"{float(c['mid']['h']):.1f}",
            f"{float(c['mid']['l']):.1f}",
            f"{float(c['mid']['c']):.1f}",
            vol,
            over,
            sentiment
        ])
        
        last_summary = {
            "time": t_ist.strftime("%Y-%m-%d %I:%M %p"),
            "bucket": bucket,
            "volume": vol,
            "avg": avg,
            "threshold": threshold,
            "multiplier": mult,
            "over": over,
            "sentiment": sentiment
        }
        
        if c in last_two_candles and over:
            candle_id = f"{name}_{c['time']}_{round(float(c['mid']['o']), 2)}"
            if candle_id not in alerted_candles:
                spikes_last_two.append({
                    "name": name,
                    "time": t_ist.strftime('%I:%M %p'),
                    "volume": vol,
                    "spike": spike_diff,
                    "sentiment": sentiment
                })
                alerted_candles.add(candle_id)
    
    return rows, spikes_last_two, last_summary, threshold_value

def render_modern_card(name, rows, bucket_minutes, summary, threshold, is_4h_mode=False):
    """Render modern instrument card with enhanced UI"""
    
    # Header with gradient background
    st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 12px 20px; 
                    border-radius: 12px 12px 0 0; 
                    margin: -15px -15px 20px -15px;">
            <h3 style="color: white; margin: 0;">{name}</h3>
        </div>
    """, unsafe_allow_html=True)
    
    # Market session
    session, session_type = get_market_session(datetime.now(IST))
    
    # Status badges
    if summary:
        badges = [
            f'<span class="badge {session_type}">{session}</span>',
            f'<span class="badge {"success pulse" if summary["over"] else "info"}">{"🚨 SPIKE" if summary["over"] else "Normal"}</span>',
            f'<span class="badge primary">×{summary["multiplier"]:.2f}</span>',
        ]
        st.markdown(f'<div style="margin-bottom: 20px;">{" ".join(badges)}</div>', unsafe_allow_html=True)
        
        # Metrics with modern styling
        cols = st.columns(4)
        cols[0].metric("📊 Volume", f"{summary['volume']:,}", 
                      delta=f"+{summary['volume'] - summary['avg']:.0f}" if summary['over'] else None)
        cols[1].metric("📈 Average", f"{summary['avg']:.0f}")
        cols[2].metric("🎯 Threshold", f"{summary['threshold']:.0f}")
        cols[3].metric("⚡ Status", "Spike!" if summary['over'] else "Normal",
                      delta_color="normal" if summary['over'] else "off")
    
    # Chart
    if st.session_state.show_charts:
        chart = create_volume_chart(name, rows, threshold, summary.get("over", False))
        if chart:
            st.plotly_chart(chart, use_container_width=True, config={'displayModeBar': False})
    
    # Data table with modern styling
    with st.expander("📋 Detailed Data", expanded=False):
        # Prepare DataFrame
        df_data = []
        for row in rows[-DISPLAY_ROWS:]:
            df_data.append({
                "Time": row[0],
                "Bucket": row[1],
                "Open": row[2],
                "High": row[3],
                "Low": row[4],
                "Close": row[5],
                "Volume": row[6],
                "Spike": "Yes" if row[7] else "No",
                "Sentiment": row[8]
            })
        
        df = pd.DataFrame(df_data)
        
        st.dataframe(df, height=300, use_container_width=True)
        
        # Export button
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="💾 Export CSV",
            data=csv,
            file_name=f"{name}_volume_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

# ====== MAIN DASHBOARD ======
def run_modern_dashboard():
    """Main dashboard with modern UI and optimizations"""
    reset_if_new_day()
    alerted_candles = load_alerted_candles()
    
    # Sound alert container
    if st.session_state.enable_sound_alerts:
        st.markdown(create_sound_alert(), unsafe_allow_html=True)
    
    # Header
    st.markdown("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 16px; margin-bottom: 30px;">
            <h1 style="color: white; font-size: 2.5rem; margin: 0;">📊 Volume Spike Dashboard Pro</h1>
            <p style="color: rgba(255,255,255,0.9); margin-top: 10px;">Real-time Volume Anomaly Detection System</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Loading bar animation
    st.markdown('<div class="loading-bar"></div>', unsafe_allow_html=True)
    
    if not st.session_state.selected_instruments:
        st.warning("⚠️ Please select at least one instrument from the sidebar")
        return
    
    # Settings summary
    col1, col2 = st.columns([3, 1])
    with col1:
        now_ist = datetime.now(IST).strftime("%Y-%m-%d %I:%M %p")
        session, session_type = get_market_session(datetime.now(IST))
        
        badges = [
            f'<span class="badge info">🕒 {now_ist} IST</span>',
            f'<span class="badge {session_type}">{session}</span>',
            f'<span class="badge primary">Candle: {st.session_state.candle_size}</span>',
            f'<span class="badge {"success" if st.session_state.enable_sound_alerts else "warning"}">🔊 {"ON" if st.session_state.enable_sound_alerts else "OFF"}</span>',
            f'<span class="badge {"success" if st.session_state.enable_telegram_alerts else "warning"}">📱 {"ON" if st.session_state.enable_telegram_alerts else "OFF"}</span>',
        ]
        st.markdown(f'<div style="margin-bottom: 20px;">{" ".join(badges)}</div>', unsafe_allow_html=True)
    
    with col2:
        if st.button("🔄 Refresh Now", use_container_width=True):
            st.rerun()
    
    # Determine settings
    if st.session_state.candle_size == "4 hour":
        granularity = "H4"
        bucket_minutes = 240
        is_4h_mode = True
    else:
        granularity = "M15"
        bucket_minutes = {"15 min": 15, "30 min": 30, "1 hour": 60}[st.session_state.bucket_choice]
        is_4h_mode = False
    
    # Batch fetch data for all instruments
    selected_inst = {name: INSTRUMENTS[name] for name in st.session_state.selected_instruments}
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    all_spike_alerts = []
    spike_detected = False
    
    # Process instruments
    cols = st.columns(2) if len(selected_inst) > 1 else [st.container()]
    
    for idx, (name, code) in enumerate(selected_inst.items()):
        progress = (idx + 1) / len(selected_inst)
        progress_bar.progress(progress)
        status_text.text(f"Processing {name}...")
        
        with cols[idx % len(cols)]:
            with st.container():
                st.markdown("""
                    <div style="padding: 20px; 
                                background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%); 
                                border-radius: 16px; 
                                border: 1px solid rgba(148, 163, 184, 0.1);
                                margin-bottom: 20px;">
                """, unsafe_allow_html=True)
                
                rows, spikes, summary, threshold = process_instrument_optimized(
                    name, code, bucket_minutes, granularity, alerted_candles
                )
                
                if rows:
                    render_modern_card(name, rows, bucket_minutes, summary, threshold, is_4h_mode)
                    
                    if spikes:
                        all_spike_alerts.extend(spikes)
                        spike_detected = True
                else:
                    st.error(f"No data available for {name}")
                
                st.markdown("</div>", unsafe_allow_html=True)
    
    progress_bar.empty()
    status_text.empty()
    
    # Handle spike alerts
    if spike_detected and all_spike_alerts:
        # Play sound alert
        if st.session_state.enable_sound_alerts:
            st.markdown('<script>playAlert();</script>', unsafe_allow_html=True)
        
        # Show notification
        st.markdown(f"""
            <div class="notification">
                🚨 Volume Spike Detected! Check {len(all_spike_alerts)} alert(s)
            </div>
        """, unsafe_allow_html=True)
        
        # Send Telegram alert
        if st.session_state.enable_telegram_alerts:
            alert_msg = "⚡ *Volume Spike Alert*\n\n"
            for spike in all_spike_alerts:
                alert_msg += f"📊 *{spike['name']}*\n"
                alert_msg += f"🕒 Time: {spike['time']}\n"
                alert_msg += f"📈 Volume: {spike['volume']:,} {spike['spike']}\n"
                alert_msg += f"📊 Sentiment: {spike['sentiment']}\n\n"
            send_telegram_alert(alert_msg)
    
    # Save state
    save_alerted_candles(alerted_candles)
    
    # Footer
    st.markdown("""
        <div style="text-align: center; 
                    padding: 20px; 
                    margin-top: 40px; 
                    border-top: 1px solid rgba(148, 163, 184, 0.1);">
            <p style="color: #94a3b8; font-size: 0.9rem;">
                Volume Spike Dashboard Pro v2.0 | Auto-refresh: {refresh} min | 
                Last update: {time}
            </p>
        </div>
    """.format(
        refresh=st.session_state.refresh_minutes,
        time=datetime.now(IST).strftime("%I:%M:%S %p")
    ), unsafe_allow_html=True)

# ====== MAIN EXECUTION ======
if __name__ == "__main__":
    run_modern_dashboard()

