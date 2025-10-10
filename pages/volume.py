import requests, json, os
import streamlit as st
from datetime import datetime, timedelta, time
import pytz
import pandas as pd
from collections import defaultdict
import wcwidth
from streamlit_autorefresh import st_autorefresh

# ====== PAGE CONFIG ======
st.set_page_config(
    page_title="Volume Spike Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ====== SUPABASE SETUP ======
SUPABASE_URL = st.secrets.get("SUPABASE_URL", "")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", "")

@st.cache_resource
def init_supabase():
    """Initialize custom Supabase client with httpx"""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None
    
    try:
        import httpx
        
        class SimpleSupabaseClient:
            def __init__(self, url, key):
                self.url = url.rstrip('/')
                self.key = key
                self.headers = {
                    "apikey": key,
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                    "Prefer": "return=representation"
                }
                self.client = httpx.Client(timeout=30.0)
            
            def table(self, table_name):
                return SupabaseTable(self.url, table_name, self.headers, self.client)
        
        class SupabaseTable:
            def __init__(self, base_url, table_name, headers, client):
                self.url = f"{base_url}/rest/v1/{table_name}"
                self.headers = headers.copy()
                self.client = client
                self.filters = {}
                self.order_by = None
                self.limit_val = None
                self.columns = "*"
            
            def select(self, columns="*", count=None):
                self.columns = columns
                if count:
                    self.headers['Prefer'] = f"count={count}"
                return self
            
            def eq(self, column, value):
                self.filters[column] = f"eq.{value}"
                return self
            
            def gte(self, column, value):
                self.filters[column] = f"gte.{value}"
                return self
            
            def lte(self, column, value):
                self.filters[column] = f"lte.{value}"
                return self
            
            def neq(self, column, value):
                self.filters[column] = f"neq.{value}"
                return self
            
            def order(self, column, desc=False):
                self.order_by = f"{column}.{'desc' if desc else 'asc'}"
                return self
            
            def limit(self, n):
                self.limit_val = n
                return self
            
            def execute(self):
                params = {'select': self.columns}
                
                for col, filter_val in self.filters.items():
                    params[col] = filter_val
                
                if self.order_by:
                    params['order'] = self.order_by
                
                if self.limit_val:
                    params['limit'] = self.limit_val
                
                try:
                    response = self.client.get(self.url, headers=self.headers, params=params)
                    response.raise_for_status()
                    
                    class Result:
                        def __init__(self, data):
                            self.data = data
                            self.count = len(data) if data else 0
                    
                    return Result(response.json())
                except:
                    class Result:
                        def __init__(self):
                            self.data = []
                            self.count = 0
                    return Result()
            
            def upsert(self, data, on_conflict=None):
                try:
                    headers = self.headers.copy()
                    headers['Prefer'] = "resolution=merge-duplicates"
                    
                    response = self.client.post(
                        self.url,
                        headers=headers,
                        json=data if isinstance(data, list) else [data]
                    )
                    response.raise_for_status()
                    
                    class Result:
                        def __init__(self, data):
                            self.data = data
                    
                    return Result(response.json())
                except:
                    class Result:
                        def __init__(self):
                            self.data = []
                    return Result()
            
            def delete(self):
                try:
                    params = {}
                    for col, filter_val in self.filters.items():
                        params[col] = filter_val
                    
                    response = self.client.delete(self.url, headers=self.headers, params=params)
                    response.raise_for_status()
                    
                    class Result:
                        def __init__(self):
                            self.data = []
                    
                    return Result()
                except:
                    class Result:
                        def __init__(self):
                            self.data = []
                    return Result()
        
        return SimpleSupabaseClient(SUPABASE_URL, SUPABASE_KEY)
    
    except:
        return None

supabase = init_supabase()

# ====== THEME/STYLE ======
BADGE_CSS = """
<style>
:root {
    --chip-bg: rgba(2,132,199,.12);
    --chip-fg: #0284c7;
    --chip-br: rgba(2,132,199,.25);
    --chip2-bg: rgba(148,163,184,.12);
    --chip2-fg: #334155;
    --chip2-br: rgba(148,163,184,.25);
    --ok-bg: rgba(16,185,129,.12);
    --ok-fg: #059669;
    --ok-br: rgba(16,185,129,.25);
    --warn-bg: rgba(234,179,8,.12);
    --warn-fg: #a16207;
    --warn-br: rgba(234,179,8,.25);
}
.badges { margin: 2px 0 12px 0; }
.badge {
    display:inline-block;
    padding:4px 10px;
    border-radius:999px;
    font-size:12px;
    background: var(--chip-bg);
    color: var(--chip-fg);
    margin-right:6px;
    border:1px solid var(--chip-br)
}
.badge.neutral {
    background: var(--chip2-bg);
    color:var(--chip2-fg);
    border-color:var(--chip2-br);
}
.badge.ok {
    background: var(--ok-bg);
    color:var(--ok-fg);
    border-color:var(--ok-br);
}
.badge.warn {
    background: var(--warn-bg);
    color:var(--warn-fg);
    border-color:var(--warn-br);
}
</style>
"""
st.markdown(BADGE_CSS, unsafe_allow_html=True)

# ====== CONFIG ======
API_KEY = st.secrets["API_KEY"]
ACCOUNT_ID = st.secrets["ACCOUNT_ID"]
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

# ====== SUPABASE CACHE FUNCTIONS ======
def get_cached_candles(instrument, granularity, start_time, end_time):
    """Retrieve cached candles from Supabase"""
    if not supabase:
        return []
    
    try:
        response = supabase.table('candles_cache').select('*').eq(
            'instrument', instrument
        ).eq(
            'granularity', granularity
        ).gte(
            'time', start_time.isoformat()
        ).lte(
            'time', end_time.isoformat()
        ).order('time').execute()
        
        return response.data if response.data else []
    except:
        return []

def save_candles_to_cache(instrument, granularity, candles):
    """Save candles to Supabase cache"""
    if not supabase or not candles:
        return
    
    try:
        records = []
        for c in candles:
            try:
                t_utc = datetime.strptime(c["time"], "%Y-%m-%dT%H:%M:%S.%f000Z")
            except ValueError:
                t_utc = datetime.strptime(c["time"], "%Y-%m-%dT%H:%M:%S.000Z")
            
            records.append({
                'instrument': instrument,
                'granularity': granularity,
                'time': t_utc.isoformat(),
                'open': float(c["mid"]["o"]),
                'high': float(c["mid"]["h"]),
                'low': float(c["mid"]["l"]),
                'close': float(c["mid"]["c"]),
                'volume': c["volume"],
                'complete': c.get("complete", True),
                'raw_data': json.dumps(c)
            })
        
        if records:
            supabase.table('candles_cache').upsert(records, on_conflict='instrument,granularity,time').execute()
    except:
        pass

def get_cached_averages(instrument, granularity, bucket_type):
    """Get pre-calculated averages from Supabase"""
    if not supabase:
        return {}
    
    try:
        response = supabase.table('volume_averages').select('*').eq(
            'instrument', instrument
        ).eq(
            'granularity', granularity
        ).eq(
            'bucket_type', bucket_type
        ).gte(
            'calculated_at', (datetime.now(UTC) - timedelta(hours=1)).isoformat()
        ).execute()
        
        if response.data and len(response.data) > 0:
            return json.loads(response.data[0]['averages'])
        return {}
    except:
        return {}

def save_averages_to_cache(instrument, granularity, bucket_type, averages):
    """Save calculated averages to Supabase"""
    if not supabase or not averages:
        return
    
    try:
        record = {
            'instrument': instrument,
            'granularity': granularity,
            'bucket_type': bucket_type,
            'averages': json.dumps(averages),
            'calculated_at': datetime.now(UTC).isoformat()
        }
        
        supabase.table('volume_averages').upsert(record, on_conflict='instrument,granularity,bucket_type').execute()
    except:
        pass

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
st.sidebar.title("🔧 Settings")

if "selected_instruments" not in st.session_state:
    st.session_state.selected_instruments = list(INSTRUMENTS.keys())
if "refresh_minutes" not in st.session_state:
    st.session_state.refresh_minutes = 5
if "bucket_choice" not in st.session_state:
    st.session_state.bucket_choice = "1 hour"
if "enable_telegram_alerts" not in st.session_state:
    st.session_state.enable_telegram_alerts = True
if "candle_size" not in st.session_state:
    st.session_state.candle_size = "15 min"
if "skip_weekends" not in st.session_state:
    st.session_state.skip_weekends = True
if "alert_multiplier" not in st.session_state:
    st.session_state.alert_multiplier = 1.618

TELEGRAM_BOT_TOKEN = st.secrets.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = st.secrets.get("TELEGRAM_CHAT_ID", "")

st.sidebar.multiselect(
    "Select Instruments to Monitor",
    options=list(INSTRUMENTS.keys()),
    default=st.session_state.selected_instruments,
    key="selected_instruments"
)

st.sidebar.slider(
    "Auto-refresh interval (minutes)",
    min_value=1, max_value=15,
    value=st.session_state.refresh_minutes,
    key="refresh_minutes"
)

st.sidebar.radio(
    "📏 Candle Size",
    ["15 min", "4 hour"],
    index=["15 min", "4 hour"].index(st.session_state.candle_size),
    key="candle_size"
)

if st.session_state.candle_size == "15 min":
    st.sidebar.radio(
        "🕒 Select Time Bucket",
        ["15 min", "30 min", "1 hour"],
        index=["15 min", "30 min", "1 hour"].index(st.session_state.bucket_choice),
        key="bucket_choice"
    )

st.sidebar.markdown("---")

st.sidebar.toggle(
    "Enable Telegram Alerts",
    value=st.session_state.enable_telegram_alerts,
    key="enable_telegram_alerts"
)

st.sidebar.slider(
    "📢 Alert Multiplier (Volume/Avg)",
    min_value=1.0,
    max_value=3.0,
    step=0.1,
    value=st.session_state.alert_multiplier,
    key="alert_multiplier"
)

st.sidebar.toggle(
    "Skip Weekends in Average",
    value=st.session_state.skip_weekends,
    key="skip_weekends"
)

if supabase:
    st.sidebar.success("✅ Supabase Connected")
else:
    st.sidebar.info("ℹ️ File cache mode")

# ====== AUTO-REFRESH ======
refresh_ms = st.session_state.refresh_minutes * 60 * 1000
refresh_count = st_autorefresh(interval=refresh_ms, limit=None, key="volume-refresh")


# ====== TELEGRAM ALERT ======
def send_telegram_alert(message):
    if not st.session_state.enable_telegram_alerts:
        return False
    
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True
    }
    
    try:
        resp = requests.post(url, data=payload, timeout=10)
        return resp.status_code == 200
    except:
        return False

# ====== SMART OANDA DATA FETCH WITH CACHE ======
@st.cache_resource
def get_session():
    s = requests.Session()
    s.headers.update(headers)
    return s

def fetch_candles_smart(instrument_code, from_time, to_time, granularity="M15"):
    """Smart fetch: use cache first, fetch missing from OANDA"""
    now_utc = datetime.now(UTC)
    from_time = min(from_time, now_utc)
    to_time = min(to_time, now_utc)
    
    # Try cache first
    cached = get_cached_candles(instrument_code, granularity, from_time, to_time)
    
    if cached and len(cached) > 0:
        last_cached_time = datetime.fromisoformat(cached[-1]['time'].replace('Z', ''))
        
        # If cache is recent (within last hour), use it
        if (now_utc - last_cached_time).total_seconds() < 3600:
            candles = []
            for c in cached:
                candles.append({
                    "time": c['time'],
                    "volume": c['volume'],
                    "complete": c['complete'],
                    "mid": {
                        "o": str(c['open']),
                        "h": str(c['high']),
                        "l": str(c['low']),
                        "c": str(c['close'])
                    }
                })
            return candles
    
    # Fetch from OANDA
    params = {
        "granularity": granularity,
        "price": "M",
        "from": from_time.isoformat(),
        "to": to_time.isoformat()
    }
    url = f"{BASE_URL}/accounts/{ACCOUNT_ID}/instruments/{instrument_code}/candles"
    
    try:
        s = get_session()
        resp = s.get(url, params=params, timeout=20)
    except Exception as e:
        st.error(f"❌ Network error for {instrument_code}: {e}")
        return []
    
    if resp.status_code != 200:
        st.error(f"❌ Failed to fetch {instrument_code} data: {resp.text}")
        return []
    
