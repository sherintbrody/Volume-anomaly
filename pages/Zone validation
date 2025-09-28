import streamlit as st
import requests
import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import pytz
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import uuid
from typing import List, Dict, Optional, Tuple

# --- Page Config ---
st.set_page_config(
    page_title="Pattern Validator Pro v2.0",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Enhanced Custom CSS ---
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        background: linear-gradient(45deg, #E74C3C, #C0392B);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(45deg, #C0392B, #E74C3C);
        box-shadow: 0 4px 8px rgba(231, 76, 60, 0.3);
        transform: translateY(-2px);
    }
    .success-box {
        padding: 15px;
        border-radius: 10px;
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .error-box {
        padding: 15px;
        border-radius: 10px;
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        border: 1px solid #f5c6cb;
        color: #721c24;
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
        margin: 15px 0;
        border: 1px solid #dee2e6;
    }
    .warning-box {
        padding: 15px;
        border-radius: 10px;
        background: linear-gradient(135deg, #fff3cd, #ffeaa7);
        border: 1px solid #ffeaa7;
        color: #856404;
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .info-box {
        padding: 15px;
        border-radius: 10px;
        background: linear-gradient(135deg, #d1ecf1, #bee5eb);
        border: 1px solid #bee5eb;
        color: #0c5460;
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# --- Timezone Setup ---
IST = pytz.timezone('Asia/Kolkata')
UTC = pytz.UTC

# --- OANDA API Configuration ---
API_KEY = st.secrets["API_KEY"]
ACCOUNT_ID = st.secrets["ACCOUNT_ID"]
BASE_URL = "https://api-fxpractice.oanda.com/v3"

# OANDA instrument mapping
SYMBOL_MAPPING = {
    "NAS100": "NAS100_USD",
    "US30": "US30_USD",
    "XAU/USD": "XAU_USD",
    "SPX500": "SPX500_USD",
    "XAG/USD": "XAG_USD",
    "EUR/USD": "EUR_USD",
    "GBP/USD": "GBP_USD", 
    "USD/JPY": "USD_JPY",
    "USD/CHF": "USD_CHF",
    "AUD/USD": "AUD_USD",
    "USD/CAD": "USD_CAD",
    "NZD/USD": "NZD_USD",
    "EUR/GBP": "EUR_GBP",
    "EUR/JPY": "EUR_JPY",
    "GBP/JPY": "GBP_JPY",
    "UK100": "UK100_GBP",
    "DE30": "DE30_EUR",
    "FR40": "FR40_EUR",
    "AU200": "AU200_AUD",
    "BCO/USD": "BCO_USD",
    "WTI/USD": "WTICO_USD"
}
# --- Pattern Success Tracking System ---
class PatternTracker:
    """Track pattern validation history and calculate success rates"""
    
    def __init__(self):
        self.history_file = "pattern_history.json"
        self.load_history()
    
    def load_history(self):
        """Load historical pattern data"""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    self.history = json.load(f)
            else:
                self.history = []
        except:
            self.history = []
    
    def save_history(self):
        """Save pattern history to file"""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.history, f, default=str)
        except Exception as e:
            st.error(f"Error saving history: {e}")
    
    def add_pattern_validation(self, symbol, pattern_type, candles, is_valid, atr, 
                              start_price, end_price, additional_data=None):
        """Add a new pattern validation to history"""
        entry = {
            'id': str(uuid.uuid4()),
            'timestamp': datetime.now(IST).isoformat(),
            'symbol': symbol,
            'pattern_type': pattern_type,
            'candles': candles,
            'is_valid': is_valid,
            'atr': atr,
            'start_price': start_price,
            'end_price': end_price,
            'price_change': end_price - start_price,
            'price_change_pct': ((end_price - start_price) / start_price) * 100,
            'outcome': None,
            'outcome_price': None,
            'outcome_time': None,
            'additional_data': additional_data
        }
        
        self.history.append(entry)
        self.save_history()
        return entry['id']
    
    def update_pattern_outcome(self, pattern_id, outcome_price, outcome_time, success=None):
        """Update the outcome of a pattern after it plays out"""
        for entry in self.history:
            if entry['id'] == pattern_id:
                entry['outcome_price'] = outcome_price
                entry['outcome_time'] = outcome_time
                
                if success is None:
                    if entry['pattern_type'] == 'rally':
                        success = outcome_price > entry['end_price']
                    elif entry['pattern_type'] == 'drop':
                        success = outcome_price < entry['end_price']
                    else:  # base
                        range_size = entry['atr'] * 1.5
                        success = abs(outcome_price - entry['end_price']) <= range_size
                
                entry['success'] = success
                self.save_history()
                return True
        return False
    
    def get_success_statistics(self, symbol=None, pattern_type=None, days_back=30):
        """Calculate success statistics for patterns"""
        cutoff_date = datetime.now(IST) - timedelta(days=days_back)
        
        filtered_history = []
        for entry in self.history:
            try:
                entry_date = datetime.fromisoformat(entry['timestamp'])
                if hasattr(entry_date, 'tzinfo') and entry_date.tzinfo is None:
                    entry_date = IST.localize(entry_date)
            except:
                continue
            
            if entry_date < cutoff_date:
                continue
            if symbol and entry['symbol'] != symbol:
                continue
            if pattern_type and entry['pattern_type'] != pattern_type:
                continue
            if not entry.get('is_valid', False):
                continue
            if 'success' not in entry:
                continue
                
            filtered_history.append(entry)
        
        if not filtered_history:
            return {
                'total_patterns': 0,
                'successful': 0,
                'failed': 0,
                'success_rate': 0,
                'avg_profit_pct': 0,
                'avg_loss_pct': 0,
                'best_trade': 0,
                'worst_trade': 0,
                'profit_factor': 0
            }
        
        successful = [e for e in filtered_history if e.get('success', False)]
        failed = [e for e in filtered_history if not e.get('success', True)]
        
        profits = []
        losses = []
        
        for entry in filtered_history:
            if entry.get('outcome_price'):
                pnl_pct = ((entry['outcome_price'] - entry['end_price']) / entry['end_price']) * 100
                
                if entry['pattern_type'] == 'drop':
                    pnl_pct = -pnl_pct
                
                if pnl_pct > 0:
                    profits.append(pnl_pct)
                else:
                    losses.append(abs(pnl_pct))
        
        stats = {
            'total_patterns': len(filtered_history),
            'successful': len(successful),
            'failed': len(failed),
            'success_rate': (len(successful) / len(filtered_history)) * 100 if filtered_history else 0,
            'avg_profit_pct': np.mean(profits) if profits else 0,
            'avg_loss_pct': np.mean(losses) if losses else 0,
            'best_trade': max(profits) if profits else 0,
            'worst_trade': max(losses) if losses else 0,
            'profit_factor': (np.mean(profits) * len(profits)) / (np.mean(losses) * len(losses)) 
                           if profits and losses and len(losses) > 0 and np.mean(losses) > 0 else 0
        }
        
        return stats

@st.cache_resource
def get_pattern_tracker():
    return PatternTracker()
# --- Helper Functions ---
def is_candle_complete(candle_time, interval_hours=4):
    """Check if a candle is complete based on current time"""
    current_time = datetime.now(UTC)
    candle_end_time = candle_time + timedelta(hours=interval_hours)
    return current_time >= candle_end_time

def convert_symbol_to_oanda(symbol):
    """Convert display symbol to OANDA instrument format"""
    return SYMBOL_MAPPING.get(symbol, symbol.replace("/", "_"))

# --- FIXED: ATR Calculation Functions ---
def calculate_atr(df, period=21):
    """Calculate Average True Range excluding incomplete candles"""
    df_copy = df.copy()
    
    incomplete_candle_exists = False
    if len(df_copy) > 0 and 'is_complete' in df_copy.columns:
        incomplete_candle_exists = not df_copy.iloc[-1]['is_complete']
    
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

def calculate_atr_for_range(df, selected_start, selected_end, period=21):
    """
    Calculate ATR properly for a selected time range
    Ensures we have enough historical data for accurate ATR calculation
    """
    df_with_history = df[df['datetime_ist'] <= selected_end].copy()
    
    if len(df_with_history) < period:
        st.warning(f"⚠️ Insufficient data for {period}-period ATR calculation. Using available {len(df_with_history)} periods.")
    
    df_with_history = calculate_atr(df_with_history, period=period)
    
    selected_df = df_with_history[
        (df_with_history['datetime_ist'] >= selected_start) & 
        (df_with_history['datetime_ist'] <= selected_end)
    ].copy()
    
    if len(selected_df) > 0:
        relevant_atr = selected_df['atr'].iloc[0]
        
        if pd.isna(relevant_atr) and len(df_with_history) > 0:
            pre_range_df = df_with_history[df_with_history['datetime_ist'] < selected_start]
            if len(pre_range_df) > 0 and 'atr' in pre_range_df.columns:
                last_valid_atr = pre_range_df['atr'].dropna()
                if len(last_valid_atr) > 0:
                    relevant_atr = last_valid_atr.iloc[-1]
    else:
        relevant_atr = None
    
    return selected_df, relevant_atr

# --- OANDA API Data Fetching ---
@st.cache_data(ttl=300)
def fetch_ohlc(symbol, start, end, interval="4h"):
    """Fetch 4H OHLC data from OANDA API within UTC range."""
    
    instrument = convert_symbol_to_oanda(symbol)
    
    granularity_map = {
        "1h": "H1",
        "4h": "H4", 
        "1d": "D",
        "1w": "W"
    }
    granularity = granularity_map.get(interval, "H4")
    
    from_time = start.strftime("%Y-%m-%dT%H:%M:%S.000000Z")
    to_time = end.strftime("%Y-%m-%dT%H:%M:%S.000000Z")
    
    url = f"{BASE_URL}/instruments/{instrument}/candles"
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    params = {
        "granularity": granularity,
        "from": from_time,
        "to": to_time,
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
                    "volume": candle.get("volume", 0),
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
        
    except requests.exceptions.RequestException as e:
        st.error(f"OANDA API Error: {e}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Data processing error: {e}")
        return pd.DataFrame()
# --- Pattern Rules ---
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

# --- Pattern Validation Helper Functions ---
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
        if (candles[i]["open"] > candles[i-1]["close"] and 
            candles[i]["close"] < candles[i-1]["open"]):
            return True
    return False

def check_bullish_engulfing(candles):
    for i in range(1, len(candles)):
        if (candles[i]["open"] < candles[i-1]["close"] and 
            candles[i]["close"] > candles[i-1]["open"]):
            return True
    return False

# --- Core Pattern Validation Function ---
def validate_pattern_detailed(candles, atr, pattern):
    n = len(candles)
    if n < 1 or n > 6:
        return False, f"Invalid number of candles: {n}. Must be between 1 and 6.", {}
    
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
            core_criteria['no_new_extreme'] = (new_high <= rules["no_new_extreme_atr"] and 
                                              new_low <= rules["no_new_extreme_atr"])
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
            results['hh_hl'] = (candles[1]["high"] > candles[0]["high"] and 
                               candles[1]["low"] > candles[0]["low"])
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
            results['lh_ll'] = (candles[1]["high"] < candles[0]["high"] and 
                               candles[1]["low"] < candles[0]["low"])
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

# --- Pattern Detection Algorithm ---
def auto_detect_patterns(df, min_confidence=0.7):
    """
    Automatically detect all valid patterns in the dataset
    Returns patterns with confidence scores
    """
    patterns_found = []
    
    if 'atr' not in df.columns or df['atr'].isna().all():
        df = calculate_atr(df, period=21)
    
    complete_df = df[df.get('is_complete', True)].copy()
    
    if len(complete_df) < 21:
        return patterns_found
    
    for window_size in range(1, 7):
        for i in range(len(complete_df) - window_size + 1):
            if i < 20:
                continue
                
            segment = complete_df.iloc[i:i+window_size]
            pattern_atr = segment.iloc[0]['atr']
            
            if pd.isna(pattern_atr) or pattern_atr <= 0:
                continue
            
            candles = [
                dict(open=row.open, high=row.high, low=row.low, close=row.close)
                for _, row in segment.iterrows()
            ]
            
            for pattern_type in ['rally', 'drop', 'base']:
                is_valid, message, details = validate_pattern_detailed(candles, pattern_atr, pattern_type)
                
                if is_valid:
                    if details:
                        confidence = sum(1 for v in details.values() if v) / len(details)
                    else:
                        confidence = 1.0
                    
                    if confidence >= min_confidence:
                        price_change = segment.iloc[-1]['close'] - segment.iloc[0]['open']
                        price_change_pct = (price_change / segment.iloc[0]['open']) * 100
                        
                        patterns_found.append({
                            'start_time': segment.iloc[0]['datetime_ist'],
                            'end_time': segment.iloc[-1]['datetime_ist'],
                            'pattern_type': pattern_type,
                            'candles': window_size,
                            'confidence': confidence,
                            'atr': pattern_atr,
                            'price_change': price_change,
                            'price_change_pct': price_change_pct,
                            'details': details,
                            'candle_data': segment[['open', 'high', 'low', 'close']].to_dict('records')
                        })
    
    return remove_overlapping_patterns(patterns_found)

def remove_overlapping_patterns(patterns):
    """Remove overlapping patterns, keeping the ones with highest confidence"""
    if not patterns:
        return patterns
    
    sorted_patterns = sorted(patterns, key=lambda x: x['confidence'], reverse=True)
    
    non_overlapping = []
    for pattern in sorted_patterns:
        overlaps = False
        for selected in non_overlapping:
            if (pattern['start_time'] <= selected['end_time'] and 
                pattern['end_time'] >= selected['start_time']):
                overlaps = True
                break
        
        if not overlaps:
            non_overlapping.append(pattern)
    
    return sorted(non_overlapping, key=lambda x: x['start_time'])
# --- Plot Functions ---
def plot_combined_chart(df, selected_candles_df=None, show_atr=True):
    """Create enhanced combined chart"""
    
    df_with_atr = df[df['atr'].notna()].copy() if 'atr' in df.columns else df.copy()
    
    if len(df_with_atr) > 42:
        atr_data = df_with_atr.tail(42)
    else:
        atr_data = df_with_atr
    
    atr_data = atr_data.dropna(subset=['atr']) if 'atr' in atr_data.columns else atr_data
    
    rows = 2 if show_atr else 1
    row_heights = [0.65, 0.35] if show_atr else [1.0]
    
    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
        subplot_titles=(
            None,
            '<b style="color:#E74C3C; font-size:16px;">📈 ATR Volatility Indicator (21-Period)</b>'
        ) if show_atr else (None,)
    )
    
    bullish_color = '#00D4AA'
    bearish_color = '#FF6B6B'
    
    fig.add_trace(go.Candlestick(
        x=df['datetime_ist'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='OHLC Data',
        increasing_line_color=bullish_color,
        decreasing_line_color=bearish_color,
        increasing_fillcolor=bullish_color,
        decreasing_fillcolor=bearish_color,
        line=dict(width=1.5),
        hoverinfo='all'
    ), row=1, col=1)
    
    if 'is_complete' in df.columns:
        incomplete_df = df[~df['is_complete']]
        if not incomplete_df.empty:
            fig.add_trace(go.Scatter(
                x=incomplete_df['datetime_ist'],
                y=incomplete_df['high'] * 1.008,
                mode='markers+text',
                marker=dict(
                    symbol='circle',
                    size=16,
                    color='#FF9500',
                    line=dict(color='white', width=2)
                ),
                text='🔄',
                textfont=dict(size=12),
                textposition="middle center",
                name='Forming Candle',
                showlegend=True,
                hovertemplate='<b>Status</b>: Candle Still Forming<br><extra></extra>'
            ), row=1, col=1)
    
    if selected_candles_df is not None and not selected_candles_df.empty:
        min_time = selected_candles_df['datetime_ist'].min()
        max_time = selected_candles_df['datetime_ist'].max()
        min_price = selected_candles_df['low'].min()
        max_price = selected_candles_df['high'].max()
        
        fig.add_shape(
            type="rect",
            x0=min_time,
            y0=min_price * 0.9985,
            x1=max_time,
            y1=max_price * 1.0015,
            line=dict(
                color="#FFD700",
                width=3,
                dash="dot"
            ),
            fillcolor="rgba(255, 215, 0, 0.1)",
            row=1, col=1
        )
        
        pattern_info = f"Pattern: {len(selected_candles_df)} candles"
        fig.add_annotation(
            x=min_time,
            y=max_price * 1.01,
            text=f"<b>{pattern_info}</b>",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor='#FFD700',
            font=dict(size=11, color='#FFD700', family="Arial"),
            bgcolor='rgba(255, 215, 0, 0.2)',
            bordercolor='#FFD700',
            borderwidth=1,
            borderpad=4,
            row=1, col=1
        )
    
    if show_atr and 'atr' in atr_data.columns and not atr_data['atr'].isna().all():
        fig.add_trace(go.Scatter(
            x=atr_data['datetime_ist'],
            y=atr_data['atr'],
            mode='lines',
            name='ATR Background',
            line=dict(
                color='rgba(46, 134, 193, 0)',
                width=0
            ),
            fill='tozeroy',
            fillcolor='rgba(46, 134, 193, 0.15)',
            showlegend=False,
            hoverinfo='skip'
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=atr_data['datetime_ist'],
            y=atr_data['atr'],
            mode='lines+markers',
            name='ATR (21)',
            line=dict(
                color='#2E86C1',
                width=3,
                shape='spline'
            ),
            marker=dict(
                size=6,
                color='#2E86C1',
                line=dict(color='white', width=1)
            ),
            showlegend=True,
            hovertemplate='<b>ATR Value</b>: %{y:.4f}<br><b>Time</b>: %{x}<br><extra></extra>'
        ), row=2, col=1)
        
        if 'atr_projected' in atr_data.columns:
            projected_atr = atr_data[atr_data['atr_projected']]
            if not projected_atr.empty:
                fig.add_trace(go.Scatter(
                    x=projected_atr['datetime_ist'],
                    y=projected_atr['atr'],
                    mode='markers',
                    marker=dict(
                        symbol='diamond',
                        size=14,
                        color='#F39C12',
                        line=dict(color='white', width=2)
                    ),
                    name='ATR Projected',
                    showlegend=True,
                    hovertemplate='<b>Projected ATR</b>: %{y:.4f}<br><extra></extra>'
                ), row=2, col=1)
        
        if not atr_data['atr'].isna().all():
            current_atr_row = atr_data[~atr_data.get('atr_projected', False)]
            if not current_atr_row.empty:
                current_atr = current_atr_row['atr'].iloc[-1]
            else:
                current_atr = atr_data['atr'].iloc[-1]
            
            fig.add_trace(go.Scatter(
                x=[atr_data['datetime_ist'].iloc[0], atr_data['datetime_ist'].iloc[-1]],
                y=[current_atr, current_atr],
                mode='lines',
                line=dict(
                    color='#E74C3C',
                    width=2,
                    dash='dashdot'
                ),
                name='Current ATR',
                showlegend=True,
                hovertemplate=f'<b>Current ATR Level</b>: {current_atr:.4f}<br><extra></extra>'
            ), row=2, col=1)
        
        atr_min = atr_data['atr'].min() * 0.95
        atr_max = atr_data['atr'].max() * 1.05
        
        fig.update_yaxes(
            range=[atr_min, atr_max],
            row=2, col=1,
            fixedrange=False
        )
        
        fig.update_xaxes(
            range=[atr_data['datetime_ist'].iloc[0], atr_data['datetime_ist'].iloc[-1]],
            row=2, col=1
        )
    
    fig.update_layout(
        height=850,
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(255, 255, 255, 0.1)',
            bordercolor='rgba(255, 255, 255, 0.2)',
            borderwidth=1,
            font=dict(size=11, color='white'),
            itemsizing="constant"
        ),
        hovermode='x unified',
        margin=dict(l=70, r=100, t=100, b=60),
        font=dict(
            family="Arial, sans-serif",
            size=12,
            color="white"
        )
    )
    
    fig.update_xaxes(
        title_text="<b>Time (IST)</b>",
        row=rows, col=1,
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(255, 255, 255, 0.1)',
        showline=True,
        linewidth=2,
        linecolor='rgba(255, 255, 255, 0.3)',
        title_font=dict(size=14, color='white'),
        tickfont=dict(size=11, color='white')
    )
    
    fig.update_yaxes(
        title_text="<b>Price Level</b>",
        row=1, col=1,
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(255, 255, 255, 0.1)',
        showline=True,
        linewidth=2,
        linecolor='rgba(255, 255, 255, 0.3)',
        title_font=dict(size=14, color='white'),
        tickfont=dict(size=11, color='white')
    )
    
    if show_atr:
        fig.update_yaxes(
            title_text="<b>ATR Value</b>",
            row=2, col=1,
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(255, 255, 255, 0.1)',
            showline=True,
            linewidth=2,
            linecolor='rgba(255, 255, 255, 0.3)',
            tickformat='.4f',
            title_font=dict(size=14, color='white'),
            tickfont=dict(size=11, color='white')
        )
    
    fig.update_layout(xaxis_rangeslider_visible=False)
    
    return fig

# --- Display Functions ---
def display_validation_results(is_valid, message, pattern, details=None):
    """Display enhanced validation results"""
    pattern_emoji = {"rally": "🚀", "drop": "📉", "base": "⚖️"}
    pattern_name = {"rally": "Rally Pattern (Bullish Impulse)", "drop": "Drop Pattern (Bearish Impulse)", "base": "Base Pattern (Consolidation Zone)"}
    pattern_colors = {"rally": "#00D4AA", "drop": "#FF6B6B", "base": "#4ECDC4"}
    
    if is_valid:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #d4edda, #c3e6cb);
            border: 2px solid {pattern_colors[pattern]};
            border-radius: 15px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        ">
            <h2 style="color: #155724; margin: 0; display: flex; align-items: center; gap: 10px;">
                {pattern_emoji[pattern]} {pattern_name[pattern]} 
                <span style="background: #28a745; color: white; padding: 5px 15px; border-radius: 20px; font-size: 14px; font-weight: bold;">✅ VALID</span>
            </h2>
            <p style="color: #155724; margin: 10px 0 0 0; font-size: 16px;">{message}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #f8d7da, #f5c6cb);
            border: 2px solid #dc3545;
            border-radius: 15px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        ">
            <h2 style="color: #721c24; margin: 0; display: flex; align-items: center; gap: 10px;">
                {pattern_emoji[pattern]} {pattern_name[pattern]} 
                <span style="background: #dc3545; color: white; padding: 5px 15px; border-radius: 20px; font-size: 14px; font-weight: bold;">❌ INVALID</span>
            </h2>
            <p style="color: #721c24; margin: 10px 0 0 0; font-size: 16px;">{message}</p>
        </div>
        """, unsafe_allow_html=True)
    
    if details:
        with st.expander("🔍 **Detailed Validation Breakdown**", expanded=True):
            total_checks = len(details)
            passed_checks = sum(details.values())
            
            progress = passed_checks / total_checks if total_checks > 0 else 0
            color = "#28a745" if progress == 1.0 else "#ffc107" if progress >= 0.5 else "#dc3545"
            
            st.markdown(f"""
            <div style="margin: 15px 0;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span><b>Validation Progress</b></span>
                    <span><b>{passed_checks}/{total_checks} checks passed</b></span>
                </div>
                <div style="background: #e9ecef; border-radius: 10px; overflow: hidden;">
                    <div style="background: {color}; width: {progress*100}%; height: 20px; border-radius: 10px; transition: width 0.3s ease;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            cols = st.columns(2)
            for i, (check, result) in enumerate(details.items()):
                col = cols[i % 2]
                icon = "✅" if result else "❌"
                color = "#28a745" if result else "#dc3545"
                check_name = check.replace('_', ' ').title()
                
                col.markdown(f"""
                <div style="
                    background: rgba(248, 249, 250, 0.8);
                    border-left: 4px solid {color};
                    padding: 10px;
                    margin: 5px 0;
                    border-radius: 5px;
                ">
                    <span style="font-size: 16px;">{icon}</span>
                    <strong style="margin-left: 8px;">{check_name}</strong>
                    <br><small style="color: {color}; margin-left: 24px;">{'✓ Passed' if result else '✗ Failed'}</small>
                </div>
                """, unsafe_allow_html=True)

def display_pattern_metrics(df, selected_candles, atr, incomplete_warning=False):
    """Display enhanced pattern metrics"""
    st.markdown("### 📊 Pattern Analysis Metrics")
    
    if incomplete_warning:
        st.markdown("""
        <div class="warning-box">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 24px;">⚠️</span>
                <div>
                    <strong>Live Market Data Notice</strong><br>
                    <small>Current candle is still forming. ATR calculation excludes incomplete data for accuracy.</small>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    if selected_candles:
        ranges = [(c["high"] - c["low"]) / atr for c in selected_candles]
        avg_range = np.mean(ranges)
        max_range = max(ranges)
        min_range = min(ranges)
        net_move = net_move_atr(selected_candles, atr)
        n_candles = len(selected_candles)
        
        cols = st.columns(4)
        
        with cols[0]:
            st.metric(
                label="🕯️ **Candles Count**",
                value=f"{n_candles}",
                help="Number of candles in pattern"
            )
        
        with cols[1]:
            st.metric(
                label="📏 **Avg Range (ATR)**",
                value=f"{avg_range:.2f}",
                delta=f"Max: {max_range:.2f}",
                help="Average range relative to ATR"
            )
        
        with cols[2]:
            st.metric(
                label="🎯 **Net Move (ATR)**",
                value=f"{net_move:.2f}",
                help="Net directional movement in ATR units"
            )
        
        with cols[3]:
            st.metric(
                label="📊 **Current ATR**",
                value=f"{atr:.4f}",
                help="21-period Average True Range"
            )

def display_pattern_statistics():
    """Display pattern success statistics dashboard"""
    tracker = get_pattern_tracker()
    
    st.markdown("### 📊 **Pattern Success Analytics**")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        stats_symbol = st.selectbox(
            "Symbol Filter",
            ["All"] + list(SYMBOL_MAPPING.keys()),
            key="stats_symbol"
        )
    with col2:
        stats_pattern = st.selectbox(
            "Pattern Filter",
            ["All", "rally", "drop", "base"],
            key="stats_pattern"
        )
    with col3:
        days_back = st.selectbox(
            "Time Period",
            [7, 14, 30, 60, 90],
            index=2,
            format_func=lambda x: f"Last {x} days",
            key="stats_days"
        )
    
    stats = tracker.get_success_statistics(
        symbol=None if stats_symbol == "All" else stats_symbol,
        pattern_type=None if stats_pattern == "All" else stats_pattern,
        days_back=days_back
    )
    
    if stats['total_patterns'] > 0:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Patterns",
                stats['total_patterns'],
                help="Total valid patterns analyzed"
            )
        
        with col2:
            success_color = "🟢" if stats['success_rate'] >= 60 else "🟡" if stats['success_rate'] >= 40 else "🔴"
            st.metric(
                f"{success_color} Success Rate",
                f"{stats['success_rate']:.1f}%",
                f"{stats['successful']}/{stats['total_patterns']}",
                help="Percentage of successful patterns"
            )
        
        with col3:
            st.metric(
                "Avg Profit",
                f"+{stats['avg_profit_pct']:.2f}%",
                help="Average profit on successful patterns"
            )
        
        with col4:
            st.metric(
                "Avg Loss",
                f"-{stats['avg_loss_pct']:.2f}%",
                help="Average loss on failed patterns"
            )
    else:
        st.info("📊 No pattern data available for the selected filters. Start validating patterns to build statistics!")
# --- Main UI ---
st.markdown("""
<div class="main-header">
    <h1 style="margin: 0; font-size: 2.5rem;">🎯 Advanced Pattern Validator Pro v2.0</h1>
    <p style="margin: 10px 0 0 0; font-size: 1.2rem; opacity: 0.9;">Professional Pattern Analysis with OANDA Real-Time Data & Success Tracking</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.markdown("## ⚙️ **Analysis Configuration**")
    
    st.markdown("### 📈 **Trading Symbol**")
    
    symbol_category = st.selectbox(
        "**Category**",
        options=["Forex", "Metals", "Indices", "Custom"],
        help="Choose symbol category"
    )
    
    if symbol_category == "Forex":
        symbol = st.selectbox("**Forex Pairs**", ["EUR/USD", "GBP/USD", "USD/JPY"])
    elif symbol_category == "Metals":
        symbol = st.selectbox("**Precious Metals**", ["XAU/USD", "XAG/USD"])
    elif symbol_category == "Indices":
        symbol = st.selectbox("**Market Indices**", ["US30", "NAS100", "SPX500"])
    else:
        symbol = st.text_input("**Enter Custom Symbol**", value="EUR/USD")
    
    pattern = st.selectbox(
        "🎨 **Pattern Type**", 
        ["rally", "drop", "base"],
        format_func=lambda x: {"rally": "🚀 Rally (Bullish)", "drop": "📉 Drop (Bearish)", "base": "⚖️ Base (Neutral)"}[x]
    )
    
    st.markdown("---")
    
    mode = st.radio(
        "🎯 **Selection Mode**",
        ["Automatic (Last N Candles)", "Manual Time Range", "Custom Candle Selection"]
    )
    
    st.markdown("---")
    
    st.markdown("### 📊 **ATR Configuration**")
    use_auto_atr = st.checkbox("🤖 Auto-detect ATR", value=True)
    if not use_auto_atr:
        current_atr = st.number_input(
            "📏 Manual ATR Value", 
            min_value=0.0001, 
            value=0.0075, 
            step=0.0001, 
            format="%.4f"
        )

# Main Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 **Analysis**", 
    "📈 **Chart**", 
    "📊 **Data**", 
    "🔎 **Scanner**",
    "📉 **Statistics**"
])

with tab1:
    if mode == "Automatic (Last N Candles)":
        st.markdown("### 🔄 **Automatic Pattern Analysis**")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            n_candles = st.slider("**Number of candles to analyze**", 1, 6, 3)
        with col2:
            analyze_btn = st.button("**Analyze Pattern**", type="primary")
        
        if analyze_btn:
            end_utc = datetime.now(UTC)
            start_utc = end_utc - timedelta(days=10)
            
            with st.spinner("🔄 Fetching OANDA market data..."):
                df = fetch_ohlc(symbol, start_utc, end_utc)
            
            if not df.empty:
                st.session_state['df'] = df
                
                has_incomplete = 'is_complete' in df.columns and not df.iloc[-1]['is_complete']
                
                if use_auto_atr:
                    if has_incomplete and len(df) > 1:
                        current_atr = df['atr'].iloc[-2]
                    else:
                        current_atr = df['atr'].iloc[-1] if not df['atr'].isna().all() else 0.0075
                    
                    st.markdown(f"""
                    <div class="info-box">
                        📊 <strong>Auto-detected ATR:</strong> {current_atr:.4f}
                    </div>
                    """, unsafe_allow_html=True)
                
                if has_incomplete:
                    analysis_df = df[df['is_complete']].tail(n_candles)
                else:
                    analysis_df = df.tail(n_candles)
                
                candles = [
                    dict(open=row.open, high=row.high, low=row.low, close=row.close)
                    for _, row in analysis_df.iterrows()
                ]
                
                st.session_state['selected_candles'] = analysis_df
                
                ok, message, details = validate_pattern_detailed(candles, current_atr, pattern)
                display_validation_results(ok, message, pattern, details)
                display_pattern_metrics(df, candles, current_atr, incomplete_warning=has_incomplete)
                
                # Track the pattern
                if ok:
                    tracker = get_pattern_tracker()
                    pattern_id = tracker.add_pattern_validation(
                        symbol=symbol,
                        pattern_type=pattern,
                        candles=len(candles),
                        is_valid=ok,
                        atr=current_atr,
                        start_price=analysis_df.iloc[0]['open'],
                        end_price=analysis_df.iloc[-1]['close']
                    )
                    st.success(f"✅ Pattern tracked! ID: {pattern_id[:8]}...")
    
    elif mode == "Manual Time Range":
        st.markdown("### 🕐 **Manual Time Range Selection**")
        
        st.markdown("""
        <div class="info-box">
            ⏰ <strong>Time Zone Notice:</strong> All times should be entered in Indian Standard Time (IST)<br>
            📊 <strong>ATR Notice:</strong> System will fetch additional historical data for accurate ATR calculation
        </div>
        """, unsafe_allow_html=True)
        
        if 'manual_start_date' not in st.session_state:
            st.session_state.manual_start_date = datetime.now(IST).date() - timedelta(days=3)
        if 'manual_start_time' not in st.session_state:
            st.session_state.manual_start_time = datetime.strptime("00:00", "%H:%M").time()
        if 'manual_end_date' not in st.session_state:
            st.session_state.manual_end_date = datetime.now(IST).date()
        if 'manual_end_time' not in st.session_state:
            st.session_state.manual_end_time = datetime.strptime("16:00", "%H:%M").time()
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**📅 Start Time**")
            start_date = st.date_input("Date", value=st.session_state.manual_start_date)
            start_time = st.time_input("Time (IST)", value=st.session_state.manual_start_time)
            
        with col2:
            st.markdown("**📅 End Time**")
            end_date = st.date_input("Date ", value=st.session_state.manual_end_date)
            end_time = st.time_input("Time (IST) ", value=st.session_state.manual_end_time)
        
        st.session_state.manual_start_date = start_date
        st.session_state.manual_start_time = start_time
        st.session_state.manual_end_date = end_date
        st.session_state.manual_end_time = end_time
        
        if st.button("**Validate Time Range**", type="primary"):
            try:
                start_ist = IST.localize(datetime.combine(start_date, start_time))
                end_ist = IST.localize(datetime.combine(end_date, end_time))
                
                if end_ist <= start_ist:
                    st.error("❌ End time must be after start time!")
                    st.stop()
                
                # Fetch extra historical data for ATR
                atr_buffer_days = 10
                start_utc_with_buffer = start_ist.astimezone(UTC) - timedelta(days=atr_buffer_days)
                end_utc = end_ist.astimezone(UTC)
                
                with st.spinner("🔄 Fetching market data with historical context..."):
                    df = fetch_ohlc(symbol, start_utc_with_buffer, end_utc)
                
                if not df.empty:
                    st.session_state['df'] = df
                    
                    selected_df, calculated_atr = calculate_atr_for_range(df, start_ist, end_ist)
                    
                    if not use_auto_atr:
                        calculated_atr = current_atr
                    
                    if calculated_atr is None or pd.isna(calculated_atr):
                        st.warning("⚠️ Unable to calculate ATR for this range. Using fallback value.")
                        calculated_atr = 0.0075
                    
                    st.markdown(f"""
                    <div class="success-box">
                        ✅ <strong>Data Loaded Successfully</strong><br>
                        📊 <strong>ATR for selected range:</strong> {calculated_atr:.4f}<br>
                        📈 <strong>Candles in range:</strong> {len(selected_df)} complete candles
                    </div>
                    """, unsafe_allow_html=True)
                    
                    complete_selected = selected_df[selected_df.get('is_complete', True)].copy()
                    
                    if not complete_selected.empty and len(complete_selected) <= 6:
                        candles = [
                            dict(open=row.open, high=row.high, low=row.low, close=row.close)
                            for _, row in complete_selected.iterrows()
                        ]
                        
                        st.session_state['selected_candles'] = complete_selected
                        
                        ok, message, details = validate_pattern_detailed(candles, calculated_atr, pattern)
                        display_validation_results(ok, message, pattern, details)
                        display_pattern_metrics(df, candles, calculated_atr)
                        
                        if ok:
                            tracker = get_pattern_tracker()
                            pattern_id = tracker.add_pattern_validation(
                                symbol=symbol,
                                pattern_type=pattern,
                                candles=len(candles),
                                is_valid=ok,
                                atr=calculated_atr,
                                start_price=complete_selected.iloc[0]['open'],
                                end_price=complete_selected.iloc[-1]['close']
                            )
                            st.success(f"✅ Pattern tracked! ID: {pattern_id[:8]}...")
                    
                    elif len(complete_selected) > 6:
                        st.warning(f"⚠️ Selected range contains {len(complete_selected)} candles. Maximum allowed is 6.")
                    else:
                        st.warning("⚠️ No complete candles found in the selected time range.")
                        
            except Exception as e:
                st.error(f"❌ Error processing time range: {e}")

with tab2:
    st.markdown("### 📈 **Interactive Market Analysis Chart**")
    
    if 'df' in st.session_state and st.session_state['df'] is not None:
        df = st.session_state['df']
        sel_df = st.session_state.get('selected_candles', None)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            show_atr = st.checkbox("📊 Show ATR Panel", value=True)
        
        fig = plot_combined_chart(df, sel_df, show_atr=show_atr)
        
        chart_config = {
            'displayModeBar': True,
            'toImageButtonOptions': {
                'format': 'png',
                'filename': f'{symbol}_analysis',
                'height': 800,
                'width': 1200,
                'scale': 1
            }
        }
        
        st.plotly_chart(fig, use_container_width=True, config=chart_config)
    else:
        st.info("📊 Load market data from the Analysis tab to view charts.")

with tab3:
    st.markdown("### 📊 **Market Data Explorer**")
    
    if 'df' in st.session_state and st.session_state['df'] is not None:
        df = st.session_state['df']
        
        show_last_n = st.number_input("**Show Last N Candles**", min_value=5, max_value=100, value=25)
        
        display_df = df.tail(show_last_n).copy()
        display_df['datetime_ist'] = display_df['datetime_ist'].dt.strftime('%Y-%m-%d %H:%M IST')
        
        st.dataframe(display_df[['datetime_ist', 'open', 'high', 'low', 'close', 'atr']], height=500)
    else:
        st.info("📊 Load market data from the Analysis tab to explore data.")

with tab4:
    st.markdown("### 🔍 **Automatic Pattern Scanner**")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        scan_symbol = st.selectbox("Symbol to Scan", list(SYMBOL_MAPPING.keys()))
    with col2:
        scan_days = st.number_input("Days to Scan", min_value=1, max_value=30, value=7)
    with col3:
        min_confidence = st.slider("Min Confidence", 0.5, 1.0, 0.7, 0.05)
    
    if st.button("🚀 **Start Pattern Scan**", type="primary"):
        end_utc = datetime.now(UTC)
        start_utc = end_utc - timedelta(days=scan_days)
        
        with st.spinner(f"🔄 Scanning {scan_symbol} for patterns..."):
            df = fetch_ohlc(scan_symbol, start_utc, end_utc)
            
            if not df.empty:
                patterns = auto_detect_patterns(df, min_confidence=min_confidence)
                
                if patterns:
                    st.success(f"✅ Found {len(patterns)} patterns!")
                    
                    for i, pattern in enumerate(patterns, 1):
                        pattern_emoji = {"rally": "🚀", "drop": "📉", "base": "⚖️"}[pattern['pattern_type']]
                        
                        with st.expander(f"{pattern_emoji} Pattern #{i} - {pattern['pattern_type'].upper()} ({pattern['confidence']*100:.1f}% confidence)"):
                            st.markdown(f"""
                            **Time Range:** {pattern['start_time'].strftime('%Y-%m-%d %H:%M')} to {pattern['end_time'].strftime('%Y-%m-%d %H:%M')}  
                            **Candles:** {pattern['candles']}  
                            **ATR:** {pattern['atr']:.4f}  
                            **Price Change:** {pattern['price_change']:.4f} ({pattern['price_change_pct']:+.2f}%)
                            """)
                else:
                    st.info(f"No patterns found with confidence >= {min_confidence*100:.0f}%")

with tab5:
    display_pattern_statistics()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 10px;">
    <div style="color: white;">
        <h4 style="margin: 0;">🎯 Pattern Validator Pro v2.0</h4>
        <p style="margin: 5px 0 0 0; opacity: 0.9;">With Enhanced ATR Calculation & Pattern Success Tracking</p>
    </div>
</div>
""", unsafe_allow_html=True)
