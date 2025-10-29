import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import ta
from dataclasses import dataclass
from typing import List, Tuple, Optional
import json

# Page Configuration
st.set_page_config(
    page_title="Professional S&D Zone Detector",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {background-color: #0e1117;}
    .stMetric {background-color: #1e2130; padding: 15px; border-radius: 5px;}
    .zone-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        color: white;
    }
    .supply-badge {
        background-color: #ef4444;
        padding: 5px 10px;
        border-radius: 15px;
        font-weight: bold;
    }
    .demand-badge {
        background-color: #10b981;
        padding: 5px 10px;
        border-radius: 15px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# DATA CLASSES
# ============================================

@dataclass
class Zone:
    """Supply/Demand Zone with institutional-grade metrics"""
    zone_type: str  # 'supply' or 'demand'
    top: float
    bottom: float
    creation_time: datetime
    creation_index: int
    strength: float
    volume_ratio: float
    touches: int
    tested: bool
    broken: bool
    age_bars: int
    distance_from_price: float
    rr_ratio: float  # Risk/Reward from current price
    
    def get_strength_label(self) -> str:
        if self.strength >= 2.5:
            return "⭐⭐⭐ Institutional"
        elif self.strength >= 2.0:
            return "⭐⭐ Strong"
        elif self.strength >= 1.5:
            return "⭐ Moderate"
        else:
            return "Weak"
    
    def get_zone_height(self) -> float:
        return abs(self.top - self.bottom)
    
    def is_price_in_zone(self, price: float) -> bool:
        return self.bottom <= price <= self.top

# ============================================
# OANDA API CLIENT
# ============================================

class OANDAClient:
    """Professional OANDA API client with error handling"""
    
    def __init__(self, api_key: str, account_id: str, practice: bool = True):
        self.api_key = st.secrets["API_KEY"]
        self.account_id = st.secrets["ACCOUNT_ID"]
        self.base_url = "https://api-fxpractice.oanda.com" if practice else "https://api-fxtrade.oanda.com"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def get_candles(self, instrument: str, granularity: str, count: int = 500) -> pd.DataFrame:
        """
        Fetch candlestick data from OANDA
        
        Granularities: S5,S10,S15,S30,M1,M2,M4,M5,M10,M15,M30,H1,H2,H3,H4,H6,H8,H12,D,W,M
        """
        try:
            endpoint = f"{self.base_url}/v3/instruments/{instrument}/candles"
            params = {
                "count": count,
                "granularity": granularity,
                "price": "MBA"  # Mid, Bid, Ask
            }
            
            response = requests.get(endpoint, headers=self.headers, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            candles = data.get('candles', [])
            
            if not candles:
                return None
            
            # Parse candles
            df_data = []
            for candle in candles:
                if candle['complete']:
                    df_data.append({
                        'time': pd.to_datetime(candle['time']),
                        'open': float(candle['mid']['o']),
                        'high': float(candle['mid']['h']),
                        'low': float(candle['mid']['l']),
                        'close': float(candle['mid']['c']),
                        'volume': int(candle['volume'])
                    })
            
            df = pd.DataFrame(df_data)
            df.set_index('time', inplace=True)
            return df
            
        except requests.exceptions.RequestException as e:
            st.error(f"API Error: {str(e)}")
            return None
        except Exception as e:
            st.error(f"Data Processing Error: {str(e)}")
            return None
    
    def get_available_instruments(self) -> List[str]:
        """Fetch available trading instruments"""
        try:
            endpoint = f"{self.base_url}/v3/accounts/{self.account_id}/instruments"
            response = requests.get(endpoint, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            instruments = [inst['name'] for inst in data.get('instruments', [])]
            return sorted(instruments)
        except:
            # Return common forex pairs if API fails
            return [
                "XAU_USD", "US30_USD", "NAS100_USD",
                "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD",
                "NZD_USD", "EUR_GBP", "EUR_JPY", "GBP_JPY"
            ]

# ============================================
# ZONE DETECTION ENGINE
# ============================================

class ZoneDetector:
    """Institutional-grade Supply & Demand zone detection"""
    
    def __init__(self, df: pd.DataFrame, config: dict):
        self.df = df.copy()
        self.config = config
        self.zones = []
        
        # Calculate technical indicators
        self._calculate_indicators()
    
    def get_enhanced_dataframe(self):
        """Return the dataframe with calculated indicators"""
        return self.df
    
    def _calculate_indicators(self):
        """Calculate all technical indicators"""
        # ATR for volatility
        self.df['atr'] = ta.volatility.average_true_range(
            self.df['high'], self.df['low'], self.df['close'], window=14
        )
        
        # Volume analysis
        self.df['volume_sma'] = self.df['volume'].rolling(window=20).mean()
        self.df['volume_ratio'] = self.df['volume'] / self.df['volume_sma']
        
        # Candle characteristics
        self.df['body'] = abs(self.df['close'] - self.df['open'])
        self.df['upper_wick'] = self.df['high'] - self.df[['open', 'close']].max(axis=1)
        self.df['lower_wick'] = self.df[['open', 'close']].min(axis=1) - self.df['low']
        self.df['total_wick'] = self.df['upper_wick'] + self.df['lower_wick']
        
        # Candle type
        self.df['is_bullish'] = self.df['close'] > self.df['open']
        self.df['is_bearish'] = self.df['close'] < self.df['open']
        
        # Price momentum
        self.df['rsi'] = ta.momentum.rsi(self.df['close'], window=14)
        
        # Support/Resistance levels
        self.df['swing_high'] = self.df['high'].rolling(window=5, center=True).max() == self.df['high']
        self.df['swing_low'] = self.df['low'].rolling(window=5, center=True).min() == self.df['low']
    
    def detect_zones(self) -> List[Zone]:
        """Main zone detection logic"""
        zones = []
        df_len = len(self.df)
        
        for i in range(1, df_len - 1):
            # Supply Zone Detection
            if self._is_supply_zone(i):
                zone = self._create_supply_zone(i)
                if zone and self._validate_zone(zone):
                    zones.append(zone)
            
            # Demand Zone Detection
            if self._is_demand_zone(i):
                zone = self._create_demand_zone(i)
                if zone and self._validate_zone(zone):
                    zones.append(zone)
        
        # Update zone metrics
        current_price = self.df['close'].iloc[-1]
        for zone in zones:
            self._update_zone_metrics(zone, current_price)
        
        # Filter and rank zones
        zones = self._filter_zones(zones)
        zones = sorted(zones, key=lambda z: z.strength, reverse=True)
        
        return zones[:self.config['max_zones']]
    
    def _is_supply_zone(self, i: int) -> bool:
        """Detect supply zone formation"""
        current = self.df.iloc[i]
        prev = self.df.iloc[i-1]
        
        # Basic conditions
        if not (current['is_bearish'] and prev['is_bullish']):
            return False
        
        # Size comparison
        size_ratio = current['body'] / (prev['body'] + 0.0001)
        if size_ratio < self.config['candle_multiplier']:
            return False
        
        # Volume confirmation
        if self.config['use_volume_filter']:
            if current['volume_ratio'] < self.config['volume_threshold']:
                return False
        
        # Additional quality filters
        if self.config['advanced_filters']:
            # Reject if too much wick (indecision)
            if current['total_wick'] > current['body'] * 1.5:
                return False
            
            # Prefer zones at swing highs
            if not current['swing_high']:
                return False
        
        return True
    
    def _is_demand_zone(self, i: int) -> bool:
        """Detect demand zone formation"""
        current = self.df.iloc[i]
        prev = self.df.iloc[i-1]
        
        # Basic conditions
        if not (current['is_bullish'] and prev['is_bearish']):
            return False
        
        # Size comparison
        size_ratio = current['body'] / (prev['body'] + 0.0001)
        if size_ratio < self.config['candle_multiplier']:
            return False
        
        # Volume confirmation
        if self.config['use_volume_filter']:
            if current['volume_ratio'] < self.config['volume_threshold']:
                return False
        
        # Additional quality filters
        if self.config['advanced_filters']:
            # Reject if too much wick
            if current['total_wick'] > current['body'] * 1.5:
                return False
            
            # Prefer zones at swing lows
            if not current['swing_low']:
                return False
        
        return True
    
    def _create_supply_zone(self, i: int) -> Optional[Zone]:
        """Create supply zone object"""
        current = self.df.iloc[i]
        prev = self.df.iloc[i-1]
        
        zone_top = current['high']
        zone_bottom = max(prev['open'], prev['close'])
        
        # Calculate strength
        strength = self._calculate_zone_strength(i, 'supply')
        
        return Zone(
            zone_type='supply',
            top=zone_top,
            bottom=zone_bottom,
            creation_time=self.df.index[i],
            creation_index=i,
            strength=strength,
            volume_ratio=current['volume_ratio'],
            touches=0,
            tested=False,
            broken=False,
            age_bars=0,
            distance_from_price=0.0,
            rr_ratio=0.0
        )
    
    def _create_demand_zone(self, i: int) -> Optional[Zone]:
        """Create demand zone object"""
        current = self.df.iloc[i]
        prev = self.df.iloc[i-1]
        
        zone_top = min(prev['open'], prev['close'])
        zone_bottom = current['low']
        
        # Calculate strength
        strength = self._calculate_zone_strength(i, 'demand')
        
        return Zone(
            zone_type='demand',
            top=zone_top,
            bottom=zone_bottom,
            creation_time=self.df.index[i],
            creation_index=i,
            strength=strength,
            volume_ratio=current['volume_ratio'],
            touches=0,
            tested=False,
            broken=False,
            age_bars=0,
            distance_from_price=0.0,
            rr_ratio=0.0
        )
    
    def _calculate_zone_strength(self, i: int, zone_type: str) -> float:
        """Calculate multi-factor zone strength score (0-3)"""
        current = self.df.iloc[i]
        strength = 0.0
        
        # Factor 1: Body to Wick Ratio (max 1.0)
        body_wick_ratio = current['body'] / (current['total_wick'] + 0.0001)
        strength += min(body_wick_ratio / 3, 1.0)
        
        # Factor 2: Volume Strength (max 1.0)
        vol_score = min(current['volume_ratio'] / 2, 1.0)
        strength += vol_score
        
        # Factor 3: ATR Comparison (max 1.0)
        atr_ratio = current['body'] / (current['atr'] + 0.0001)
        strength += min(atr_ratio / 1.5, 1.0)
        
        # Bonus: RSI extreme
        if zone_type == 'supply' and current['rsi'] > 70:
            strength += 0.3
        elif zone_type == 'demand' and current['rsi'] < 30:
            strength += 0.3
        
        return min(strength, 3.0)
    
    def _validate_zone(self, zone: Zone) -> bool:
        """Validate zone meets minimum requirements"""
        # Minimum height check
        zone_height = zone.get_zone_height()
        current_price = self.df['close'].iloc[-1]
        height_percent = (zone_height / current_price) * 100
        
        if height_percent < self.config['min_zone_height']:
            return False
        
        # Maximum height check (avoid huge zones)
        if height_percent > 5.0:
            return False
        
        return True
    
    def _update_zone_metrics(self, zone: Zone, current_price: float):
        """Update zone with current metrics"""
        # Calculate age
        zone.age_bars = len(self.df) - zone.creation_index - 1
        
        # Calculate distance from price
        if current_price > zone.top:
            zone.distance_from_price = ((current_price - zone.top) / current_price) * 100
        elif current_price < zone.bottom:
            zone.distance_from_price = ((zone.bottom - current_price) / current_price) * 100
        else:
            zone.distance_from_price = 0.0
        
        # Check if price tested the zone
        for i in range(zone.creation_index + 1, len(self.df)):
            candle = self.df.iloc[i]
            
            if zone.zone_type == 'supply':
                if candle['high'] >= zone.bottom and candle['high'] <= zone.top:
                    zone.touches += 1
                    zone.tested = True
                if candle['close'] > zone.top:
                    zone.broken = True
                    break
            else:  # demand
                if candle['low'] <= zone.top and candle['low'] >= zone.bottom:
                    zone.touches += 1
                    zone.tested = True
                if candle['close'] < zone.bottom:
                    zone.broken = True
                    break
        
        # Calculate Risk/Reward ratio
        zone_height = zone.get_zone_height()
        if zone.zone_type == 'supply':
            potential_move = current_price - zone.bottom
            zone.rr_ratio = abs(potential_move / zone_height) if zone_height > 0 else 0
        else:
            potential_move = zone.top - current_price
            zone.rr_ratio = abs(potential_move / zone_height) if zone_height > 0 else 0
    
    def _filter_zones(self, zones: List[Zone]) -> List[Zone]:
        """Filter out broken and weak zones"""
        filtered = []
        
        for zone in zones:
            # Remove broken zones
            if zone.broken:
                continue
            
            # Remove old untested zones
            if self.config['remove_untested'] and not zone.tested:
                if zone.age_bars > self.config['untested_bar_limit']:
                    continue
            
            # Keep zone
            filtered.append(zone)
        
        return filtered

# ============================================
# VISUALIZATION
# ============================================

def create_advanced_chart(df: pd.DataFrame, zones: List[Zone], instrument: str, timeframe: str):
    """Create professional trading chart with zones"""
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=(f'{instrument} - {timeframe}', 'Volume', 'RSI')
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ),
        row=1, col=1
    )
    
    # Add zones
    for zone in zones:
        color = 'rgba(239, 68, 68, 0.3)' if zone.zone_type == 'supply' else 'rgba(16, 185, 129, 0.3)'
        border_color = '#ef4444' if zone.zone_type == 'supply' else '#10b981'
        
        # Zone rectangle
        fig.add_shape(
            type="rect",
            x0=zone.creation_time,
            x1=df.index[-1],
            y0=zone.bottom,
            y1=zone.top,
            fillcolor=color,
            line=dict(color=border_color, width=2 if zone.tested else 1),
            row=1, col=1
        )
        
        # Zone label
        label_text = f"{zone.zone_type.upper()}<br>{zone.get_strength_label()}"
        fig.add_annotation(
            x=df.index[-1],
            y=zone.top if zone.zone_type == 'supply' else zone.bottom,
            text=label_text,
            showarrow=False,
            bgcolor=border_color,
            font=dict(color='white', size=9),
            row=1, col=1
        )
    
    # Volume bars
    colors = ['#26a69a' if close >= open else '#ef5350' 
              for close, open in zip(df['close'], df['open'])]
    
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['volume'],
            name='Volume',
            marker_color=colors,
            showlegend=False
        ),
        row=2, col=1
    )
    
    # RSI - check if RSI exists in dataframe
    if 'rsi' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['rsi'],
                name='RSI',
                line=dict(color='#2196F3', width=2)
            ),
            row=3, col=1
        )
        
        # RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=3, col=1)
        fig.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.3, row=3, col=1)
    
    # Layout
    fig.update_layout(
        height=900,
        template='plotly_dark',
        showlegend=True,
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )
    
    fig.update_xaxes(showgrid=True, gridcolor='rgba(128, 128, 128, 0.2)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(128, 128, 128, 0.2)')
    
    return fig

# ============================================
# MAIN APPLICATION
# ============================================

def main():
    st.title("🎯 Professional Supply & Demand Zone Detector")
    st.markdown("*Institutional-Grade Analysis with OANDA Real-Time Data*")
    
    # Get API credentials from Streamlit secrets
    try:
        API_KEY = st.secrets["API_KEY"]
        ACCOUNT_ID = st.secrets["ACCOUNT_ID"]
        is_practice = True  # Using practice account
    except:
        # Fallback to hardcoded values if secrets not configured
        API_KEY = "b3f49c357df0852d6141377a821e7a67-20514dacb28f665d453d071d57ed67c9"
        ACCOUNT_ID = "101-001-37134715-001"
        is_practice = True
    
    # Initialize OANDA client
    client = OANDAClient(API_KEY, ACCOUNT_ID, is_practice)
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Market Selection
        st.subheader("📊 Market Selection")
        
        # Primary instruments
        primary_instruments = ["XAU_USD", "US30_USD", "NAS100_USD"]
        other_instruments = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD"]
        all_instruments = primary_instruments + other_instruments
        
        instrument = st.selectbox(
            "Select Instrument",
            all_instruments,
            index=0,
            format_func=lambda x: {
                "XAU_USD": "🥇 Gold (XAU/USD)",
                "US30_USD": "📈 US30 (Dow Jones)",
                "NAS100_USD": "💻 NAS100 (Nasdaq)",
                "EUR_USD": "💶 EUR/USD",
                "GBP_USD": "💷 GBP/USD",
                "USD_JPY": "💴 USD/JPY",
                "AUD_USD": "🇦🇺 AUD/USD",
                "USD_CAD": "🇨🇦 USD/CAD"
            }.get(x, x)
        )
        
        # Timeframe selection
        timeframe = st.selectbox(
            "Timeframe",
            ["H4", "D"],
            index=0,
            format_func=lambda x: "⏰ 4 Hour" if x == "H4" else "📅 Daily"
        )
        
        candle_count = st.slider("Historical Candles", 100, 1000, 500)
        
        st.divider()
        
        # Zone Detection Settings
        st.subheader("🎛️ Detection Settings")
        
        strength_preset = st.select_slider(
            "Zone Strength",
            options=["Aggressive", "Balanced", "Conservative", "Ultra Conservative"],
            value="Balanced"
        )
        
        # Map preset to multiplier
        strength_map = {
            "Aggressive": 1.5,
            "Balanced": 2.0,
            "Conservative": 2.5,
            "Ultra Conservative": 3.0
        }
        candle_multiplier = strength_map[strength_preset]
        
        use_volume_filter = st.checkbox("Volume Filter", value=True)
        volume_threshold = st.slider("Volume Threshold", 1.0, 3.0, 1.3, 0.1) if use_volume_filter else 1.0
        
        advanced_filters = st.checkbox("Advanced Filters", value=True, 
                                       help="Swing high/low, wick analysis")
        
        min_zone_height = st.slider("Min Zone Height %", 0.1, 2.0, 0.3, 0.1)
        
        max_zones = st.slider("Maximum Zones", 5, 30, 15)
        
        remove_untested = st.checkbox("Remove Untested Zones", value=True)
        untested_bar_limit = st.slider("Untested Bar Limit", 20, 200, 100) if remove_untested else 999
        
        st.divider()
        
        # Auto-refresh option
        auto_refresh = st.checkbox("Auto Refresh", value=False)
        if auto_refresh:
            refresh_interval = st.slider("Refresh Interval (seconds)", 30, 300, 60)
            st.info(f"🔄 Auto-refresh every {refresh_interval} seconds")
        
        analyze_button = st.button("🚀 Analyze Market", type="primary", use_container_width=True)
    
    # Main Content Area
    if analyze_button or auto_refresh:
        with st.spinner(f"🔄 Fetching {instrument} data..."):
            df = client.get_candles(instrument, timeframe, candle_count)
            
            if df is None or df.empty:
                st.error("❌ Failed to fetch data. Check your network connection.")
                return
            
            st.success(f"✅ Loaded {len(df)} candles for {instrument}")
        
        with st.spinner("🔍 Detecting zones..."):
            config = {
                'candle_multiplier': candle_multiplier,
                'use_volume_filter': use_volume_filter,
                'volume_threshold': volume_threshold,
                'advanced_filters': advanced_filters,
                'min_zone_height': min_zone_height,
                'max_zones': max_zones,
                'remove_untested': remove_untested,
                'untested_bar_limit': untested_bar_limit
            }
            
            detector = ZoneDetector(df, config)
            zones = detector.detect_zones()
            df_enhanced = detector.get_enhanced_dataframe()
            
            st.success(f"✅ Detected {len(zones)} high-quality zones")
        
        # Key Metrics
        st.subheader("📊 Market Overview")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        current_price = df['close'].iloc[-1]
        price_change = ((df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]) * 100
        
        supply_zones = [z for z in zones if z.zone_type == 'supply']
        demand_zones = [z for z in zones if z.zone_type == 'demand']
        
        with col1:
            st.metric("Current Price", f"{current_price:.5f}", f"{price_change:+.2f}%")
        
        with col2:
            st.metric("Supply Zones", len(supply_zones), "Resistance")
        
        with col3:
            st.metric("Demand Zones", len(demand_zones), "Support")
        
        with col4:
            avg_strength = np.mean([z.strength for z in zones]) if zones else 0
            st.metric("Avg Strength", f"{avg_strength:.2f}", "Quality")
        
        with col5:
            # Get ATR from enhanced dataframe
            if 'atr' in df_enhanced.columns and not df_enhanced['atr'].empty:
                current_atr = df_enhanced['atr'].iloc[-1]
                if pd.notna(current_atr):
                    st.metric("ATR", f"{current_atr:.5f}", "Volatility")
                else:
                    st.metric("ATR", "N/A", "Volatility")
            else:
                st.metric("Volume", f"{df['volume'].iloc[-1]:,}", "Activity")
        
        # Chart
        st.subheader("📈 Price Action & Zones")
        fig = create_advanced_chart(df_enhanced, zones, instrument, timeframe)
        st.plotly_chart(fig, use_container_width=True)
        
        # Zone Details
        st.subheader("🎯 Zone Analysis")
        
        # Tabs for Supply and Demand
        tab1, tab2 = st.tabs(["📍 All Zones", "📋 Detailed Report"])
        
        with tab1:
            if zones:
                for zone in zones:
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        badge_class = "supply-badge" if zone.zone_type == 'supply' else "demand-badge"
                        distance_text = f"{zone.distance_from_price:.2f}% {'above' if zone.zone_type == 'supply' else 'below'} price"
                        
                        st.markdown(f"""
                        <div class="zone-card">
                            <span class="{badge_class}">{zone.zone_type.upper()}</span>
                            <strong> {zone.get_strength_label()}</strong>
                            <br/>
                            <small>
                            📍 Zone: {zone.bottom:.5f} - {zone.top:.5f} | 
                            📏 Height: {zone.get_zone_height():.5f} | 
                            📊 {distance_text}
                            </small>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.metric("Touches", zone.touches)
                        st.metric("Age", f"{zone.age_bars} bars")
                        st.metric("R:R", f"{zone.rr_ratio:.2f}")
            else:
                st.info("No zones detected with current settings. Try adjusting the detection parameters.")
        
        with tab2:
            if zones:
                report_data = []
                for zone in zones:
                    report_data.append({
                        'Type': zone.zone_type.upper(),
                        'Top': f"{zone.top:.5f}",
                        'Bottom': f"{zone.bottom:.5f}",
                        'Strength': zone.get_strength_label(),
                        'Touches': zone.touches,
                        'Tested': '✅' if zone.tested else '❌',
                        'Distance %': f"{zone.distance_from_price:.2f}%",
                        'Age': f"{zone.age_bars} bars",
                        'R:R Ratio': f"{zone.rr_ratio:.2f}",
                        'Volume': f"{zone.volume_ratio:.2f}x"
                    })
                
                report_df = pd.DataFrame(report_data)
                st.dataframe(report_df, use_container_width=True, hide_index=True)
                
                # Download report
                csv = report_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Report (CSV)",
                    data=csv,
                    file_name=f"{instrument}_{timeframe}_zones_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No zones to report")
        
        # Trading Recommendations
        st.subheader("💡 Trading Insights")
        
        # Find nearest zones
        nearest_supply = None
        nearest_demand = None
        
        for zone in supply_zones:
            if zone.bottom > current_price:
                if nearest_supply is None or zone.bottom < nearest_supply.bottom:
                    nearest_supply = zone
        
        for zone in demand_zones:
            if zone.top < current_price:
                if nearest_demand is None or zone.top > nearest_demand.top:
                    nearest_demand = zone
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔴 Nearest Supply (Resistance)")
            if nearest_supply:
                distance = ((nearest_supply.bottom - current_price) / current_price) * 100
                st.markdown(f"""
                - **Level**: {nearest_supply.bottom:.5f} - {nearest_supply.top:.5f}
                - **Distance**: {distance:.2f}% above price
                - **Strength**: {nearest_supply.get_strength_label()}
                - **Touches**: {nearest_supply.touches}
                - **Risk/Reward**: {nearest_supply.rr_ratio:.2f}
                
                **Strategy**: Consider short positions if price reaches zone with bearish confirmation.
                """)
            else:
                st.info("No supply zone above current price")
        
        with col2:
            st.markdown("#### 🟢 Nearest Demand (Support)")
            if nearest_demand:
                distance = ((current_price - nearest_demand.top) / current_price) * 100
                st.markdown(f"""
                - **Level**: {nearest_demand.bottom:.5f} - {nearest_demand.top:.5f}
                - **Distance**: {distance:.2f}% below price
                - **Strength**: {nearest_demand.get_strength_label()}
                - **Touches**: {nearest_demand.touches}
                - **Risk/Reward**: {nearest_demand.rr_ratio:.2f}
                
                **Strategy**: Consider long positions if price reaches zone with bullish confirmation.
                """)
            else:
                st.info("No demand zone below current price")
        
        # Market Bias
        st.markdown("#### 🎯 Market Bias Analysis")
        
        # Calculate bias based on zone positioning
        supply_count_above = len([z for z in supply_zones if z.bottom > current_price])
        demand_count_below = len([z for z in demand_zones if z.top < current_price])
        
        # RSI
        current_rsi = df_enhanced['rsi'].iloc[-1] if 'rsi' in df_enhanced.columns else 50
        
        # Price position analysis
        recent_high = df['high'].tail(20).max()
        recent_low = df['low'].tail(20).min()
        price_position = ((current_price - recent_low) / (recent_high - recent_low)) * 100
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("RSI (14)", f"{current_rsi:.1f}", 
                     "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral")
        
        with col2:
            st.metric("Price Position", f"{price_position:.1f}%", 
                     "Upper range" if price_position > 70 else "Lower range" if price_position < 30 else "Mid range")
        
        with col3:
            bias = "Bullish 📈" if demand_count_below > supply_count_above else "Bearish 📉" if supply_count_above > demand_count_below else "Neutral ↔️"
            st.metric("Zone Bias", bias)
        
        # Advanced Statistics
        with st.expander("📊 Advanced Statistics"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Supply Zone Metrics**")
                if supply_zones:
                    avg_supply_strength = np.mean([z.strength for z in supply_zones])
                    avg_supply_touches = np.mean([z.touches for z in supply_zones])
                    tested_supply = len([z for z in supply_zones if z.tested])
                    
                    st.write(f"- Average Strength: {avg_supply_strength:.2f}")
                    st.write(f"- Average Touches: {avg_supply_touches:.1f}")
                    st.write(f"- Tested Zones: {tested_supply}/{len(supply_zones)}")
                else:
                    st.write("No supply zones detected")
            
            with col2:
                st.markdown("**Demand Zone Metrics**")
                if demand_zones:
                    avg_demand_strength = np.mean([z.strength for z in demand_zones])
                    avg_demand_touches = np.mean([z.touches for z in demand_zones])
                    tested_demand = len([z for z in demand_zones if z.tested])
                    
                    st.write(f"- Average Strength: {avg_demand_strength:.2f}")
                    st.write(f"- Average Touches: {avg_demand_touches:.1f}")
                    st.write(f"- Tested Zones: {tested_demand}/{len(demand_zones)}")
                else:
                    st.write("No demand zones detected")
        
        # Risk Management Calculator
        with st.expander("⚠️ Risk Management Calculator"):
            st.markdown("### Position Size Calculator")
            
            col1, col2 = st.columns(2)
            
            with col1:
                account_size = st.number_input("Account Size ($)", min_value=100, value=10000, step=100)
                risk_percent = st.slider("Risk Per Trade (%)", 0.5, 5.0, 1.0, 0.5)
                entry_price = st.number_input("Entry Price", value=float(current_price), format="%.5f")
            
            with col2:
                trade_direction = st.radio("Trade Direction", ["Long (Buy)", "Short (Sell)"])
                
                if trade_direction == "Long (Buy)" and nearest_demand:
                    suggested_sl = nearest_demand.bottom
                    suggested_tp = nearest_supply.bottom if nearest_supply else entry_price * 1.02
                else:
                    suggested_sl = nearest_supply.top if nearest_supply else entry_price * 1.01
                    suggested_tp = nearest_demand.top if nearest_demand else entry_price * 0.98
                
                stop_loss = st.number_input("Stop Loss", value=float(suggested_sl), format="%.5f")
                take_profit = st.number_input("Take Profit", value=float(suggested_tp), format="%.5f")
            
            # Calculations
            risk_amount = account_size * (risk_percent / 100)
            
            if trade_direction == "Long (Buy)":
                sl_distance = entry_price - stop_loss
                tp_distance = take_profit - entry_price
            else:
                sl_distance = stop_loss - entry_price
                tp_distance = entry_price - take_profit
            
            if sl_distance > 0:
                position_size = risk_amount / sl_distance
                rr_ratio = tp_distance / sl_distance if sl_distance > 0 else 0
                potential_profit = position_size * tp_distance
                
                st.markdown("---")
                st.markdown("### 📊 Trade Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Position Size", f"{position_size:,.0f} units")
                with col2:
                    st.metric("Risk Amount", f"${risk_amount:,.2f}")
                with col3:
                    st.metric("Potential Profit", f"${potential_profit:,.2f}")
                with col4:
                    st.metric("R:R Ratio", f"{rr_ratio:.2f}")
                
                # Trade quality assessment
                if rr_ratio >= 2.0:
                    quality = "🟢 Excellent"
                elif rr_ratio >= 1.5:
                    quality = "🟡 Good"
                else:
                    quality = "🔴 Poor"
                
                st.info(f"**Trade Quality**: {quality} (Minimum recommended R:R is 1.5:1)")
            else:
                st.error("Invalid stop loss position")
        
        # Footer with timestamp
        st.markdown("---")
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Data source: OANDA | Instrument: {instrument} | Timeframe: {timeframe}")
        
        # Auto-refresh
        if auto_refresh:
            import time
            time.sleep(refresh_interval)
            st.rerun()
    
    else:
        # Welcome screen
        st.info("👈 Click 'Analyze Market' in the sidebar to begin analysis")
        
        st.markdown("### 📖 Quick Start Guide")
        st.markdown(f"""
        1. **Select Instrument**: Choose from Gold (XAU/USD), US30, NAS100, or Forex pairs
        2. **Choose Timeframe**: 4-Hour or Daily charts available
        3. **Configure Settings**: Adjust detection sensitivity and filters
        4. **Analyze**: Click 'Analyze Market' button to detect zones
        5. **Review Results**: Study the zones, metrics, and trading insights
        """)
        
        st.markdown("### ✨ Key Features")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**🎯 Smart Detection**")
            st.markdown("Multi-factor zone analysis with institutional-grade algorithms")
        with col2:
            st.markdown("**📊 Real-Time Data**")
            st.markdown("Live OANDA market feed with automatic updates")
        with col3:
            st.markdown("**💎 Professional Tools**")
            st.markdown("Risk management calculator and position sizing")
        
        # Market status
        st.markdown("### 🌍 Market Status")
        market_status = {
            "Forex": "Open 24/5",
            "Gold": "Open 23/5",
            "Indices": "Check market hours"
        }
        for market, status in market_status.items():
            st.write(f"• **{market}**: {status}")

# ============================================
# RUN APPLICATION
# ============================================

if __name__ == "__main__":
    # Add footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### 📚 Resources
    - [Trading Guide](https://www.investopedia.com/articles/trading/09/supply-demand-trading.asp)
    - [Risk Management](https://www.investopedia.com/articles/forex/06/riskmanagement.asp)
    
    ### ℹ️ About
    Professional S&D Zone Detector v2.0
    
    Built with Streamlit + OANDA API
    
    *For educational purposes*
    """)
    
    main()
