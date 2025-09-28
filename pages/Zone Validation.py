import streamlit as st
import requests
import pandas as pd
from dataclasses import dataclass
from datetime import datetime, timedelta

# --- Twelve Data API ---
API_KEY = st.secrets["TWELVE_DATA"]["9497fde228f249b88beeb24558f14f12"]
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

# --- Candle Structure ---
@dataclass
class Candle:
    name: int
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

def validate_rally_drop(candles, atr_series, current_atr):
    n = len(candles)
    if n < 1 or n > 6:
        return False, "Candle count out of range"

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
                return False, "Mixed directional bias"
        else:
            if prev and is_inside(prev, c):
                inside_cnt += 1
            else:
                return False, "Non-valid non-inside candle"

    ok = (
        valid >= rule["valid"] and
        inside_cnt <= rule["inside"] and
        atr_exp >= rule["atr_exp"]
    )
    return ok, f"{n}-candle {'rally' if direction else 'drop'} validation: {ok}"

# --- Execution Example ---
if __name__ == "__main__":
    symbol = "XAU/USD"
    end = datetime.utcnow()
    start = end - timedelta(days=3)
    current_atr = 0.75  # manually entered

    df = fetch_ohlc(symbol, start, end)
    df["atr"] = (df["high"] - df["low"]).abs()
    atr_series = df["atr"].to_dict()

    # Convert to Candle objects
    candles = [
        Candle(name=i, open=row.open, high=row.high, low=row.low, close=row.close)
        for i, row in df.tail(6).iterrows()
    ]

    result, message = validate_rally_drop(candles, atr_series, current_atr)
    print(f"\n🔍 {message}")
