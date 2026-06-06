import requests
import pandas as pd
from datetime import datetime
import streamlit as st
import time

# ============================================================
# CONFIGURATION
# ============================================================
API_KEY    = st.secrets["API_KEY"]
ACCOUNT_ID = st.secrets["ACCOUNT_ID"]
BASE_URL   = "https://api-fxpractice.oanda.com/v3"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# ============================================================
# INSTRUMENTS - NAS100, US30, USOIL, XAUUSD ONLY
# ============================================================
INSTRUMENTS = [
    "NAS100_USD",   # NASDAQ 100
    "US30_USD",     # Dow Jones US30
    "WTICO_USD",    # WTI Crude Oil (USOIL)
    "XAU_USD",      # Gold vs USD
]

# ── Friendly display names ───────────────────────────────────
DISPLAY_NAMES = {
    "NAS100_USD" : "NASDAQ 100  (NAS100)",
    "US30_USD"   : "Dow Jones   (US30)",
    "WTICO_USD"  : "WTI Crude   (USOIL)",
    "XAU_USD"    : "Gold        (XAUUSD)",
}

GRANULARITY = "M"    # Monthly
COUNT       = 500    # Covers 2005 → present
PRICE_TYPE  = "MBA"  # Mid + Bid + Ask

# ============================================================
# FETCH MONTHLY CANDLES
# ============================================================
def fetch_monthly_candles(
    instrument  : str,
    granularity : str = "M",
    count       : int = 500,
    from_date   : str = None,
    to_date     : str = None
) -> pd.DataFrame:
    """
    Fetch monthly candlestick data from OANDA API.

    Parameters
    ----------
    instrument  : str  - OANDA instrument code e.g. 'XAU_USD'
    granularity : str  - 'M' monthly | 'W' weekly | 'D' daily
    count       : int  - number of candles (max 5000)
    from_date   : str  - 'YYYY-MM-DD' optional start date
    to_date     : str  - 'YYYY-MM-DD' optional end date

    Returns
    -------
    pd.DataFrame with OHLCV + Bid + Ask columns
    """

    endpoint = f"{BASE_URL}/instruments/{instrument}/candles"

    params = {
        "granularity" : granularity,
        "price"       : PRICE_TYPE,
        "count"       : count
    }

    # ── Date range overrides count ───────────────────────────
    if from_date:
        from_dt        = datetime.strptime(from_date, "%Y-%m-%d")
        params["from"] = from_dt.strftime("%Y-%m-%dT%H:%M:%S.000000000Z")
        params.pop("count", None)   # OANDA: cannot combine count + from

    if to_date:
        to_dt        = datetime.strptime(to_date, "%Y-%m-%d")
        params["to"] = to_dt.strftime("%Y-%m-%dT%H:%M:%S.000000000Z")

    label = DISPLAY_NAMES.get(instrument, instrument)
    print(f"\n📡 Fetching [{granularity}] candles for {label} ...")
    print(f"   Endpoint → {endpoint}")
    print(f"   Params   → {params}")

    # ── HTTP Request ─────────────────────────────────────────
    try:
        response = requests.get(
            endpoint,
            headers = HEADERS,
            params  = params,
            timeout = 30
        )
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Request failed: {e}")
        return pd.DataFrame()

    # ── HTTP Error handling ──────────────────────────────────
    if response.status_code != 200:
        print(f"   ❌ HTTP {response.status_code}: {response.text}")
        return pd.DataFrame()

    data    = response.json()
    candles = data.get("candles", [])

    if not candles:
        print(f"   ⚠️  No candle data returned for {instrument}")
        return pd.DataFrame()

    # ── Parse candles ────────────────────────────────────────
    rows = []
    for candle in candles:

        # Skip incomplete current candle
        if not candle.get("complete", True):
            continue

        time_str = candle.get("time", "")
        mid      = candle.get("mid", {})
        bid      = candle.get("bid", {})
        ask      = candle.get("ask", {})

        row = {
            "datetime"    : pd.to_datetime(time_str),
            "instrument"  : instrument,
            "display_name": DISPLAY_NAMES.get(instrument, instrument),
            # ── Mid OHLCV ─────────────────────────────────────
            "open"        : float(mid["o"]) if mid else None,
            "high"        : float(mid["h"]) if mid else None,
            "low"         : float(mid["l"]) if mid else None,
            "close"       : float(mid["c"]) if mid else None,
            # ── Bid ───────────────────────────────────────────
            "bid_open"    : float(bid["o"]) if bid else None,
            "bid_high"    : float(bid["h"]) if bid else None,
            "bid_low"     : float(bid["l"]) if bid else None,
            "bid_close"   : float(bid["c"]) if bid else None,
            # ── Ask ───────────────────────────────────────────
            "ask_open"    : float(ask["o"]) if ask else None,
            "ask_high"    : float(ask["h"]) if ask else None,
            "ask_low"     : float(ask["l"]) if ask else None,
            "ask_close"   : float(ask["c"]) if ask else None,
            # ── Volume ────────────────────────────────────────
            "volume"      : int(candle.get("volume", 0)),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        print(f"   ⚠️  DataFrame empty after parsing {instrument}")
        return df

    # ── Set datetime index ───────────────────────────────────
    df.set_index("datetime", inplace=True)

    if df.index.tzinfo is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    print(f"   ✅ {len(df)} candles | "
          f"{df.index[0].date()} → {df.index[-1].date()}")

    return df


# ============================================================
# TECHNICAL INDICATORS (pure pandas)
# ============================================================
def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to monthly OHLCV DataFrame."""

    if df.empty:
        return df

    # ── Moving Averages ──────────────────────────────────────
    df["SMA_3"]  = df["close"].rolling(window=3).mean()
    df["SMA_6"]  = df["close"].rolling(window=6).mean()
    df["SMA_12"] = df["close"].rolling(window=12).mean()
    df["EMA_6"]  = df["close"].ewm(span=6,  adjust=False).mean()
    df["EMA_12"] = df["close"].ewm(span=12, adjust=False).mean()

    # ── RSI 14-period ────────────────────────────────────────
    delta    = df["close"].diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs       = avg_gain / avg_loss
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # ── MACD ─────────────────────────────────────────────────
    ema12           = df["close"].ewm(span=12, adjust=False).mean()
    ema26           = df["close"].ewm(span=26, adjust=False).mean()
    df["MACD"]      = ema12 - ema26
    df["Signal"]    = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["Signal"]

    # ── Bollinger Bands 20-period ─────────────────────────────
    sma20           = df["close"].rolling(window=20).mean()
    std20           = df["close"].rolling(window=20).std()
    df["BB_Upper"]  = sma20 + (2 * std20)
    df["BB_Middle"] = sma20
    df["BB_Lower"]  = sma20 - (2 * std20)
    df["BB_Width"]  = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Middle"]

    # ── ATR 14-period ────────────────────────────────────────
    high_low   = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close  = (df["low"]  - df["close"].shift()).abs()
    true_range = pd.concat(
        [high_low, high_close, low_close], axis=1
    ).max(axis=1)
    df["ATR_14"] = true_range.ewm(com=13, adjust=False).mean()

    # ── Monthly Returns ──────────────────────────────────────
    df["Monthly_Return"]    = df["close"].pct_change() * 100
    df["Cumulative_Return"] = (1 + df["close"].pct_change()).cumprod() - 1

    # ── Price Range ──────────────────────────────────────────
    df["Monthly_Range"]     = df["high"] - df["low"]
    df["Monthly_Range_Pct"] = (df["Monthly_Range"] / df["open"]) * 100

    # ── Spread (Ask - Bid) ───────────────────────────────────
    if "ask_close" in df.columns and "bid_close" in df.columns:
        df["Spread"] = df["ask_close"] - df["bid_close"]

    return df


# ============================================================
# SUMMARY STATISTICS
# ============================================================
def print_summary(df: pd.DataFrame, instrument: str):
    """Print formatted summary for each instrument."""
    if df.empty:
        return

    label = DISPLAY_NAMES.get(instrument, instrument)

    print(f"\n{'='*60}")
    print(f"  📊  {label}")
    print(f"{'='*60}")
    print(f"  Period       : {df.index[0].date()} → {df.index[-1].date()}")
    print(f"  Total Months : {len(df)}")
    print(f"  First Open   : {df['open'].iloc[0]:>12.4f}")
    print(f"  Last Close   : {df['close'].iloc[-1]:>12.4f}")
    print(f"  All-Time High: {df['high'].max():>12.4f}")
    print(f"  All-Time Low : {df['low'].min():>12.4f}")
    print(f"  Avg Monthly  : {df['Monthly_Return'].mean():>10.3f}%")
    print(f"  Best Month   : {df['Monthly_Return'].max():>10.3f}%"
          f"  ({df['Monthly_Return'].idxmax().date()})")
    print(f"  Worst Month  : {df['Monthly_Return'].min():>10.3f}%"
          f"  ({df['Monthly_Return'].idxmin().date()})")
    print(f"  Avg Volume   : {df['volume'].mean():>12,.0f}")
    print(f"  RSI (latest) : {df['RSI_14'].iloc[-1]:>10.2f}")
    print(f"  MACD         : {df['MACD'].iloc[-1]:>12.4f}")
    if "Spread" in df.columns:
        print(f"  Avg Spread   : {df['Spread'].mean():>12.5f}")
    print(f"{'='*60}")


# ============================================================
# SAVE TO CSV
# ============================================================
def save_to_csv(df: pd.DataFrame, instrument: str, folder: str = "."):
    """Save individual instrument DataFrame to CSV."""
    if df.empty:
        return

    filename = f"{folder}/{instrument}_monthly.csv"
    df.to_csv(filename)
    print(f"   💾 Saved → {filename}")


# ============================================================
# MAIN RUNNER
# ============================================================
def main():
    print("\n🚀 OANDA Monthly Data Fetcher")
    print("   Instruments: NAS100 | US30 | USOIL | XAUUSD")
    print("=" * 60)

    all_data = {}

    for instrument in INSTRUMENTS:
        try:
            # ── Fetch candles ─────────────────────────────────
            df = fetch_monthly_candles(
                instrument  = instrument,
                granularity = GRANULARITY,
                count       = COUNT,
                # ── Optional: use date range instead of count ──
                # from_date = "2005-01-01",
                # to_date   = "2024-12-31",
            )

            if df.empty:
                print(f"   ⚠️  Skipping {instrument} — no data returned")
                continue

            # ── Add indicators ────────────────────────────────
            df = add_technical_indicators(df)

            # ── Print summary ─────────────────────────────────
            print_summary(df, instrument)

            # ── Save individual CSV ───────────────────────────
            save_to_csv(df, instrument)

            all_data[instrument] = df

            # ── Rate limit delay ──────────────────────────────
            time.sleep(0.5)

        except Exception as e:
            print(f"   ❌ Exception [{instrument}]: {e}")
            continue

    # ── Save combined CSV ─────────────────────────────────────
    if all_data:
        combined = pd.concat(all_data.values(), keys=all_data.keys())
        combined.to_csv("NAS100_US30_USOIL_XAUUSD_monthly.csv")
        print(f"\n✅ Combined CSV → NAS100_US30_USOIL_XAUUSD_monthly.csv")
    else:
        print("\n⚠️  No data fetched. Check API key and instrument codes.")

    print("\n🏁 Done!")
    return all_data


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    data = main()
