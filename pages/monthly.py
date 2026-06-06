import requests
import pandas as pd
from datetime import datetime
import time

# ============================================================
# CONFIGURATION
# ============================================================
API_KEY = "your_api_key_here"
ACCOUNT_TYPE = "practice"  # "practice" or "live"

BASE_URL = (
    "https://api-fxtrade.oanda.com"
    if ACCOUNT_TYPE == "live"
    else "https://api-fxpractice.oanda.com"
)

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# ============================================================
# INSTRUMENTS TO FETCH
# ============================================================
INSTRUMENTS = [
    "EUR_USD",
    "GBP_USD",
    "USD_JPY",
    "AUD_USD",
    "USD_CAD",
    "USD_CHF",
    "NZD_USD"
]

GRANULARITY = "M"   # Monthly
COUNT       = 500   # Max safe count (covers 2005 to present)
PRICE_TYPE  = "MBA" # M=Mid, B=Bid, A=Ask

# ============================================================
# FETCH MONTHLY CANDLES
# ============================================================
def fetch_monthly_candles(instrument: str,
                          granularity: str = "M",
                          count: int = 500,
                          from_date: str = None,
                          to_date: str = None) -> pd.DataFrame:
    """
    Fetch monthly candlestick data from OANDA API.

    Parameters
    ----------
    instrument  : str  - e.g. 'EUR_USD'
    granularity : str  - 'M' monthly | 'W' weekly | 'D' daily
    count       : int  - number of candles (max 5000)
    from_date   : str  - 'YYYY-MM-DD' optional start date
    to_date     : str  - 'YYYY-MM-DD' optional end date

    Returns
    -------
    pd.DataFrame with OHLCV columns
    """
    endpoint = f"{BASE_URL}/v3/instruments/{instrument}/candles"

    params = {
        "granularity": granularity,
        "price":       PRICE_TYPE,
        "count":       count
    }

    # Override count if date range is supplied
    if from_date:
        from_dt    = datetime.strptime(from_date, "%Y-%m-%d")
        params["from"] = from_dt.strftime("%Y-%m-%dT%H:%M:%S.000000000Z")
        params.pop("count", None)   # OANDA: cannot use count + from together

    if to_date:
        to_dt        = datetime.strptime(to_date, "%Y-%m-%d")
        params["to"] = to_dt.strftime("%Y-%m-%dT%H:%M:%S.000000000Z")

    print(f"\n📡 Fetching {granularity} candles for {instrument} ...")
    print(f"   Params → {params}")

    response = requests.get(endpoint, headers=HEADERS, params=params)

    # ── Error handling ──────────────────────────────────────
    if response.status_code != 200:
        print(f"   ❌ Error {response.status_code}: {response.text}")
        return pd.DataFrame()

    data     = response.json()
    candles  = data.get("candles", [])

    if not candles:
        print(f"   ⚠️  No candle data returned for {instrument}")
        return pd.DataFrame()

    # ── Parse candles ───────────────────────────────────────
    rows = []
    for candle in candles:
        if not candle.get("complete", True):
            continue                          # skip incomplete current candle

        time_str = candle["time"]
        mid      = candle.get("mid", {})
        bid      = candle.get("bid", {})
        ask      = candle.get("ask", {})

        row = {
            "datetime": pd.to_datetime(time_str),
            "instrument": instrument,
            # Mid prices
            "open":   float(mid.get("o", 0)) if mid else None,
            "high":   float(mid.get("h", 0)) if mid else None,
            "low":    float(mid.get("l", 0)) if mid else None,
            "close":  float(mid.get("c", 0)) if mid else None,
            # Bid prices
            "bid_open":  float(bid.get("o", 0)) if bid else None,
            "bid_high":  float(bid.get("h", 0)) if bid else None,
            "bid_low":   float(bid.get("l", 0)) if bid else None,
            "bid_close": float(bid.get("c", 0)) if bid else None,
            # Ask prices
            "ask_open":  float(ask.get("o", 0)) if ask else None,
            "ask_high":  float(ask.get("h", 0)) if ask else None,
            "ask_low":   float(ask.get("l", 0)) if ask else None,
            "ask_close": float(ask.get("c", 0)) if ask else None,
            # Volume
            "volume": int(candle.get("volume", 0)),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        print(f"   ⚠️  DataFrame is empty after parsing for {instrument}")
        return df

    df.set_index("datetime", inplace=True)
    df.index = df.index.tz_convert("UTC")

    print(f"   ✅ {len(df)} candles fetched | "
          f"{df.index[0].date()} → {df.index[-1].date()}")

    return df


# ============================================================
# TECHNICAL INDICATORS  (pure pandas / no TA-lib needed)
# ============================================================
def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add common technical indicators to monthly OHLCV dataframe."""

    if df.empty:
        return df

    # ── Moving Averages ─────────────────────────────────────
    df["SMA_3"]  = df["close"].rolling(window=3).mean()
    df["SMA_6"]  = df["close"].rolling(window=6).mean()
    df["SMA_12"] = df["close"].rolling(window=12).mean()
    df["EMA_6"]  = df["close"].ewm(span=6,  adjust=False).mean()
    df["EMA_12"] = df["close"].ewm(span=12, adjust=False).mean()

    # ── RSI (14-period) ─────────────────────────────────────
    delta = df["close"].diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs        = avg_gain / avg_loss
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # ── MACD ────────────────────────────────────────────────
    ema12         = df["close"].ewm(span=12, adjust=False).mean()
    ema26         = df["close"].ewm(span=26, adjust=False).mean()
    df["MACD"]    = ema12 - ema26
    df["Signal"]  = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"]= df["MACD"] - df["Signal"]

    # ── Bollinger Bands (20-period) ─────────────────────────
    sma20               = df["close"].rolling(window=20).mean()
    std20               = df["close"].rolling(window=20).std()
    df["BB_Upper"]      = sma20 + (2 * std20)
    df["BB_Middle"]     = sma20
    df["BB_Lower"]      = sma20 - (2 * std20)
    df["BB_Width"]      = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Middle"]

    # ── ATR (Average True Range) ────────────────────────────
    high_low   = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close  = (df["low"]  - df["close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR_14"] = true_range.ewm(com=13, adjust=False).mean()

    # ── Monthly Returns ─────────────────────────────────────
    df["Monthly_Return"]    = df["close"].pct_change() * 100
    df["Cumulative_Return"] = (1 + df["close"].pct_change()).cumprod() - 1

    # ── Price Range ─────────────────────────────────────────
    df["Monthly_Range"]     = df["high"] - df["low"]
    df["Monthly_Range_Pct"] = (df["Monthly_Range"] / df["open"]) * 100

    return df


# ============================================================
# SUMMARY STATISTICS
# ============================================================
def print_summary(df: pd.DataFrame, instrument: str):
    """Print a clean summary of the fetched data."""
    if df.empty:
        return

    print(f"\n{'='*60}")
    print(f"  📊  {instrument} Monthly Summary")
    print(f"{'='*60}")
    print(f"  Period      : {df.index[0].date()} → {df.index[-1].date()}")
    print(f"  Total Months: {len(df)}")
    print(f"  Open        : {df['open'].iloc[0]:.5f}  →  {df['open'].iloc[-1]:.5f}")
    print(f"  Close Range : {df['close'].min():.5f} – {df['close'].max():.5f}")
    print(f"  Avg Monthly : {df['Monthly_Return'].mean():.3f}%")
    print(f"  Best Month  : {df['Monthly_Return'].max():.3f}%  "
          f"({df['Monthly_Return'].idxmax().date()})")
    print(f"  Worst Month : {df['Monthly_Return'].min():.3f}%  "
          f"({df['Monthly_Return'].idxmin().date()})")
    print(f"  Avg Volume  : {df['volume'].mean():,.0f}")
    print(f"  RSI (latest): {df['RSI_14'].iloc[-1]:.2f}")
    print(f"  MACD        : {df['MACD'].iloc[-1]:.6f}")
    print(f"{'='*60}")


# ============================================================
# SAVE TO CSV
# ============================================================
def save_to_csv(df: pd.DataFrame, instrument: str, folder: str = "."):
    """Save dataframe to CSV file."""
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
    print("=" * 60)

    all_data = {}

    for instrument in INSTRUMENTS:
        try:
            # ── Fetch raw candles ────────────────────────────
            df = fetch_monthly_candles(
                instrument  = instrument,
                granularity = GRANULARITY,
                count       = COUNT,
                # Uncomment below to use date range instead of count:
                # from_date = "2005-01-01",
                # to_date   = "2024-12-31"
            )

            if df.empty:
                continue

            # ── Add indicators ───────────────────────────────
            df = add_technical_indicators(df)

            # ── Summary ──────────────────────────────────────
            print_summary(df, instrument)

            # ── Save ─────────────────────────────────────────
            save_to_csv(df, instrument)

            all_data[instrument] = df

            # ── Polite delay between requests ─────────────────
            time.sleep(0.5)

        except Exception as e:
            print(f"   ❌ Exception for {instrument}: {e}")
            continue

    # ── Combined export ──────────────────────────────────────
    if all_data:
        combined = pd.concat(all_data.values(), keys=all_data.keys())
        combined.to_csv("all_instruments_monthly.csv")
        print(f"\n✅ Combined file saved → all_instruments_monthly.csv")

    print("\n🏁 Done!")
    return all_data


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    data = main()
