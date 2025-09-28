import pandas as pd
from datetime import datetime, timedelta

# --- CONFIG ---
VALID_MODES = ["rally", "drop"]

def validate_rally(df, min_atr):
    """Check if close is strictly increasing and ATR meets manual threshold."""
    closes = df["close"].values
    atrs = (df["high"] - df["low"]).abs()
    return all(closes[i] < closes[i+1] for i in range(len(closes)-1)) and all(atrs >= min_atr)

def validate_drop(df, min_atr):
    """Check if close is strictly decreasing and ATR meets manual threshold."""
    closes = df["close"].values
    atrs = (df["high"] - df["low"]).abs()
    return all(closes[i] > closes[i+1] for i in range(len(closes)-1)) and all(atrs >= min_atr)

def validate_sets(df, mode="rally", min_atr=0.5):
    """Validate all sets from 1 to 6 candles with manual ATR."""
    results = {}
    for n in range(1, 7):
        subset = df.tail(n)
        if len(subset) < n:
            results[n] = False
            continue
        if mode == "rally":
            results[n] = validate_rally(subset, min_atr)
        elif mode == "drop":
            results[n] = validate_drop(subset, min_atr)
    return results

# --- EXAMPLE USAGE ---
if __name__ == "__main__":
    # Simulated 4H OHLC data (replace with real fetch)
    data = {
        "datetime": [datetime.utcnow() - timedelta(hours=4*i) for i in range(20)],
        "open": [100 + i for i in range(20)],
        "high": [101 + i for i in range(20)],
        "low": [99 + i for i in range(20)],
        "close": [100 + i for i in range(20)],
    }
    df = pd.DataFrame(data).sort_values("datetime")

    # Filter by custom UTC range
    start = datetime.utcnow() - timedelta(days=2)
    end = datetime.utcnow()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    filtered = df[(df["datetime"] >= start) & (df["datetime"] <= end)]

    # Manual input
    mode = "rally"  # or "drop"
    manual_atr = 0.6  # entered by you

    result = validate_sets(filtered, mode=mode, min_atr=manual_atr)

    print(f"\n🔍 Validation results for {mode.upper()} zone with manual ATR ≥ {manual_atr}:")
    for k, v in result.items():
        print(f"  - {k}-candle set: {'✅ Valid' if v else '❌ Invalid'}")
