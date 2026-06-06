import streamlit as st
from datetime import datetime, timezone, timedelta
import requests
import csv
import os
import pandas as pd
import calendar

# 🧭 Page Setup (modern wide layout)
st.set_page_config(
    page_title="Pivot Calculator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 🔐 OANDA Credentials from Secrets
try:
    API_KEY = st.secrets["API_KEY"]
    ACCOUNT_ID = st.secrets["ACCOUNT_ID"]
    BASE_URL = "https://api-fxpractice.oanda.com/v3"
except Exception:
    st.error("🔐 API credentials not found in secrets. Please configure `API_KEY` and `ACCOUNT_ID`.")
    st.stop()

HEADERS = {"Authorization": f"Bearer {API_KEY}"}
BASE_URL = "https://api-fxpractice.oanda.com/v3/instruments/{}/candles"
LOG_FILE = "pivot_log.csv"

# 📈 Instruments
INSTRUMENTS = {
    "USOIL": "WTICO_USD",
    "GOLD": "XAU_USD",
    "NAS100": "NAS100_USD",
    "US30": "US30_USD",
}

# Price source (Mid, to match your reference behavior)
PRICE_TYPE = "M"  # "M"=mid, "B"=bid, "A"=ask
OHLC_KEY = {"B": "bid", "M": "mid", "A": "ask"}[PRICE_TYPE]

# 💄 Simple badges for a modern header
BADGE_CSS = """
<style>
.badge {
  display:inline-block; padding:4px 10px; border-radius:999px; font-size:12px;
  background: rgba(2,132,199,.12); color:#0284c7; margin-right:6px; border:1px solid rgba(2,132,199,.25)
}
.badge.neutral {
  background: rgba(148,163,184,.12); color:#334155; border-color: rgba(148,163,184,.25);
}
.badge.demark {
  background: rgba(168,85,247,.12); color:#7c3aed; border-color: rgba(168,85,247,.25);
}
.badge.monthly {
  background: rgba(34,197,94,.12); color:#15803d; border-color: rgba(34,197,94,.25);
}
</style>
"""
st.markdown(BADGE_CSS, unsafe_allow_html=True)


def iso_midnight_utc(d):
    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


def first_day_of_month(d):
    """Return first day of the month for a given date."""
    return d.replace(day=1)


def first_day_of_prev_month(d):
    """Return first day of the previous month."""
    first = d.replace(day=1)
    prev = first - timedelta(days=1)
    return prev.replace(day=1)


# ⚡️ Cached session + cached HTTP calls for snappy UX
@st.cache_resource
def get_session():
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


@st.cache_data(show_spinner=False, ttl=60)
def _request_candles_cached(instrument, params_tuple):
    params = dict(params_tuple)
    url = BASE_URL.format(instrument)
    s = get_session()
    r = s.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json().get("candles", [])


def _request_candles(instrument, params):
    params_tuple = tuple(sorted(params.items()))
    return _request_candles_cached(instrument, params_tuple)


def _extract_ohlc(candle):
    ohlc = candle[OHLC_KEY]
    return float(ohlc["o"]), float(ohlc["h"]), float(ohlc["l"]), float(ohlc["c"])


# 🔍 Last completed candle (native OANDA D/W candles)
def fetch_last_completed_candle(instrument, granularity="D"):
    params = {"granularity": granularity, "price": PRICE_TYPE, "count": 2}
    candles = _request_candles(instrument, params)
    if len(candles) < 2:
        raise ValueError("Not enough candles returned")
    c = candles[-2]  # last completed
    o, h, l, c_close = _extract_ohlc(c)
    return o, h, l, c_close, c["time"][:10]


# 🔁 Prior completed candle strictly before a selected date
def fetch_prior_candle_before_date(instrument, granularity, selected_date):
    if granularity == "D":
        params = {
            "granularity": "D",
            "price": PRICE_TYPE,
            "to": iso_midnight_utc(selected_date),
            "count": 1,
        }
        candles = _request_candles(instrument, params)
        if candles:
            c = candles[-1]
            o, h, l, c_close = _extract_ohlc(c)
            return o, h, l, c_close, c["time"][:10]
        # Fallback
        t = selected_date - timedelta(days=1)
        for _ in range(9):
            params["to"] = iso_midnight_utc(t)
            candles = _request_candles(instrument, params)
            if candles:
                c = candles[-1]
                o, h, l, c_close = _extract_ohlc(c)
                return o, h, l, c_close, c["time"][:10]
            t -= timedelta(days=1)
        raise ValueError(
            f"No prior daily candle found before {selected_date} for {instrument}"
        )

    elif granularity == "W":
        params = {
            "granularity": "W",
            "price": PRICE_TYPE,
            "to": iso_midnight_utc(selected_date),
            "count": 1,
        }
        candles = _request_candles(instrument, params)
        if candles:
            c = candles[-1]
            o, h, l, c_close = _extract_ohlc(c)
            return o, h, l, c_close, c["time"][:10]
        # Fallback: step back by weeks
        t = selected_date - timedelta(days=7)
        for _ in range(5):
            params["to"] = iso_midnight_utc(t)
            candles = _request_candles(instrument, params)
            if candles:
                c = candles[-1]
                o, h, l, c_close = _extract_ohlc(c)
                return o, h, l, c_close, c["time"][:10]
            t -= timedelta(days=7)
        raise ValueError(
            f"No prior weekly candle found before {selected_date} for {instrument}"
        )


# 📅 Fetch prior completed monthly candle
# OANDA does not have "M" granularity natively, so we aggregate daily candles
def fetch_monthly_candle(instrument, custom_date=None):
    """
    Fetch the prior completed month's OHLC by aggregating daily candles.
    If custom_date is provided, use the month before that date's month.
    Otherwise use the previous calendar month.
    """
    today = datetime.now(timezone.utc).date()
    ref_date = custom_date if custom_date else today

    # Previous month boundaries
    first_of_ref_month = ref_date.replace(day=1)
    last_day_of_prev_month = first_of_ref_month - timedelta(days=1)
    first_day_of_prev = last_day_of_prev_month.replace(day=1)

    from_dt = iso_midnight_utc(first_day_of_prev)
    # Add one day to last day so 'to' is exclusive end-of-month boundary
    to_dt = iso_midnight_utc(first_of_ref_month)

    params = {
        "granularity": "D",
        "price": PRICE_TYPE,
        "from": from_dt,
        "to": to_dt,
    }

    candles = _request_candles(instrument, params)
    if not candles:
        raise ValueError(
            f"No daily candles found for {instrument} in {first_day_of_prev} to {last_day_of_prev_month}"
        )

    # Aggregate: Open of first candle, High/Low across all, Close of last candle
    opens  = [_extract_ohlc(c)[0] for c in candles]
    highs  = [_extract_ohlc(c)[1] for c in candles]
    lows   = [_extract_ohlc(c)[2] for c in candles]
    closes = [_extract_ohlc(c)[3] for c in candles]

    o = opens[0]
    h = max(highs)
    l = min(lows)
    c = closes[-1]

    used_label = f"{first_day_of_prev} → {last_day_of_prev_month}"
    return o, h, l, c, used_label


# 📊 Pivot Logic (Classic)
def calculate_pivots_classic(high, low, close):
    pivot = (high + low + close) / 3
    r1 = 2 * pivot - low
    r2 = pivot + (high - low)
    r3 = high + 2 * (pivot - low)
    s1 = 2 * pivot - high
    s2 = pivot - (high - low)
    s3 = low - 2 * (high - pivot)
    return {
        "R3": round(r3, 4),
        "R2": round(r2, 4),
        "R1": round(r1, 4),
        "Pivot": round(pivot, 4),
        "S1": round(s1, 4),
        "S2": round(s2, 4),
        "S3": round(s3, 4),
    }


# 📊 Pivot Logic (Demark)
def calculate_pivots_demark(open_price, high, low, close):
    if close < open_price:
        x = high + 2 * low + close
    elif close > open_price:
        x = 2 * high + low + close
    else:
        x = high + low + 2 * close

    pivot = x / 4
    r1 = x / 2 - low
    s1 = x / 2 - high

    return {
        "R1": round(r1, 4),
        "Pivot": round(pivot, 4),
        "S1": round(s1, 4),
    }


# 🧾 Log to CSV
def log_to_csv(name, date, o, h, l, c, pivots, pivot_type, timeframe):
    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                [
                    "Name", "Date", "Timeframe", "Type",
                    "Open", "High", "Low", "Close",
                    "R3", "R2", "R1", "Pivot", "S1", "S2", "S3",
                ]
            )

        if pivot_type == "Classic":
            row = [
                name, date, timeframe, pivot_type, o, h, l, c,
                pivots["R3"], pivots["R2"], pivots["R1"],
                pivots["Pivot"], pivots["S1"], pivots["S2"], pivots["S3"],
            ]
        else:  # Demark
            row = [
                name, date, timeframe, pivot_type, o, h, l, c,
                "", "", pivots["R1"],
                pivots["Pivot"], pivots["S1"], "", "",
            ]

        writer.writerow(row)


def fmt_val(v, d=4):
    try:
        return f"{float(v):.{d}f}"
    except Exception:
        return str(v)


# 🧰 Native, theme-aware pivot list with per-level copy
def render_pivot_levels_native(rows, d=4):
    h1, h2, h3 = st.columns([1.0, 1.6, 1.0])
    h1.markdown("**Level**")
    h2.markdown("**Value**")
    h3.markdown("**Copy**")
    for lvl, val in rows:
        val_str = fmt_val(val, d)
        c1, c2, c3 = st.columns([1.0, 1.6, 1.0])
        c1.markdown(f"**{lvl}**")
        c2.markdown(f"`{val_str}`")
        c3.code(val_str, language="text")
    with st.expander("Copy all levels"):
        all_text = "\n".join(f"{lvl}: {fmt_val(val, d)}" for lvl, val in rows)
        st.code(all_text, language="text")


# 🚀 Run Pivot Calculation (modern UI)
def run_pivot(
    granularity="D",
    custom_date=None,
    selected_names=None,
    decimals=4,
    pivot_type="Classic",
):
    today = datetime.now(timezone.utc).date()
    label_map = {"D": "Daily", "W": "Weekly", "M": "Monthly"}
    label = label_map.get(granularity, granularity)
    pivot_date = custom_date if custom_date else today

    # Header + badges
    st.subheader("📈 Pivot Dashboard")
    badge_class = "badge demark" if pivot_type == "Demark" else "badge"
    tf_badge    = "badge monthly" if granularity == "M" else "badge neutral"
    st.markdown(
        f'<span class="badge">Pivot date: {pivot_date}</span>'
        f'<span class="{badge_class}">Type: {pivot_type}</span>'
        f'<span class="{tf_badge}">Timeframe: {label}</span>'
        f'<span class="badge neutral">Price: {PRICE_TYPE}</span>'
        f'<span class="badge neutral">Decimals: {decimals}</span>',
        unsafe_allow_html=True,
    )

    names = selected_names if selected_names else list(INSTRUMENTS.keys())
    results = []

    for name in names:
        symbol = INSTRUMENTS[name]
        try:
            # ── Fetch OHLC based on granularity ──────────────────────────────
            if granularity == "M":
                o, h, l, c, used_date = fetch_monthly_candle(symbol, custom_date=custom_date)

            elif custom_date:
                if granularity == "D":
                    query_date = custom_date - timedelta(days=1)
                else:  # W
                    query_date = custom_date - timedelta(days=7)
                o, h, l, c, used_date = fetch_prior_candle_before_date(
                    symbol, granularity, query_date
                )
            else:
                o, h, l, c, used_date = fetch_last_completed_candle(symbol, granularity)

            # ── Calculate pivots ──────────────────────────────────────────────
            if pivot_type == "Classic":
                pivots = calculate_pivots_classic(h, l, c)
                pivot_rows = [
                    ("R3", pivots["R3"]),
                    ("R2", pivots["R2"]),
                    ("R1", pivots["R1"]),
                    ("Pivot", pivots["Pivot"]),
                    ("S1", pivots["S1"]),
                    ("S2", pivots["S2"]),
                    ("S3", pivots["S3"]),
                ]
            else:  # Demark
                pivots = calculate_pivots_demark(o, h, l, c)
                pivot_rows = [
                    ("R1", pivots["R1"]),
                    ("Pivot", pivots["Pivot"]),
                    ("S1", pivots["S1"]),
                ]

            # ── Card UI ───────────────────────────────────────────────────────
            with st.container(border=True):
                st.markdown(f"### {name}")
                st.caption(f"Candle used: {used_date}")

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Open",  fmt_val(o, decimals))
                m2.metric("High",  fmt_val(h, decimals))
                m3.metric("Low",   fmt_val(l, decimals))
                delta = float(c - o)
                m4.metric("Close", fmt_val(c, decimals), delta=f"{delta:+.{decimals}f}")

                st.markdown("#### 📌 Pivot Levels")
                render_pivot_levels_native(pivot_rows, d=decimals)

            # ── Collect for export & logging ──────────────────────────────────
            result = {
                "Name": name,
                "CandleDate": used_date,
                "Timeframe": label,
                "Type": pivot_type,
                "Open": o,
                "High": h,
                "Low": l,
                "Close": c,
            }
            result.update(pivots)
            results.append(result)

            log_to_csv(name, used_date, o, h, l, c, pivots, pivot_type, label)

        except Exception as e:
            st.error(f"{name}: Failed — {e}")

    # ── Export (CSV) ──────────────────────────────────────────────────────────
    if results:
        df = pd.DataFrame(results)
        st.download_button(
            "⬇️ Download pivots (CSV)",
            data=df.to_csv(index=False),
            file_name=f"pivots_{pivot_type.lower()}_{label.lower()}_{pivot_date}.csv",
            mime="text/csv",
            use_container_width=True,
        )


# 📂 View Logs
def view_logs():
    if not os.path.exists(LOG_FILE):
        st.warning("⚠️ No logs found.")
        return
    try:
        df = pd.read_csv(LOG_FILE)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if "Name" in df.columns:
                names_filter = st.multiselect(
                    "Filter by Instrument",
                    options=df["Name"].unique(),
                    default=df["Name"].unique(),
                )
                df = df[df["Name"].isin(names_filter)]
        with col2:
            if "Timeframe" in df.columns:
                tf_filter = st.multiselect(
                    "Filter by Timeframe",
                    options=df["Timeframe"].unique(),
                    default=df["Timeframe"].unique(),
                )
                df = df[df["Timeframe"].isin(tf_filter)]
        with col3:
            if "Type" in df.columns:
                type_filter = st.multiselect(
                    "Filter by Type",
                    options=df["Type"].unique(),
                    default=df["Type"].unique(),
                )
                df = df[df["Type"].isin(type_filter)]
        with col4:
            st.write(f"Total records: {len(df)}")

        st.dataframe(df, use_container_width=True)

        st.download_button(
            "⬇️ Download logs (CSV)",
            data=df.to_csv(index=False),
            file_name="pivot_logs_filtered.csv",
            mime="text/csv",
        )
    except Exception as e:
        st.error(f"Failed to read logs: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 🧭 Sidebar Controls
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.title("📈 Pivot Dashboard")
action = st.sidebar.radio("Choose Action", ["Calculate Pivots", "View Logs"])

if action == "Calculate Pivots":

    # Pivot Type
    pivot_type = st.sidebar.radio(
        "Pivot Type",
        ["Classic", "Demark"],
        horizontal=True,
        help="Classic: 7 levels (R3-R1, Pivot, S1-S3) | Demark: 3 levels (R1, Pivot, S1)",
    )

    # Timeframe  ← Monthly added here
    timeframe = st.sidebar.radio(
        "Select Timeframe",
        ["Daily", "Weekly", "Monthly"],
        horizontal=True,
    )
    granularity_map = {"Daily": "D", "Weekly": "W", "Monthly": "M"}
    granularity = granularity_map[timeframe]

    # Custom date (not relevant for Monthly – we always use prior full month)
    if granularity != "M":
        use_custom = st.sidebar.toggle("Use custom date", value=False)
        custom_date = (
            st.sidebar.date_input("Select date", value=datetime.now(timezone.utc).date())
            if use_custom
            else None
        )
    else:
        use_custom = st.sidebar.toggle(
            "Use custom reference month",
            value=False,
            help="Pick any date; the calculator will use the month *before* that date's month.",
        )
        custom_date = (
            st.sidebar.date_input(
                "Reference date (prior month will be used)",
                value=datetime.now(timezone.utc).date(),
            )
            if use_custom
            else None
        )

    # Instruments
    selected_names = st.sidebar.multiselect(
        "Instruments",
        options=list(INSTRUMENTS.keys()),
        default=list(INSTRUMENTS.keys()),
    )

    # Decimals
    decimals = st.sidebar.slider("Decimals", 2, 6, 4, help="Number of decimals to display")

    # Info panel
    with st.sidebar.expander("ℹ️ About Pivot Types"):
        st.markdown(
            """
**Classic Pivots:**
- Standard pivot calculation
- 7 levels: R3, R2, R1, Pivot, S1, S2, S3
- Based on High, Low, Close

**Demark Pivots:**
- Developed by Tom DeMark
- 3 levels: R1, Pivot, S1
- Considers Open-Close relationship
- Often more responsive to trends

**Monthly Pivots:**
- Uses the prior complete calendar month's OHLC
- Daily candles are aggregated:
  - Open = first day's open
  - High = highest daily high
  - Low  = lowest daily low
  - Close = last day's close
"""
        )

    run_pivot(
        granularity,
        custom_date=custom_date,
        selected_names=selected_names,
        decimals=decimals,
        pivot_type=pivot_type,
    )

else:
    view_logs()
