import streamlit as st
from datetime import datetime, timezone, timedelta
import requests
import csv
import os
from uuid import uuid4
import streamlit.components.v1 as components

# 🧭 Page Setup
st.set_page_config(page_title="Pivot Calculator", page_icon="📈")

# 🔐 OANDA Credentials from Secrets
try:
    API_KEY = st.secrets["API_KEY"]
    ACCOUNT_ID = st.secrets["ACCOUNT_ID"]
except Exception:
    st.error("🔐 API credentials not found in secrets. Please configure `API_KEY` and `ACCOUNT_ID`.")
    st.stop()

HEADERS = {"Authorization": f"Bearer {API_KEY}"}
BASE_URL = f"https://api-fxpractice.oanda.com/v3/instruments/{{}}/candles"
LOG_FILE = "pivot_log.csv"

# 📈 Instruments
INSTRUMENTS = {
    "GOLD": "XAU_USD",
    "NAS100": "NAS100_USD",
    "US30": "US30_USD"
}

# 🔍 Fetch OHLC (daily or weekly)
def fetch_ohlc(instrument, granularity="D"):
    params = {"granularity": granularity, "count": 2, "price": "M"}
    url = BASE_URL.format(instrument)
    r = requests.get(url, headers=HEADERS, params=params)
    r.raise_for_status()
    candles = r.json().get('candles', [])
    if len(candles) < 2:
        raise ValueError("Not enough candles returned")
    prev = candles[-2]
    date = prev['time'][:10]
    ohlc = prev['mid']
    return float(ohlc['o']), float(ohlc['h']), float(ohlc['l']), float(ohlc['c']), date

# 📊 Pivot Logic
def calculate_pivots(high, low, close):
    pivot = (high + low + close) / 3
    r1 = 2 * pivot - low
    r2 = pivot + (high - low)
    r3 = high + 2 * (pivot - low)
    s1 = 2 * pivot - high
    s2 = pivot - (high - low)
    s3 = low - 2 * (high - pivot)
    return round(r3, 4), round(r2, 4), round(r1, 4), round(pivot, 4), round(s1, 4), round(s2, 4), round(s3, 4)

# 🧾 Log to CSV
def log_to_csv(name, date, o, h, l, c, pivots):
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, date, o, h, l, c] + list(pivots))

# 🧰 Render pivot table with copy buttons
def render_pivot_table(name, levels):
    """
    levels: list of tuples [(label, value), ...]
    """
    table_id = f"tbl_{name}_{uuid4().hex}"
    rows_html = []
    for i, (lvl, val) in enumerate(levels):
        # Ensure string with 4 decimals
        val_str = f"{float(val):.4f}" if isinstance(val, (int, float)) or str(val).replace('.', '', 1).isdigit() else str(val)
        btn_id = f"btn_{i}_{uuid4().hex}"
        msg_id = f"msg_{i}_{uuid4().hex}"
        rows_html.append(f"""
            <tr>
              <td style="padding:8px 12px; font-weight:600">{lvl}</td>
              <td style="padding:8px 12px; font-family:ui-monospace, SFMono-Regular, Menlo, monospace">{val_str}</td>
              <td style="padding:8px 12px">
                <button id="{btn_id}" class="copy-btn" data-value="{val_str}">Copy</button>
                <span id="{msg_id}" class="copy-msg" style="margin-left:8px; color:#16a34a; font-size:12px;"></span>
              </td>
            </tr>
        """)
    rows_html = "\n".join(rows_html)

    html = f"""
    <style>
    #{table_id} .copy-btn {{
        background: #0ea5e9;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 6px 10px;
        cursor: pointer;
    }}
    #{table_id} .copy-btn:hover {{ background: #0284c7; }}
    #{table_id} table {{
        border-collapse: collapse;
        width: 100%;
        margin-top: 6px;
    }}
    #{table_id} tr:nth-child(even) {{ background: rgba(0,0,0,0.03); }}
    #{table_id} th, #{table_id} td {{
        border-bottom: 1px solid rgba(0,0,0,0.08);
    }}
    #{table_id} th {{
        text-align: left;
        padding: 8px 12px;
        font-weight: 700;
        font-size: 14px;
    }}
    </style>
    <div id="{table_id}">
      <table>
        <thead>
          <tr>
            <th>Level</th>
            <th>Value</th>
            <th>Copy</th>
          </tr>
        </thead>
        <tbody>
          {rows_html}
        </tbody>
      </table>
    </div>
    <script>
      const root = document.getElementById("{table_id}");
      const buttons = Array.from(root.querySelectorAll('.copy-btn'));
      buttons.forEach((btn) => {{
          btn.addEventListener('click', () => {{
              const val = btn.getAttribute('data-value');
              navigator.clipboard.writeText(val).then(() => {{
                  const msg = btn.nextElementSibling;
                  if (msg) {{
                      msg.textContent = 'Copied!';
                      setTimeout(() => {{ msg.textContent = ''; }}, 1200);
                  }}
              }});
          }});
      }});
    </script>
    """
    # Height: header + 7 rows approx 40px each
    components.html(html, height=320, scrolling=False)

# 🚀 Run Pivot Calculation
def run_pivot(granularity="D"):
    today = datetime.now(timezone.utc).date()
    label = "Daily" if granularity == "D" else "Weekly"
    st.subheader(f"📅 {label} Pivot Levels for {today}")

    for name, symbol in INSTRUMENTS.items():
        try:
            o, h, l, c, _ = fetch_ohlc(symbol, granularity)
            pivots = calculate_pivots(h, l, c)
            log_to_csv(name, today - timedelta(days=1), o, h, l, c, pivots)
            r3, r2, r1, p, s1, s2, s3 = pivots

            st.markdown(f"### 📊 {name}")

            color = "green" if c > o else "red"
            ohlc_html = f"""
            <div style='color:{color}; font-size:18px; font-weight:bold'>
            Open: {o:.2f} &nbsp;&nbsp; High: {h:.2f} &nbsp;&nbsp; Low: {l:.2f} &nbsp;&nbsp; Close: {c:.2f}
            </div>
            """
            st.markdown(ohlc_html, unsafe_allow_html=True)
            st.markdown(" ")

            # 🧱 Table with copy buttons
            rows = [("R3", r3), ("R2", r2), ("R1", r1), ("Pivot", p),
                    ("S1", s1), ("S2", s2), ("S3", s3)]
            render_pivot_table(name, rows)

            st.markdown("---")
        except Exception as e:
            st.error(f"{name}: Failed — {e}")

# 📂 View Logs
def view_logs():
    if not os.path.exists(LOG_FILE):
        st.warning("⚠️ No logs found.")
        return
    with open(LOG_FILE, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            name, date, o, h, l, c, r3, r2, r1, p, s1, s2, s3 = row
            st.markdown(f"### 📊 {name} — {date}")
            st.write(f"Open: {o}  High: {h}  Low: {l}  Close: {c}")
            st.write(f"R3: {r3}  R2: {r2}  R1: {r1}  Pivot: {p}")
            st.write(f"S1: {s1}  S2: {s2}  S3: {s3}")
            st.markdown("---")

# 🧭 Sidebar Controls
st.sidebar.title("📈 Pivot Dashboard")
action = st.sidebar.radio("Choose Action", ["Calculate Pivots", "View Logs"])
if action == "Calculate Pivots":
    timeframe = st.sidebar.radio("Select Timeframe", ["Daily", "Weekly"], horizontal=True)
    granularity = "D" if timeframe == "Daily" else "W"
    run_pivot(granularity)
else:
    view_logs()
