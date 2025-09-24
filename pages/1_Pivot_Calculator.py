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

# 📊 Pivot Logic (Classic)
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
    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Name", "Date", "Open", "High", "Low", "Close", "R3", "R2", "R1", "Pivot", "S1", "S2", "S3"])
        writer.writerow([name, date, o, h, l, c] + list(pivots))

# 🧰 Render pivot table with fixed-layout copy buttons
def render_pivot_table(name, levels):
    """
    levels: list of tuples [(label, value), ...]
    """
    table_id = f"tbl_{name}_{uuid4().hex}"
    rows_html = []
    for i, (lvl, val) in enumerate(levels):
        try:
            val_str = f"{float(val):.4f}"
        except Exception:
            val_str = str(val)
        btn_id = f"btn_{i}_{uuid4().hex}"
        rows_html.append(f"""
            <tr>
              <td class="lvl-td">{lvl}</td>
              <td class="val-td">{val_str}</td>
              <td class="copy-td">
                <button id="{btn_id}" class="copy-btn" data-value="{val_str}">
                  <span class="btn-label">Copy</span>
                </button>
              </td>
            </tr>
        """)
    rows_html = "\n".join(rows_html)

    html = f"""
    <style>
    #{table_id} table {{
        width: 100%;
        border-collapse: collapse;
        table-layout: fixed; /* prevents reflow/shift */
        margin-top: 6px;
    }}
    #{table_id} col.col-level {{ width: 20%; }}
    #{table_id} col.col-value {{ width: 55%; }}
    #{table_id} col.col-copy  {{ width: 25%; }}

    #{table_id} th, #{table_id} td {{
        border-bottom: 1px solid rgba(0,0,0,0.08);
        padding: 8px 12px;
    }}
    #{table_id} tr:nth-child(even) {{ background: rgba(0,0,0,0.03); }}
    #{table_id} th {{
        text-align: left;
        font-weight: 700;
        font-size: 14px;
    }}
    #{table_id} .lvl-td {{ font-weight: 600; }}
    #{table_id} .val-td {{
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        white-space: nowrap;
    }}
    #{table_id} .copy-td {{
        display: flex;
        align-items: center;
    }}
    /* Fixed-width button to avoid any nudge when text changes */
    #{table_id} .copy-btn {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 96px; /* keep button width constant */
        height: 32px;
        padding: 0 10px;
        border-radius: 6px;
        background: #0ea5e9;
        color: #fff;
        border: 2px solid transparent; /* reserve space for focus border */
        cursor: pointer;
        font-weight: 600;
        white-space: nowrap;
    }}
    #{table_id} .copy-btn:hover {{ background: #0284c7; }}
    #{table_id} .copy-btn:focus-visible {{
        border-color: #38bdf8; /* accessible focus without layout shift */
        outline: none;
    }}
    #{table_id} .copy-btn.copied {{ background: #16a34a; }}
    </style>

    <div id="{table_id}">
      <table>
        <colgroup>
          <col class="col-level" />
          <col class="col-value" />
          <col class="col-copy" />
        </colgroup>
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
      (function() {{
        const root = document.getElementById("{table_id}");
        const buttons = root.querySelectorAll('.copy-btn');
        buttons.forEach((btn) => {{
          btn.addEventListener('click', () => {{
            const val = btn.getAttribute('data-value');
            const label = btn.querySelector('.btn-label');
            const original = label.textContent;

            const setCopied = () => {{
              btn.classList.add('copied');
              label.textContent = 'Copied!';
              setTimeout(() => {{
                btn.classList.remove('copied');
                label.textContent = original;
              }}, 1200);
            }};

            if (navigator.clipboard && window.isSecureContext) {{
              navigator.clipboard.writeText(val).then(setCopied).catch(setCopied);
            }} else {{
              // Fallback for non-secure contexts
              const ta = document.createElement('textarea');
              ta.value = val;
              ta.style.position = 'fixed';
              ta.style.opacity = '0';
              document.body.appendChild(ta);
              ta.focus();
              ta.select();
              try {{ document.execCommand('copy'); }} catch (e) {{}}
              document.body.removeChild(ta);
              setCopied();
            }}
          }});
        }});
      }})();
    </script>
    """
    # ~7 rows + header
    components.html(html, height=360, scrolling=False)

# 🚀 Run Pivot Calculation
def run_pivot(granularity="D"):
    today = datetime.now(timezone.utc).date()
    label = "Daily" if granularity == "D" else "Weekly"
    st.subheader(f"📅 {label} Pivot Levels for {today}")

    for name, symbol in INSTRUMENTS.items():
        try:
            o, h, l, c, candle_date = fetch_ohlc(symbol, granularity)
            pivots = calculate_pivots(h, l, c)
            log_to_csv(name, candle_date, o, h, l, c, pivots)
            r3, r2, r1, p, s1, s2, s3 = pivots

            st.markdown(f"### 📊 {name}")

            color = "green" if c > o else "red"
            ohlc_html = f"""
            <div style='color:{color}; font-size:18px; font-weight:bold'>
            Open: {o:.2f} &nbsp;&nbsp; High: {h:.2f} &nbsp;&nbsp; Low: {l:.2f} &nbsp;&nbsp; Close: {c:.2f}
            </div>
            """
            st.markdown(ohlc_html, unsafe_allow_html=True)
            st.markdown("")

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
        header = next(reader, None)
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
