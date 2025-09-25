import streamlit as st
from typing import Optional, Dict, List
import pandas as pd

# ---------- Page + Style ----------
st.set_page_config(page_title="COT Spec Index Analyzer", page_icon="📊", layout="wide")

BADGE_CSS = """
<style>
.badge { display:inline-block; padding:3px 8px; border-radius:999px; font-size:12px;
         margin-right:6px; border:1px solid rgba(148,163,184,.35); color:#334155; background:rgba(148,163,184,.12);}
.badge.ok { color:#065f46; background:rgba(16,185,129,.12); border-color:rgba(16,185,129,.35);}
.badge.warn { color:#92400e; background:rgba(245,158,11,.12); border-color:rgba(245,158,11,.35);}
.badge.err { color:#991b1b; background:rgba(239,68,68,.12); border-color:rgba(239,68,68,.35);}
.small { font-size:12px; color:#475569; }
</style>
"""
st.markdown(BADGE_CSS, unsafe_allow_html=True)

st.title("COT Spec Index Analyzer")
st.caption("Inputs mapped to MarketBulls • Legacy → Large Speculators → Percent of OI, plus COT Index 6M/36M gauges")

# ---------- pp vs % CAUTION ----------
with st.container():
    st.warning(
        "Heads‑up: pp = percentage points (absolute difference), not percent change.\n"
        "- COT 6M 62% → 67% = +5 pp (not +8.1%)\n"
        "- Specs Net %OI 51.6% → 49.9% = −1.7 pp\n"
        "- Open Interest WoW is expressed as percent change (e.g., +2.0%), not pp",
        icon="⚠️"
    )

# ---------- Helpers ----------
def parse_flt(s: str) -> Optional[float]:
    """Parse user text like '63.3' or '63.3%' to float. Returns None if blank."""
    if s is None: return None
    s = str(s).strip().replace("%", "")
    if s == "": return None
    try:
        return float(s)
    except Exception:
        return None

def fmt(x: Optional[float], nd: int = 1, suffix: str = "") -> str:
    return "n/a" if x is None else f"{x:.{nd}f}{suffix}"

def net_pct_of_oi(long_pct: Optional[float], short_pct: Optional[float],
                  net_pct: Optional[float] = None) -> Optional[float]:
    # Specs Net %OI = Long% - Short%
    if net_pct is not None:
        return float(net_pct)
    if long_pct is None or short_pct is None:
        return None
    return float(long_pct) - float(short_pct)

def wow_pp(current: Optional[float], previous: Optional[float]) -> Optional[float]:
    # Week-over-week absolute difference in percentage points
    if current is None or previous is None:
        return None
    return float(current) - float(previous)

def pct_change(current: Optional[float], previous: Optional[float]) -> Optional[float]:
    # Relative percent change (for Open Interest)
    if current is None or previous is None or previous == 0:
        return None
    return (float(current) - float(previous)) / float(previous) * 100.0

def bias_from_indices(cot6: Optional[float], cot36: Optional[float]) -> str:
    # Specs-only, relative filter
    # Longs allowed if 6M > 60 and 36M > 50
    # Shorts allowed if 6M < 40 and 36M < 50
    # Else Neutral
    if cot6 is None or cot36 is None:
        return "Neutral"
    if cot6 > 60 and cot36 > 50:
        return "Longs allowed"
    if cot6 < 40 and cot36 < 50:
        return "Shorts allowed"
    return "Neutral"

def flags_and_grade(
    bias: str,
    cot6: Optional[float],
    cot36: Optional[float],
    d_cot6_pp: Optional[float],
    oi_wow_pct: Optional[float],
    net_wow_pp: Optional[float],
    net_now: Optional[float]
) -> Dict[str, str]:
    # Extremes: 6M or 36M > 85 or < 15 (trend maturity; avoid chasing)
    extremes = False
    if cot6 is not None and cot36 is not None:
        extremes = (cot6 > 85 or cot36 > 85 or cot6 < 15 or cot36 < 15)

    # Conflict: bias vs sign of net% OI
    conflict = "No"
    if net_now is not None and bias != "Neutral":
        if (bias == "Shorts allowed" and net_now > 0) or (bias == "Longs allowed" and net_now < 0):
            conflict = "Yes"

    # Momentum support rules
    def mom_ok_long():
        return ((d_cot6_pp is not None and d_cot6_pp >= 5)
                or (oi_wow_pct is not None and oi_wow_pct >= 0)
                or (net_wow_pp is not None and net_wow_pp >= 0))

    def mom_ok_short():
        return ((d_cot6_pp is not None and d_cot6_pp <= -5)
                or (oi_wow_pct is not None and oi_wow_pct <= 0)
                or (net_wow_pp is not None and net_wow_pp <= 0))

    if bias == "Neutral":
        return {"extremes": "Yes" if extremes else "No", "conflict": conflict, "grade": "C", "note": "Directional filter off"}

    if bias == "Longs allowed":
        supp = mom_ok_long()
        if extremes and not supp: return {"extremes": "Yes", "conflict": conflict, "grade": "B-", "note": "Extreme + weak momentum"}
        if extremes:               return {"extremes": "Yes", "conflict": conflict, "grade": "B",  "note": "Extreme"}
        if supp:                   return {"extremes": "No",  "conflict": conflict, "grade": "A",  "note": "Momentum supportive"}
        return {"extremes": "No",  "conflict": conflict, "grade": "B",  "note": "Momentum not confirmed"}

    # Shorts allowed
    supp = mom_ok_short()
    if extremes and not supp: return {"extremes": "Yes", "conflict": conflict, "grade": "B-", "note": "Extreme + weak momentum"}
    if extremes:               return {"extremes": "Yes", "conflict": conflict, "grade": "B",  "note": "Extreme"}
    if supp:                   return {"extremes": "No",  "conflict": conflict, "grade": "A",  "note": "Momentum supportive"}
    return {"extremes": "No",  "conflict": conflict, "grade": "B",  "note": "Momentum not confirmed"}

def action_suggestions(bias: str, extremes: str) -> Dict[str, str]:
    if bias == "Longs allowed":
        if extremes == "Yes":
            return {
                "action_1": "Buy pullbacks into demand/pivot clusters; avoid chasing fresh breakouts",
                "action_2": "Require trigger (absorption/retest fail); reduce size"
            }
        return {
            "action_1": "Trade with trend; buy pullbacks or breakout‑retest holds",
            "action_2": "Normal size if spreads/news OK"
        }
    if bias == "Shorts allowed":
        if extremes == "Yes":
            return {
                "action_1": "Sell rallies into supply/pivot clusters; avoid chasing fresh breakdowns",
                "action_2": "Require trigger (rejection/retest fail); reduce size"
            }
        return {
            "action_1": "Trade with trend; short failed retests or breakdown‑retest holds",
            "action_2": "Normal size if spreads/news OK"
        }
    return {
        "action_1": "No directional filter; treat setups as tactical only",
        "action_2": "Demand A+ price confirmation (zone + rejection/volume)"
    }

def analyze_from_marketbulls_legacy(
    name: str,
    cot6: Optional[float], cot36: Optional[float],
    spec_long_pct_oi: Optional[float], spec_short_pct_oi: Optional[float],
    # Optional current/previous for WoW
    prev_cot6: Optional[float] = None,
    prev_spec_long_pct_oi: Optional[float] = None, prev_spec_short_pct_oi: Optional[float] = None,
    oi_current: Optional[float] = None, oi_prev: Optional[float] = None,
    # Optional: “Changes” in contracts (sanity)
    spec_long_change: Optional[float] = None, spec_short_change: Optional[float] = None,
) -> Dict[str, str]:

    net_now = net_pct_of_oi(spec_long_pct_oi, spec_short_pct_oi)
    net_prev = net_pct_of_oi(prev_spec_long_pct_oi, prev_spec_short_pct_oi)
    net_wow = wow_pp(net_now, net_prev)

    d_cot6_pp = wow_pp(cot6, prev_cot6)
    oi_wow = pct_change(oi_current, oi_prev)

    # Optional: net Δ contracts
    net_change_contracts = None
    if spec_long_change is not None or spec_short_change is not None:
        try:
            net_change_contracts = float(spec_long_change or 0) - float(spec_short_change or 0)
        except Exception:
            net_change_contracts = None

    bias = bias_from_indices(cot6, cot36)
    g = flags_and_grade(bias, cot6, cot36, d_cot6_pp, oi_wow, net_wow, net_now)
    act = action_suggestions(bias, g["extremes"])

    result = {
        "Instrument": name,
        "COT 6M": cot6, "COT 6M WoW (pp)": d_cot6_pp, "COT 36M": cot36,
        "Specs Long %OI": spec_long_pct_oi, "Specs Short %OI": spec_short_pct_oi,
        "Specs Net %OI": net_now, "Specs Net %OI WoW (pp)": net_wow,
        "OI WoW (%)": oi_wow,
        "Bias": bias, "Grade": g["grade"],
        "Flags": f"Extremes={g['extremes']}; Conflict={g['conflict']}; {g['note']}",
        "Action 1": act["action_1"], "Action 2": act["action_2"],
    }
    if net_change_contracts is not None:
        result["Specs Net Δ (contracts)"] = net_change_contracts
    return result

def render_card(row: Dict[str, Optional[float]]):
    with st.container(border=True):
        st.subheader(row["Instrument"])
        chips = [
            f'<span class="badge">Bias: {row["Bias"]}</span>',
            f'<span class="badge">Grade: {row["Grade"]}</span>',
        ]
        st.markdown(" ".join(chips), unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("COT 6M", fmt(row["COT 6M"]), delta=fmt(row["COT 6M WoW (pp)"], suffix=" pp"))
        c2.metric("COT 36M", fmt(row["COT 36M"]))
        c3.metric("Specs Net %OI", fmt(row["Specs Net %OI"]), delta=fmt(row["Specs Net %OI WoW (pp)"], suffix=" pp"))
        c4.metric("OI WoW", fmt(row["OI WoW (%)"], suffix="%"))

        st.caption(row["Flags"])
        st.write("• " + row["Action 1"])
        st.write("• " + row["Action 2"])

        if "Specs Net Δ (contracts)" in row:
            st.caption(f"Specs Net Δ (contracts): {fmt(row['Specs Net Δ (contracts)'], nd=0)}")

# ---------- Input forms ----------
def instrument_form(default_name: str, key: str):
    st.markdown(f"#### {default_name}")
    with st.container(border=True):
        st.write("Current week")
        c1, c2, c3, c4, c5 = st.columns([1,1,1,1,1])
        cot6 = parse_flt(c1.text_input("COT Index 6M", value="", key=f"{key}_cot6"))
        cot36 = parse_flt(c2.text_input("COT Index 36M", value="", key=f"{key}_cot36"))
        long_pct = parse_flt(c3.text_input("Specs Long %OI", value="", key=f"{key}_long_pct"))
        short_pct = parse_flt(c4.text_input("Specs Short %OI", value="", key=f"{key}_short_pct"))
        oi_curr = parse_flt(c5.text_input("Open Interest (contracts) [optional]", value="", key=f"{key}_oi_curr"))

        st.write("Previous week (for WoW)")
        p1, p2, p3, p4 = st.columns(4)
        prev_cot6 = parse_flt(p1.text_input("Prev COT 6M", value="", key=f"{key}_prev_cot6"))
        prev_long_pct = parse_flt(p2.text_input("Prev Specs Long %OI", value="", key=f"{key}_prev_long"))
        prev_short_pct = parse_flt(p3.text_input("Prev Specs Short %OI", value="", key=f"{key}_prev_short"))
        oi_prev = parse_flt(p4.text_input("Prev Open Interest [optional]", value="", key=f"{key}_oi_prev"))

        st.write("Optional sanity (from 'Changes' row, contracts)")
        s1, s2 = st.columns(2)
        long_chg = parse_flt(s1.text_input("Specs Long Δ (contracts)", value="", key=f"{key}_long_chg"))
        short_chg = parse_flt(s2.text_input("Specs Short Δ (contracts)", value="", key=f"{key}_short_chg"))

        return {
            "name": default_name, "cot6": cot6, "cot36": cot36,
            "long_pct": long_pct, "short_pct": short_pct,
            "prev_cot6": prev_cot6, "prev_long_pct": prev_long_pct, "prev_short_pct": prev_short_pct,
            "oi_current": oi_curr, "oi_prev": oi_prev,
            "long_chg": long_chg, "short_chg": short_chg,
        }

# ---------- Tabs for instruments ----------
tabs = st.tabs(["XAUUSD", "NAS100", "US30", "Custom"])

with tabs[0]:
    xau = instrument_form("XAUUSD", key="xau")
    # Example helper (optional): fill example values by clicking the button
    if st.button("Fill XAU example from screenshot", key="xau_fill"):
        st.session_state["xau_cot6"] = "0"
        st.session_state["xau_cot36"] = "13.5"
        st.session_state["xau_long_pct"] = "63.3"
        st.session_state["xau_short_pct"] = "11.7"

with tabs[1]:
    nas = instrument_form("NAS100", key="nas")

with tabs[2]:
    dow = instrument_form("US30", key="dow")

with tabs[3]:
    cname = st.text_input("Custom instrument name", value="CUSTOM", key="custom_name")
    custom = instrument_form(cname, key="custom")

# ---------- Analyze & render ----------
def analyze_if_ready(d):
    if d["cot6"] is not None and d["cot36"] is not None and d["long_pct"] is not None and d["short_pct"] is not None:
        return analyze_from_marketbulls_legacy(
            name=d["name"], cot6=d["cot6"], cot36=d["cot36"],
            spec_long_pct_oi=d["long_pct"], spec_short_pct_oi=d["short_pct"],
            prev_cot6=d["prev_cot6"], prev_spec_long_pct_oi=d["prev_long_pct"], prev_spec_short_pct_oi=d["prev_short_pct"],
            oi_current=d["oi_current"], oi_prev=d["oi_prev"],
            spec_long_change=d["long_chg"], spec_short_change=d["short_chg"],
        )
    return None

st.divider()
st.subheader("Results")

results: List[Dict[str, Optional[float]]] = []
for d in [xau, nas, dow, custom]:
    r = analyze_if_ready(d)
    if r:
        render_card(r)
        results.append(r)
    else:
        with st.container(border=True):
            st.subheader(d["name"])
            st.info("Enter: COT 6M, COT 36M, Specs Long %OI, Specs Short %OI to analyze.", icon="ℹ️")

# ---------- Export ----------
if results:
    df = pd.DataFrame(results)
    st.download_button(
        "⬇️ Download CSV",
        data=df.to_csv(index=False),
        file_name="cot_spec_report.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ---------- Help ----------
with st.expander("How to copy numbers from MarketBulls"):
    st.markdown("""
- Go to Legacy → LARGE SPECULATORS:
  - Copy Percent of OI → Long and Short (e.g., 63.3 and 11.7)
- Read gauges on the right:
  - COT Index 6 Month and COT Index 36 Month
- Optional for WoW:
  - Navigate to previous week and copy the same Long% OI, Short% OI, and COT 6M
  - Copy Open Interest (contracts) for both weeks if you want OI WoW %
- Optional sanity:
  - From the 'Changes' row, you can paste Specs Long Δ and Specs Short Δ (contracts) — the app shows Net Δ.
""")
