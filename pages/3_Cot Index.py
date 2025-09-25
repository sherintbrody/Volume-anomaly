import streamlit as st
from typing import Optional, Dict, List
import pandas as pd

# ---------- Page + Style ----------
st.set_page_config(page_title="COT Spec Index Analyzer + Signal Score", page_icon="📊", layout="wide")

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

st.title("COT Spec Index Analyzer + Signal Score (Manual)")
st.caption("Inputs from MarketBulls (Legacy → Large Speculators %OI + COT 6M/36M). Manual confirmations for zones, pivots, fibs, volume, Bollinger, regime, and risk.")

# ---------- Tip (pp vs %) ----------
with st.container(border=True):
    st.info(
        "Tip: pp = percentage points (absolute difference), not percent change.\n"
        "• COT 6M 62% → 67% = +5 pp (not +8.1%)\n"
        "• Specs Net %OI 51.6% → 49.9% = −1.7 pp\n"
        "• Open Interest WoW is a percent change (e.g., +2.0%), not pp",
        icon="💡"
    )

# ---------- Helpers ----------
def parse_flt(s: str) -> Optional[float]:
    if s is None: return None
    s = str(s).strip().replace("%", "")
    if s == "": return None
    try: return float(s)
    except Exception: return None

def fmt(x: Optional[float], nd: int = 1, suffix: str = "") -> str:
    return "n/a" if x is None else f"{x:.{nd}f}{suffix}"

def net_pct_of_oi(long_pct: Optional[float], short_pct: Optional[float],
                  net_pct: Optional[float] = None) -> Optional[float]:
    if net_pct is not None:
        return float(net_pct)
    if long_pct is None or short_pct is None:
        return None
    return float(long_pct) - float(short_pct)

def wow_pp(current: Optional[float], previous: Optional[float]) -> Optional[float]:
    if current is None or previous is None: return None
    return float(current) - float(previous)

def pct_change(current: Optional[float], previous: Optional[float]) -> Optional[float]:
    if current is None or previous is None or previous == 0: return None
    return (float(current) - float(previous)) / float(previous) * 100.0

def bias_from_indices(cot6: Optional[float], cot36: Optional[float]) -> str:
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
    extremes = False
    if cot6 is not None and cot36 is not None:
        extremes = (cot6 > 85 or cot36 > 85 or cot6 < 15 or cot36 < 15)

    conflict = "No"
    if net_now is not None and bias != "Neutral":
        if (bias == "Shorts allowed" and net_now > 0) or (bias == "Longs allowed" and net_now < 0):
            conflict = "Yes"

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

    supp = mom_ok_short()
    if extremes and not supp: return {"extremes": "Yes", "conflict": conflict, "grade": "B-", "note": "Extreme + weak momentum"}
    if extremes:               return {"extremes": "Yes", "conflict": conflict, "grade": "B",  "note": "Extreme"}
    if supp:                   return {"extremes": "No",  "conflict": conflict, "grade": "A",  "note": "Momentum supportive"}
    return {"extremes": "No",  "conflict": conflict, "grade": "B",  "note": "Momentum not confirmed"}

# ---------- Weights (adjustable) ----------
st.sidebar.header("Weights & Penalties")
w_bias   = st.sidebar.slider("Weight: COT Bias (on/off)", 0.0, 6.0, 4.0, 0.25)
w_zone   = st.sidebar.slider("Weight: In HTF zone (D/H4)", 0.0, 5.0, 3.0, 0.25)
w_fresh  = st.sidebar.slider("Bonus: Zone fresh (≤2 retests)", 0.0, 1.5, 0.5, 0.25)
w_conf   = st.sidebar.slider("Weight: Pivot/Fib confluence (max)", 0.0, 2.0, 1.5, 0.25)
w_vol15  = st.sidebar.slider("Weight: 15m Volume spike (P95)", 0.0, 3.0, 2.0, 0.25)
w_vol1h  = st.sidebar.slider("Bonus: 1h Volume support (P90)", 0.0, 1.5, 0.5, 0.25)
w_boll   = st.sidebar.slider("Weight: Bollinger confirmation", 0.0, 2.0, 1.0, 0.25)
w_trend  = st.sidebar.slider("Weight: Trend regime alignment", 0.0, 2.0, 1.0, 0.25)
w_volreg = st.sidebar.slider("Weight: Volatility state helpful", 0.0, 1.5, 0.5, 0.25)
w_sess   = st.sidebar.slider("Weight: Session context helpful", 0.0, 1.0, 0.25, 0.25)
p_ext    = st.sidebar.slider("Penalty: Extremes (mature trend)", 0.0, 1.5, 0.5, 0.25)
p_conf   = st.sidebar.slider("Penalty: Conflict (bias vs net%OI)", 0.0, 1.5, 0.5, 0.25)
p_news   = st.sidebar.slider("Penalty: High-impact news soon", 0.0, 2.0, 1.0, 0.25)
p_spread = st.sidebar.slider("Penalty: Wide spread/liquidity risk", 0.0, 2.0, 0.5, 0.25)

MAX_SCORE = 10.0

def compute_signal_score_and_breakdown(
    bias: str,
    extremes: str,
    conflict: str,
    in_htf_zone: bool,
    fresh_zone: bool,
    pivot_confluence: bool,
    fib_confluence: bool,
    vol15_spike: bool,
    vol1h_support: bool,
    bb_confirm: bool,
    trend_align: str,        # "Aligned", "Against", "Range/NA"
    vol_state: str,          # "Expansion", "Compression/Neutral"
    session_help: bool,      # True if London/NY open helps; else False
    news_risk: bool,
    spread_risk: bool
) -> Dict[str, float]:
    contrib = {}
    score = 0.0

    # Bias
    if bias in ("Longs allowed", "Shorts allowed"):
        contrib["COT bias"] = w_bias; score += w_bias
    else:
        contrib["COT bias"] = 0.0

    # Location
    if in_htf_zone:
        contrib["HTF zone"] = w_zone; score += w_zone
        if fresh_zone:
            contrib["Fresh zone"] = w_fresh; score += w_fresh
        else:
            contrib["Fresh zone"] = 0.0
    else:
        contrib["HTF zone"] = 0.0; contrib["Fresh zone"] = 0.0

    # Confluence
    conf_sum = 0.0
    if pivot_confluence: conf_sum += min(w_conf/2, w_conf)  # split conf weight across pivot+fib
    if fib_confluence:   conf_sum += min(w_conf/2, w_conf)
    conf_sum = min(conf_sum, w_conf)
    contrib["Pivot/Fib confluence"] = conf_sum; score += conf_sum

    # Volume
    if vol15_spike:
        contrib["15m volume spike"] = w_vol15; score += w_vol15
        if vol1h_support:
            contrib["1h volume bonus"] = w_vol1h; score += w_vol1h
        else:
            contrib["1h volume bonus"] = 0.0
    else:
        contrib["15m volume spike"] = 0.0; contrib["1h volume bonus"] = 0.0

    # Bollinger
    if bb_confirm:
        contrib["Bollinger confirm"] = w_boll; score += w_boll
    else:
        contrib["Bollinger confirm"] = 0.0

    # Regime
    if trend_align == "Aligned":
        contrib["Trend regime"] = w_trend; score += w_trend
    elif trend_align == "Against":
        contrib["Trend regime"] = -w_trend; score -= w_trend
    else:
        contrib["Trend regime"] = 0.0

    if vol_state == "Expansion":
        contrib["Volatility helpful"] = w_volreg; score += w_volreg
    else:
        contrib["Volatility helpful"] = 0.0

    if session_help:
        contrib["Session helpful"] = w_sess; score += w_sess
    else:
        contrib["Session helpful"] = 0.0

    # Penalties
    if extremes == "Yes":
        contrib["Extreme penalty"] = -p_ext; score -= p_ext
    else:
        contrib["Extreme penalty"] = 0.0

    if conflict == "Yes":
        contrib["Conflict penalty"] = -p_conf; score -= p_conf
    else:
        contrib["Conflict penalty"] = 0.0

    if news_risk:
        contrib["News penalty"] = -p_news; score -= p_news
    else:
        contrib["News penalty"] = 0.0

    if spread_risk:
        contrib["Spread penalty"] = -p_spread; score -= p_spread
    else:
        contrib["Spread penalty"] = 0.0

    # Clamp and tier
    score = max(0.0, min(MAX_SCORE, score))
    tier = "A" if score >= 8.5 else ("B" if score >= 6.5 else "C")
    contrib["TOTAL"] = score
    return {"score": score, "tier": tier, "contrib": contrib}

# ---------- Analyzer ----------
def analyze_from_marketbulls_legacy(
    name: str,
    cot6: Optional[float], cot36: Optional[float],
    spec_long_pct_oi: Optional[float], spec_short_pct_oi: Optional[float],
    prev_cot6: Optional[float] = None,
    prev_spec_long_pct_oi: Optional[float] = None, prev_spec_short_pct_oi: Optional[float] = None,
    oi_current: Optional[float] = None, oi_prev: Optional[float] = None,
) -> Dict[str, Optional[float]]:
    net_now = net_pct_of_oi(spec_long_pct_oi, spec_short_pct_oi)
    net_prev = net_pct_of_oi(prev_spec_long_pct_oi, prev_spec_short_pct_oi)
    net_wow = wow_pp(net_now, net_prev)
    d_cot6_pp = wow_pp(cot6, prev_cot6)
    oi_wow = pct_change(oi_current, oi_prev)

    bias = bias_from_indices(cot6, cot36)
    g = flags_and_grade(bias, cot6, cot36, d_cot6_pp, oi_wow, net_wow, net_now)

    return {
        "Instrument": name,
        "COT 6M": cot6, "COT 6M WoW (pp)": d_cot6_pp, "COT 36M": cot36,
        "Specs Long %OI": spec_long_pct_oi, "Specs Short %OI": spec_short_pct_oi,
        "Specs Net %OI": net_now, "Specs Net %OI WoW (pp)": net_wow,
        "OI WoW (%)": oi_wow,
        "Bias": bias, "Grade": g["grade"],
        "Extremes": g["extremes"], "Conflict": g["conflict"], "Note": g["note"],
    }

# ---------- Input forms ----------
def instrument_form(default_name: str, key: str):
    st.markdown(f"#### {default_name}")
    with st.container(border=True):
        st.write("Current week (MarketBulls → Legacy → Large Speculators)")
        c1, c2, c3, c4, c5 = st.columns([1,1,1,1,1])
        cot6 = parse_flt(c1.text_input("COT Index 6M", value="", key=f"{key}_cot6"))
        cot36 = parse_flt(c2.text_input("COT Index 36M", value="", key=f"{key}_cot36"))
        long_pct = parse_flt(c3.text_input("Specs Long %OI", value="", key=f"{key}_long_pct"))
        short_pct = parse_flt(c4.text_input("Specs Short %OI", value="", key=f"{key}_short_pct"))
        oi_curr = parse_flt(c5.text_input("Open Interest (contracts) [optional]", value="", key=f"{key}_oi_curr"))

        show_wow = st.checkbox("Show previous week (WoW) fields", value=False, key=f"{key}_show_wow")
        if show_wow:
            st.write("Previous week (for WoW)")
            p1, p2, p3, p4 = st.columns(4)
            prev_cot6 = parse_flt(p1.text_input("Prev COT 6M", value="", key=f"{key}_prev_cot6"))
            prev_long_pct = parse_flt(p2.text_input("Prev Specs Long %OI", value="", key=f"{key}_prev_long"))
            prev_short_pct = parse_flt(p3.text_input("Prev Specs Short %OI", value="", key=f"{key}_prev_short"))
            oi_prev = parse_flt(p4.text_input("Prev Open Interest [optional]", value="", key=f"{key}_oi_prev"))
        else:
            prev_cot6 = prev_long_pct = prev_short_pct = oi_prev = None

        st.write("Checklist (manual confirmations; no auto-import)")
        q1, q2, q3 = st.columns(3)
        in_zone  = q1.checkbox("In HTF zone (Daily/H4)", value=False, key=f"{key}_in_zone",
                               help="Price is inside a marked Daily/H4 Supply or Demand zone.")
        fresh    = q1.checkbox("Zone fresh (≤2 retests)", value=False, key=f"{key}_fresh",
                               help="Zone hasn’t been hit more than twice since it formed.")
        pivot_cf = q2.checkbox("Pivot confluence", value=False, key=f"{key}_pivot",
                               help="Zone overlaps DP/R1/S1 (within ~0.15–0.20 ATR).")
        fib_cf   = q2.checkbox("Fib confluence", value=False, key=f"{key}_fib",
                               help="At 38.2–61.8% pullback or 127–161.8% extension from last impulse.")
        vol15    = q3.checkbox("15m volume spike ≥ P95", value=False, key=f"{key}_vol15",
                               help="Current 15m volume is in top 5% for that bucket over ~21 days (manual).")
        vol1h    = q3.checkbox("1h volume supportive (P90)", value=False, key=f"{key}_vol1h",
                               help="Sum of last 4×15m bars in top 10% for that hour-of-day (manual).")
        bb_conf  = q3.checkbox("Bollinger confirmation", value=False, key=f"{key}_bb",
                               help="Re-entry (wick back inside band) or breakout-retest hold with 20/2σ bands.")

        st.write("Regime & Risk (manual)")
        r1, r2, r3, r4, r5 = st.columns(5)
        trend_align = r1.selectbox("Trend regime vs intended bias",
                                   ["Range/NA", "Aligned", "Against"], index=0,
                                   help="Use HTF SMAs/structure: is your intended direction aligned?")
        vol_state = r2.selectbox("Volatility state", ["Compression/Neutral", "Expansion"], index=0,
                                 help="ATR rising or Bollinger Bandwidth expanding?")
        session_help = r3.checkbox("Session helpful (London/NY)", value=False,
                                   help="Session context likely to amplify your setup.")
        news_risk = r4.checkbox("High-impact news soon (60–90m)", value=False,
                                 help="If checked, a penalty is applied.")
        spread_risk = r5.checkbox("Wide spread/liquidity risk", value=False,
                                  help="Illiquid conditions penalty.")

        notes = st.text_area("Notes (optional)", value="", key=f"{key}_notes")

        return {
            "name": default_name, "cot6": cot6, "cot36": cot36,
            "long_pct": long_pct, "short_pct": short_pct,
            "prev_cot6": prev_cot6, "prev_long_pct": prev_long_pct, "prev_short_pct": prev_short_pct,
            "oi_current": oi_curr, "oi_prev": oi_prev,
            "in_zone": in_zone, "fresh": fresh,
            "pivot_cf": pivot_cf, "fib_cf": fib_cf,
            "vol15": vol15, "vol1h": vol1h, "bb_conf": bb_conf,
            "trend_align": trend_align, "vol_state": vol_state, "session_help": session_help,
            "news_risk": news_risk, "spread_risk": spread_risk,
            "notes": notes,
        }

# ---------- Tabs ----------
tabs = st.tabs(["XAUUSD", "NAS100", "US30", "Custom"])

with tabs[0]:
    xau = instrument_form("XAUUSD", key="xau")
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
        )
    return None

def action_lines_from_score(direction: str, tier: str, extremes: str) -> List[str]:
    if direction == "Long":
        if extremes == "Yes":
            return ["Buy pullbacks into demand/pivot clusters; avoid chasing fresh breakouts",
                    "Require trigger (absorption/retest fail); reduce size"]
        return ["Trade with trend; buy pullbacks or breakout‑retest holds",
                "Normal size if spreads/news OK" if tier in ("A","B") else "Tactical only; smaller size"]
    if direction == "Short":
        if extremes == "Yes":
            return ["Sell rallies into supply/pivot clusters; avoid chasing fresh breakdowns",
                    "Require trigger (rejection/retest fail); reduce size"]
        return ["Trade with trend; short failed retests or breakdown‑retest holds",
                "Normal size if spreads/news OK" if tier in ("A","B") else "Tactical only; smaller size"]
    return ["No directional filter; treat setups as tactical only",
            "Demand A+ price confirmation (zone + rejection/volume)"]

def render_card(row: Dict[str, Optional[float]], d: Dict[str, any]):
    direction = "Long" if row["Bias"] == "Longs allowed" else ("Short" if row["Bias"] == "Shorts allowed" else "Neutral")

    sb = compute_signal_score_and_breakdown(
        bias=row["Bias"], extremes=row["Extremes"], conflict=row["Conflict"],
        in_htf_zone=d["in_zone"], fresh_zone=d["fresh"],
        pivot_confluence=d["pivot_cf"], fib_confluence=d["fib_cf"],
        vol15_spike=d["vol15"], vol1h_support=d["vol1h"], bb_confirm=d["bb_conf"],
        trend_align=d["trend_align"], vol_state=d["vol_state"], session_help=d["session_help"],
        news_risk=d["news_risk"], spread_risk=d["spread_risk"]
    )

    acts = action_lines_from_score(direction, sb["tier"], row["Extremes"])

    with st.container(border=True):
        st.subheader(row["Instrument"])
        chips = [
            f'<span class="badge">Bias: {row["Bias"]}</span>',
            f'<span class="badge">Grade: {row["Grade"]}</span>',
            f'<span class="badge">Signal Score: {sb["score"]:.1f} ({sb["tier"]})</span>',
        ]
        st.markdown(" ".join(chips), unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("COT 6M", fmt(row["COT 6M"]), delta=fmt(row["COT 6M WoW (pp)"], suffix=" pp"))
        m2.metric("COT 36M", fmt(row["COT 36M"]))
        m3.metric("Specs Net %OI", fmt(row["Specs Net %OI"]), delta=fmt(row["Specs Net %OI WoW (pp)"], suffix=" pp"))
        m4.metric("OI WoW", fmt(row["OI WoW (%)"], suffix="%"))

        st.caption(f"Flags: Extremes={row['Extremes']}; Conflict={row['Conflict']}; {row['Note']}")

        st.write("• " + acts[0])
        st.write("• " + acts[1])

        with st.expander("Score breakdown"):
            # Factor contributions table
            contrib_items = [{"Factor": k, "Points": v} for k, v in sb["contrib"].items()]
            dfc = pd.DataFrame(contrib_items)
            st.dataframe(dfc, hide_index=True, width="stretch")

        with st.expander("Checklist state & Notes"):
            cc_df = pd.DataFrame([{
                "In HTF zone": d["in_zone"],
                "Fresh zone (≤2)": d["fresh"],
                "Pivot confluence": d["pivot_cf"],
                "Fib confluence": d["fib_cf"],
                "15m vol spike (P95)": d["vol15"],
                "1h vol supportive (P90)": d["vol1h"],
                "Bollinger confirmation": d["bb_conf"],
                "Trend regime": d["trend_align"],
                "Volatility state": d["vol_state"],
                "Session helpful": d["session_help"],
                "News risk": d["news_risk"],
                "Spread risk": d["spread_risk"],
            }])
            st.dataframe(cc_df, hide_index=True, width="stretch")
            if d["notes"]:
                st.caption(f"Notes: {d['notes']}")

    # Append extras for export
    row.update({
        "Direction": direction,
        "Signal Score": sb["score"],
        "Signal Tier": sb["tier"],
        "In HTF zone": d["in_zone"],
        "Fresh zone": d["fresh"],
        "Pivot confluence": d["pivot_cf"],
        "Fib confluence": d["fib_cf"],
        "Vol15 spike": d["vol15"],
        "Vol1h support": d["vol1h"],
        "Bollinger confirm": d["bb_conf"],
        "Trend regime": d["trend_align"],
        "Volatility state": d["vol_state"],
        "Session helpful": d["session_help"],
        "News risk": d["news_risk"],
        "Spread risk": d["spread_risk"],
        "Notes": d["notes"],
    })
    return row

# ---------- Tabs ----------
tabs = st.tabs(["XAUUSD", "NAS100", "US30", "Custom"])

with tabs[0]:
    xau = instrument_form("XAUUSD", key="xau")
with tabs[1]:
    nas = instrument_form("NAS100", key="nas")
with tabs[2]:
    dow = instrument_form("US30", key="dow")
with tabs[3]:
    cname = st.text_input("Custom instrument name", value="CUSTOM", key="custom_name")
    custom = instrument_form(cname, key="custom")

# ---------- Run ----------
def analyze_if_ready(d):
    if d["cot6"] is not None and d["cot36"] is not None and d["long_pct"] is not None and d["short_pct"] is not None:
        return analyze_from_marketbulls_legacy(
            name=d["name"], cot6=d["cot6"], cot36=d["cot36"],
            spec_long_pct_oi=d["long_pct"], spec_short_pct_oi=d["short_pct"],
            prev_cot6=d["prev_cot6"], prev_spec_long_pct_oi=d["prev_long_pct"], prev_spec_short_pct_oi=d["prev_short_pct"],
            oi_current=d["oi_current"], oi_prev=d["oi_prev"],
        )
    return None

st.divider()
st.subheader("Results")

results: List[Dict[str, Optional[float]]] = []
for d in [xau, nas, dow, custom]:
    r = analyze_if_ready(d)
    if r:
        enriched = render_card(r, d)
        results.append(enriched)
    else:
        with st.container(border=True):
            st.subheader(d["name"])
            st.info("Enter: COT 6M, COT 36M, Specs Long %OI, Specs Short %OI to analyze.", icon="ℹ️")

# ---------- Export ----------
if results:
    df = pd.DataFrame(results)
    order_cols = [c for c in [
        "Instrument", "Direction", "Bias", "Grade", "Signal Score", "Signal Tier",
        "COT 6M", "COT 6M WoW (pp)", "COT 36M",
        "Specs Long %OI", "Specs Short %OI", "Specs Net %OI", "Specs Net %OI WoW (pp)",
        "OI WoW (%)",
        "Extremes", "Conflict", "Note",
        "In HTF zone", "Fresh zone", "Pivot confluence", "Fib confluence",
        "Vol15 spike", "Vol1h support", "Bollinger confirm",
        "Trend regime", "Volatility state", "Session helpful",
        "News risk", "Spread risk", "Notes"
    ] if c in df.columns]
    df = df[order_cols]
    st.download_button(
        "⬇️ Download CSV",
        data=df.to_csv(index=False),
        file_name="cot_signal_report.csv",
        mime="text/csv",
        width="stretch",
    )
