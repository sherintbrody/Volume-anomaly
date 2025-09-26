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

st.title("COT Spec Index Analyzer + Signal Score")
st.caption("Manual inputs (no auto‑import). Use MarketBulls → Legacy → Large Speculators → Percent of OI and the 6M/36M gauges.")

# ---------- Playbook toggle ----------
def render_playbook():
    st.subheader("Step‑down trade playbook")
    st.markdown("**1) Weekly bias (COT first, specs only)**")
    st.markdown("""
- Read COT Index 6M and 36M (relative position in last 26/156 weeks)
  - Longs allowed: 6M > 60 AND 36M > 50
  - Shorts allowed: 6M < 40 AND 36M < 50
  - Neutral: otherwise
- Extremes filter: if 6M or 36M > 85 (crowded long) or < 15 (crowded short) → trend mature; prefer pullbacks and reduce size
- Momentum upgrade (optional): WoW Δ COT 6M ≥ +5 (longs) or ≤ −5 (shorts), or Open Interest WoW in the bias direction
    """)
    st.markdown("**2) Location (where to do business)**")
    st.markdown("""
- Mark Daily and H4 Supply/Demand zones
  - Freshness: 1–2 retests max
  - Reaction quality: base → strong departure; for absorption, wick ≥ 40% of range and body ≤ 40%
- Add confluence:
  - Fibonacci: pullback longs at 38.2–61.8% of last impulse; shorts mirrored. Extensions 127/161.8% are target magnets
  - Pivots: Daily/Weekly Pivot, R1/R2/R3/S1/S2/S3
- Rule of thumb: Prefer trades where a D/H4 zone overlaps a pivot band (or a Fib cluster) within 0.15–0.20 ATR(14)
    """)
    st.markdown("**3) Intraday context (pivots tell you the day type)**")
    st.markdown("""
- Above DP and holding → trend‑day up bias; below DP → trend‑day down bias
- Rotation day: DP magnet, fade toward S1/R1 with confirmation
- Best confluence: HTF zone ± pivot band ± Fib level
    """)
    st.markdown("**4) Confirmation (only then pull the trigger)**")
    st.markdown("""
- Volume spike (session‑aware)
  - Compute per‑bucket percentile (e.g., 15m 95th percentile over last ~21 trading days). Spike if current volume ≥ P95 for that 15m slot
  - Higher quality: last four 15m bars sum ≥ hourly P90 (dual‑TF confirmation)
- Bollinger Bands (20 SMA, 2σ)
  - Reversal/absorption: breach outside band then close back inside, ideally at HTF zone/pivot
  - Breakout/continuation: full‑body close beyond band with rising Bandwidth (vol expansion)
    """)
    st.markdown("**5) Entry tactics (pick 1)**")
    st.markdown("""
- Reversal at zone (absorption)
  - Long: at demand zone + pivot/Fib, rejection candle (long lower wick), volume ≥ P95, close back inside lower band
  - Short: mirror at supply zone
- Breakout‑retest (continuation)
  - Close through level (zone edge/pivot), volume ≥ P95, fast retest holds (lower wick for longs, upper wick for shorts), then go
    """)
    st.markdown("**6) Risk, stop, targets**")
    st.markdown("""
- Initial stop: beyond the zone edge or 1.0× ATR(14) beyond trigger low/high (whichever is farther but logical)
- Position size: 0.5–1.0R at extremes; 1.0–1.25R when not extreme with momentum upgrade
- Targets:
  - T1 = nearest opposing pivot (DP/R1/S1) or prior swing → take 50% off
  - T2 = next pivot/HTF level or Fib 127/161.8% extension
  - If trend‑day (holding DP): let a runner trail by last swing lows (longs) or 1× ATR(14); otherwise trail by 20 SMA (Bollinger midline)
    """)
    st.markdown("**7) Management rules**")
    st.markdown("""
- Time‑stop: if no progress after 3–5 bars on your execution timeframe, reduce or exit
- Break‑even: after +1R, move stop to entry
- No chase: never enter when price is >1–1.5 ATR below/above the 20 SMA or outside bands—wait for a pullback or a retest
    """)
    st.markdown("**8) Two quick flows to copy**")
    st.markdown("""
- Bullish bias (COT 6M>60, 36M>50)
  1) Mark D/H4 demand; draw Fib of last daily upswing; add pivots  
  2) Wait for price into zone ± DP/S1; require either:  
     • Absorption: long lower wick + close back inside lower band + 15m vol ≥ P95  
     • Breakout‑retest: reclaim DP with high vol; retest DP holds (lower wick)  
  3) Long; stop under zone; T1 = DP/R1; T2 = R2 or Fib 127/161.8; trail by swing/ATR
- Bearish bias (COT 6M<40, 36M<50)
  1) Mark D/H4 supply; draw Fib of last daily downswing; add pivots  
  2) Price rallies into zone ± DP/R1; require:  
     • Absorption: upper wick + close back inside upper band + vol ≥ P95  
     • Breakout‑retest down: lose DP on vol; retest DP fails (upper wick)  
  3) Short; stop above zone; T1 = DP/S1; T2 = S2 or Fib 127/161.8; trail by swing/ATR
    """)
    st.markdown("**9) Pre‑trade audit (60 seconds)**")
    st.markdown("""
- News in next 60–90 min? If yes, smaller size or skip  
- Spread/latency OK? If not, skip  
- Session fit: London/NY opens for breakouts; off‑hours for fades
    """)
    st.markdown("**10) Common pitfalls (and fixes)**")
    st.markdown("""
- COT extreme and you chase: at 6M/36M >85 or <15, only sell rallies or buy dips—don’t chase breaks  
- Using average volume, not percentiles: switch to percentile threshold per 15m bucket to avoid session bias  
- Zone freshness ignored: if a zone has >2 retests, odds drop—demand stronger confirmation  
- Pivot mismatch: fading against trend when price is holding above/below DP—align with the day type
    """)

show_playbook = st.checkbox("Show trade playbook", value=False, key="show_playbook")
if show_playbook:
    render_playbook()

# ---------- Tip (pp vs %) ----------
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

# ---------- Signal Score ----------
def compute_signal_score(
    bias: str,
    extremes: str,
    conflict: str,
    in_htf_zone: bool,
    fresh_zone: bool,
    pivot_confluence: bool,
    fib_confluence: bool,
    vol15_spike: bool,
    vol1h_support: bool,
    bb_confirm: bool
) -> Dict[str, float]:
    score = 0.0
    if bias in ("Longs allowed", "Shorts allowed"):
        score += 4.0
    if in_htf_zone:
        score += 3.0
        if fresh_zone:
            score += 0.5
    if pivot_confluence: score += 0.75
    if fib_confluence:   score += 0.75
    if vol15_spike:
        score += 2.0
        if vol1h_support: score += 0.5
    if bb_confirm: score += 1.0
    if extremes == "Yes": score -= 0.5
    if conflict == "Yes": score -= 0.5
    score = max(0.0, min(10.0, score))
    tier = "A" if score >= 8.5 else ("B" if score >= 6.5 else "C")
    return {"score": score, "tier": tier}

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

# ---------- Analyzer tailored to manual MarketBulls inputs ----------
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

# ---------- Input forms (manual only) ----------
def instrument_form(default_name: str, key: str):
    st.markdown(f"#### {default_name}")
    with st.container():
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
            prev_long_pct = parse_flt(p2.text_input("Prev Specs Long %OI", value="", key=f"{key}_prev_long_pct"))
            prev_short_pct = parse_flt(p3.text_input("Prev Specs Short %OI", value="", key=f"{key}_prev_short_pct"))
            oi_prev = parse_flt(p4.text_input("Prev Open Interest [optional]", value="", key=f"{key}_oi_prev"))
        else:
            prev_cot6 = prev_long_pct = prev_short_pct = oi_prev = None

        st.write("Checklist (manual confirmations; no auto‑import)")
        q1, q2, q3 = st.columns(3)
        in_zone  = q1.checkbox("In HTF zone (Daily/H4)", value=False, key=f"{key}_in_zone", help="Price is inside a marked Daily/H4 Supply or Demand zone.")
        fresh    = q1.checkbox("Zone fresh (≤2 retests)", value=False, key=f"{key}_fresh", help="Zone hasn’t been hit more than twice since it formed.")
        pivot_cf = q2.checkbox("Pivot confluence", value=False, key=f"{key}_pivot_cf", help="Zone overlaps DP/R1/S1 (within ~0.15–0.20 ATR).")
        fib_cf   = q2.checkbox("Fib confluence", value=False, key=f"{key}_fib_cf", help="At 38.2–61.8% pullback or 127–161.8% extension from last impulse.")
        vol15    = q3.checkbox("15m volume spike ≥ P95", value=False, key=f"{key}_vol15", help="Current 15m volume is in top 5% for that bucket over ~21 days (manual).")
        vol1h    = q3.checkbox("1h volume supportive (P90)", value=False, key=f"{key}_vol1h", help="Sum of last 4×15m bars in top 10% for that hour‑of‑day (manual).")
        bb_conf  = q3.checkbox("Bollinger confirmation", value=False, key=f"{key}_bb_conf", help="Re‑entry (wick back inside band) or breakout‑retest hold with 20/2σ bands.")

        notes = st.text_area("Notes (optional)", value="", key=f"{key}_notes")

        return {
            "name": default_name, "cot6": cot6, "cot36": cot36,
            "long_pct": long_pct, "short_pct": short_pct,
            "prev_cot6": prev_cot6, "prev_long_pct": prev_long_pct, "prev_short_pct": prev_short_pct,
            "oi_current": oi_curr, "oi_prev": oi_prev,
            "in_zone": in_zone, "fresh": fresh,
            "pivot_cf": pivot_cf, "fib_cf": fib_cf,
            "vol15": vol15, "vol1h": vol1h, "bb_conf": bb_conf,
            "notes": notes,
        }

# ---------- Tabs (single creation) ----------
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

# ---------- Render card ----------
def render_card(row: Dict[str, Optional[float]], checklist: Dict[str, bool], notes: str):
    direction = "Long" if row["Bias"] == "Longs allowed" else ("Short" if row["Bias"] == "Shorts allowed" else "Neutral")
    sc = compute_signal_score(
        bias=row["Bias"],
        extremes=row["Extremes"],
        conflict=row["Conflict"],
        in_htf_zone=checklist["in_zone"],
        fresh_zone=checklist["fresh"],
        pivot_confluence=checklist["pivot_cf"],
        fib_confluence=checklist["fib_cf"],
        vol15_spike=checklist["vol15"],
        vol1h_support=checklist["vol1h"],
        bb_confirm=checklist["bb_conf"]
    )
    acts = action_lines_from_score(direction, sc["tier"], row["Extremes"])

    with st.container():
        st.subheader(row["Instrument"])
        chips = [
            f'<span class="badge">Bias: {row["Bias"]}</span>',
            f'<span class="badge">Grade: {row["Grade"]}</span>',
            f'<span class="badge">Signal Score: {sc["score"]:.1f} ({sc["tier"]})</span>',
        ]
        st.markdown(" ".join(chips), unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("COT 6M", fmt(row["COT 6M"]), delta=fmt(row.get("COT 6M WoW (pp)"), nd=1) + (" pp" if row.get("COT 6M WoW (pp)") is not None else ""))
        m2.metric("COT 36M", fmt(row["COT 36M"]))
        m3.metric("Specs Net %OI", fmt(row["Specs Net %OI"]), delta=fmt(row.get("Specs Net %OI WoW (pp)"), nd=1) + (" pp" if row.get("Specs Net %OI WoW (pp)") is not None else ""))
        m4.metric("OI WoW", fmt(row.get("OI WoW (%)"), nd=1) + ("% " if row.get("OI WoW (%)") is not None else ""))

        st.caption(f"Flags: Extremes={row['Extremes']}; Conflict={row['Conflict']}; {row['Note']}")
        st.write("• " + acts[0])
        st.write("• " + acts[1])

        with st.expander("Checklist state"):
            cc_df = pd.DataFrame([{
                "In HTF zone": checklist["in_zone"],
                "Fresh zone (≤2)": checklist["fresh"],
                "Pivot confluence": checklist["pivot_cf"],
                "Fib confluence": checklist["fib_cf"],
                "15m vol spike (P95)": checklist["vol15"],
                "1h vol supportive (P90)": checklist["vol1h"],
                "Bollinger confirmation": checklist["bb_conf"],
            }])
            st.dataframe(cc_df, hide_index=True, width="100%")
            if notes:
                st.caption(f"Notes: {notes}")

    row["Signal Score"] = sc["score"]
    row["Signal Tier"]  = sc["tier"]
    row["Direction"]    = direction
    row.update({
        "In HTF zone": checklist["in_zone"],
        "Fresh zone": checklist["fresh"],
        "Pivot confluence": checklist["pivot_cf"],
        "Fib confluence": checklist["fib_cf"],
        "Vol15 spike": checklist["vol15"],
        "Vol1h support": checklist["vol1h"],
        "Bollinger confirm": checklist["bb_conf"],
        "Notes": notes,
    })
    return row

# ---------- Analyze helper ----------
def analyze_if_ready(d):
    if d["cot6"] is not None and d["cot36"] is not None and d["long_pct"] is not None and d["short_pct"] is not None:
        return analyze_from_marketbulls_legacy(
            name=d["name"], cot6=d["cot6"], cot36=d["cot36"],
            spec_long_pct_oi=d["long_pct"], spec_short_pct_oi=d["short_pct"],
            prev_cot6=d["prev_cot6"], prev_spec_long_pct_oi=d["prev_long_pct"], prev_spec_short_pct_oi=d["prev_short_pct"],
            oi_current=d["oi_current"], oi_prev=d["oi_prev"],
        )
    return None

# ---------- Run ----------
st.divider()
st.subheader("Results")

results: List[Dict[str, Optional[float]]] = []
for d in [xau, nas, dow, custom]:
    r = analyze_if_ready(d)
    if r:
        enriched = render_card(
            r,
            {
                "in_zone": d["in_zone"], "fresh": d["fresh"],
                "pivot_cf": d["pivot_cf"], "fib_cf": d["fib_cf"],
                "vol15": d["vol15"], "vol1h": d["vol1h"], "bb_conf": d["bb_conf"],
            },
            d["notes"]
        )
        results.append(enriched)
    else:
        with st.container():
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
        "Notes"
    ] if c in df.columns]
    df = df[order_cols]
    st.download_button(
        "⬇️ Download CSV",
        data=df.to_csv(index=False),
        file_name="cot_signal_report.csv",
        mime="text/csv",
        key="download_csv"
    )
