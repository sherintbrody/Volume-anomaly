# cot_mb_analyzer.py
from typing import Optional, Dict

# ---------- Core math helpers ----------
def net_pct_of_oi(long_pct: Optional[float], short_pct: Optional[float],
                  net_pct: Optional[float] = None) -> Optional[float]:
    """Specs Net % of OI = Long% - Short%. If net_pct provided, return as is."""
    if net_pct is not None:
        return float(net_pct)
    if long_pct is None or short_pct is None:
        return None
    return float(long_pct) - float(short_pct)

def wow_pp(current: Optional[float], previous: Optional[float]) -> Optional[float]:
    """Week-over-week change in percentage points (indexes or % values)."""
    if current is None or previous is None:
        return None
    return float(current) - float(previous)

def pct_change(current: Optional[float], previous: Optional[float]) -> Optional[float]:
    """Percentage change (for Open Interest)."""
    if current is None or previous is None or previous == 0:
        return None
    return (float(current) - float(previous)) / float(previous) * 100.0

def fmt(x: Optional[float], nd: int = 1, suffix: str = "") -> str:
    return "n/a" if x is None else f"{x:.{nd}f}{suffix}"

# ---------- Bias + grading ----------
def bias_from_indices(cot6: Optional[float], cot36: Optional[float]) -> str:
    """
    Specs-only, relative bias (simple, robust):
      - Longs allowed if 6M > 60 and 36M > 50
      - Shorts allowed if 6M < 40 and 36M < 50
      - Else Neutral
    """
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
    """
    Extremes: 6M or 36M > 85 or < 15  (trend maturity / don't chase)
    Momentum:
      - Longs: ΔCOT6 ≥ +5 pp OR OI WoW ≥ 0% OR Net%OI WoW ≥ 0 pp
      - Shorts: ΔCOT6 ≤ −5 pp OR OI WoW ≤ 0% OR Net%OI WoW ≤ 0 pp
    Grade:
      - A: bias allowed + momentum supportive + not extreme
      - B: bias allowed but either extreme or weak momentum
      - C: Neutral
    Extra flag: conflict if bias says "Shorts" while net_now > 0 (or vice versa).
    """
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

# ---------- Main analyzer tailored to MarketBulls Legacy (Large Specs) ----------
def analyze_from_marketbulls_legacy(
    name: str,
    # current week (from the MarketBulls dashboard)
    cot6: float,                      # COT Index 6 Month (gauge)
    cot36: float,                     # COT Index 36 Month (gauge)
    spec_long_pct_oi: float,          # Percent of OI → LARGE SPECULATORS → Long
    spec_short_pct_oi: float,         # Percent of OI → LARGE SPECULATORS → Short
    # optional: current OI and "Changes" (Δ contracts) row for specs
    oi_current: Optional[float] = None,
    spec_long_change: Optional[float] = None,   # from "Changes" row (contracts)
    spec_short_change: Optional[float] = None,  # from "Changes" row (contracts)
    # previous week (if you can navigate previous week on the dashboard and read the same fields)
    prev_cot6: Optional[float] = None,
    prev_spec_long_pct_oi: Optional[float] = None,
    prev_spec_short_pct_oi: Optional[float] = None,
    oi_prev: Optional[float] = None
) -> Dict[str, str]:
    # Net % OI (current and previous) + WoW
    net_now = net_pct_of_oi(spec_long_pct_oi, spec_short_pct_oi)
    net_prev = net_pct_of_oi(prev_spec_long_pct_oi, prev_spec_short_pct_oi) if (prev_spec_long_pct_oi is not None and prev_spec_short_pct_oi is not None) else None
    net_wow = wow_pp(net_now, net_prev)

    # WoW for COT 6M and OI
    d_cot6_pp = wow_pp(cot6, prev_cot6)
    oi_wow = pct_change(oi_current, oi_prev)

    # Quick net-change sanity from "Changes" row (if supplied)
    net_change_contracts = None
    if spec_long_change is not None or spec_short_change is not None:
        try:
            net_change_contracts = float(spec_long_change or 0) - float(spec_short_change or 0)
        except Exception:
            net_change_contracts = None

    # Bias/grade/actions
    bias = bias_from_indices(cot6, cot36)
    g = flags_and_grade(bias, cot6, cot36, d_cot6_pp, oi_wow, net_wow, net_now)
    act = action_suggestions(bias, g["extremes"])

    # Build result
    result = {
        "instrument": name,
        "COT_6M": fmt(cot6),
        "COT_6M_WoW_pp": fmt(d_cot6_pp),
        "COT_36M": fmt(cot36),
        "Specs_Long_%OI": fmt(spec_long_pct_oi),
        "Specs_Short_%OI": fmt(spec_short_pct_oi),
        "Specs_Net_%OI": fmt(net_now),
        "Specs_Net_%OI_WoW_pp": fmt(net_wow),
        "OI_WoW_%": fmt(oi_wow, nd=1, suffix="%"),
        "Bias": bias,
        "Grade": g["grade"],
        "Flags": f"Extremes={g['extremes']}; Conflict={g['conflict']}; {g['note']}",
        "Action_1": act["action_1"],
        "Action_2": act["action_2"],
    }
    if net_change_contracts is not None:
        result["Specs_Net_ΔContracts"] = fmt(net_change_contracts, nd=0)
    return result

def format_report(row: Dict[str, str]) -> str:
    lines = [
        f"{row['instrument']}",
        f"- COT 6M: {row['COT_6M']} (WoW {row['COT_6M_WoW_pp']} pp) | COT 36M: {row['COT_36M']}",
        f"- Specs %OI: Long {row['Specs_Long_%OI']}, Short {row['Specs_Short_%OI']} → Net {row['Specs_Net_%OI']} (WoW {row['Specs_Net_%OI_WoW_pp']} pp) | OI WoW: {row['OI_WoW_%']}",
        f"- Bias: {row['Bias']} | Grade: {row['Grade']} | {row['Flags']}",
        f"- {row['Action_1']}",
        f"- {row['Action_2']}",
    ]
    if "Specs_Net_ΔContracts" in row:
        lines.insert(3, f"- Specs Net Δ (contracts): {row['Specs_Net_ΔContracts']}")
    return "\n".join(lines)

# ---------- Example with your GOLD screenshot ----------
if __name__ == "__main__":
    # From your image (Legacy • Large Speculators):
    # Percent of OI: Long 63.3%, Short 11.7% ⇒ Net ≈ +51.6% (computed by script)
    # COT Index gauges: 6M = 0.0, 36M = 13.5
    gold = analyze_from_marketbulls_legacy(
        name="XAUUSD",
        cot6=0.0,
        cot36=13.5,
        spec_long_pct_oi=63.3,
        spec_short_pct_oi=11.7,
        # Optional extras if you have them for better WoW and sanity checks:
        # oi_current=520000, oi_prev=510000,
        # spec_long_change=+1903, spec_short_change=-2767,
        # prev_cot6=5.0, prev_spec_long_pct_oi=64.0, prev_spec_short_pct_oi=10.5,
    )
    print(format_report(gold))
