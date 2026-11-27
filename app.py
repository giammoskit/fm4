"""Streamlit app for the Trend Filtered Multi Asset Portfolio.

Run locally with
    pip install -r requirements.txt
    streamlit run app.py

All calculations come from strategy.py, the same code as the Colab notebook.
This file only builds the interface.
"""
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import strategy as s

st.set_page_config(page_title="Trend Filtered Portfolio", page_icon=":material/trending_up:", layout="wide",
                   initial_sidebar_state="collapsed")

# ---------------------------------------------------------------- design tokens (dark)
BG, CARD, RAISED, RAISED_2 = "#0a0a0a", "#141414", "#212121", "#2c2c2c"
BORDER, GRID, HERO_GRID = "#232323", "#242424", "#1a1a1a"
INK, INK_2 = "#f4f4f5", "#8e8e93"
GOOD, BAD = "#22c55e", "#f04444"
ACCENT = "#338ced"
COLORS = {s.STRATEGY: ACCENT, s.BENCH_6040: "#e76624", s.BENCH_EW: "#00a888", s.BENCH_SPY: "#d28e05"}
ASSET_COLORS = {"SPY": "#338ced", "EFA": "#e76624", "IEF": "#00a888", "GLD": "#d28e05", "CASH": "#e24d8b"}
ORDER = [s.STRATEGY, s.BENCH_6040, s.BENCH_EW, s.BENCH_SPY]
FONT = "Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif"
ASSET_INFO = {
    "SPY": ("US equities", "S&P 500 ETF"),
    "EFA": ("Developed markets", "MSCI EAFE ETF, outside the US"),
    "IEF": ("US Treasuries", "7 to 10 year Treasury ETF"),
    "GLD": ("Gold", "Physical gold ETF"),
}
SHORT_STRESS = {
    "Global financial crisis (Nov 2007 to Feb 2009)": "Financial crisis 2007 to 2009",
    "Rebound after the crisis (Mar 2009 to Dec 2009)": "Rebound 2009",
    "Euro area debt crisis (May 2011 to Sep 2011)": "Euro debt crisis 2011",
    "Equity sell off (Oct 2018 to Dec 2018)": "Sell off end of 2018",
    "Covid crash (Feb 2020 to Mar 2020)": "Covid crash 2020",
    "Covid rebound (Apr 2020 to Dec 2020)": "Covid rebound 2020",
    "Inflation and rate shock (Jan 2022 to Dec 2022)": "Rate shock 2022",
}
FIRST_YEAR = pd.Timestamp(s.BACKTEST_START).year
LAST_YEAR = pd.Timestamp(s.BACKTEST_END).year
RANGES = ["1Y", "3Y", "5Y", "10Y", "Max", "Custom"]
CHART_TYPES = ["Line", "Candles"]
CHART_LABELS = {"Line": ":material/show_chart:", "Candles": ":material/candlestick_chart:"}
SPIKE = "rgba(244,244,245,0.35)"  # thin, translucent hover line
VIEWS = ["Growth", "Drawdowns", "Holdings history", "Years", "Stress tests", "Risk table", "Robustness", "Method"]
BENCHMARKS = [s.BENCH_6040, s.BENCH_EW, s.BENCH_SPY]
SNAPSHOT = Path(__file__).parent / "data" / "prices_snapshot.csv"
REPO_URL = "https://github.com/giammoskit/fm4"

DEFAULTS = {"window": s.SMA_MONTHS, "cost": int(s.COST_BPS), "use_filter": True, "benchmark": s.PRIMARY_BENCHMARK,
            "range": "Max", "years": (FIRST_YEAR, LAST_YEAR), "amount": 10_000, "view": "Growth",
            "chart_type": "Line", "show_benchmark": True}
# Settings edited inside the quick action popovers. A closed popover is not drawn in the browser, so its controls
# would miss a reset and send their old value back on the next rerun. These settings therefore live under their own
# key, the controls use a key with a generation number, and a reset moves to a new generation of controls.
POPOVER_SETTINGS = ["window", "cost", "use_filter", "benchmark", "amount"]
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)
st.session_state.setdefault("generation", 0)


def control_key(name):
    return f"{name}_control_{st.session_state.generation}"


def sync_setting(name):
    """Copy a popover control to its setting. An emptied chip choice falls back to the current setting."""
    picked = st.session_state[control_key(name)]
    if picked is None:
        st.session_state[control_key(name)] = st.session_state[name]
    else:
        st.session_state[name] = picked


def reset_settings():
    for key, value in DEFAULTS.items():
        st.session_state[key] = value
        st.session_state[f"_last_{key}"] = value
    st.session_state.pop("holding_month", None)
    st.session_state.generation += 1


def keep_selected(key):
    """Clicking the active chip would clear the choice. Restore the previous choice instead."""
    if st.session_state.get(key) is None:
        st.session_state[key] = st.session_state.get(f"_last_{key}", DEFAULTS[key])
    else:
        st.session_state[f"_last_{key}"] = st.session_state[key]


# ---------------------------------------------------------------- styling
# Layout ideas. A borderless hero with one large number and a clean chart, period chips under the chart
# (Trade Republic). Round quick action buttons that open each setting (Revolut). Section titles above
# rounded widget cards, a key figures grid and an allocation donut with the holdings list (getquin).
ACT = '.st-key-actions [data-testid="stPopoverButton"]'
ICON_BOX = 'span:has(> [data-testid="stIconMaterial"])'
st.markdown(f"""<style>
.stApp {{ background:{BG}; }}
[data-testid="stHeader"] {{ background:transparent; }}
[data-testid="stMain"] {{ background:radial-gradient(ellipse 900px 300px at 22% 0%, rgba(51,140,237,0.15) 0%,
  rgba(51,140,237,0.05) 45%, rgba(51,140,237,0) 72%) no-repeat; background-attachment:local; }}
[data-testid="stMainBlockContainer"], .block-container {{ max-width:1180px; padding-top:1.1rem; padding-bottom:4rem; }}

/* top bar */
.tf-brand {{ display:flex; align-items:center; gap:12px; min-width:0; }}
.tf-logo {{ flex:0 0 auto; width:40px; height:40px; border-radius:12px; display:flex; align-items:center;
           justify-content:center; background:linear-gradient(135deg, #338ced 0%, #9887ef 100%);
           box-shadow:0 8px 28px rgba(51,140,237,0.30); }}
.tf-name {{ font-size:1.02rem; font-weight:700; letter-spacing:-0.01em; color:{INK}; line-height:1.25; }}
.tf-sub {{ font-size:0.78rem; color:{INK_2}; line-height:1.35; }}
.tf-badge {{ display:inline-flex; align-items:center; gap:7px; font-size:0.78rem; font-weight:600; color:{INK};
            background:{RAISED}; padding:7px 12px; border-radius:999px; white-space:nowrap; }}
.tf-dot {{ display:inline-block; flex:0 0 auto; width:8px; height:8px; border-radius:50%; }}
.st-key-topbar {{ margin-bottom:10px; }}
.st-key-topbar button {{ border-radius:999px; min-height:34px; padding:4px 14px; background:{RAISED}; border:none;
                        font-size:0.8rem; }}
.st-key-topbar button:hover {{ background:{RAISED_2}; color:{INK}; }}

/* hero */
.tf-eyebrow {{ display:flex; align-items:center; gap:8px; font-size:0.88rem; color:{INK_2}; font-weight:500; }}
.tf-hero {{ font-size:clamp(2.7rem, 7.4vw, 4.1rem); font-weight:700; letter-spacing:-0.045em; line-height:1;
           margin:12px 0 12px 0; color:{INK}; font-variant-numeric:tabular-nums; }}
.tf-change {{ display:flex; flex-wrap:wrap; align-items:center; gap:4px 10px; font-size:1rem; font-weight:600;
             font-variant-numeric:tabular-nums; }}
.tf-change.good {{ color:{GOOD}; }} .tf-change.bad {{ color:{BAD}; }}
.tf-meta {{ font-size:0.86rem; color:{INK_2}; margin-top:6px; font-variant-numeric:tabular-nums; }}
.tf-vs {{ display:flex; flex-wrap:wrap; align-items:center; gap:4px 8px; margin-top:10px; font-size:0.88rem;
         color:{INK_2}; font-variant-numeric:tabular-nums; }}
.tf-vs b {{ color:{INK}; font-weight:600; }}
.tf-vs .good {{ color:{GOOD}; }} .tf-vs .bad {{ color:{BAD}; }}

/* round quick actions. ACT is the button that opens each popover. Older Streamlit puts the popover content
   next to that button inside the same row, so every rule targets the button itself and never other buttons. */
.st-key-actions {{ gap:6px !important; justify-content:flex-end; flex-wrap:nowrap !important; }}
{ACT} {{ --circle:50px; background:transparent !important; border:none !important; box-shadow:none !important;
  padding:0 !important; min-height:0 !important; width:84px; color:{INK} !important; }}
{ACT}, {ACT} > div {{ flex-direction:column; align-items:center; justify-content:flex-start; gap:8px; }}
{ACT} span[data-has-shortcut], {ACT} > div > div:first-child, {ACT} > div > span {{ display:flex;
  flex-direction:column; align-items:center; gap:8px; }}
{ACT} > div > div[aria-hidden="true"], {ACT} > div:has(> svg) {{ display:none; }}  /* dropdown arrow */
/* the circle is the small wrapper Streamlit puts around the icon. :has() finds it without its generated class */
{ACT} {ICON_BOX}:not(:has([data-testid="stMarkdownContainer"])) {{ display:flex; align-items:center;
  justify-content:center; flex:0 0 auto; width:var(--circle); height:var(--circle); border-radius:50%;
  background:{RAISED}; margin:0; transition:background .15s ease, transform .15s ease; }}
{ACT} [data-testid="stIconMaterial"] {{ font-size:22px; width:auto; height:auto; color:{INK}; }}
{ACT}:hover {ICON_BOX} {{ background:{RAISED_2}; }}
{ACT}:active {ICON_BOX} {{ transform:scale(0.94); }}
{ACT}[aria-expanded="true"] {ICON_BOX}, [aria-expanded="true"] > {ACT} {ICON_BOX} {{ background:{ACCENT}; }}
{ACT} [data-testid="stMarkdownContainer"] p {{ font-size:0.76rem; font-weight:500; line-height:1.25; color:{INK};
  text-align:center; white-space:normal; margin:0; }}
[data-testid="stPopoverBody"] {{ background:#1a1a1a; border:1px solid #2a2a2a; border-radius:20px;
  box-shadow:0 24px 64px rgba(0,0,0,0.6); padding:18px 18px 10px 18px; width:min(340px, calc(100vw - 32px)); }}
[data-testid="stPopoverBody"] > div {{ background:transparent; border:none; box-shadow:none; padding:0; }}
.tf-pop-title {{ font-size:1rem; font-weight:700; color:{INK}; }}
.tf-pop-text {{ font-size:0.8rem; color:{INK_2}; line-height:1.45; margin:2px 0 4px 0; }}

/* chips. Period and chart type look like Trade Republic chips, the analysis views like tabs */
[data-testid="stButtonGroup"] button {{ border:none !important; background:transparent !important;
  border-radius:999px !important; color:{INK_2} !important; min-height:34px; padding:4px 13px !important;
  box-shadow:none !important; font-weight:600; }}
[data-testid="stButtonGroup"] button p {{ font-weight:600; font-size:0.86rem; }}
[data-testid="stButtonGroup"] button:hover {{ color:{INK} !important; background:rgba(255,255,255,0.05) !important; }}
[data-testid="stButtonGroup"] button[aria-checked="true"],
[data-testid="stButtonGroup"] button[data-testid$="Active"] {{ background:{RAISED} !important;
  color:{INK} !important; }}
[data-testid="stButtonGroup"] [role="radiogroup"], [data-testid="stButtonGroup"] > div {{ gap:2px; }}
.st-key-view [data-testid="stButtonGroup"], .st-key-view [data-testid="stButtonGroup"] > div {{ width:100%; }}
.st-key-view [data-testid="stButtonGroup"] > div:last-child {{ flex-wrap:nowrap; overflow-x:auto; gap:22px;
  border-bottom:1px solid {BORDER}; scrollbar-width:none; }}
.st-key-view [data-testid="stButtonGroup"] > div:last-child::-webkit-scrollbar {{ display:none; }}
.st-key-view [data-testid="stButtonGroup"] button {{ flex:0 0 auto; border-radius:0 !important;
  padding:8px 0 10px 0 !important;
  background:transparent !important; border-bottom:2px solid transparent !important; margin-bottom:-1px; }}
.st-key-view [data-testid="stButtonGroup"] button:hover {{ background:transparent !important; }}
.st-key-view [data-testid="stButtonGroup"] button[aria-checked="true"],
.st-key-view [data-testid="stButtonGroup"] button[data-testid$="Active"] {{ background:transparent !important;
  border-bottom-color:{ACCENT} !important; }}
.st-key-view [data-testid="stButtonGroup"] button p {{ font-size:0.9rem; white-space:nowrap; }}

/* keyboard focus. The chips, tabs and round buttons above drop Streamlit's own ring, so they get a clear one back */
[data-testid="stButtonGroup"] button:focus-visible, .st-key-topbar button:focus-visible,
.st-key-holdnav button:focus-visible, .tf-info summary:focus-visible, .tf-foot a:focus-visible {{
  outline:2px solid {ACCENT} !important; outline-offset:2px; }}
.st-key-view [data-testid="stButtonGroup"] button:focus-visible {{ outline-offset:-2px; }}
{ACT}:focus-visible {{ outline:none !important; }}
{ACT}:focus-visible {ICON_BOX} {{ box-shadow:0 0 0 2px {BG}, 0 0 0 4px {ACCENT}; }}

/* holdings month stepper */
.st-key-holdnav {{ flex-wrap:nowrap !important; gap:10px !important; }}
.st-key-holdnav button {{ width:40px; height:40px; min-height:40px; padding:0; border-radius:50%; border:none;
  background:{RAISED}; }}
.st-key-holdnav button:hover:not(:disabled) {{ background:{RAISED_2}; color:{INK}; }}
.st-key-holdnav button:disabled {{ background:{RAISED} !important; opacity:0.35; }}
.st-key-holdnav button [data-testid="stMarkdownContainer"] {{ position:absolute; width:1px; height:1px;
  overflow:hidden; clip:rect(0 0 0 0); white-space:nowrap; }}  /* the words stay for screen readers */
.st-key-holdnav > div:has([data-testid="stSlider"]) {{ flex:1 1 auto !important; min-width:0 !important; }}

/* sections and cards */
.tf-sec {{ display:flex; justify-content:space-between; align-items:baseline; gap:4px 14px; flex-wrap:wrap;
          margin:26px 0 0 0; }}
.tf-sec-title {{ font-size:1.3rem; font-weight:700; letter-spacing:-0.02em; color:{INK}; }}
.tf-sec-meta {{ font-size:0.82rem; color:{INK_2}; }}
div[class*="st-key-card"] {{ background:{CARD}; border:1px solid {BORDER}; border-radius:24px;
  padding:20px 22px 16px 22px; }}
.tf-label {{ font-size:0.84rem; color:{INK_2}; font-weight:500; }}
.tf-caption {{ font-size:0.78rem; color:{INK_2}; margin:0 0 8px 0; line-height:1.5; }}
.tf-t1 {{ font-weight:600; color:{INK}; font-size:0.94rem; line-height:1.3; }}
.tf-t2 {{ color:{INK_2}; font-size:0.79rem; line-height:1.4; }}
.tf-empty {{ color:{INK_2}; font-size:0.86rem; padding:14px 2px; }}

/* key figures grid, split by hairlines */
.tf-stats {{ display:grid; grid-template-columns:repeat(4, minmax(0, 1fr)); gap:1px; background:{BORDER};
            border:1px solid {BORDER}; border-radius:24px; overflow:hidden; }}
.tf-stat {{ background:{CARD}; padding:18px 20px 18px 20px; min-width:0; }}
.tf-stat .v {{ font-size:1.6rem; font-weight:700; letter-spacing:-0.025em; margin:6px 0 10px 0; color:{INK};
              font-variant-numeric:tabular-nums; }}
.tf-stat .c {{ font-size:0.78rem; color:{INK_2}; display:flex; align-items:center; gap:6px 8px; flex-wrap:wrap;
              font-variant-numeric:tabular-nums; }}
.tf-pill {{ display:inline-flex; align-items:center; gap:5px; padding:3px 9px; border-radius:999px; font-weight:600;
           font-size:0.76rem; color:{INK}; font-variant-numeric:tabular-nums; white-space:nowrap; }}
.tf-pill.good {{ background:rgba(34,197,94,0.14); color:{GOOD}; }}
.tf-pill.bad {{ background:rgba(240,68,68,0.15); color:{BAD}; }}
.tf-pill.neutral {{ background:{RAISED}; }}
.tf-plain {{ font-size:0.78rem; color:{INK_2}; line-height:1.45; margin-top:8px; }}
.tf-info summary {{ list-style:none; cursor:pointer; display:block; border-radius:6px; }}
.tf-info summary::-webkit-details-marker {{ display:none; }}
.tf-info summary .i {{ display:inline-flex; align-items:center; justify-content:center; width:16px; height:16px;
  border-radius:50%; border:1px solid {INK_2}; color:{INK_2}; font-size:0.66rem; font-weight:700;
  font-style:normal; line-height:1; margin-left:6px; vertical-align:1px; }}
.tf-info summary:hover .i, .tf-info[open] summary .i {{ border-color:{INK}; color:{INK}; }}
.tf-info-box {{ font-size:0.78rem; color:{INK}; line-height:1.45; background:{RAISED}; border-radius:12px;
  padding:9px 11px; margin:8px 0 0 0; }}
.tf-keys {{ display:flex; flex-wrap:wrap; gap:6px 16px; font-size:0.8rem; color:{INK_2}; margin:2px 0 2px 0; }}
.tf-keys span {{ display:inline-flex; align-items:center; gap:7px; }}
.tf-keys .sw {{ display:inline-block; width:16px; height:3px; border-radius:2px; }}
.tf-rule {{ font-size:0.84rem; color:{INK_2}; line-height:1.5; margin-top:12px; max-width:560px; }}
.tf-error {{ max-width:560px; margin:48px auto 12px auto; text-align:center; }}
.tf-error .ic {{ width:52px; height:52px; border-radius:50%; margin:0 auto 14px auto; display:flex; align-items:center;
  justify-content:center; background:rgba(210,142,5,0.16); color:#d28e05; font-size:1.5rem; font-weight:800; }}
.tf-error .t {{ font-size:1.25rem; font-weight:700; color:{INK}; margin-bottom:6px; }}
.tf-error .d {{ font-size:0.9rem; color:{INK_2}; line-height:1.55; }}
.st-key-error_actions {{ justify-content:center; }}
.st-key-error_details {{ max-width:560px; margin:0 auto; }}

/* holdings and activity lists */
.tf-donut-text .big {{ font-size:1.65rem; font-weight:700; letter-spacing:-0.02em; color:{INK};
                      font-variant-numeric:tabular-nums; margin:4px 0 8px 0; }}
.tf-legend {{ display:flex; flex-wrap:wrap; gap:6px 14px; font-size:0.8rem; color:{INK_2}; margin-bottom:10px; }}
.tf-legend span {{ display:inline-flex; align-items:center; gap:6px; }}
.tf-legend b {{ color:{INK}; font-weight:600; font-variant-numeric:tabular-nums; }}
.tf-item {{ display:grid; grid-template-columns:44px minmax(0, 1fr) 112px 128px; align-items:center; gap:14px;
           padding:12px 0; border-top:1px solid {BORDER}; }}
.tf-avatar {{ width:44px; height:44px; border-radius:50%; display:flex; align-items:center; justify-content:center;
             font-weight:700; font-size:0.72rem; letter-spacing:0.02em; }}
.tf-right {{ display:flex; flex-direction:column; align-items:flex-end; gap:3px; text-align:right;
            font-variant-numeric:tabular-nums; }}
.tf-w {{ font-weight:600; color:{INK}; font-size:0.94rem; }}
.tf-w.off {{ color:{INK_2}; }}
.tf-trend {{ font-size:0.78rem; font-weight:500; white-space:nowrap; }}
.tf-trend.good {{ color:{GOOD}; }} .tf-trend.bad {{ color:{BAD}; }}
.tf-group {{ font-size:0.78rem; font-weight:600; color:{INK_2}; padding:14px 0 4px 0; }}
.tf-group:first-child {{ padding-top:2px; }}
.tf-act {{ display:grid; grid-template-columns:40px minmax(0, 1fr) auto; gap:12px; align-items:center; padding:9px 0; }}
.tf-act-ic {{ width:40px; height:40px; border-radius:50%; display:flex; align-items:center; justify-content:center;
             font-weight:800; font-size:1rem; }}
.tf-amt {{ font-weight:600; font-size:0.94rem; color:{INK}; }}
.tf-amt.in {{ color:{GOOD}; }}

/* notices and footer */
.tf-note {{ display:flex; gap:10px; align-items:center; padding:11px 14px; border-radius:16px; font-size:0.84rem;
           color:{INK}; line-height:1.45; margin-top:6px; }}
.tf-note.warn {{ background:rgba(210,142,5,0.12); border:1px solid rgba(210,142,5,0.28); }}
.tf-note.info {{ background:rgba(51,140,237,0.10); border:1px solid rgba(51,140,237,0.26); }}
.tf-foot {{ color:{INK_2}; font-size:0.76rem; margin-top:28px; line-height:1.6; max-width:760px; }}
.tf-foot a {{ color:{INK}; text-decoration:underline; text-underline-offset:2px; }}
div[class*="st-key-card"] [data-testid="stExpander"] details {{ border:1px solid {BORDER}; border-radius:14px; }}
.st-key-method {{ max-width:780px; }}
.st-key-method p {{ line-height:1.6; }}
div[class*="st-key-switch_"] {{ flex-wrap:nowrap !important; align-items:center !important; margin:2px 0 4px 0; }}
div[class*="st-key-switch_"] > div[data-testid="stElementContainer"]:last-child {{ flex:1 1 auto !important;
  min-width:0 !important; width:auto !important; }}

@media (max-width: 900px) {{
  .tf-stats {{ grid-template-columns:repeat(2, minmax(0, 1fr)); }}
}}
@media (max-width: 640px) {{
  [data-testid="stMainBlockContainer"], .block-container {{ padding-left:16px; padding-right:16px;
    padding-top:0.6rem; }}
  .st-key-actions {{ display:grid !important; grid-template-columns:repeat(5, minmax(0, 1fr)); gap:0 !important;
                    margin-top:8px; }}
  .st-key-actions > div {{ width:auto !important; min-width:0 !important; }}
  {ACT} {{ width:100%; --circle:46px; }}
  .tf-sub {{ display:none; }}
  .tf-stat {{ padding:14px 14px; }}
  .tf-stat .v {{ font-size:1.3rem; }}
  .tf-item {{ grid-template-columns:40px minmax(0, 1fr) auto; gap:12px; }}
  .tf-avatar {{ width:40px; height:40px; }}
  .tf-spark {{ display:none; }}
  div[class*="st-key-card"] {{ padding:16px 14px 12px 14px; border-radius:20px; }}
  .tf-sec-title {{ font-size:1.15rem; }}
}}
</style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------- data and calculations
@st.cache_data(ttl=60 * 60 * 24, show_spinner="Downloading prices from Yahoo Finance")
def load_daily_prices():
    """Live download from the Yahoo Finance API. A saved copy is used only if Yahoo is unreachable."""
    try:
        return s.download_daily_prices(s.DATA_START, s.BACKTEST_END, retries=3, pause=3), "live"
    except Exception:
        if SNAPSHOT.exists():
            return pd.read_csv(SNAPSHOT, index_col=0, parse_dates=True), "snapshot"
        raise


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def market_data():
    daily, source = load_daily_prices()
    return s.prepare_monthly_data(daily), source


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def backtest(window: int, cost: float, use_filter: bool):
    data, _ = market_data()
    return s.run_backtest(data, sma_months=window, cost_bps=cost, use_filter=use_filter)


def in_period(frame, years):
    return frame.loc[f"{years[0]}-01-01":f"{years[1]}-12-31"]


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def window_table(cost: float, years: tuple, benchmark: str):
    rows = {}
    for n in range(2, 13):
        bt_n = backtest(n, cost, True)
        rows[n] = s.performance_table(in_period(bt_n.returns, years), in_period(bt_n.rf, years), benchmark)[s.STRATEGY]
    return pd.DataFrame(rows).T


def resample(frame, kind):
    """Resampler for weekly, monthly or quarterly bins, for old and new pandas."""
    rules = {"week": ["W-FRI"], "month": ["ME", "M"], "quarter": ["QE", "Q"]}[kind]
    for rule in rules:
        try:
            return frame.resample(rule)
        except ValueError:
            continue
    raise ValueError(f"no resample rule for {kind}")


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def daily_paths(window: int, cost: float, use_filter: bool, names: tuple):
    """Daily value of 1 invested at the start of the backtest, for each portfolio in `names`.

    During each month a portfolio holds the weights set at the previous month end, so its value moves with the
    daily ETF prices and the Treasury bill rate. At every month end the path equals the monthly backtest exactly.
    """
    daily, _ = load_daily_prices()
    data, _ = market_data()
    bt_ = backtest(window, cost, use_filter)
    raw = daily[s.RISKY_ASSETS]
    prices = raw.ffill()[raw.notna().any(axis=1)]
    months = prices.index.to_period("M")
    out = {}
    for name in names:
        weights, details = bt_.weights[name], bt_.details[name]
        value, pieces = 1.0, []
        for month_end in weights.index:
            days = prices[months == month_end.to_period("M")]
            start_prices = data.prices.loc[month_end - pd.offsets.MonthEnd(1), s.RISKY_ASSETS]
            w = weights.loc[month_end]
            accrual = np.arange(1, len(days) + 1) / len(days)
            growth_in_month = ((days / start_prices).to_numpy() @ w[s.RISKY_ASSETS].to_numpy()
                               + w["CASH"] * (1.0 + bt_.rf.loc[month_end] * accrual))
            path = value * (1.0 - details.loc[month_end, "cost"]) * growth_in_month
            pieces.append(pd.Series(path, index=days.index))
            value = float(path[-1])
        out[name] = pd.concat(pieces)
    return pd.DataFrame(out)


def candle_kind(n_months):
    return "week" if n_months <= 18 else "month" if n_months <= 72 else "quarter"


def ohlc(path, start, end, kind):
    """Open, high, low and close of a value path per week, month or quarter, rebased to 1 at `start`.

    Each candle opens at the previous close, so the candles join up like the line chart.
    """
    base = path.loc[:start].iloc[-1] if path.index[0] <= start else 1.0
    seg = path.loc[start + pd.Timedelta(days=1):end] / base
    r = resample(seg, kind)
    frame = pd.DataFrame({"high": r.max(), "low": r.min(), "close": r.last()}).dropna()
    frame["open"] = frame["close"].shift(1).fillna(1.0)
    frame["high"] = frame[["high", "open"]].max(axis=1)
    frame["low"] = frame[["low", "open"]].min(axis=1)
    return frame


def period_years(range_key, custom):
    if range_key == "Custom":
        return tuple(custom)
    if range_key in (None, "Max"):
        return FIRST_YEAR, LAST_YEAR
    n = int(range_key.rstrip("Y"))
    return max(FIRST_YEAR, LAST_YEAR - n + 1), LAST_YEAR


# ---------------------------------------------------------------- small formatting helpers
def rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


def pct(v, d=1, sign=False):
    if pd.isna(v):
        return "n/a"
    txt = f"{v * 100:+.{d}f}%" if sign else f"{v * 100:.{d}f}%"
    return txt.replace("-", "−")


def num(v, d=2, sign=False):
    if pd.isna(v):
        return "n/a"
    return (f"{v:+.{d}f}" if sign else f"{v:.{d}f}").replace("-", "−")


def money(v):
    return f"${v:,.0f}"


def pill(text, kind="neutral", icon=""):
    ic = f'<span class="ic">{icon}</span>' if icon else ""
    return f'<span class="tf-pill {kind}">{ic}{escape(text)}</span>'


def compare_pill(delta, higher_is_better=True, points=True):
    """Gap to the benchmark in plain words, such as "3.0 pts better".

    Measures in percent differ in percentage points (pts), ratios such as the Sharpe ratio in plain units.
    """
    size = abs(delta) * 100 if points else abs(delta)
    if pd.isna(delta) or round(size, 1 if points else 2) == 0:
        return pill("equal", "neutral")
    better = (delta > 0) == higher_is_better
    text = f"{size:.1f} pts" if points else f"{size:.2f}"
    return pill(f"{text} {'better' if better else 'worse'}", "good" if better else "bad")


def explained(label, text):
    """Label with an (i) that opens a short explanation. A native disclosure works with touch, mouse and keys."""
    return (f'<details class="tf-info"><summary><span class="tf-label">{escape(label)}</span><span class="i">i</span>'
            f'</summary><div class="tf-info-box">{escape(text)}</div></details>')


def chart_keys(names, dashed=()):
    """Compact legend above a chart. It wraps on a phone instead of taking space from the plot."""
    items = []
    for n in names:
        style = (f"background:repeating-linear-gradient(90deg,{COLORS[n]} 0 3px,transparent 3px 5px)" if n in dashed
                 else f"background:{COLORS[n]}")
        items.append(f'<span><span class="sw" style="{style}"></span>{escape(n)}</span>')
    return f'<div class="tf-keys">{"".join(items)}</div>'


def dot(color):
    return f'<span class="tf-dot" style="background:{color}"></span>'


def switch(name, title, note=None, **control):
    """Toggle that reacts only to the switch. The words beside it are plain text, not part of the control."""
    with st.container(horizontal=True, vertical_alignment="center", gap="small", key=f"switch_{name}"):
        st.toggle(title, label_visibility="collapsed", width="content", **(control or {"key": name}))
        note_html = f'<div class="tf-t2">{escape(note)}</div>' if note else ""
        # width="stretch" keeps the words on the same row as the switch and wraps them inside the space left
        st.markdown(f'<div class="tf-switch-text"><div class="tf-t1">{escape(title)}</div>{note_html}</div>',
                    unsafe_allow_html=True, width="stretch")


def bound(name):
    """Keyword arguments that tie a popover control to its setting (see POPOVER_SETTINGS)."""
    key = control_key(name)
    st.session_state.setdefault(key, st.session_state[name])
    return dict(key=key, on_change=sync_setting, args=(name,))


def table_height(frame):
    return 38 + 35 * len(frame)


def html(markup, **kwargs):
    st.markdown(markup, unsafe_allow_html=True, **kwargs)


def section(title, meta=""):
    """Section title above a card, with an optional note on the right."""
    html(f'<div class="tf-sec"><div class="tf-sec-title">{escape(title)}</div>'
         f'<div class="tf-sec-meta">{meta}</div></div>')


def notice(text, kind="info"):
    icon = "ⓘ" if kind == "info" else "⚠"
    html(f'<div class="tf-note {kind}"><span>{icon}</span><span>{escape(text)}</span></div>')


# ---------------------------------------------------------------- chart helpers
def base_layout(fig, height=320, yformat=None, yprefix="", right_margin=16, legend=True, xhover="%b %Y", bg=CARD,
                grid=GRID):
    # Plotly draws a backdrop under the hover line in the chart background colour. A solid colour equal to the
    # surface behind the chart keeps that backdrop invisible, so only the 1px translucent line shows.
    fig.update_layout(
        height=height, paper_bgcolor=bg, plot_bgcolor=bg,
        font=dict(family=FONT, size=13, color=INK), margin=dict(l=4, r=right_margin, t=8, b=4),
        hovermode="x unified", showlegend=legend,
        hoverlabel=dict(bgcolor="rgba(30,30,30,0.94)", bordercolor="rgba(255,255,255,0.08)",
                        font=dict(family=FONT, color=INK, size=12)),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, title=None,
                    font=dict(color=INK_2, size=12), bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(automargin=True, showgrid=False, showline=False, zeroline=False, ticks="",
                     tickfont=dict(color=INK_2, size=12),
                     showspikes=True, spikecolor=SPIKE, spikethickness=1, spikedash="solid", spikemode="across",
                     hoverformat=xhover)
    fig.update_yaxes(automargin=True, showgrid=True, gridcolor=grid, gridwidth=1, zeroline=False, showline=False,
                     ticks="",
                     tickfont=dict(color=INK_2, size=12), tickformat=yformat, tickprefix=yprefix, side="left")
    return fig


def end_labels(fig, frame, names, fmt, colors=None):
    """Direct labels at the right end of each line, nudged apart and joined by thin leaders."""
    lo, hi = frame[names].min().min(), frame[names].max().max()
    gap = 0.075 * (hi - lo) if hi > lo else 0.1
    last = None
    for name in sorted(names, key=lambda n: frame[n].iloc[-1]):
        y = frame[name].iloc[-1]
        y_text = y if last is None or y - last >= gap else last + gap
        last = y_text
        fig.add_annotation(x=frame.index[-1], y=y, text=fmt(name, y), showarrow=True, arrowhead=0, arrowwidth=1,
                           arrowcolor=INK_2, axref="pixel", ax=18, ayref="y", ay=y_text, xanchor="left",
                           font=dict(color=(colors or {}).get(name, INK), size=11.5), bgcolor="rgba(0,0,0,0)")


def show(fig):
    # Drag to zoom is off. On a phone a sideways swipe zoomed a chart by accident, and without the toolbar there
    # was no visible way back. Hover and tap still show the values. The period chips set the time window instead.
    fig.update_layout(dragmode=False)
    st.plotly_chart(fig, theme=None, config={"displayModeBar": False, "scrollZoom": False, "doubleClick": False})


def sparkline(prices, sma, color, w=112, h=38):
    """Inline SVG of the last two years of month end prices, with the moving average dashed."""
    p = prices.to_numpy(dtype=float)
    m = sma.to_numpy(dtype=float)
    both = np.concatenate([p, m[~np.isnan(m)]])
    lo, hi = float(np.nanmin(both)), float(np.nanmax(both))
    span = hi - lo if hi > lo else 1.0

    def points(arr):
        xs = np.linspace(2, w - 2, len(arr))
        return " ".join(f"{x:.1f},{h - 3 - (y - lo) / span * (h - 6):.1f}" for x, y in zip(xs, arr) if not np.isnan(y))

    return (f'<svg class="tf-spark" width="{w}" height="{h}" viewBox="0 0 {w} {h}" aria-hidden="true">'
            f'<polyline points="{points(m)}" fill="none" stroke="{INK_2}" stroke-width="1" stroke-dasharray="3 3"/>'
            f'<polyline points="{points(p)}" fill="none" stroke="{color}" stroke-width="2" '
            f'stroke-linejoin="round" stroke-linecap="round"/></svg>')


def held_runs(held: pd.Series):
    runs, start, prev_end = [], None, None
    for month_end, flag in held.items():
        m_start, m_end = month_end - pd.offsets.MonthBegin(1), month_end + pd.Timedelta(days=1)
        if flag and start is None:
            start = m_start
        if not flag and start is not None:
            runs.append((start, prev_end))
            start = None
        prev_end = m_end
    if start is not None:
        runs.append((start, prev_end))
    return runs


# ---------------------------------------------------------------- load data
logo = ('<svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden="true">'
        '<path d="M3 17 L9 11 L13 14 L21 6" stroke="white" stroke-width="2.4" stroke-linecap="round" '
        'stroke-linejoin="round"/><circle cx="21" cy="6" r="2" fill="white"/></svg>')
brand = (f'<div class="tf-brand"><div class="tf-logo">{logo}</div><div>'
         f'<div class="tf-name">Trend Filtered Portfolio</div>'
         f'<div class="tf-sub">Backtest explorer for the investor prospectus · 2270 Financial Modeling, Nova SBE</div>'
         f'</div></div>')
try:
    data, source = market_data()
except Exception as err:
    # Friendly page for visitors. The technical message stays one click away for the owner of the app.
    html(brand)
    html('<div class="tf-error"><div class="ic">!</div><div class="t">The prices are not available right now</div>'
         '<div class="d">Yahoo Finance, which supplies the prices, did not answer. This usually passes within a '
         'minute or two. Press Try again in a moment.</div></div>')
    with st.container(horizontal=True, key="error_actions"):
        if st.button("Try again", icon=":material/refresh:", type="primary"):
            st.cache_data.clear()
            st.rerun()
    with st.container(key="error_details"), st.expander("Technical details"):
        st.markdown("The app keeps working when Yahoo Finance is down if the repository contains a saved copy of "
                    "the prices in `data/prices_snapshot.csv`. The last cell of the Colab notebook creates it.")
        st.code(str(err), language=None, wrap_lines=True)
    st.stop()

window, cost, use_filter = st.session_state.window, st.session_state.cost, st.session_state.use_filter
benchmark = st.session_state.benchmark or s.PRIMARY_BENCHMARK
years = period_years(st.session_state.range, st.session_state.years)
amount = float(st.session_state.amount or 10_000)

bt = backtest(window, float(cost), use_filter)
returns, rf = in_period(bt.returns, years), in_period(bt.rf, years)
perf = s.performance_table(returns, rf, benchmark)
growth = s.growth_of_one(returns)
start_label = f"{growth.index[0]:%B %Y}"
period_label = f"{returns.index[0]:%b %Y} to {returns.index[-1]:%b %Y}"
bname = escape(benchmark)
is_default = (window == DEFAULTS["window"] and cost == DEFAULTS["cost"] and use_filter
              and benchmark == DEFAULTS["benchmark"] and years == (FIRST_YEAR, LAST_YEAR))

# ---------------------------------------------------------------- top bar
with st.container(horizontal=True, vertical_alignment="center", key="topbar"):
    html(brand, width="stretch")
    if source == "snapshot":
        html(f'<span class="tf-badge" title="Yahoo Finance did not respond, so the app uses a saved copy of the same '
             f'Yahoo Finance prices.">{dot("#d28e05")}Saved prices</span>', width="content")
    if is_default:
        html(f'<span class="tf-badge">{dot(ACCENT)}Prospectus settings</span>', width="content")
    else:
        html(f'<span class="tf-badge">{dot("#e76624")}Custom settings</span>', width="content")
        st.button("Reset", icon=":material/restart_alt:", on_click=reset_settings, key="reset",
                  help="Back to the settings used in the prospectus")

# ---------------------------------------------------------------- hero
show_bench = bool(st.session_state.show_benchmark)
on_chart = [s.STRATEGY, benchmark] if show_bench else [s.STRATEGY]
value = amount * growth[s.STRATEGY].iloc[-1]
total = growth[s.STRATEGY].iloc[-1] - 1
b_value = amount * growth[benchmark].iloc[-1]
b_total = growth[benchmark].iloc[-1] - 1

col_num, col_act = st.columns([1.1, 1], vertical_alignment="bottom", gap="medium")
with col_num:
    up = value >= amount
    vs_html = (f'<div class="tf-vs">{dot(COLORS[benchmark])}<span>{bname}</span><b>{money(b_value)}</b>'
               f'<span class="{"good" if b_total >= 0 else "bad"}">{pct(b_total, 1, sign=True)}</span>'
               f'<span>· {pct(perf.loc["CAGR", benchmark], 1)} a year</span></div>' if show_bench else "")
    html(f'<div class="tf-eyebrow">{dot(ACCENT)}Strategy · {money(amount)} invested at the end of {start_label}</div>'
         f'<div class="tf-hero">{money(value)}</div>'
         f'<div class="tf-change {"good" if up else "bad"}"><span>{"▲" if up else "▼"} {money(abs(value - amount))}'
         f'</span><span>{pct(total, 1, sign=True)}</span></div>'
         f'<div class="tf-meta">{pct(perf.loc["CAGR", s.STRATEGY], 1)} a year · {period_label}</div>{vs_html}')
    # The rule in one sentence, so a first time visitor knows what the number belongs to
    rule = (f"The strategy holds SPY, EFA, IEF and GLD at 25% each. At every month end, an ETF below its {window} "
            f"month average moves to Treasury bills until its price is back above it." if use_filter else
            "The trend filter is off, so the strategy simply holds SPY, EFA, IEF and GLD at 25% each every month.")
    html(f'<div class="tf-rule">{rule}</div>')

with col_act:
    # Revolut style quick actions. Each round button shows the current setting and opens its control.
    with st.container(horizontal=True, key="actions"):
        with st.popover(f"{window} month trend", icon=":material/timeline:"):
            html('<div class="tf-pop-title">Trend window</div><div class="tf-pop-text">Each month end price is '
                 'compared with the average of the last N month end prices. Above the average the ETF keeps its '
                 '25%, below it the 25% moves to Treasury bills. The prospectus uses 10 months.</div>')
            st.slider("Moving average window (months)", 2, 12, format="%d months", label_visibility="collapsed",
                      **bound("window"))
        with st.popover(f"{cost} bp costs", icon=":material/receipt_long:"):
            html('<div class="tf-pop-title">Trading cost</div><div class="tf-pop-text">Cost of buying or selling '
                 'ETFs, as a share of the amount traded. 10 basis points equal 0.10%. Treasury bill trades are free. '
                 'The prospectus uses 10.</div>')
            st.slider("Trading cost (basis points per trade)", 0, 100, step=5, format="%d bp",
                      label_visibility="collapsed", **bound("cost"))
        with st.popover(f"Filter {'on' if use_filter else 'off'}",
                        icon=":material/filter_alt:" if use_filter else ":material/filter_alt_off:"):
            html('<div class="tf-pop-title">Trend filter</div><div class="tf-pop-text">On moves an ETF to Treasury '
                 'bills while it trades below its trend. Off holds 25% in each ETF every month, like the equal '
                 'weight benchmark.</div>')
            switch("use_filter", "Use the trend filter", **bound("use_filter"))
        with st.popover(f"vs {benchmark}", icon=":material/compare_arrows:"):
            html('<div class="tf-pop-title">Benchmark</div><div class="tf-pop-text">Alpha, beta, tracking error and '
                 'information ratio are measured against this portfolio. The prospectus uses 60/40.</div>')
            st.segmented_control("Benchmark", BENCHMARKS, label_visibility="collapsed", **bound("benchmark"))
        with st.popover(money(amount), icon=":material/payments:"):
            html('<div class="tf-pop-title">Invested amount</div><div class="tf-pop-text">Any amount from $100. It '
                 'scales the dollar values only, not the returns.</div>')
            st.number_input("Invested amount ($)", min_value=100, max_value=10_000_000, step=1000,
                            label_visibility="collapsed", **bound("amount"))

chart_type = st.session_state.chart_type or "Line"
if chart_type == "Line":
    v = growth[on_chart] * amount
    lo, hi = min(v.min().min(), amount), max(v.max().max(), amount)
    floor = lo - 0.08 * (hi - lo)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=v.index, y=[floor] * len(v), mode="lines", line=dict(width=0),
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(
        x=v.index, y=v[s.STRATEGY], name=s.STRATEGY, mode="lines", fill="tonexty",
        line=dict(color=COLORS[s.STRATEGY], width=2.4),
        fillgradient=dict(type="vertical", colorscale=[[0, rgba(COLORS[s.STRATEGY], 0.0)],
                                                        [1, rgba(COLORS[s.STRATEGY], 0.30)]]),
        hovertemplate="$%{y:,.0f}"))
    if show_bench:
        fig.add_trace(go.Scatter(x=v.index, y=v[benchmark], name=benchmark, mode="lines",
                                 line=dict(color=COLORS[benchmark], width=1.8, dash="dot"),
                                 hovertemplate="$%{y:,.0f}"))
    y_range = [floor, hi + 0.06 * (hi - lo)]
    x_range = [v.index[0], v.index[-1]]
    caption = "Monthly values in US dollars, net of trading costs."
else:
    kind = candle_kind(len(returns))
    paths = daily_paths(window, float(cost), use_filter, tuple(on_chart))
    start, end = growth.index[0], returns.index[-1]
    c_s = ohlc(paths[s.STRATEGY], start, end, kind) * amount
    c_b = ohlc(paths[benchmark], start, end, kind)["close"] * amount if show_bench else None
    half = {"week": pd.Timedelta(days=3.5), "month": pd.Timedelta(days=15), "quarter": pd.Timedelta(days=45)}[kind]
    x = c_s.index - half
    if kind == "week":
        names_ = [f"Week to {min(d, end):%d %b %Y}" for d in c_s.index]
    elif kind == "month":
        names_ = [f"{d:%b %Y}" for d in c_s.index]
    else:
        names_ = [f"Q{d.quarter} {d.year}" for d in c_s.index]
    bench_vals = c_b.to_numpy() if show_bench else [None] * len(c_s)
    texts = [f"<b>{n}</b>  {(row.close / row.open - 1) * 100:+.1f}%".replace("-", "−")
             + f"<br>Open {money(row.open)}  Close {money(row.close)}"
             + f"<br>High {money(row.high)}  Low {money(row.low)}"
             + (f"<br>{escape(benchmark)} {money(bc)}" if show_bench else "")
             for n, row, bc in zip(names_, c_s.itertuples(), bench_vals)]
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=x, open=c_s["open"], high=c_s["high"], low=c_s["low"], close=c_s["close"], name=s.STRATEGY,
        increasing=dict(line=dict(color=GOOD, width=1.2), fillcolor=rgba(GOOD, 0.28)),
        decreasing=dict(line=dict(color=BAD, width=1.2), fillcolor=BAD),
        whiskerwidth=0, hoverinfo="text", hovertext=texts))
    lo, hi = min(c_s["low"].min(), amount), max(c_s["high"].max(), amount)
    if show_bench:
        fig.add_trace(go.Scatter(x=x, y=c_b, name=benchmark, mode="lines", hoverinfo="skip",
                                 line=dict(color=COLORS[benchmark], width=1.6, dash="dot")))
        lo, hi = min(lo, c_b.min()), max(hi, c_b.max())
    y_range = [lo - 0.06 * (hi - lo), hi + 0.06 * (hi - lo)]
    right_edge = min(x[-1] + half, end + pd.Timedelta(hours=23, minutes=59))  # no ticks after the data ends
    x_range = [x[0] - half, right_edge]
    caption = (f"Each candle is one {kind} of the strategy value, built from daily prices and net of trading costs. "
               f"Green candles closed above their open, red candles below.")

# Trade Republic style chart. Price axis on the right, faint grid, and a dotted line at the amount invested.
fig.add_hline(y=amount, line=dict(color=rgba(INK, 0.28), width=1, dash="dot"))
fig.add_annotation(x=1, xref="paper", y=amount, yref="y", text=f"Invested {money(amount)}", showarrow=False,
                   xanchor="right", yanchor="bottom", yshift=2, font=dict(color=INK_2, size=11))
base_layout(fig, height=340, yformat="~s", yprefix="$", right_margin=4, legend=False, bg=BG, grid=HERO_GRID)
fig.update_layout(hovermode="x" if chart_type == "Candles" else "x unified", margin=dict(l=0, r=0, t=12, b=4))
fig.update_yaxes(side="right", range=y_range, tickfont=dict(color=INK_2, size=11), nticks=6)
fig.update_xaxes(range=x_range, rangeslider_visible=False, tickfont=dict(color=INK_2, size=11))
show(fig)

c_range, c_tools = st.columns([1.25, 1], vertical_alignment="center", gap="small")
with c_range:
    st.segmented_control("Period", RANGES, key="range", label_visibility="collapsed",
                         on_change=keep_selected, args=("range",))
with c_tools:
    with st.container(horizontal=True, horizontal_alignment="right", vertical_alignment="center", key="charttools"):
        st.toggle(f"Compare {benchmark}", key="show_benchmark")
        st.segmented_control("Chart type", CHART_TYPES, key="chart_type", label_visibility="collapsed",
                             format_func=lambda c: CHART_LABELS[c], on_change=keep_selected, args=("chart_type",))
if st.session_state.range == "Custom":
    st.slider("Years to evaluate", FIRST_YEAR, LAST_YEAR, key="years")
bench_sentence = f" The dotted line is the {bname} benchmark." if show_bench else ""
html(f'<div class="tf-caption">{period_label}. {caption}{bench_sentence}</div>')

if len(returns) < 36:
    notice("The selected period has fewer than 36 months, so the measures below are very noisy.", "warn")
if not use_filter:
    notice("The trend filter is off. The strategy holds 25% in each ETF every month and equals the equal weight "
           "benchmark.", "info")

# ---------------------------------------------------------------- key figures
p, b = perf[s.STRATEGY], perf[benchmark]
trade_bt = s.BacktestResult(returns=returns, rf=rf, details={k: in_period(d, years) for k, d in bt.details.items()},
                            weights={k: in_period(w, years) for k, w in bt.weights.items()})
trade = s.trading_summary(trade_bt, s.STRATEGY)
t_alpha = p["Alpha t statistic"]
sig = "significant at 5%" if abs(t_alpha) >= 1.96 else "not significant"


EXPLAIN = {
    "Average annual return": "The average monthly return multiplied by 12. Higher is better.",
    "Volatility": "How much the monthly returns swing, scaled to one year. Lower means a calmer ride.",
    "Sharpe ratio": "Return above the Treasury bill rate for each unit of volatility. Higher means more reward for "
                    "the risk taken.",
    "Maximum drawdown": "The largest fall from a previous high before the value recovered. Closer to zero is better.",
    "Alpha, annualised": "Yearly return that the benchmark does not explain. A t statistic further from zero than "
                         "1.96 means the alpha is unlikely to be luck.",
    "Beta": "How strongly the strategy follows the benchmark. A beta of 0.5 means it moved about half as much as "
            "the benchmark.",
    "Information ratio": "Average return above the benchmark divided by the tracking error, which measures how far "
                         "the two drift apart. Higher is better.",
    "Time invested in ETFs": "Average share of the portfolio held in ETFs rather than Treasury bills. A switch is one "
                             "ETF moving in or out.",
}


def stat(label, value, compare_html, plain=""):
    plain_html = f'<div class="tf-plain">{plain}</div>' if plain else ""
    return (f'<div class="tf-stat">{explained(label, EXPLAIN[label])}<div class="v">{value}</div>'
            f'<div class="c">{compare_html}</div>{plain_html}</div>')


dd_s, dd_b = p["Maximum drawdown"], b["Maximum drawdown"]
stats = [
    stat("Average annual return", pct(p["Annualised average return"], 2),
         f'<span>{bname} {pct(b["Annualised average return"], 2)}</span>'
         + compare_pill(p["Annualised average return"] - b["Annualised average return"], True)),
    stat("Volatility", pct(p["Annualised volatility"], 2),
         f'<span>{bname} {pct(b["Annualised volatility"], 2)}</span>'
         + compare_pill(p["Annualised volatility"] - b["Annualised volatility"], False)),
    stat("Sharpe ratio", num(p["Sharpe ratio"]),
         f'<span>{bname} {num(b["Sharpe ratio"])}</span>'
         + compare_pill(p["Sharpe ratio"] - b["Sharpe ratio"], True, points=False)),
    stat("Maximum drawdown", pct(dd_s, 1), f'<span>{bname} {pct(dd_b, 1)}</span>' + compare_pill(dd_s - dd_b, True),
         plain=f"Like {money(amount)} falling to {money(amount * (1 + dd_s))}, against "
               f"{money(amount * (1 + dd_b))} for {bname}"),
    stat("Alpha, annualised", pct(p["Alpha (annualised)"], 2),
         f'<span>t statistic {num(t_alpha)}</span>' + pill(sig, "neutral")),
    stat("Beta", num(p["Beta"]), f'<span>Against {bname}, whose beta is 1.00</span>'),
    stat("Information ratio", num(p["Information ratio"]),
         f'<span>Tracking error {pct(p["Tracking error"], 1)} a year</span>'),
    stat("Time invested in ETFs", pct(trade["Average share invested in ETFs"], 0),
         f'<span>{trade["Signal switches per year"]:.1f} switches and {pct(trade["Annual trading cost"], 2)} '
         f'costs a year</span>'),
]
section("Key figures", f"Strategy against {bname} · {period_label}")
html('<div class="tf-stats">' + "".join(stats) + "</div>")

# ---------------------------------------------------------------- holdings and activity
HOLDING_KEY = "holding_month"


def step_month(labels, step):
    """Previous or next month. On a phone the slider packs 241 months into a few hundred pixels."""
    i = labels.index(st.session_state[HOLDING_KEY]) + step
    st.session_state[HOLDING_KEY] = labels[min(max(i, 0), len(labels) - 1)]


@st.fragment
def holdings_card(bt, data, window, use_filter):
    """Holdings in any month of the backtest. Only this card reruns when its month slider moves."""
    w_hist = bt.weights[s.STRATEGY]
    # every month of the backtest (Max window), plus the month after it, set at the last rebalance
    months = list(w_hist.index) + [w_hist.index[-1] + pd.offsets.MonthEnd(1)]
    labels = [f"{m:%b %Y}" for m in months]
    if st.session_state.get(HOLDING_KEY) not in labels:
        st.session_state[HOLDING_KEY] = labels[-1]
    picked = months[labels.index(st.session_state[HOLDING_KEY])]
    section("Holdings", f"{picked:%B %Y}" + (" · latest" if picked == months[-1] else ""))
    with st.container(key="card_holdings"):
        html(f'<div class="tf-caption">Move the slider or step one month at a time to see what the portfolio held in '
             f'any month from {labels[0]} to {labels[-1]}.</div>')
        at = labels.index(st.session_state[HOLDING_KEY])
        with st.container(horizontal=True, vertical_alignment="center", key="holdnav"):
            st.button("Previous month", icon=":material/chevron_left:", key="hold_prev", on_click=step_month,
                      args=(labels, -1), disabled=at == 0)
            choice = st.select_slider("Month", options=labels, key=HOLDING_KEY, label_visibility="collapsed")
            st.button("Next month", icon=":material/chevron_right:", key="hold_next", on_click=step_month,
                      args=(labels, 1), disabled=at == len(labels) - 1)
        month = months[labels.index(choice)]
        rebalance = month - pd.offsets.MonthEnd(1)
        if month in w_hist.index:
            weights = w_hist.loc[month].to_dict()
        else:
            sig_ = bt.signals.loc[rebalance]
            weights = {a: (0.25 if (sig_[a] == 1 or not use_filter) else 0.0) for a in s.RISKY_ASSETS}
            weights["CASH"] = 1 - sum(weights.values())
        sma_at, price_at, irx_at = bt.sma.loc[rebalance], data.prices.loc[rebalance], data.irx.loc[rebalance]
        keys = s.RISKY_ASSETS + ["CASH"]
        d1, d2 = st.columns([1, 1.7], gap="small", vertical_alignment="center")
        with d1:
            donut = go.Figure(go.Pie(labels=[a if a != "CASH" else "T-bills" for a in keys],
                                     values=[weights[a] for a in keys], hole=0.74, sort=False, direction="clockwise",
                                     marker=dict(colors=[ASSET_COLORS[a] for a in keys], line=dict(color=CARD, width=3)),
                                     textinfo="none", hovertemplate="%{label} %{value:.0%}<extra></extra>"))
            donut.add_annotation(text=f"<b>{pct(1 - weights['CASH'], 0)}</b><br><span style='font-size:12px;"
                                      f"color:{INK_2}'>invested</span>", showarrow=False,
                                 font=dict(size=22, color=INK, family=FONT))
            donut.update_layout(height=176, margin=dict(l=4, r=4, t=8, b=8), showlegend=False,
                                paper_bgcolor=CARD, font=dict(family=FONT))
            show(donut)
        with d2:
            n_held = sum(1 for a in s.RISKY_ASSETS if weights[a] > 0)
            legend = "".join(f'<span>{dot(ASSET_COLORS[a])}{a if a != "CASH" else "T-bills"} '
                             f'<b>{pct(weights[a], 0)}</b></span>' for a in keys)
            html(f'<div class="tf-donut-text"><div class="tf-label">Held during {month:%B %Y}</div>'
                 f'<div class="big">{n_held} of 4 ETFs</div><div class="tf-legend">{legend}</div>'
                 f'<div class="tf-t2">Set at the rebalance on {rebalance:%d %b %Y}. Cash earns the Treasury bill rate, '
                 f'{irx_at:.2f}% a year at that date.</div></div>')
        rows = []
        for a in s.RISKY_ASSETS:
            name, desc = ASSET_INFO[a]
            color = ASSET_COLORS[a]
            dist = price_at[a] / sma_at[a] - 1
            held = weights[a] > 0
            trend = (f'<span class="tf-trend {"good" if dist >= 0 else "bad"}">{"▲" if dist >= 0 else "▼"} '
                     f'{pct(abs(dist), 1)} {"above" if dist >= 0 else "below"} trend</span>')
            spark = sparkline(data.prices[a].loc[:rebalance].tail(24), bt.sma[a].loc[:rebalance].tail(24), color)
            rows.append(f'<div class="tf-item"><div class="tf-avatar" style="background:{rgba(color, 0.16)};'
                        f'color:{color}">{a}</div><div><div class="tf-t1">{name}</div><div class="tf-t2">{a} · '
                        f'{desc}</div></div>{spark}<div class="tf-right"><span class="tf-w{"" if held else " off"}">'
                        f'{pct(weights[a], 0)}</span>{trend}</div></div>')
        cash_color = ASSET_COLORS["CASH"]
        rows.append(f'<div class="tf-item"><div class="tf-avatar" style="background:{rgba(cash_color, 0.16)};'
                    f'color:{cash_color}">T</div><div><div class="tf-t1">Treasury bills</div><div class="tf-t2">'
                    f'3 month US government bills</div></div><div class="tf-spark"></div><div class="tf-right">'
                    f'<span class="tf-w{"" if weights["CASH"] > 0 else " off"}">{pct(weights["CASH"], 0)}</span>'
                    f'<span class="tf-t2">{irx_at:.2f}% a year</span></div></div>')
        html("".join(rows))
        html(f'<div class="tf-caption" style="margin-top:4px">Sparklines show the 24 month end prices up to the '
             f'rebalance, with the {window} month moving average (the trend) dashed.</div>')


col_hold, col_act = st.columns([1.4, 1], gap="medium")
with col_hold:
    holdings_card(bt, data, window, use_filter)

with col_act:
    w_all = bt.weights[s.STRATEGY][s.RISKY_ASSETS]
    w_per = in_period(w_all, years)
    prev = w_all.shift(1).reindex(w_per.index)
    trades = []
    for date in w_per.index:
        for a in s.RISKY_ASSETS:
            before, after = prev.loc[date, a], w_per.loc[date, a]
            if pd.isna(before) or before == after:
                continue
            trades.append((date - pd.offsets.MonthEnd(1), "Bought" if after > before else "Sold", a))
    section("Activity", f"{len(trades)} trades · {period_label}")
    with st.container(key="card_activity"):
        if trades:
            # Revolut style list. Trades grouped by month, money moving in shown in green.
            items, last_group = [], None
            for date, action, a in trades[::-1][:6]:
                group = f"{date:%B %Y}"
                if group != last_group:
                    items.append(f'<div class="tf-group">{group}</div>')
                    last_group = group
                color = ASSET_COLORS[a]
                bought = action == "Bought"
                why = (f"Rose above its {window} month average" if bought
                       else f"Fell below its {window} month average")
                items.append(f'<div class="tf-act"><div class="tf-act-ic" style="background:{rgba(color, 0.16)};'
                             f'color:{color}">{"↗" if bought else "↘"}</div><div><div class="tf-t1">{action} {a}'
                             f'</div><div class="tf-t2">{why}</div></div><div class="tf-right">'
                             f'<span class="tf-amt{" in" if bought else ""}">{"+" if bought else "−"}25%</span>'
                             f'<span class="tf-t2">{"from" if bought else "to"} T-bills</span></div></div>')
            html("".join(items))
            with st.expander(f"See all {len(trades)} trades"):
                st.dataframe(pd.DataFrame([{"Rebalance": f"{d:%d %b %Y}", "Trade": f"{act} {a}",
                                            "Weight": "+25%" if act == "Bought" else "−25%"}
                                           for d, act, a in trades[::-1]]),
                             hide_index=True, height=min(420, 38 + 35 * len(trades)))
            html('<div class="tf-caption" style="margin-top:8px">Trades happen at month end prices.</div>')
        else:
            html('<div class="tf-empty">No trades. Every ETF kept its weight during this period.</div>')

# ---------------------------------------------------------------- analysis views
section("Analysis", period_label)
with st.container(key="card_analysis"):
    view = st.segmented_control("View", VIEWS, key="view", label_visibility="collapsed",
                                on_change=keep_selected, args=("view",)) or "Growth"

    if view == "Growth":
        html(f'<div class="tf-caption">Growth of 1 invested at the end of {start_label}. All four portfolios start '
             f'at 1. With the prospectus settings this is Figure 1 of the notebook.</div>')
        fig = go.Figure()
        for name in reversed(ORDER):
            fig.add_trace(go.Scatter(x=growth.index, y=growth[name], name=name, mode="lines",
                                     line=dict(color=COLORS[name], width=2.8 if name == s.STRATEGY else 1.8),
                                     hovertemplate="%{y:.2f}"))
        end_labels(fig, growth, ORDER, lambda n, y: f"{y:.2f}", colors=COLORS)
        base_layout(fig, height=420, yformat=".2f", right_margin=44, legend=False)
        fig.update_xaxes(range=[growth.index[0], growth.index[-1]])
        html(chart_keys(ORDER))
        show(fig)

    elif view == "Drawdowns":
        html('<div class="tf-caption">Percentage fall of each portfolio from its previous highest value.</div>')
        dd = s.drawdowns(returns)
        names = [s.BENCH_SPY] + ([benchmark] if benchmark != s.BENCH_SPY else []) + [s.STRATEGY]
        fig = go.Figure()
        for name in names:
            is_s = name == s.STRATEGY
            fig.add_trace(go.Scatter(
                x=dd.index, y=dd[name], name=name, mode="lines", line=dict(color=COLORS[name], width=2.6 if is_s else 1.6),
                fill="tozeroy" if is_s else None,
                fillgradient=dict(type="vertical", colorscale=[[0, rgba(COLORS[name], 0.30)], [1, rgba(COLORS[name], 0.02)]])
                if is_s else None,
                hovertemplate="%{y:.1%}"))
        base_layout(fig, height=380, yformat=".0%", legend=False)
        html(chart_keys(names[::-1]))
        show(fig)

    elif view == "Holdings history":
        html('<div class="tf-caption">Coloured bars mark the months in which an ETF kept its 25% weight. Gaps are '
             'months in Treasury bills.</div>')
        w = in_period(bt.weights[s.STRATEGY], years)
        held = w[s.RISKY_ASSETS] > 0
        rows = [{"ETF": f"{a}  {held[a].mean():.0%} of months", "Start": a0, "End": a1, "Asset": a}
                for a in s.RISKY_ASSETS for a0, a1 in held_runs(held[a])]
        if rows:
            tl = px.timeline(pd.DataFrame(rows), x_start="Start", x_end="End", y="ETF", color="Asset",
                             color_discrete_map=ASSET_COLORS)
            tl.update_traces(marker_line_color=CARD, marker_line_width=1, hovertemplate="%{y}<extra></extra>")
            order = [f"{a}  {held[a].mean():.0%} of months" for a in s.RISKY_ASSETS]
            base_layout(tl, height=300, legend=False)
            tl.update_yaxes(categoryorder="array", categoryarray=list(reversed(order)), title=None, showgrid=False,
                            side="left", tickfont=dict(color=INK, size=13))
            tl.update_xaxes(title=None, showspikes=False,
                            range=[w.index[0] - pd.offsets.MonthBegin(1), w.index[-1] + pd.Timedelta(days=1)])
            tl.update_layout(hovermode="closest", bargap=0.35)
            show(tl)
        cash = list(w["CASH"]) + [w["CASH"].iloc[-1]]
        starts = list(w.index - pd.offsets.MonthBegin(1)) + [w.index[-1] + pd.Timedelta(days=1)]
        cf = go.Figure(go.Scatter(x=starts, y=cash, mode="lines", name="Treasury bills",
                                  line=dict(color=ASSET_COLORS["CASH"], width=2, shape="hv"), fill="tozeroy",
                                  fillcolor=rgba(ASSET_COLORS["CASH"], 0.14), hovertemplate="%{y:.0%}"))
        base_layout(cf, height=200, yformat=".0%", legend=False)
        cf.update_yaxes(range=[0, 1.05])
        cf.update_xaxes(range=[starts[0], starts[-1]])
        html('<div class="tf-caption" style="margin-top:6px">Share of the portfolio in Treasury bills.</div>')
        show(cf)

    elif view == "Years":
        yr = s.calendar_year_returns(returns)
        html(f'<div class="tf-caption">Return in each calendar year, strategy against the {bname} benchmark.</div>')
        fig = go.Figure()
        for name in [s.STRATEGY, benchmark]:
            fig.add_trace(go.Bar(x=yr.index.astype(str), y=yr[name], name=name, marker_color=COLORS[name],
                                 marker_cornerradius=4, hovertemplate="%{y:.1%}"))
        base_layout(fig, height=360, yformat=".0%", legend=False)
        fig.update_layout(barmode="group", bargap=0.32, bargroupgap=0.1)
        fig.update_xaxes(showspikes=False)
        html(chart_keys([s.STRATEGY, benchmark]))
        show(fig)
        st.dataframe(s.format_percent(yr).rename_axis("Year").reset_index(), height=table_height(yr), hide_index=True)

    elif view == "Stress tests":
        stress = s.period_returns(returns)
        html('<div class="tf-caption">Compounded return in the main stress periods and in the recoveries after '
             'them. Periods outside the selected years are left out.</div>')
        if len(stress):
            short = [SHORT_STRESS.get(i, i) for i in stress.index]
            fig = go.Figure()
            for name in [benchmark, s.STRATEGY]:
                fig.add_trace(go.Bar(y=short, x=stress[name], name=name, orientation="h", marker_color=COLORS[name],
                                     marker_cornerradius=4, hovertemplate="%{x:.1%}"))
            base_layout(fig, height=40 + 62 * len(stress), yformat=None, legend=False)
            fig.update_layout(barmode="group", bargap=0.3, hovermode="y unified")
            fig.update_xaxes(tickformat=".0%", showgrid=True, gridcolor=GRID, showspikes=False)
            fig.update_yaxes(autorange="reversed", showgrid=False, side="left", tickfont=dict(color=INK, size=12))
            html(chart_keys([s.STRATEGY, benchmark]))
            show(fig)
            st.dataframe(s.format_percent(stress).rename_axis("Period").reset_index(), height=table_height(stress),
                         hide_index=True)
        else:
            html('<div class="tf-empty">None of the stress periods falls fully inside the selected years.</div>')

    elif view == "Risk table":
        html(f'<div class="tf-caption">{period_label}. Alpha, beta, tracking error and information ratio against '
             f'the {bname} benchmark. With the prospectus settings this table equals Table 1 of the notebook.</div>')
        st.dataframe(s.format_metrics(perf), height=table_height(perf))
        html('<div class="tf-caption" style="margin-top:10px">The strategy against each benchmark.</div>')
        rel = s.relative_table(returns, rf, s.STRATEGY)
        st.dataframe(s.format_metrics(rel), height=table_height(rel))
        with st.expander("How the measures are calculated"):
            st.markdown(
                "- All returns are monthly, in US dollars and net of trading costs.\n"
                "- Average annual return equals 12 times the mean monthly return.\n"
                "- Volatility equals the square root of 12 times the standard deviation of monthly returns.\n"
                "- The Sharpe ratio uses monthly excess returns over the 3 month Treasury bill rate (Yahoo ticker "
                "^IRX), annualised with the square root of 12.\n"
                "- Alpha and beta come from an OLS regression of monthly portfolio excess returns on monthly "
                "benchmark excess returns. Alpha is multiplied by 12. Beta is not annualised.\n"
                "- The information ratio equals the annualised mean active return (portfolio minus benchmark) "
                "divided by the annualised tracking error.\n"
                "- Maximum drawdown is the largest fall of the month end value from a previous peak.")
        csv = returns.assign(**{"Risk free rate": rf}).to_csv().encode()
        st.download_button("Download the monthly returns (CSV)", csv, "monthly_returns.csv", "text/csv",
                           icon=":material/download:")

    elif view == "Robustness":
        html(f'<div class="tf-caption">Strategy results for moving average windows from 2 to 12 months, with the '
             f'current trading cost and period. The bright bar is the selected window. The line is the {bname} '
             f'benchmark.</div>')
        wt = window_table(float(cost), years, benchmark)
        c1, c2 = st.columns(2, gap="medium")
        for col, m, fmt in [(c1, "Sharpe ratio", ".2f"), (c2, "Maximum drawdown", ".1%")]:
            fig = go.Figure(go.Bar(
                x=wt.index, y=wt[m], marker_cornerradius=4, name=m,
                marker_color=[COLORS[s.STRATEGY] if n == window else rgba(COLORS[s.STRATEGY], 0.35) for n in wt.index],
                hovertemplate=f"%{{x}} months<br>%{{y:{fmt}}}<extra></extra>"))
            ref = perf.loc[m, benchmark]
            fig.add_hline(y=ref, line=dict(color=COLORS[benchmark], width=2),
                          annotation_text=f"{benchmark} {num(ref) if m == 'Sharpe ratio' else pct(ref)}",
                          annotation_position="top left", annotation_font=dict(color=INK, size=12))
            base_layout(fig, height=300, yformat=".2f" if m == "Sharpe ratio" else ".0%", legend=False)
            fig.update_layout(hovermode="closest", margin=dict(b=44))
            lo_r, hi_r = min(0.0, wt[m].min(), ref), max(0.0, wt[m].max(), ref)
            fig.update_yaxes(range=[lo_r - 0.16 * (hi_r - lo_r), hi_r + 0.16 * (hi_r - lo_r)])
            fig.update_xaxes(dtick=1, showspikes=False, title=dict(text="Window (months)", font=dict(color=INK_2, size=12)))
            with col:
                html(f'<div class="tf-t1" style="margin:4px 0 0 0">{m}</div>')
                show(fig)
        st.dataframe(pd.DataFrame({
            "Average annual return": wt["Annualised average return"].map(pct),
            "Volatility": wt["Annualised volatility"].map(pct),
            "Sharpe ratio": wt["Sharpe ratio"].map(num),
            "Alpha": wt["Alpha (annualised)"].map(lambda x: pct(x, 2)),
            "Information ratio": wt["Information ratio"].map(num),
            "Maximum drawdown": wt["Maximum drawdown"].map(pct),
        }).rename_axis("Window (months)").reset_index(), height=table_height(wt), hide_index=True)

    else:  # Method
        with st.container(key="method"):
            st.markdown(f"""
**Target investor.** Cautious retail investors aged roughly 50 to 60 who save for retirement over five to ten
years. They want returns above inflation and deposit rates, but large losses would push them to sell at the worst
moment. Limiting deep losses comes first.

**Assets.** SPY (US equities), EFA (developed equities outside the US), IEF (US Treasuries with 7 to 10 years to
maturity) and GLD (gold). Cash earns the 3 month Treasury bill rate.

**Rule.** Each ETF has a target weight of 25%. At every month end the price of each ETF is compared with its moving
average over the chosen number of month end prices. Above the average, the ETF keeps its 25% for the next month.
Below it, the 25% moves to Treasury bills. The portfolio returns to these weights every month end. The weights for a
month use only prices up to the end of the previous month, so the backtest has no look ahead bias.

**Data.** Daily prices from the Yahoo Finance API through the yfinance package, adjusted for splits and dividends and
converted to month end values. Strategy returns run from January 2006 to December 2025, and prices from 2004 serve
only to compute the first moving averages. The risk free rate is the 13 week Treasury bill yield (^IRX) at the
previous month end, divided by 12.

**Costs.** Every ETF purchase or sale costs the chosen number of basis points. Treasury bill trades are free. The
benchmarks are rebalanced monthly and pay the same costs. Taxes, advisory fees and currency effects are not included.
A European investor would hold the UCITS versions of these ETFs and face EUR/USD currency risk.

**Main risks.** The filter reacts with a delay, so it can sell after a fall and buy back after a rebound (whipsaw). It
can lag in strong bull markets because part of the portfolio sits in cash. The 10 month window comes from past
research and may not suit future markets. Results also depend on four US listed ETFs over one historical period.

**Consistency with the notebook.** The app runs `strategy.py`, which contains the same code as the engine cells of
the Colab notebook. With the prospectus settings it reproduces Table 1 of the notebook.

**Source code.** [{REPO_URL.removeprefix("https://")}]({REPO_URL})
""")

html('<div class="tf-foot">Data from Yahoo Finance through the yfinance package. Monthly returns in US dollars, net of '
     'trading costs. Past performance does not predict future returns. Educational project for 2270 Financial '
     f'Modeling at Nova SBE, not investment advice. <a href="{REPO_URL}" target="_blank" rel="noopener">Source code '
     'on GitHub</a></div>')
