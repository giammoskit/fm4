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

st.set_page_config(page_title="Trend Filtered Portfolio", layout="wide", initial_sidebar_state="collapsed")

# ---------------------------------------------------------------- design tokens (dark)
CARD, RAISED, BORDER = "#191919", "#222222", "#2a2a2a"
INK, INK_2, GRID = "#efefef", "#9c9c9c", "#2b2b2b"
GOOD, BAD = "#0aac23", "#dc3c3d"
COLORS = {s.STRATEGY: "#338ced", s.BENCH_6040: "#e76624", s.BENCH_EW: "#00a888", s.BENCH_SPY: "#d28e05"}
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
CHART_LABELS = {"Line": ":material/show_chart: Line", "Candles": ":material/candlestick_chart: Candles"}
SPIKE = "rgba(239,239,239,0.35)"  # thin, translucent hover line
VIEWS = ["Growth", "Drawdowns", "Holdings history", "Years", "Stress tests", "Risk table", "Robustness", "Method"]
BENCHMARKS = [s.BENCH_6040, s.BENCH_EW, s.BENCH_SPY]
SNAPSHOT = Path(__file__).parent / "data" / "prices_snapshot.csv"
REPO_URL = "add the GitHub repository URL here"

DEFAULTS = {"window": s.SMA_MONTHS, "cost": int(s.COST_BPS), "use_filter": True, "benchmark": s.PRIMARY_BENCHMARK,
            "range": "Max", "years": (FIRST_YEAR, LAST_YEAR), "amount": 10_000, "view": "Growth",
            "chart_type": "Line", "show_benchmark": True}
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)


def reset_settings():
    for key, value in DEFAULTS.items():
        st.session_state[key] = value
        st.session_state[f"_last_{key}"] = value
    st.session_state.pop("holding_month", None)


def keep_selected(key):
    """Clicking the active chip would clear the choice. Restore the previous choice instead."""
    if st.session_state.get(key) is None:
        st.session_state[key] = st.session_state.get(f"_last_{key}", DEFAULTS[key])
    else:
        st.session_state[f"_last_{key}"] = st.session_state[key]


# ---------------------------------------------------------------- styling
st.markdown(f"""<style>
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stMainBlockContainer"], .block-container {{ max-width: 1220px; padding-top: 1.4rem; padding-bottom: 3rem; }}
div[class*="st-key-card"] {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 24px; padding: 20px 22px 14px 22px; }}
.tf-top {{ display:flex; justify-content:space-between; align-items:center; gap:14px; margin:0 0 16px 0; flex-wrap:wrap; }}
.tf-brand {{ display:flex; align-items:center; gap:12px; }}
.tf-logo {{ width:42px; height:42px; border-radius:13px; display:flex; align-items:center; justify-content:center;
           background: linear-gradient(135deg, #338ced 0%, #9887ef 100%); box-shadow: 0 8px 28px rgba(51,140,237,0.28); }}
.tf-name {{ font-size:1.1rem; font-weight:700; letter-spacing:-0.01em; color:{INK}; line-height:1.2; }}
.tf-sub {{ font-size:0.8rem; color:{INK_2}; }}
.tf-badge {{ font-size:0.78rem; font-weight:600; color:{INK}; background:{RAISED}; border:1px solid {BORDER};
            padding:6px 12px; border-radius:999px; white-space:nowrap; }}
.tf-dot {{ display:inline-block; width:7px; height:7px; border-radius:50%; margin-right:7px; vertical-align:1px; }}
.tf-label {{ font-size:0.86rem; color:{INK_2}; font-weight:500; }}
.tf-hero {{ font-size:clamp(2.3rem, 5.2vw, 3.4rem); font-weight:750; letter-spacing:-0.035em; line-height:1.05;
           margin:6px 0 12px 0; color:{INK}; font-variant-numeric:tabular-nums; }}
.tf-row {{ display:flex; flex-wrap:wrap; align-items:center; gap:10px; font-size:0.88rem; color:{INK_2}; }}
.tf-pill {{ display:inline-flex; align-items:center; gap:6px; padding:4px 10px; border-radius:999px; font-weight:600;
           font-size:0.82rem; color:{INK}; font-variant-numeric:tabular-nums; white-space:nowrap; }}
.tf-pill.good {{ background: rgba(10,172,35,0.16); }} .tf-pill.good .ic {{ color:{GOOD}; }}
.tf-pill.bad {{ background: rgba(220,60,61,0.18); }} .tf-pill.bad .ic {{ color:{BAD}; }}
.tf-pill.neutral {{ background:{RAISED}; }}
.tf-section {{ font-size:1.02rem; font-weight:700; color:{INK}; margin:0 0 2px 0; }}
.tf-caption {{ font-size:0.8rem; color:{INK_2}; margin:0 0 8px 0; }}
.tf-grid {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(210px, 1fr)); gap:12px; margin:14px 0 14px 0; }}
.tf-metric {{ background:{CARD}; border:1px solid {BORDER}; border-radius:20px; padding:16px 18px; }}
.tf-metric .v {{ font-size:1.65rem; font-weight:700; letter-spacing:-0.02em; margin:6px 0 8px 0; color:{INK};
                font-variant-numeric:tabular-nums; }}
.tf-metric .c {{ font-size:0.8rem; color:{INK_2}; display:flex; align-items:center; gap:8px; flex-wrap:wrap; }}
.tf-item {{ display:grid; grid-template-columns:44px 1fr 112px 136px; align-items:center; gap:14px; padding:11px 2px;
           border-bottom:1px solid {BORDER}; }}
.tf-item:last-child {{ border-bottom:none; }}
.tf-avatar {{ width:44px; height:44px; border-radius:50%; display:flex; align-items:center; justify-content:center;
             font-weight:700; font-size:0.74rem; letter-spacing:0.02em; }}
.tf-t1 {{ font-weight:600; color:{INK}; font-size:0.94rem; }}
.tf-t2 {{ color:{INK_2}; font-size:0.79rem; }}
.tf-right {{ display:flex; flex-direction:column; align-items:flex-end; gap:5px; text-align:right; }}
.tf-act {{ display:grid; grid-template-columns:38px 1fr auto; gap:12px; align-items:center; padding:10px 2px;
          border-bottom:1px solid {BORDER}; }}
.tf-act:last-child {{ border-bottom:none; }}
.tf-act-ic {{ width:38px; height:38px; border-radius:12px; display:flex; align-items:center; justify-content:center;
             font-weight:800; font-size:1rem; }}
.tf-empty {{ color:{INK_2}; font-size:0.86rem; padding:14px 2px; }}
.tf-foot {{ color:{INK_2}; font-size:0.76rem; margin-top:20px; line-height:1.55; }}
div[class*="st-key-switch_"] {{ flex-wrap:nowrap !important; align-items:center !important; margin:2px 0 8px 0; }}
div[class*="st-key-switch_"] > div[data-testid="stElementContainer"]:last-child {{ flex:1 1 auto !important;
  min-width:0 !important; width:auto !important; }}
.tf-switch-text .tf-t1 {{ font-size:0.9rem; font-weight:500; }}
.tf-switch-text {{ line-height:1.3; }}
.tf-donut-text .big {{ font-size:1.5rem; font-weight:700; color:{INK}; font-variant-numeric:tabular-nums; }}
@media (max-width: 640px) {{
  .tf-grid {{ grid-template-columns:repeat(2, minmax(0, 1fr)); gap:10px; }}
  .tf-metric {{ padding:13px 13px; border-radius:18px; }}
  .tf-metric .v {{ font-size:1.3rem; }}
  .tf-item {{ grid-template-columns:40px 1fr 112px; }}
  .tf-spark {{ display:none; }}
  div[class*="st-key-card"] {{ padding:16px 14px 10px 14px; border-radius:20px; }}
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


def money_short(v):
    if abs(v) < 1e3:
        return f"${v:,.0f}"
    if abs(v) >= 1e6:
        return f"${v / 1e6:.2f}M"
    return f"${v / 1e3:.1f}k"


def pill(text, kind="neutral", icon=""):
    ic = f'<span class="ic">{icon}</span>' if icon else ""
    return f'<span class="tf-pill {kind}">{ic}{escape(text)}</span>'


def compare_pill(delta, higher_is_better=True, fmt=pct):
    """Arrow and colour show whether the strategy did better or worse than the benchmark."""
    if pd.isna(delta) or abs(delta) < 1e-12:
        return pill("equal", "neutral")
    better = (delta > 0) == higher_is_better
    return pill(fmt(delta, sign=True), "good" if better else "bad", "▲" if delta > 0 else "▼")


def switch(key, title, note=None):
    """Toggle that reacts only to the switch. The words beside it are plain text, not part of the control."""
    with st.container(horizontal=True, vertical_alignment="center", gap="small", key=f"switch_{key}"):
        st.toggle(title, key=key, label_visibility="collapsed", width="content")
        note_html = f'<div class="tf-t2">{escape(note)}</div>' if note else ""
        # width="stretch" keeps the words on the same row as the switch and wraps them inside the space left
        st.markdown(f'<div class="tf-switch-text"><div class="tf-t1">{escape(title)}</div>{note_html}</div>',
                    unsafe_allow_html=True, width="stretch")


def table_height(frame):
    return 38 + 35 * len(frame)


def html(markup):
    st.markdown(markup, unsafe_allow_html=True)


# ---------------------------------------------------------------- chart helpers
def base_layout(fig, height=320, yformat=None, yprefix="", right_margin=16, legend=True, xhover="%b %Y"):
    # Plotly draws a backdrop under the hover line in the chart background colour. A solid card colour keeps that
    # backdrop invisible, so only the 1px translucent line shows.
    fig.update_layout(
        height=height, paper_bgcolor=CARD, plot_bgcolor=CARD,
        font=dict(family=FONT, size=13, color=INK), margin=dict(l=4, r=right_margin, t=8, b=4),
        hovermode="x unified", showlegend=legend,
        hoverlabel=dict(bgcolor="rgba(34,34,34,0.90)", bordercolor="rgba(255,255,255,0.08)",
                        font=dict(family=FONT, color=INK, size=12)),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, title=None,
                    font=dict(color=INK_2, size=12), bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(automargin=True, showgrid=False, showline=False, zeroline=False, ticks="",
                     tickfont=dict(color=INK_2, size=12),
                     showspikes=True, spikecolor=SPIKE, spikethickness=1, spikedash="solid", spikemode="across",
                     hoverformat=xhover)
    fig.update_yaxes(automargin=True, showgrid=True, gridcolor=GRID, gridwidth=1, zeroline=False, showline=False,
                     ticks="",
                     tickfont=dict(color=INK_2, size=12), tickformat=yformat, tickprefix=yprefix, side="left")
    return fig


def end_labels(fig, frame, names, fmt):
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
                           font=dict(color=INK, size=11.5), bgcolor="rgba(0,0,0,0)")


def show(fig):
    st.plotly_chart(fig, theme=None, config={"displayModeBar": False})


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
try:
    data, source = market_data()
except Exception as err:
    st.error("The prices could not be downloaded from Yahoo Finance. Wait a minute and press Retry. "
             f"Details. {err}")
    if st.button("Retry"):
        st.cache_data.clear()
        st.rerun()
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
is_default = (window == DEFAULTS["window"] and cost == DEFAULTS["cost"] and use_filter
              and benchmark == DEFAULTS["benchmark"] and years == (FIRST_YEAR, LAST_YEAR))

# ---------------------------------------------------------------- top bar
logo = ('<svg width="24" height="24" viewBox="0 0 24 24" fill="none" aria-hidden="true">'
        '<path d="M3 17 L9 11 L13 14 L21 6" stroke="white" stroke-width="2.4" stroke-linecap="round" '
        'stroke-linejoin="round"/><circle cx="21" cy="6" r="2" fill="white"/></svg>')
badge = (f'<span class="tf-badge"><span class="tf-dot" style="background:#338ced"></span>Prospectus settings</span>'
         if is_default else
         f'<span class="tf-badge"><span class="tf-dot" style="background:#e76624"></span>Custom settings</span>')
html(f'<div class="tf-top"><div class="tf-brand"><div class="tf-logo">{logo}</div><div>'
     f'<div class="tf-name">Trend Filtered Portfolio</div>'
     f'<div class="tf-sub">Backtest explorer for the investor prospectus. 2270 Financial Modeling, Nova SBE</div>'
     f'</div></div>{badge}</div>')
if source == "snapshot":
    st.warning("Yahoo Finance did not respond, so the app uses a saved copy of the same Yahoo Finance prices.")

# ---------------------------------------------------------------- hero and settings
col_hero, col_set = st.columns([2.2, 1], gap="medium")

with col_hero:
    with st.container(key="card_hero"):
        value = amount * growth[s.STRATEGY].iloc[-1]
        total = growth[s.STRATEGY].iloc[-1] - 1
        b_value = amount * growth[benchmark].iloc[-1]
        b_total = growth[benchmark].iloc[-1] - 1
        show_bench = bool(st.session_state.show_benchmark)
        on_chart = [s.STRATEGY, benchmark] if show_bench else [s.STRATEGY]
        bench_text = (f'<span>·</span><span>{escape(benchmark)} {money(b_value)} '
                      f'({pct(b_total, 1, sign=True)})</span>' if show_bench else "")
        html(f'<div class="tf-label">Value of {money(amount)} invested at the end of {start_label}</div>'
             f'<div class="tf-hero">{money(value)}</div>'
             f'<div class="tf-row">{compare_pill(total, True)}'
             f'<span>{pct(perf.loc["CAGR", s.STRATEGY], 1, sign=True)} a year</span>{bench_text}</div>')
        bench_sentence = f" The dotted line is the {escape(benchmark)} benchmark." if show_bench else ""
        chart_type = st.session_state.chart_type or "Line"
        if chart_type == "Line":
            v = growth[on_chart] * amount
            lo, hi = v.min().min(), v.max().max()
            floor = lo - 0.08 * (hi - lo)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=v.index, y=[floor] * len(v), mode="lines", line=dict(width=0),
                                     hoverinfo="skip", showlegend=False))
            fig.add_trace(go.Scatter(
                x=v.index, y=v[s.STRATEGY], name=s.STRATEGY, mode="lines", fill="tonexty",
                line=dict(color=COLORS[s.STRATEGY], width=2.6),
                fillgradient=dict(type="vertical", colorscale=[[0, rgba(COLORS[s.STRATEGY], 0.0)],
                                                                [1, rgba(COLORS[s.STRATEGY], 0.32)]]),
                hovertemplate="$%{y:,.0f}"))
            if show_bench:
                fig.add_trace(go.Scatter(x=v.index, y=v[benchmark], name=benchmark, mode="lines",
                                         line=dict(color=COLORS[benchmark], width=1.8, dash="dot"),
                                         hovertemplate="$%{y:,.0f}"))
            end_labels(fig, v, on_chart, lambda n, y: f"{n} {money_short(y)}")
            base_layout(fig, height=300, yformat="~s", yprefix="$", right_margin=104, legend=show_bench)
            fig.update_layout(legend=dict(traceorder="normal"))
            fig.update_yaxes(range=[floor, hi + 0.06 * (hi - lo)])
            fig.update_xaxes(range=[v.index[0], v.index[-1]])
            caption = f"{period_label}. Monthly values in US dollars, net of trading costs.{bench_sentence}"
        else:
            kind = candle_kind(len(returns))
            paths = daily_paths(window, float(cost), use_filter, tuple(on_chart))
            start, end = growth.index[0], returns.index[-1]
            c_s = ohlc(paths[s.STRATEGY], start, end, kind) * amount
            c_b = ohlc(paths[benchmark], start, end, kind)["close"] * amount if show_bench else None
            half = {"week": pd.Timedelta(days=3.5), "month": pd.Timedelta(days=15),
                    "quarter": pd.Timedelta(days=45)}[kind]
            x = c_s.index - half
            if kind == "week":
                names_ = [f"Week to {min(d, end):%d %b %Y}" for d in c_s.index]
            elif kind == "month":
                names_ = [f"{d:%b %Y}" for d in c_s.index]
            else:
                names_ = [f"Q{d.quarter} {d.year}" for d in c_s.index]
            bench_vals = c_b.to_numpy() if show_bench else [None] * len(c_s)
            texts = [f"<b>{n}</b>  {(row.close / row.open - 1) * 100:+.1f}%".replace("-", "\u2212")
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
            ends = pd.DataFrame({s.STRATEGY: c_s["close"].to_numpy()}, index=x)
            lo, hi = c_s["low"].min(), c_s["high"].max()
            if show_bench:
                fig.add_trace(go.Scatter(x=x, y=c_b, name=benchmark, mode="lines", hoverinfo="skip",
                                         line=dict(color=COLORS[benchmark], width=1.6, dash="dot")))
                ends[benchmark] = c_b.to_numpy()
                lo, hi = min(lo, c_b.min()), max(hi, c_b.max())
            end_labels(fig, ends, on_chart, lambda n, y: f"{n} {money_short(y)}")
            base_layout(fig, height=300, yformat="~s", yprefix="$", right_margin=104, legend=show_bench)
            fig.update_layout(hovermode="x", legend=dict(traceorder="normal"))
            fig.update_yaxes(range=[lo - 0.06 * (hi - lo), hi + 0.06 * (hi - lo)])
            right_edge = min(x[-1] + half, end + pd.Timedelta(hours=23, minutes=59))  # no ticks after the data ends
            fig.update_xaxes(rangeslider_visible=False, range=[x[0] - half, right_edge])
            caption = (f"{period_label}. Each candle is one {kind} of the strategy value, built from daily prices "
                       f"and net of trading costs. Green candles closed above their open, red candles below."
                       + (f" The dotted line is the {escape(benchmark)} benchmark at each close." if show_bench else ""))
        show(fig)
        c_range, c_type = st.columns([3, 1.5], gap="small", vertical_alignment="center")
        with c_range:
            st.segmented_control("Period", RANGES, key="range", label_visibility="collapsed",
                                 on_change=keep_selected, args=("range",))
        with c_type:
            st.segmented_control("Chart type", CHART_TYPES, key="chart_type", label_visibility="collapsed",
                                 format_func=lambda c: CHART_LABELS[c], on_change=keep_selected, args=("chart_type",))
        switch("show_benchmark", f"Compare with the {benchmark} benchmark")
        html(f'<div class="tf-caption">{caption}</div>')

with col_set:
    with st.container(key="card_settings"):
        html('<div class="tf-section">Strategy settings</div>'
             '<div class="tf-caption">Change an assumption and every number updates.</div>')
        st.slider("Moving average window (months)", 2, 12, key="window",
                  help="Each month end price is compared with the average of the last N month end prices. "
                       "The prospectus uses 10 months.")
        st.slider("Trading cost (basis points per trade)", 0, 100, step=5, key="cost",
                  help="Cost of buying or selling ETFs, as a share of the amount traded. 10 basis points equal "
                       "0.10%. The prospectus uses 10.")
        switch("use_filter", "Trend filter", "Off holds 25% in each ETF every month, like the equal weight "
                                             "benchmark.")
        st.segmented_control("Benchmark", BENCHMARKS, key="benchmark", on_change=keep_selected, args=("benchmark",),
                             format_func=lambda x: "Equal" if x == s.BENCH_EW else x,
                             help="Alpha, beta, tracking error and information ratio are measured against this "
                                  "portfolio. The prospectus uses 60/40.")
        st.number_input("Invested amount ($)", min_value=100, max_value=10_000_000, step=100, key="amount",
                        help="Any amount from $100. It scales the dollar values only, not the returns.")
        if st.session_state.range == "Custom":
            st.slider("Years to evaluate", FIRST_YEAR, LAST_YEAR, key="years")
        st.button("Reset to prospectus settings", on_click=reset_settings, width="stretch")

if len(returns) < 36:
    st.warning("The selected period has fewer than 36 months, so the measures below are very noisy.")
if not use_filter:
    st.info("The trend filter is off. The strategy holds 25% in each ETF every month and equals the equal weight "
            "benchmark.")

# ---------------------------------------------------------------- metric cards
p, b = perf[s.STRATEGY], perf[benchmark]
trade_bt = s.BacktestResult(returns=returns, rf=rf, details={k: in_period(d, years) for k, d in bt.details.items()},
                            weights={k: in_period(w, years) for k, w in bt.weights.items()})
trade = s.trading_summary(trade_bt, s.STRATEGY)
t_alpha = p["Alpha t statistic"]
sig = "significant at 5%" if abs(t_alpha) >= 1.96 else "not significant"


def card(label, value, compare_html):
    return (f'<div class="tf-metric"><div class="tf-label">{escape(label)}</div><div class="v">{value}</div>'
            f'<div class="c">{compare_html}</div></div>')


bname = escape(benchmark)
cards = [
    card("Average annual return", pct(p["Annualised average return"], 2),
         f'<span>{bname} {pct(b["Annualised average return"], 2)}</span>'
         + compare_pill(p["Annualised average return"] - b["Annualised average return"], True)),
    card("Volatility", pct(p["Annualised volatility"], 2),
         f'<span>{bname} {pct(b["Annualised volatility"], 2)}</span>'
         + compare_pill(p["Annualised volatility"] - b["Annualised volatility"], False)),
    card("Sharpe ratio", num(p["Sharpe ratio"]),
         f'<span>{bname} {num(b["Sharpe ratio"])}</span>'
         + compare_pill(p["Sharpe ratio"] - b["Sharpe ratio"], True, fmt=num)),
    card("Maximum drawdown", pct(p["Maximum drawdown"], 1),
         f'<span>{bname} {pct(b["Maximum drawdown"], 1)}</span>'
         + compare_pill(p["Maximum drawdown"] - b["Maximum drawdown"], True)),
    card("Alpha, annualised", pct(p["Alpha (annualised)"], 2),
         f'<span>t statistic {num(t_alpha)}</span>' + pill(sig, "neutral")),
    card("Beta", num(p["Beta"]), f'<span>Against {bname}, whose beta is 1.00</span>'),
    card("Information ratio", num(p["Information ratio"]),
         f'<span>Tracking error {pct(p["Tracking error"], 1)} a year</span>'),
    card("Time invested in ETFs", pct(trade["Average share invested in ETFs"], 0),
         f'<span>{trade["Signal switches per year"]:.1f} switches and {pct(trade["Annual trading cost"], 2)} '
         f'costs a year</span>'),
]
html('<div class="tf-grid">' + "".join(cards) + "</div>")

# ---------------------------------------------------------------- holdings and activity
HOLDING_KEY = "holding_month"


@st.fragment
def holdings_card(bt, data, window, use_filter):
    """Holdings in any month of the backtest. Only this card reruns when its month slider moves."""
    w_hist = bt.weights[s.STRATEGY]
    # every month of the backtest (Max window), plus the month after it, set at the last rebalance
    months = list(w_hist.index) + [w_hist.index[-1] + pd.offsets.MonthEnd(1)]
    labels = [f"{m:%b %Y}" for m in months]
    if st.session_state.get(HOLDING_KEY) not in labels:
        st.session_state[HOLDING_KEY] = labels[-1]
    with st.container(key="card_holdings"):
        html(f'<div class="tf-section">Holdings</div><div class="tf-caption">Move the slider to see what the '
             f'portfolio held in any month from {labels[0]} to {labels[-1]}. Positions are set at the rebalance on '
             f'the last day of the month before.</div>')
        choice = st.select_slider("Month", options=labels, key=HOLDING_KEY, label_visibility="collapsed")
        month = months[labels.index(choice)]
        rebalance = month - pd.offsets.MonthEnd(1)
        if month in w_hist.index:
            weights = w_hist.loc[month].to_dict()
        else:
            sig = bt.signals.loc[rebalance]
            weights = {a: (0.25 if (sig[a] == 1 or not use_filter) else 0.0) for a in s.RISKY_ASSETS}
            weights["CASH"] = 1 - sum(weights.values())
        sma_at, price_at, irx_at = bt.sma.loc[rebalance], data.prices.loc[rebalance], data.irx.loc[rebalance]
        latest = " (latest)" if month == months[-1] else ""
        d1, d2 = st.columns([1, 1.6], gap="small")
        with d1:
            keys = s.RISKY_ASSETS + ["CASH"]
            donut = go.Figure(go.Pie(labels=[a if a != "CASH" else "T-bills" for a in keys],
                                     values=[weights[a] for a in keys], hole=0.72, sort=False, direction="clockwise",
                                     marker=dict(colors=[ASSET_COLORS[a] for a in keys], line=dict(color=CARD, width=3)),
                                     textinfo="none", hovertemplate="%{label} %{value:.0%}<extra></extra>"))
            donut.add_annotation(text=f"<b>{pct(1 - weights['CASH'], 0)}</b><br><span style='font-size:12px;"
                                      f"color:{INK_2}'>invested</span>", showarrow=False,
                                 font=dict(size=22, color=INK, family=FONT))
            donut.update_layout(height=184, margin=dict(l=8, r=8, t=12, b=12), showlegend=False,
                                paper_bgcolor=CARD, font=dict(family=FONT))
            show(donut)
        with d2:
            n_held = sum(1 for a in s.RISKY_ASSETS if weights[a] > 0)
            html(f'<div class="tf-donut-text" style="padding-top:16px"><div class="tf-label">Held during '
                 f'{month:%B %Y}{latest}</div><div class="big">{n_held} of 4 ETFs</div><div class="tf-t2" '
                 f'style="margin-top:8px">Set at the rebalance on {rebalance:%d %b %Y}. The rest earns the Treasury '
                 f'bill rate, {irx_at:.2f}% a year at that date.</div></div>')
        rows = []
        for a in s.RISKY_ASSETS:
            name, desc = ASSET_INFO[a]
            color = ASSET_COLORS[a]
            dist = price_at[a] / sma_at[a] - 1
            status = pill("Invested 25%", "good", "\u25cf") if weights[a] > 0 else pill("In T-bills", "neutral")
            trend = f'<span class="tf-t2">{pct(dist, 1, sign=True)} vs trend</span>'
            spark = sparkline(data.prices[a].loc[:rebalance].tail(24), bt.sma[a].loc[:rebalance].tail(24), color)
            rows.append(f'<div class="tf-item"><div class="tf-avatar" style="background:{rgba(color, 0.16)};'
                        f'color:{color}">{a}</div><div><div class="tf-t1">{name}</div><div class="tf-t2">{a} \u00b7 '
                        f'{desc}</div></div>{spark}<div class="tf-right">{status}{trend}</div></div>')
        cash_color = ASSET_COLORS["CASH"]
        rows.append(f'<div class="tf-item"><div class="tf-avatar" style="background:{rgba(cash_color, 0.16)};'
                    f'color:{cash_color}">T</div><div><div class="tf-t1">Treasury bills</div><div class="tf-t2">'
                    f'3 month US government bills</div></div><div class="tf-spark"></div><div class="tf-right">'
                    f'{pill(pct(weights["CASH"], 0), "neutral")}<span class="tf-t2">{irx_at:.2f}% a year'
                    f'</span></div></div>')
        html("".join(rows))
        html(f'<div class="tf-caption" style="margin-top:8px">Sparklines show the 24 month end prices up to the '
             f'rebalance, with the {window} month moving average (the trend) dashed.</div>')


col_hold, col_act = st.columns([1.35, 1], gap="medium")
with col_hold:
    holdings_card(bt, data, window, use_filter)

with col_act:
    with st.container(key="card_activity"):
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
        html(f'<div class="tf-section">Recent trades</div><div class="tf-caption">{len(trades)} signal changes in '
             f'the selected period. Trades happen at month end prices.</div>')
        if trades:
            items = []
            for date, action, a in trades[::-1][:7]:
                color = ASSET_COLORS[a]
                arrow = "↗" if action == "Bought" else "↘"
                why = (f"Price rose above its {window} month average" if action == "Bought"
                       else f"Price fell below its {window} month average. 25% moved to T-bills")
                items.append(f'<div class="tf-act"><div class="tf-act-ic" style="background:{rgba(color, 0.16)};'
                             f'color:{color}">{arrow}</div><div><div class="tf-t1">{action} {a}</div>'
                             f'<div class="tf-t2">{why}</div></div><div class="tf-t2">{date:%d %b %Y}</div></div>')
            html("".join(items))
        else:
            html('<div class="tf-empty">No trades. Every ETF kept its weight during this period.</div>')

# ---------------------------------------------------------------- analysis views
with st.container(key="card_analysis"):
    html('<div class="tf-section">Analysis</div>')
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
        end_labels(fig, growth, ORDER, lambda n, y: f"{n} {y:.2f}")
        base_layout(fig, height=420, yformat=".2f", right_margin=128)
        fig.update_layout(legend=dict(traceorder="reversed"))
        fig.update_xaxes(range=[growth.index[0], growth.index[-1]])
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
        base_layout(fig, height=380, yformat=".0%")
        fig.update_layout(legend=dict(traceorder="reversed"))
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
        base_layout(fig, height=360, yformat=".0%")
        fig.update_layout(barmode="group", bargap=0.32, bargroupgap=0.1)
        fig.update_xaxes(showspikes=False)
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
            base_layout(fig, height=60 + 62 * len(stress), yformat=None)
            fig.update_layout(barmode="group", bargap=0.3, hovermode="y unified", legend=dict(traceorder="reversed"))
            fig.update_xaxes(tickformat=".0%", showgrid=True, gridcolor=GRID, showspikes=False)
            fig.update_yaxes(autorange="reversed", showgrid=False, side="left", tickfont=dict(color=INK, size=12))
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
        st.download_button("Download the monthly returns (CSV)", csv, "monthly_returns.csv", "text/csv")

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

**Source code.** {REPO_URL}
""")

html('<div class="tf-foot">Data from Yahoo Finance through the yfinance package. Monthly returns in US dollars, net of '
     'trading costs. Past performance does not predict future returns. Educational project for 2270 Financial '
     'Modeling at Nova SBE, not investment advice.</div>')
