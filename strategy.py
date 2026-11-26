# ==== SECTION: settings ====
"""
Trend Filtered Multi Asset Portfolio: shared backtest engine.

The Colab notebook and the Streamlit app run exactly this code. With the same
settings, the app therefore shows the same numbers as the notebook.

Rule in one sentence. Hold SPY, EFA, IEF and GLD at 25% each, but move any
asset whose month end price is below its 10 month simple moving average into
3 month US Treasury bills for the following month.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Investment universe (Yahoo Finance tickers)
RISKY_ASSETS = ["SPY", "EFA", "IEF", "GLD"]
RF_TICKER = "^IRX"  # 13 week US Treasury bill yield, in percent per year
ASSET_LABELS = {
    "SPY": "US equities (SPY)",
    "EFA": "Developed equities outside the US (EFA)",
    "IEF": "US Treasuries 7 to 10 years (IEF)",
    "GLD": "Gold (GLD)",
    "CASH": "Cash (3 month Treasury bills)",
}

# Sample period, set explicitly
DATA_START = "2004-01-01"      # extra history to compute the first moving averages (GLD starts November 2004)
BACKTEST_START = "2006-01-01"  # first month of portfolio returns
BACKTEST_END = "2025-12-31"    # last month of portfolio returns

# Strategy settings used in the prospectus
SMA_MONTHS = 10     # moving average window in months
COST_BPS = 10.0     # trading cost in basis points per unit of ETF value bought or sold
SIGNAL_LAG = 1      # signal at the end of month t sets the weights for month t+1
BASE_WEIGHTS = {"SPY": 0.25, "EFA": 0.25, "IEF": 0.25, "GLD": 0.25}

# Portfolio names used in every table and chart
STRATEGY = "Strategy"
BENCH_6040 = "60/40"
BENCH_EW = "Equal weight"
BENCH_SPY = "S&P 500"
BENCHMARK_WEIGHTS = {
    BENCH_6040: {"SPY": 0.60, "IEF": 0.40},  # classic balanced portfolio, rebalanced monthly
    BENCH_EW: dict(BASE_WEIGHTS),           # same four ETFs at 25%, rebalanced monthly, no filter
    BENCH_SPY: {"SPY": 1.00},               # US equity market, buy and hold
}
PRIMARY_BENCHMARK = BENCH_6040
PERIODS_PER_YEAR = 12


# ==== SECTION: data ====
def _extract_close(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Return one column of adjusted closing prices per ticker from a yfinance download."""
    if raw is None or raw.empty:
        raise RuntimeError("Yahoo Finance returned no data")
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = set(raw.columns.get_level_values(0))
        if "Close" in level0:
            close = raw["Close"]
        elif "Adj Close" in level0:
            close = raw["Adj Close"]
        else:  # columns grouped by ticker
            close = raw.xs("Close", axis=1, level=1)
    else:  # single ticker without a column MultiIndex
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})
    close = close.copy()
    close.index = pd.to_datetime(close.index)
    if close.index.tz is not None:
        close.index = close.index.tz_localize(None)
    close = close.sort_index()
    keep = [t for t in tickers if t in close.columns]
    return close[keep].astype(float)


def download_daily_prices(start: str = DATA_START, end: str = BACKTEST_END,
                          tickers: list[str] | None = None,
                          retries: int = 3, pause: float = 5.0) -> pd.DataFrame:
    """Download daily prices from the Yahoo Finance API through yfinance.

    auto_adjust=True returns closing prices adjusted for splits and dividends,
    so the returns computed from them are total returns.
    """
    import yfinance as yf

    tickers = list(tickers or RISKY_ASSETS + [RF_TICKER])
    end_exclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            raw = yf.download(tickers, start=start, end=end_exclusive, auto_adjust=True,
                              group_by="column", progress=False, threads=False)
            close = _extract_close(raw, tickers)
            missing = [t for t in tickers if t not in close.columns or close[t].dropna().empty]
            if missing:
                raise RuntimeError(f"no prices returned for {missing}")
            return close
        except Exception as err:  # network hiccups and rate limits are usually temporary
            last_error = err
            if attempt < retries:
                time.sleep(pause * attempt)
    raise RuntimeError(f"Yahoo Finance download failed after {retries} attempts. "
                       f"Wait a minute and run the cell again. Last error: {last_error}")


def to_month_end(df: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Last available observation of each calendar month."""
    try:
        return df.resample("ME").last()
    except ValueError:  # pandas older than 2.2
        return df.resample("M").last()


@dataclass
class MarketData:
    prices: pd.DataFrame   # month end adjusted prices of the four ETFs
    returns: pd.DataFrame  # monthly total returns of the four ETFs
    rf: pd.Series          # risk free return earned during each month
    irx: pd.Series         # month end 13 week Treasury bill yield, percent per year


def prepare_monthly_data(daily: pd.DataFrame) -> MarketData:
    """Turn daily prices into month end prices, monthly returns and a monthly risk free rate."""
    monthly = to_month_end(daily)
    prices = monthly[RISKY_ASSETS]
    returns = prices / prices.shift(1) - 1.0
    irx = monthly[RF_TICKER].astype(float)
    if irx.max() > 25:  # guard in case the yield is ever quoted at ten times its value
        irx = irx / 10.0
    # A Treasury bill bought at the end of month t-1 earns the yield known at that date during month t.
    rf = (irx / 100.0 / PERIODS_PER_YEAR).shift(1)
    rf.name = "RF"
    return MarketData(prices=prices, returns=returns, rf=rf, irx=irx)


def truncate_data(data: MarketData, last_date: str) -> MarketData:
    """Keep only the information available up to last_date (used to test for look ahead bias)."""
    return MarketData(prices=data.prices.loc[:last_date], returns=data.returns.loc[:last_date],
                      rf=data.rf.loc[:last_date], irx=data.irx.loc[:last_date])


# ==== SECTION: rules ====
def trend_signals(prices: pd.DataFrame, window: int = SMA_MONTHS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Signal = 1 when the month end price is above its moving average, 0 otherwise.

    The moving average uses the last `window` month end prices, including the
    current one. The signal is missing until enough history exists.
    """
    sma = prices.rolling(window=window, min_periods=window).mean()
    signal = (prices > sma).astype(float)
    signal = signal.where(sma.notna() & prices.notna())
    return signal, sma


def strategy_weights(signals: pd.DataFrame, base_weights: dict | None = None,
                     use_filter: bool = True, lag: int = SIGNAL_LAG) -> pd.DataFrame:
    """Target weights held during each month, including the cash weight.

    With lag=1 the weights for month t use only prices up to the end of month t-1.
    lag=0 would use the same month's closing price and is look ahead bias. It is
    used only once in the notebook to show how much that bias would flatter results.
    """
    base = pd.Series(base_weights or BASE_WEIGHTS, dtype=float).reindex(signals.columns).fillna(0.0)
    if use_filter:
        risky = signals.mul(base, axis=1)
    else:
        risky = pd.DataFrame(np.tile(base.values, (len(signals), 1)),
                             index=signals.index, columns=signals.columns)
    risky = risky.shift(lag)
    weights = risky.copy()
    weights["CASH"] = 1.0 - risky.sum(axis=1, min_count=len(risky.columns))
    return weights


def fixed_weights(index: pd.DatetimeIndex, weights: dict) -> pd.DataFrame:
    """Constant target weights (benchmarks), rebalanced every month."""
    w = pd.DataFrame(0.0, index=index, columns=RISKY_ASSETS)
    for asset, x in weights.items():
        w[asset] = float(x)
    w["CASH"] = 1.0 - w[RISKY_ASSETS].sum(axis=1)
    return w


# ==== SECTION: engine ====
def simulate(weights: pd.DataFrame, asset_returns: pd.DataFrame, rf: pd.Series,
             cost_bps: float = COST_BPS) -> pd.DataFrame:
    """Monthly returns of a portfolio rebalanced to `weights` at every month end.

    The investor starts in cash. At each month end the portfolio trades from its
    drifted weights back to the target weights. Trading costs equal cost_bps times
    the value of ETFs bought and sold. Moving money in or out of Treasury bills
    is assumed free. Costs are paid at the start of the month.
    """
    cols = RISKY_ASSETS + ["CASH"]
    W = weights.reindex(columns=cols).astype(float)
    R = asset_returns.reindex(index=W.index, columns=RISKY_ASSETS).copy()
    R["CASH"] = rf.reindex(W.index)
    if W.isna().any().any():
        raise ValueError("Missing target weights. Not enough price history before the start date.")
    if R.isna().any().any():
        raise ValueError("Missing returns inside the backtest window.")
    w_arr, r_arr = W.to_numpy(), R.to_numpy()
    n_risky = len(RISKY_ASSETS)
    c = cost_bps / 10_000.0
    w_prev = np.zeros(len(cols))
    w_prev[-1] = 1.0  # start fully in cash
    rows = []
    for w, r in zip(w_arr, r_arr):
        turnover = float(np.abs(w[:n_risky] - w_prev[:n_risky]).sum())
        cost = c * turnover
        gross = float(w @ r)
        net = (1.0 + gross) * (1.0 - cost) - 1.0
        w_prev = w * (1.0 + r) / (1.0 + gross)  # weights drift with returns until next rebalance
        rows.append((gross, net, turnover, cost, float(w[:n_risky].sum())))
    return pd.DataFrame(rows, index=W.index, columns=["gross", "net", "turnover", "cost", "invested"])


@dataclass
class BacktestResult:
    returns: pd.DataFrame                          # monthly net returns, one column per portfolio
    rf: pd.Series                                  # monthly risk free rate over the same months
    details: dict = field(default_factory=dict)    # gross, net, turnover, cost and invested share
    weights: dict = field(default_factory=dict)    # target weights per portfolio
    signals: pd.DataFrame | None = None            # month end trend signals of the four ETFs
    sma: pd.DataFrame | None = None                # month end moving averages
    settings: dict = field(default_factory=dict)


def run_backtest(data: MarketData, sma_months: int = SMA_MONTHS, cost_bps: float = COST_BPS,
                 use_filter: bool = True, start: str = BACKTEST_START, end: str = BACKTEST_END,
                 lag: int = SIGNAL_LAG) -> BacktestResult:
    """Run the strategy and the three benchmarks over the same months."""
    signals, sma = trend_signals(data.prices, sma_months)
    w_strategy = strategy_weights(signals, BASE_WEIGHTS, use_filter=use_filter, lag=lag)
    index = data.returns.loc[start:end].index
    weights = {STRATEGY: w_strategy.reindex(index)}
    for name, bw in BENCHMARK_WEIGHTS.items():
        weights[name] = fixed_weights(index, bw)
    details = {name: simulate(w, data.returns, data.rf, cost_bps) for name, w in weights.items()}
    returns = pd.DataFrame({name: d["net"] for name, d in details.items()})
    settings = dict(sma_months=sma_months, cost_bps=cost_bps, use_filter=use_filter,
                    start=str(index[0].date()), end=str(index[-1].date()), lag=lag)
    return BacktestResult(returns=returns, rf=data.rf.reindex(index), details=details,
                          weights=weights, signals=signals, sma=sma, settings=settings)


# ==== SECTION: metrics ====
def growth_of_one(returns: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Cumulative value of 1 invested at the month end before the first return."""
    wealth = (1.0 + returns).cumprod()
    start = returns.index[0] - pd.offsets.MonthEnd(1)
    if isinstance(returns, pd.Series):
        return pd.concat([pd.Series([1.0], index=[start], name=returns.name), wealth])
    first = pd.DataFrame(1.0, index=[start], columns=returns.columns)
    return pd.concat([first, wealth])


def drawdowns(returns: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Percentage loss from the previous peak of the cumulative value (starting value counts as a peak)."""
    wealth = (1.0 + returns).cumprod()
    peak = wealth.cummax().clip(lower=1.0)
    return wealth / peak - 1.0


def ols(y: np.ndarray, x: np.ndarray) -> dict:
    """Ordinary least squares of y on a constant and x, with classic standard errors."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    n = len(y)
    X = np.column_stack([np.ones(n), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    s2 = float(resid @ resid) / (n - 2)
    cov = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    tss = float((y - y.mean()) @ (y - y.mean()))
    return {"alpha": coef[0], "beta": coef[1], "t_alpha": coef[0] / se[0], "t_beta": coef[1] / se[1],
            "r2": 1.0 - float(resid @ resid) / tss if tss > 0 else np.nan, "resid_std": np.sqrt(s2)}


METRIC_ORDER = [
    "Annualised average return", "Annualised volatility", "Sharpe ratio",
    "Alpha (annualised)", "Alpha t statistic", "Beta", "Tracking error", "Information ratio",
    "CAGR", "Maximum drawdown",
]


def performance_table(returns: pd.DataFrame, rf: pd.Series,
                      benchmark: str = PRIMARY_BENCHMARK) -> pd.DataFrame:
    """Performance measures for every portfolio. Alpha, beta and IR are measured against `benchmark`.

    Annualised average return = 12 x mean monthly return
    Annualised volatility     = sqrt(12) x standard deviation of monthly returns
    Sharpe ratio              = sqrt(12) x mean / standard deviation of monthly excess returns
    Alpha, beta               = OLS of monthly portfolio excess returns on benchmark excess returns,
                                alpha x 12, beta not annualised
    Information ratio         = 12 x mean active return / (sqrt(12) x standard deviation of active return)
    """
    k = PERIODS_PER_YEAR
    rf = rf.reindex(returns.index)
    bench = returns[benchmark]
    table = {}
    for name in returns.columns:
        r = returns[name]
        ex = r - rf
        row = {
            "Annualised average return": k * r.mean(),
            "Annualised volatility": np.sqrt(k) * r.std(ddof=1),
            "Sharpe ratio": np.sqrt(k) * ex.mean() / ex.std(ddof=1),
            "CAGR": float((1.0 + r).prod()) ** (k / len(r)) - 1.0,
            "Maximum drawdown": float(drawdowns(r).min()),
        }
        if name == benchmark:
            row.update({"Alpha (annualised)": 0.0, "Alpha t statistic": np.nan, "Beta": 1.0,
                        "Tracking error": 0.0, "Information ratio": np.nan})
        else:
            reg = ols(ex.to_numpy(), (bench - rf).to_numpy())
            active = r - bench
            te = np.sqrt(k) * active.std(ddof=1)
            row.update({"Alpha (annualised)": k * reg["alpha"], "Alpha t statistic": reg["t_alpha"],
                        "Beta": reg["beta"], "Tracking error": te,
                        "Information ratio": k * active.mean() / te if te > 0 else np.nan})
        table[name] = row
    return pd.DataFrame(table).reindex(METRIC_ORDER)


RATIO_METRICS = {"Sharpe ratio", "Alpha t statistic", "Beta", "Information ratio", "Correlation"}


def format_metrics(table: pd.DataFrame, digits: int = 2) -> pd.DataFrame:
    """Readable copy of a metrics table. Ratios as numbers, everything else as percentages."""
    def fmt(row, v):
        if pd.isna(v):
            return "n/a"
        if row in RATIO_METRICS:
            return f"{round(v, digits) + 0.0:.{digits}f}"
        return f"{round(v * 100, digits) + 0.0:.{digits}f}%"  # + 0.0 turns -0.0 into 0.0
    return pd.DataFrame({c: [fmt(r, table.loc[r, c]) for r in table.index] for c in table.columns},
                        index=table.index)


def format_percent(table: pd.DataFrame, digits: int = 1) -> pd.DataFrame:
    """Readable copy of a table of returns."""
    return table.apply(lambda col: col.map(
        lambda v: "n/a" if pd.isna(v) else f"{round(v * 100, digits) + 0.0:.{digits}f}%"))


def relative_table(returns: pd.DataFrame, rf: pd.Series, portfolio: str = STRATEGY) -> pd.DataFrame:
    """Alpha, beta and information ratio of one portfolio against each of the other portfolios."""
    rows = {}
    for bench in returns.columns:
        if bench == portfolio:
            continue
        t = performance_table(returns[[portfolio, bench]], rf, benchmark=bench)[portfolio]
        corr = returns[portfolio].corr(returns[bench])
        rows[f"vs {bench}"] = {"Alpha (annualised)": t["Alpha (annualised)"],
                               "Alpha t statistic": t["Alpha t statistic"], "Beta": t["Beta"],
                               "Correlation": corr, "Tracking error": t["Tracking error"],
                               "Information ratio": t["Information ratio"]}
    return pd.DataFrame(rows)


def trading_summary(bt: BacktestResult, name: str = STRATEGY) -> dict:
    """Average invested share, annual turnover, annual cost and number of switches."""
    d = bt.details[name]
    years = len(d) / PERIODS_PER_YEAR
    after_first = d.iloc[1:]  # the first month includes the initial purchase
    w = bt.weights[name][RISKY_ASSETS]
    switches = int((w.diff().iloc[1:].abs() > 0.05).sum().sum()) if name == STRATEGY else 0
    return {
        "Average share invested in ETFs": d["invested"].mean(),
        "Months fully in cash": int((d["invested"] < 1e-9).sum()),
        "Annual turnover (ETF trades, % of portfolio)": after_first["turnover"].sum() / years,
        "Annual trading cost": after_first["cost"].sum() / years,
        "Signal switches per year": switches / years,
    }


def calendar_year_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Compounded return of each calendar year."""
    return (1.0 + returns).groupby(returns.index.year).prod() - 1.0


STRESS_PERIODS = {
    "Global financial crisis (Nov 2007 to Feb 2009)": ("2007-11-01", "2009-02-28"),
    "Rebound after the crisis (Mar 2009 to Dec 2009)": ("2009-03-01", "2009-12-31"),
    "Euro area debt crisis (May 2011 to Sep 2011)": ("2011-05-01", "2011-09-30"),
    "Equity sell off (Oct 2018 to Dec 2018)": ("2018-10-01", "2018-12-31"),
    "Covid crash (Feb 2020 to Mar 2020)": ("2020-02-01", "2020-03-31"),
    "Covid rebound (Apr 2020 to Dec 2020)": ("2020-04-01", "2020-12-31"),
    "Inflation and rate shock (Jan 2022 to Dec 2022)": ("2022-01-01", "2022-12-31"),
}


def period_returns(returns: pd.DataFrame, periods: dict = STRESS_PERIODS) -> pd.DataFrame:
    """Compounded return of each portfolio inside each period that falls in the sample."""
    rows = {}
    for label, (a, b) in periods.items():
        chunk = returns.loc[a:b]
        expected_months = len(pd.period_range(a, b, freq="M"))
        if len(chunk) != expected_months:  # period not fully inside the sample
            continue
        rows[label] = (1.0 + chunk).prod() - 1.0
    return pd.DataFrame(rows, index=returns.columns).T if rows else pd.DataFrame(columns=returns.columns)


def rolling_beta(returns: pd.DataFrame, rf: pd.Series, portfolio: str = STRATEGY,
                 benchmark: str = PRIMARY_BENCHMARK, window: int = 36) -> pd.Series:
    """Beta of the portfolio against the benchmark over a rolling window of months."""
    y = returns[portfolio] - rf
    x = returns[benchmark] - rf
    return (y.rolling(window).cov(x) / x.rolling(window).var()).dropna()


# ==== SECTION: analysis ====
def window_sensitivity(data: MarketData, windows=range(2, 13), cost_bps: float = COST_BPS,
                       start: str = BACKTEST_START, end: str = BACKTEST_END,
                       benchmark: str = PRIMARY_BENCHMARK) -> pd.DataFrame:
    """Strategy results for different moving average windows (checks how much the 10 month choice matters)."""
    rows = {}
    for n in windows:
        bt = run_backtest(data, sma_months=n, cost_bps=cost_bps, start=start, end=end)
        perf = performance_table(bt.returns, bt.rf, benchmark)[STRATEGY]
        trade = trading_summary(bt)
        rows[n] = {
            "Annualised average return": perf["Annualised average return"],
            "Annualised volatility": perf["Annualised volatility"],
            "Sharpe ratio": perf["Sharpe ratio"],
            "Alpha (annualised)": perf["Alpha (annualised)"],
            "Information ratio": perf["Information ratio"],
            "Maximum drawdown": perf["Maximum drawdown"],
            "Annual turnover (ETF trades, % of portfolio)": trade["Annual turnover (ETF trades, % of portfolio)"],
        }
    out = pd.DataFrame(rows).T
    out.index.name = "Moving average window (months)"
    return out


def cost_sensitivity(data: MarketData, costs=(0, 5, 10, 25, 50, 100), sma_months: int = SMA_MONTHS,
                     start: str = BACKTEST_START, end: str = BACKTEST_END,
                     benchmark: str = PRIMARY_BENCHMARK) -> pd.DataFrame:
    """Strategy and benchmark results for different trading cost assumptions."""
    rows = {}
    for c in costs:
        bt = run_backtest(data, sma_months=sma_months, cost_bps=c, start=start, end=end)
        perf = performance_table(bt.returns, bt.rf, benchmark)
        rows[c] = {
            "Strategy average return": perf.loc["Annualised average return", STRATEGY],
            "Strategy Sharpe ratio": perf.loc["Sharpe ratio", STRATEGY],
            "Strategy alpha": perf.loc["Alpha (annualised)", STRATEGY],
            "Strategy information ratio": perf.loc["Information ratio", STRATEGY],
            f"{benchmark} average return": perf.loc["Annualised average return", benchmark],
        }
    out = pd.DataFrame(rows).T
    out.index.name = "Trading cost (basis points)"
    return out


def check_no_lookahead(data: MarketData, cutoff: str = "2015-06-30", sma_months: int = SMA_MONTHS) -> bool:
    """Weights up to `cutoff` must be identical when all prices after `cutoff` are deleted."""
    full, _ = trend_signals(data.prices, sma_months)
    w_full = strategy_weights(full)
    cut = truncate_data(data, cutoff)
    part, _ = trend_signals(cut.prices, sma_months)
    w_part = strategy_weights(part)
    common = w_part.index
    a = w_full.loc[common].to_numpy()
    b = w_part.loc[common].to_numpy()
    return bool(np.allclose(a, b, equal_nan=True))
