"""
Lean Hogs Futures Trading Model
=================================
A quantitative trading platform for CME Lean Hogs (HE) futures.

Tabs:
  1. Market Overview & Fair Value
  2. Signals & Seasonality
  3. Trading Strategies (Gaussian Breakout, MA, Mean Reversion)
  4. ML Weight Optimizer
  5. Backtest & Monte Carlo
  6. Futures Curve & Spread Analysis
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf
from scipy.ndimage import gaussian_filter1d
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

# ============================================================
# CONSTANTS
# ============================================================

TICKERS = {"hogs": "HE=F", "corn": "ZC=F", "meal": "ZM=F"}
CONTRACT_SIZE = 40_000      # lbs per lean hog contract
TICK_SIZE = 0.025           # cents/lb
TICK_VALUE = 10.0           # $ per tick
RISK_FREE_RATE = 0.05       # annualized

# Lean hogs only trade certain months
LEAN_HOG_MONTHS = {2: "G", 4: "J", 5: "K", 6: "M", 7: "N", 8: "Q", 10: "V", 12: "Z"}

# Seasonal premia by month (historical pattern, $/cwt relative to annual avg)
SEASONAL_PREMIA = {
    1: -4.2, 2: -2.8, 3: 0.5, 4: 3.8, 5: 6.5, 6: 7.2,
    7: 5.8, 8: 2.4, 9: -1.8, 10: -5.2, 11: -7.0, 12: -5.2
}

# ============================================================
# SECTION 1: DATA LAYER
# ============================================================

@st.cache_data(ttl=3600)
def load_market_data(lookback_days: int = 756, force_synthetic: bool = False) -> dict:
    """
    Returns dict keyed by 'hogs', 'corn', 'meal'.
    Each value is a DataFrame with DatetimeIndex and OHLCV columns.
    Falls back to synthetic data on failure.
    """
    result = {}
    for name, ticker in TICKERS.items():
        if force_synthetic:
            result[name] = _synthetic_ohlcv(
                n=lookback_days,
                start_price={"hogs": 85.0, "corn": 450.0, "meal": 320.0}[name],
                annual_vol={"hogs": 0.25, "corn": 0.22, "meal": 0.28}[name],
                seed={"hogs": 42, "corn": 7, "meal": 13}[name],
            )
            continue
        try:
            df = yf.Ticker(ticker).history(
                period=f"{max(lookback_days // 252 + 1, 2)}y",
                auto_adjust=False,
            )
            if df.empty or len(df) < 50:
                raise ValueError("Insufficient data")
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df = df.dropna(subset=["Close"]).tail(lookback_days)
            result[name] = df
        except Exception:
            result[name] = _synthetic_ohlcv(
                n=lookback_days,
                start_price={"hogs": 85.0, "corn": 450.0, "meal": 320.0}[name],
                annual_vol={"hogs": 0.25, "corn": 0.22, "meal": 0.28}[name],
                seed={"hogs": 42, "corn": 7, "meal": 13}[name],
            )
    return result


def _synthetic_ohlcv(
    n: int = 756,
    start_price: float = 85.0,
    annual_vol: float = 0.25,
    seed: int = 42,
) -> pd.DataFrame:
    """Geometric Brownian Motion with realistic OHLCV structure."""
    rng = np.random.default_rng(seed)
    dt = 1 / 252
    daily_vol = annual_vol * np.sqrt(dt)
    drift = 0.0  # futures have no drift component

    returns = rng.normal(drift, daily_vol, n)
    close = start_price * np.cumprod(1 + returns)

    # Add seasonal pattern for hogs
    if 55 < start_price < 150:  # hog price range
        dates = pd.bdate_range(end=datetime.today(), periods=n)
        seasonal = np.array([SEASONAL_PREMIA[d.month] for d in dates])
        close = close + seasonal * 0.3

    dates = pd.bdate_range(end=datetime.today(), periods=n)
    intraday_range = np.abs(rng.normal(0, daily_vol * 0.5, n))
    high = close * (1 + intraday_range)
    low = close * (1 - intraday_range)
    open_p = np.roll(close, 1)
    open_p[0] = close[0]
    volume = rng.integers(5_000, 60_000, n)

    return pd.DataFrame(
        {"Open": open_p, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


# ============================================================
# SECTION 2: INDICATOR TOOLKIT (FULLY VECTORIZED)
# ============================================================

def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mu = series.rolling(window, min_periods=window // 2).mean()
    sigma = series.rolling(window, min_periods=window // 2).std(ddof=1).clip(lower=1e-8)
    return (series - mu) / sigma


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def causal_gaussian_smooth(series: pd.Series, sigma: int = 10) -> pd.Series:
    """
    Causal Gaussian smooth via one-sided convolution — no look-ahead.
    Kernel covers 4*sigma bars to the left only.
    """
    w = int(4 * sigma)
    x = np.arange(w + 1)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    # Flip: kernel[0] = weight on current bar, kernel[w] = oldest bar
    kernel_flipped = kernel[::-1]
    vals = series.values.astype(float)
    result = np.full_like(vals, np.nan)
    for i in range(w, len(vals)):
        result[i] = np.dot(vals[i - w: i + 1], kernel_flipped)
    return pd.Series(result, index=series.index)


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    rs = gain / loss.clip(lower=1e-8)
    return 100 - 100 / (1 + rs)


def compute_bollinger(
    close: pd.Series, window: int = 20, n_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = close.rolling(window).mean()
    std = close.rolling(window).std(ddof=1)
    return mid - n_std * std, mid, mid + n_std * std


# ============================================================
# SECTION 3: SIGNAL GENERATORS
# ============================================================

def gaussian_breakout_signal(
    close: pd.Series,
    atr: pd.Series,
    bandwidth: int = 20,
    threshold_k: float = 1.5,
) -> pd.Series:
    """
    Causal Gaussian Breakout.
    +1 when close breaks above smooth + k*rolling_std
    -1 when close breaks below smooth - k*rolling_std
    Continuous signal = tanh(z_score / k) for position sizing.
    """
    smooth = causal_gaussian_smooth(close, sigma=bandwidth)
    rolling_std = close.rolling(bandwidth).std(ddof=1).clip(lower=1e-8)
    z_score = (close - smooth) / rolling_std
    # Continuous signal for position sizing
    signal = np.tanh(z_score / threshold_k)
    return signal.rename("gb_signal")


def ma_crossover_signal(
    close: pd.Series, fast: int = 10, slow: int = 40, smooth: int = 3
) -> pd.Series:
    """
    EMA crossover + momentum confirmation.
    Returns continuous signal in [-1, 1].
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    spread = ema_fast - ema_slow
    spread_norm = spread / close.rolling(slow).std(ddof=1).clip(lower=1e-8)
    signal = np.tanh(spread_norm.ewm(span=smooth, adjust=False).mean())
    return signal.rename("ma_signal")


def mean_reversion_signal(
    close: pd.Series, window: int = 20, entry_z: float = 1.5
) -> pd.Series:
    """
    Z-score based mean reversion (stateful — uses loop for path-dependency).
    Returns +1 (long), -1 (short), 0 (flat) position array.
    """
    zscores = rolling_zscore(close, window).values
    n = len(zscores)
    signals = np.zeros(n)
    position = 0
    for i in range(1, n):
        z = zscores[i]
        if np.isnan(z):
            signals[i] = 0
            continue
        if position == 0:
            if z < -entry_z:
                position = 1    # oversold — go long
            elif z > entry_z:
                position = -1   # overbought — go short
        elif position == 1 and z > -0.3:
            position = 0        # mean-revert exit
        elif position == -1 and z < 0.3:
            position = 0
        signals[i] = position
    return pd.Series(signals, index=close.index, name="mr_signal")


def seasonality_signal(close: pd.Series, years_lookback: int = 5) -> pd.Series:
    """
    For each week-of-year, compute fraction of historical years showing positive returns.
    Signal = 2*(fraction - 0.5) so range is [-1, +1].
    """
    df = pd.DataFrame({"close": close})
    df["week"] = df.index.isocalendar().week.astype(int)
    df["year"] = df.index.year
    df["weekly_ret"] = df["close"].pct_change(5)  # approx 1-week return

    cutoff = df.index.max() - pd.DateOffset(years=years_lookback)
    hist = df[df.index < cutoff]

    week_bull_frac = (
        hist.groupby("week")["weekly_ret"]
        .apply(lambda x: (x > 0).mean())
        .rename("bull_frac")
    )
    df = df.join(week_bull_frac, on="week")
    signal = 2 * (df["bull_frac"].fillna(0.5) - 0.5)
    return signal.rename("seasonal_signal")


def supply_demand_signal(
    close: pd.Series,
    corn: pd.Series | None = None,
    meal: pd.Series | None = None,
) -> pd.Series:
    """
    Proxy supply/demand signal from feed cost margins.
    High feed cost relative to hog price → bearish margin signal.
    """
    if corn is not None and meal is not None:
        corn_r = corn.reindex(close.index).ffill()
        meal_r = meal.reindex(close.index).ffill()
        # Corn ZC=F is in cents/bushel → /100 for dollars
        feed_cost_proxy = (corn_r / 100 * 6.5 + meal_r / 2000 * 1.5 * 2000) / 100
        margin = close - feed_cost_proxy
        signal = rolling_zscore(margin, 60).clip(-2, 2) / 2
    else:
        # Use price vs 120-day moving average as proxy
        ma_120 = close.rolling(120).mean()
        signal = rolling_zscore(close - ma_120, 40).clip(-2, 2) / 2
    return signal.rename("sd_signal")


def build_signal_matrix(
    close: pd.Series,
    corn_close: pd.Series | None = None,
    meal_close: pd.Series | None = None,
    gb_bandwidth: int = 20,
    gb_k: float = 1.5,
    ma_fast: int = 10,
    ma_slow: int = 40,
    mr_window: int = 20,
    mr_z: float = 1.5,
) -> pd.DataFrame:
    """Build all signals into a single DataFrame."""
    high = close.copy()  # fallback if OHLC not available
    low = close.copy()
    atr = compute_atr(high, low, close, 14)

    gb = gaussian_breakout_signal(close, atr, gb_bandwidth, gb_k)
    ma = ma_crossover_signal(close, ma_fast, ma_slow)
    mr = mean_reversion_signal(close, mr_window, mr_z)
    seas = seasonality_signal(close)
    sd = supply_demand_signal(close, corn_close, meal_close)
    rsi = compute_rsi(close).rename("rsi")
    zs = rolling_zscore(close, mr_window).rename("zscore_20")

    return pd.concat([gb, ma, mr, seas, sd, rsi, zs], axis=1).fillna(0)


# ============================================================
# SECTION 4: FAIR VALUE MODEL
# ============================================================

def compute_fair_value(
    hogs_index: pd.DatetimeIndex,
    corn_close: pd.Series | None = None,
    meal_close: pd.Series | None = None,
    other_costs: float = 35.0,
    margin_pct: float = 0.08,
) -> pd.Series:
    """
    Cost-of-production fair value for lean hogs ($/cwt).
    Feed conversion: 2.8 lbs feed per lb gain, 60% corn / 15% meal inclusion.
    Slaughter weight: 265 lbs live, 75% dressing %.
    """
    n = len(hogs_index)

    if corn_close is not None and meal_close is not None:
        # ZC=F is in cents/bushel
        corn_dol = corn_close.reindex(hogs_index).ffill() / 100.0  # $/bushel
        meal_dol = meal_close.reindex(hogs_index).ffill()          # $/ton
    else:
        # Simulate mean feed prices with slow cycle
        t = np.arange(n)
        corn_dol = 4.5 + 0.8 * np.sin(2 * np.pi * t / (252 * 3))
        meal_dol = 320 + 40 * np.cos(2 * np.pi * t / (252 * 2))
        corn_dol = pd.Series(corn_dol, index=hogs_index)
        meal_dol = pd.Series(meal_dol, index=hogs_index)

    slaughter_wt = 265.0    # lbs live weight
    dressing = 0.75
    feed_conv = 2.8         # lbs feed per lb of live gain
    corn_frac = 0.60
    meal_frac = 0.15

    lbs_feed = slaughter_wt * feed_conv
    corn_cost_head = corn_dol * (lbs_feed * corn_frac / 56)   # 56 lbs/bushel
    meal_cost_head = meal_dol * (lbs_feed * meal_frac / 2000) # 2000 lbs/ton
    total_feed_head = corn_cost_head + meal_cost_head

    cwt_produced = (slaughter_wt * dressing) / 100.0
    feed_cost_cwt = total_feed_head / cwt_produced
    fair_value = (feed_cost_cwt + other_costs) * (1 + margin_pct)

    # Seasonal adjustment
    seasonal_adj = pd.Series(
        [SEASONAL_PREMIA[d.month] for d in hogs_index], index=hogs_index
    )
    fair_value = fair_value + seasonal_adj

    return fair_value.ewm(span=10).mean().rename("fair_value")


# ============================================================
# SECTION 5: ML WALK-FORWARD WEIGHT OPTIMIZER
# ============================================================

@st.cache_data(ttl=3600)
def walk_forward_optimize(
    signal_matrix_values: np.ndarray,
    signal_names: list,
    signal_index: list,
    returns_values: np.ndarray,
    train_window: int = 252,
    test_window: int = 63,
) -> pd.DataFrame:
    """
    Walk-forward Random Forest optimizer.
    Trains on past train_window days, predicts weights for next test_window.
    Returns DataFrame with ensemble_signal and feature importances over time.
    """
    signal_matrix = pd.DataFrame(
        signal_matrix_values, index=pd.DatetimeIndex(signal_index), columns=signal_names
    )
    returns = pd.Series(returns_values, index=pd.DatetimeIndex(signal_index))

    core_signals = [c for c in ["gb_signal", "ma_signal", "mr_signal"] if c in signal_names]
    weight_cols = [c.replace("_signal", "_weight") for c in core_signals]

    n = len(signal_matrix)
    ensemble = pd.Series(np.zeros(n), index=signal_matrix.index, name="ensemble_signal")
    weights_history = pd.DataFrame(
        np.nan, index=signal_matrix.index, columns=weight_cols
    )

    if n < train_window + test_window:
        # Fallback: equal weights
        if core_signals:
            for wc in weight_cols:
                weights_history[wc] = 1.0 / len(core_signals)
            ensemble = signal_matrix[core_signals].mean(axis=1)
            ensemble.name = "ensemble_signal"
        return pd.concat([ensemble, weights_history], axis=1)

    scaler = StandardScaler()
    feature_cols = [c for c in signal_matrix.columns if c not in ["rsi"]]

    for t in range(train_window, n, test_window):
        X_train = signal_matrix.iloc[t - train_window: t][feature_cols].values
        y_train = (returns.iloc[t - train_window: t] > 0).astype(int).values

        t_end = min(t + test_window, n)
        X_test = signal_matrix.iloc[t: t_end][feature_cols].values

        X_tr_sc = scaler.fit_transform(X_train)
        X_te_sc = scaler.transform(X_test)

        rf = RandomForestClassifier(
            n_estimators=80, max_depth=4, min_samples_leaf=15,
            random_state=42, n_jobs=1
        )
        rf.fit(X_tr_sc, y_train)

        # Predicted probability of positive return
        prob_up = rf.predict_proba(X_te_sc)[:, 1]
        # Convert to [-1, 1] signal
        ens = 2 * prob_up - 1

        # Blend with individual signal weighted by feature importance
        importance = dict(zip(feature_cols, rf.feature_importances_))
        total_imp = sum(importance.get(c, 0) for c in core_signals) + 1e-8
        w = {c: importance.get(c, 0) / total_imp for c in core_signals}

        # Weighted combination of core signals
        period_signals = signal_matrix.iloc[t: t_end][core_signals]
        blended = sum(w[c] * period_signals[c] for c in core_signals)

        # Final ensemble = RF probability blend + weighted signal blend
        final = 0.5 * ens + 0.5 * blended.values
        ensemble.iloc[t: t_end] = final
        for c, wc in zip(core_signals, weight_cols):
            weights_history.iloc[t: t_end][wc] = w[c]

    weights_history = weights_history.ffill().fillna(1.0 / max(len(core_signals), 1))
    return pd.concat([ensemble, weights_history], axis=1)


# ============================================================
# SECTION 6: BACKTEST ENGINE
# ============================================================

def vectorized_backtest(
    signal: pd.Series,
    close: pd.Series,
    transaction_cost_bps: float = 10.0,
) -> dict:
    """
    Vectorized backtest with look-ahead prevention via signal.shift(1).
    Returns equity curve, daily returns, and performance metrics.
    """
    signal = signal.reindex(close.index).fillna(0)
    price_ret = close.pct_change().fillna(0)

    # CRITICAL: shift signal by 1 to avoid look-ahead bias
    position = signal.shift(1).fillna(0)
    tc_per_trade = transaction_cost_bps / 10_000
    turnover = position.diff().abs().fillna(0)

    strat_ret = position * price_ret - turnover * tc_per_trade
    buy_hold_ret = price_ret.copy()

    equity_strat = (1 + strat_ret).cumprod()
    equity_bh = (1 + buy_hold_ret).cumprod()

    metrics = _compute_metrics(strat_ret, buy_hold_ret)
    return {
        "equity_strategy": equity_strat,
        "equity_buyhold": equity_bh,
        "daily_returns": strat_ret,
        "position": position,
        "metrics": metrics,
    }


def _compute_metrics(strat_ret: pd.Series, bh_ret: pd.Series) -> dict:
    ann = 252
    s = strat_ret.dropna()
    if len(s) < 20:
        return {}

    cagr = (1 + s).prod() ** (ann / len(s)) - 1
    vol = s.std() * np.sqrt(ann)
    sharpe = (cagr - RISK_FREE_RATE) / (vol + 1e-8)

    downside = s[s < 0].std() * np.sqrt(ann)
    sortino = (cagr - RISK_FREE_RATE) / (downside + 1e-8)

    cum = (1 + s).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()
    max_dd = dd.min()
    calmar = cagr / (-max_dd + 1e-8)

    win_rate = (s > 0).mean()
    profit_factor = s[s > 0].sum() / (-s[s < 0].sum() + 1e-8)

    bh_cagr = (1 + bh_ret).prod() ** (ann / len(bh_ret)) - 1

    return {
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max Drawdown": max_dd,
        "Calmar": calmar,
        "Win Rate": win_rate,
        "Profit Factor": profit_factor,
        "B&H CAGR": bh_cagr,
    }


# ============================================================
# SECTION 7: MONTE CARLO / PERMUTATION TEST
# ============================================================

def permutation_test(
    strategy_returns: pd.Series,
    n_permutations: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Checks whether strategy Sharpe is significantly better than random.
    Uses block bootstrap (block=20 days) to preserve autocorrelation.
    """
    rng = np.random.default_rng(seed)
    ret = strategy_returns.dropna().values
    n = len(ret)
    if n < 50:
        return {}

    def sharpe(r):
        mu = r.mean() * 252
        sd = r.std() * np.sqrt(252)
        return (mu - RISK_FREE_RATE) / (sd + 1e-8)

    observed = sharpe(ret)

    # Block bootstrap
    block_size = 20
    null_dist = np.empty(n_permutations)
    for i in range(n_permutations):
        n_blocks = (n // block_size) + 2
        starts = rng.integers(0, max(n - block_size, 1), size=n_blocks)
        blocks = [ret[s: s + block_size] for s in starts]
        shuffled = np.concatenate(blocks)[:n]
        null_dist[i] = sharpe(shuffled)

    p_value = (null_dist >= observed).mean()
    return {
        "observed_sharpe": observed,
        "null_distribution": null_dist,
        "p_value": p_value,
        "percentile": 1 - p_value,
        "significant": p_value < 0.05,
        "null_median": np.median(null_dist),
        "null_95th": np.percentile(null_dist, 95),
    }


def monte_carlo_equity_paths(
    daily_returns: pd.Series,
    n_simulations: int = 500,
    horizon_days: int = 252,
    seed: int = 42,
) -> np.ndarray:
    """
    Block bootstrap equity paths. Returns (n_simulations x horizon_days) array.
    """
    rng = np.random.default_rng(seed)
    ret = daily_returns.dropna().values
    n = len(ret)
    block_size = 20
    n_blocks = (horizon_days // block_size) + 2
    max_start = max(n - block_size, 1)

    starts = rng.integers(0, max_start, size=(n_simulations, n_blocks))
    paths = np.zeros((n_simulations, horizon_days))
    for i in range(n_simulations):
        blocks = np.concatenate([ret[s: s + block_size] for s in starts[i]])
        path_ret = blocks[:horizon_days]
        paths[i] = np.cumprod(1 + path_ret)
    return paths


# ============================================================
# SECTION 8: FUTURES CURVE & SPREAD ANALYSIS
# ============================================================

def get_futures_curve_tickers(n_contracts: int = 8) -> list[str]:
    """Generate next N active lean hog contract tickers."""
    now = datetime.now()
    tickers = []
    month, year = now.month, now.year
    while len(tickers) < n_contracts:
        month += 1
        if month > 12:
            month = 1
            year += 1
        if month in LEAN_HOG_MONTHS:
            code = LEAN_HOG_MONTHS[month]
            yr_short = str(year)[-2:]
            tickers.append(f"HE{code}{yr_short}.CME")
    return tickers


def synthetic_futures_curve(
    spot_price: float,
    n_contracts: int = 8,
) -> pd.DataFrame:
    """
    Builds a realistic lean hogs forward curve using seasonal premia.
    Hogs can't be stored, so each contract reflects expected S/D at delivery.
    """
    now = datetime.now()
    rows = []
    month, year = now.month, now.year
    found = 0
    while found < n_contracts:
        month += 1
        if month > 12:
            month = 1
            year += 1
        if month in LEAN_HOG_MONTHS:
            seasonal = SEASONAL_PREMIA[month]
            months_fwd = (year - now.year) * 12 + (month - now.month)
            # Forward price = spot adjusted for seasonal + small noise
            fwd = spot_price + seasonal - SEASONAL_PREMIA[now.month]
            label = f"HE{LEAN_HOG_MONTHS[month]}{str(year)[-2:]}"
            rows.append({
                "contract": label,
                "month": month,
                "year": year,
                "months_forward": months_fwd,
                "price": round(max(fwd, 40.0), 3),
                "seasonal_premium": seasonal,
            })
            found += 1
    return pd.DataFrame(rows)


def compute_calendar_spreads(curve_df: pd.DataFrame) -> pd.DataFrame:
    """Compute M1-M2, M2-M3, ... spreads."""
    rows = []
    for i in range(len(curve_df) - 1):
        near = curve_df.iloc[i]
        far = curve_df.iloc[i + 1]
        spread = far["price"] - near["price"]
        rows.append({
            "name": f"{near['contract']} / {far['contract']}",
            "near_month": near["contract"],
            "far_month": far["contract"],
            "near_price": near["price"],
            "far_price": far["price"],
            "spread": round(spread, 3),
            "spread_pct": round(spread / near["price"] * 100, 2),
            "structure": "Contango" if spread > 0.5 else ("Backwardation" if spread < -0.5 else "Flat"),
        })
    return pd.DataFrame(rows)


# ============================================================
# SECTION 9: STREAMLIT UI — RENDER FUNCTIONS
# ============================================================

def _metric_fmt(val: float, fmt: str = ".2f", suffix: str = "") -> str:
    return f"{val:{fmt}}{suffix}"


def render_overview(hogs_df, corn_df, meal_df, signals, fair_value):
    st.header("Market Overview & Fair Value")

    close = hogs_df["Close"]
    current = float(close.iloc[-1])
    prev = float(close.iloc[-2])
    fv_current = float(fair_value.iloc[-1])
    fv_premium = current - fv_current
    fv_pct = fv_premium / fv_current * 100

    # Get current signal
    ens_col = "ensemble_signal" if "ensemble_signal" in signals.columns else "gb_signal"
    ens_sig = float(signals[ens_col].iloc[-1])
    sig_direction = "LONG" if ens_sig > 0.15 else ("SHORT" if ens_sig < -0.15 else "FLAT")
    sig_color = {"LONG": "🟢", "SHORT": "🔴", "FLAT": "🟡"}[sig_direction]

    # Key metrics row
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Spot Price", f"${current:.2f}", delta=f"{(current - prev) / prev:.1%}")
    c2.metric("Fair Value", f"${fv_current:.2f}", delta=f"Δ{fv_premium:+.2f}")
    c3.metric("FV Premium", f"{fv_pct:+.1f}%",
              delta="Overvalued" if fv_pct > 5 else ("Undervalued" if fv_pct < -5 else "Fair"),
              delta_color="inverse" if fv_pct > 5 else ("normal" if fv_pct < -5 else "off"))
    c4.metric("Signal", f"{sig_color} {sig_direction}", delta=f"{ens_sig:+.2f}")
    c5.metric("RSI", f"{signals['rsi'].iloc[-1]:.1f}")
    seas = float(signals["seasonal_signal"].iloc[-1])
    c6.metric("Seasonal", "Bullish" if seas > 0.3 else ("Bearish" if seas < -0.3 else "Neutral"),
              delta=f"{seas:+.2f}")

    # Price chart with fair value overlay
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
        subplot_titles=["Lean Hogs Price vs Fair Value", "Fair Value Premium/Discount (%)"],
        vertical_spacing=0.07,
    )
    fig.add_trace(go.Candlestick(
        x=hogs_df.index[-252:], open=hogs_df["Open"].iloc[-252:],
        high=hogs_df["High"].iloc[-252:], low=hogs_df["Low"].iloc[-252:],
        close=close.iloc[-252:], name="HE Futures"), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=close.index[-252:], y=fair_value.iloc[-252:],
        name="Fair Value", line=dict(color="orange", width=2, dash="dash")), row=1, col=1)
    fv_prem_series = ((close - fair_value) / fair_value * 100).iloc[-252:]
    colors = ["#00cc66" if v < 0 else "#ff4444" for v in fv_prem_series]
    fig.add_trace(go.Bar(
        x=fv_prem_series.index, y=fv_prem_series,
        marker_color=colors, name="FV Premium %"), row=2, col=1)
    fig.add_hline(y=0, row=2, col=1, line_color="white", line_dash="dash", line_width=1)
    fig.add_hline(y=10, row=2, col=1, line_color="red", line_dash="dot", line_width=0.8, opacity=0.5)
    fig.add_hline(y=-10, row=2, col=1, line_color="green", line_dash="dot", line_width=0.8, opacity=0.5)
    fig.update_layout(
        template="plotly_dark", height=580, showlegend=True,
        xaxis_rangeslider_visible=False, margin=dict(t=40))
    st.plotly_chart(fig, use_container_width=True)

    # Signal summary table + gauge
    col_l, col_r = st.columns([1, 1])
    with col_l:
        st.subheader("Signal Summary")
        sig_rows = []
        label_map = {
            "gb_signal": "Gaussian Breakout",
            "ma_signal": "Moving Average",
            "mr_signal": "Mean Reversion",
            "seasonal_signal": "Seasonality",
            "sd_signal": "Supply / Demand",
        }
        for key, label in label_map.items():
            if key in signals.columns:
                val = float(signals[key].iloc[-1])
                direction = "🟢 Bullish" if val > 0.15 else ("🔴 Bearish" if val < -0.15 else "🟡 Neutral")
                sig_rows.append({"Signal": label, "Score": f"{val:+.3f}", "Direction": direction})
        st.dataframe(pd.DataFrame(sig_rows), use_container_width=True, hide_index=True)

    with col_r:
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=ens_sig,
            number={"valueformat": "+.2f"},
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Composite Signal"},
            gauge={
                "axis": {"range": [-1, 1], "tickwidth": 1},
                "bar": {"color": "white", "thickness": 0.3},
                "steps": [
                    {"range": [-1, -0.5], "color": "#cc0000"},
                    {"range": [-0.5, -0.15], "color": "#ff6666"},
                    {"range": [-0.15, 0.15], "color": "#555555"},
                    {"range": [0.15, 0.5], "color": "#66cc66"},
                    {"range": [0.5, 1], "color": "#00aa44"},
                ],
                "threshold": {"line": {"color": "white", "width": 3}, "value": ens_sig},
            },
        ))
        fig_g.update_layout(template="plotly_dark", height=300, margin=dict(t=30))
        st.plotly_chart(fig_g, use_container_width=True)


def render_signals(hogs_df, signals):
    st.header("Signals & Seasonality")

    close = hogs_df["Close"]

    # Monthly seasonality bar chart
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Historical Monthly Returns")
        df_s = pd.DataFrame({"close": close, "month": close.index.month})
        df_s["ret"] = df_s["close"].pct_change() * 100
        mo_avg = df_s.groupby("month")["ret"].mean()
        mo_std = df_s.groupby("month")["ret"].std()
        month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        fig_mo = go.Figure(go.Bar(
            x=month_names,
            y=[mo_avg.get(m, 0) for m in range(1, 13)],
            error_y=dict(type="data", array=[mo_std.get(m, 0) for m in range(1, 13)], visible=True),
            marker_color=["#00cc66" if mo_avg.get(m, 0) > 0 else "#ff4444" for m in range(1, 13)],
            text=[f"{mo_avg.get(m, 0):.2f}%" for m in range(1, 13)],
            textposition="auto",
        ))
        fig_mo.update_layout(
            title="Average Daily Return by Month (%)",
            template="plotly_dark", height=360, margin=dict(t=40))
        st.plotly_chart(fig_mo, use_container_width=True)

    with col2:
        st.subheader("Year-over-Year Seasonal Pattern")
        df_yoy = pd.DataFrame({"close": close})
        df_yoy["year"] = close.index.year
        df_yoy["doy"] = close.index.dayofyear
        fig_yoy = go.Figure()
        palette = px.colors.qualitative.Set2
        last_years = sorted(df_yoy["year"].unique())[-5:]
        for i, yr in enumerate(last_years):
            yd = df_yoy[df_yoy["year"] == yr]
            norm = yd["close"] / yd["close"].iloc[0] * 100
            fig_yoy.add_trace(go.Scatter(
                x=yd["doy"], y=norm, name=str(yr),
                line=dict(color=palette[i % len(palette)])))
        fig_yoy.update_layout(
            title="Price Indexed to Year Start = 100",
            xaxis_title="Day of Year", yaxis_title="Index",
            template="plotly_dark", height=360, margin=dict(t=40))
        st.plotly_chart(fig_yoy, use_container_width=True)

    # Seasonal return heatmap (year × month)
    st.subheader("Seasonal Return Heatmap")
    df_heat = pd.DataFrame({"ret": close.pct_change() * 100})
    df_heat["year"] = close.index.year
    df_heat["month"] = close.index.month
    pivot = df_heat.groupby(["year", "month"])["ret"].sum().unstack(level=1)
    pivot.columns = month_names[:len(pivot.columns)]
    fig_heat = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale="RdYlGn",
        zmid=0,
        text=[[f"{v:.1f}%" if not np.isnan(v) else "" for v in row] for row in pivot.values],
        texttemplate="%{text}",
        colorbar=dict(title="Ret %"),
    ))
    fig_heat.update_layout(
        title="Monthly Cumulative Returns by Year (%)",
        template="plotly_dark", height=400, margin=dict(t=40))
    st.plotly_chart(fig_heat, use_container_width=True)

    # Supply/Demand signal
    st.subheader("Supply & Demand Signal")
    fig_sd = make_subplots(rows=1, cols=2, shared_xaxes=False,
                           subplot_titles=["S/D Signal (rolling)", "Seasonal Signal"])
    if "sd_signal" in signals.columns:
        fig_sd.add_trace(go.Scatter(
            x=signals.index[-252:], y=signals["sd_signal"].iloc[-252:],
            fill="tozeroy", name="S/D Signal", line=dict(color="#00bfff")), row=1, col=1)
    if "seasonal_signal" in signals.columns:
        fig_sd.add_trace(go.Scatter(
            x=signals.index[-252:], y=signals["seasonal_signal"].iloc[-252:],
            fill="tozeroy", name="Seasonal", line=dict(color="orange")), row=1, col=2)
    fig_sd.update_layout(template="plotly_dark", height=300, showlegend=False, margin=dict(t=40))
    st.plotly_chart(fig_sd, use_container_width=True)


def render_strategies(hogs_df, signals, gb_bandwidth, gb_k, ma_fast, ma_slow, mr_window, mr_z):
    st.header("Trading Strategies")

    close = hogs_df["Close"]
    tab_gb, tab_ma, tab_mr, tab_comb = st.tabs([
        "Gaussian Breakout", "Moving Average", "Mean Reversion", "Combined"
    ])

    # --- Gaussian Breakout ---
    with tab_gb:
        st.markdown("""
**Gaussian Breakout** uses a causal Gaussian-smoothed price as the "mean" and
a rolling standard deviation band. The signal is `tanh(z / K)` where
`z = (price − smooth) / σ`. A value near +1 = strong upside breakout.

- Enter long when `price > smooth + K·σ`
- Enter short when `price < smooth − K·σ`
""")
        smooth = causal_gaussian_smooth(close, sigma=gb_bandwidth)
        roll_std = close.rolling(gb_bandwidth).std().clip(lower=1e-8)
        upper = smooth + gb_k * roll_std
        lower = smooth - gb_k * roll_std
        z = (close - smooth) / roll_std

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35],
                            vertical_spacing=0.05)
        n = min(252, len(close))
        fig.add_trace(go.Scatter(x=close.index[-n:], y=close.iloc[-n:],
                                  name="Price", line=dict(color="white")), row=1, col=1)
        fig.add_trace(go.Scatter(x=close.index[-n:], y=smooth.iloc[-n:],
                                  name="Gaussian Smooth", line=dict(color="yellow", dash="dash")), row=1, col=1)
        fig.add_trace(go.Scatter(x=upper.index[-n:], y=upper.iloc[-n:],
                                  name="Upper Band", line=dict(color="#ff4444", dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=lower.index[-n:], y=lower.iloc[-n:],
                                  name="Lower Band", line=dict(color="#00cc66", dash="dot"),
                                  fill="tonexty", fillcolor="rgba(100,100,100,0.1)"), row=1, col=1)
        fig.add_trace(go.Scatter(x=z.index[-n:], y=z.iloc[-n:],
                                  fill="tozeroy", name="Z-Score",
                                  line=dict(color="cyan")), row=2, col=1)
        fig.add_hline(y=gb_k, row=2, col=1, line_color="red", line_dash="dot", opacity=0.6)
        fig.add_hline(y=-gb_k, row=2, col=1, line_color="green", line_dash="dot", opacity=0.6)
        fig.add_hline(y=0, row=2, col=1, line_color="white", line_dash="dash", opacity=0.4)
        fig.update_layout(template="plotly_dark", height=560, xaxis_rangeslider_visible=False,
                          margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)
        cc = st.columns(3)
        cc[0].metric("Z-Score (latest)", f"{z.iloc[-1]:.2f}σ")
        cc[1].metric("Upper Band", f"${upper.iloc[-1]:.2f}")
        cc[2].metric("Lower Band", f"${lower.iloc[-1]:.2f}")

    # --- Moving Average ---
    with tab_ma:
        st.markdown(f"""
**MA Crossover** compares EMA({ma_fast}) vs EMA({ma_slow}).
Signal = `tanh(normalised_spread)`. Positive = fast above slow (bullish trend).
""")
        ema_f = close.ewm(span=ma_fast, adjust=False).mean()
        ema_s = close.ewm(span=ma_slow, adjust=False).mean()
        spread = ema_f - ema_s

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            row_heights=[0.5, 0.25, 0.25], vertical_spacing=0.04)
        n = min(252, len(close))
        fig.add_trace(go.Scatter(x=close.index[-n:], y=close.iloc[-n:],
                                  name="Price", line=dict(color="white")), row=1, col=1)
        fig.add_trace(go.Scatter(x=ema_f.index[-n:], y=ema_f.iloc[-n:],
                                  name=f"EMA({ma_fast})", line=dict(color="cyan")), row=1, col=1)
        fig.add_trace(go.Scatter(x=ema_s.index[-n:], y=ema_s.iloc[-n:],
                                  name=f"EMA({ma_slow})", line=dict(color="orange")), row=1, col=1)
        fig.add_trace(go.Scatter(x=spread.index[-n:], y=spread.iloc[-n:],
                                  fill="tozeroy", name="EMA Spread",
                                  line=dict(color="purple")), row=2, col=1)
        ma_sig = signals["ma_signal"].iloc[-n:] if "ma_signal" in signals.columns else pd.Series(dtype=float)
        fig.add_trace(go.Scatter(x=ma_sig.index, y=ma_sig,
                                  fill="tozeroy", name="MA Signal",
                                  line=dict(color="lime")), row=3, col=1)
        fig.add_hline(y=0, row=2, col=1, line_color="white", line_dash="dash", opacity=0.4)
        fig.add_hline(y=0, row=3, col=1, line_color="white", line_dash="dash", opacity=0.4)
        fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False,
                          margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)
        cc = st.columns(3)
        cc[0].metric(f"EMA({ma_fast})", f"${ema_f.iloc[-1]:.2f}")
        cc[1].metric(f"EMA({ma_slow})", f"${ema_s.iloc[-1]:.2f}")
        cc[2].metric("Trend", "↑ Bullish" if float(ema_f.iloc[-1]) > float(ema_s.iloc[-1]) else "↓ Bearish")

    # --- Mean Reversion ---
    with tab_mr:
        st.markdown(f"""
**Mean Reversion** trades the return toward the mean when prices deviate beyond ±{mr_z}σ.
Combined with RSI and Bollinger %B for confirmation.
""")
        zs = rolling_zscore(close, mr_window)
        bb_l, bb_m, bb_u = compute_bollinger(close, mr_window)
        rsi = compute_rsi(close)

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            row_heights=[0.45, 0.28, 0.27], vertical_spacing=0.05)
        n = min(252, len(close))
        fig.add_trace(go.Scatter(x=close.index[-n:], y=close.iloc[-n:],
                                  name="Price", line=dict(color="white")), row=1, col=1)
        fig.add_trace(go.Scatter(x=bb_m.index[-n:], y=bb_m.iloc[-n:],
                                  name="BB Mid", line=dict(color="yellow", dash="dash")), row=1, col=1)
        fig.add_trace(go.Scatter(x=bb_u.index[-n:], y=bb_u.iloc[-n:],
                                  name="BB Upper", line=dict(color="red", dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=bb_l.index[-n:], y=bb_l.iloc[-n:],
                                  name="BB Lower", line=dict(color="green", dash="dot"),
                                  fill="tonexty", fillcolor="rgba(100,100,100,0.08)"), row=1, col=1)
        fig.add_trace(go.Scatter(x=zs.index[-n:], y=zs.iloc[-n:],
                                  fill="tozeroy", name="Z-Score",
                                  line=dict(color="cyan")), row=2, col=1)
        fig.add_hline(y=mr_z, row=2, col=1, line_color="red", line_dash="dot", opacity=0.7)
        fig.add_hline(y=-mr_z, row=2, col=1, line_color="green", line_dash="dot", opacity=0.7)
        fig.add_hline(y=0, row=2, col=1, line_color="white", line_dash="dash", opacity=0.4)
        fig.add_trace(go.Scatter(x=rsi.index[-n:], y=rsi.iloc[-n:],
                                  name="RSI(14)", line=dict(color="orange")), row=3, col=1)
        fig.add_hline(y=70, row=3, col=1, line_color="red", line_dash="dot", opacity=0.6)
        fig.add_hline(y=30, row=3, col=1, line_color="green", line_dash="dot", opacity=0.6)
        fig.add_hline(y=50, row=3, col=1, line_color="white", line_dash="dash", opacity=0.3)
        fig.update_layout(template="plotly_dark", height=620, xaxis_rangeslider_visible=False,
                          margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)
        cc = st.columns(3)
        cc[0].metric("Z-Score", f"{zs.iloc[-1]:.2f}σ")
        cc[1].metric("RSI(14)", f"{rsi.iloc[-1]:.1f}")
        mr_pos_str = "Long" if float(signals["mr_signal"].iloc[-1]) > 0.5 else (
            "Short" if float(signals["mr_signal"].iloc[-1]) < -0.5 else "Flat"
        ) if "mr_signal" in signals.columns else "N/A"
        cc[2].metric("MR Position", mr_pos_str)

    # --- Combined ---
    with tab_comb:
        st.subheader("All Signals Overlay")
        ens_col = "ensemble_signal" if "ensemble_signal" in signals.columns else "gb_signal"
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.55, 0.45],
                            vertical_spacing=0.05)
        n = min(252, len(close))
        fig.add_trace(go.Scatter(x=close.index[-n:], y=close.iloc[-n:],
                                  name="Price", line=dict(color="white")), row=1, col=1)
        colors_map = {"gb_signal": "#00bfff", "ma_signal": "#ffa500",
                      "mr_signal": "#00ff88", ens_col: "white"}
        names_map = {"gb_signal": "Gaussian", "ma_signal": "MA",
                     "mr_signal": "MR", "ensemble_signal": "Ensemble"}
        for col, lc in colors_map.items():
            if col in signals.columns and col != ens_col:
                fig.add_trace(go.Scatter(
                    x=signals.index[-n:], y=signals[col].iloc[-n:],
                    name=names_map.get(col, col), opacity=0.5,
                    line=dict(color=lc, dash="dot")), row=2, col=1)
        if ens_col in signals.columns:
            fig.add_trace(go.Scatter(
                x=signals.index[-n:], y=signals[ens_col].iloc[-n:],
                name="Composite", fill="tozeroy",
                line=dict(color="white", width=2)), row=2, col=1)
        fig.add_hline(y=0, row=2, col=1, line_color="gray", line_dash="dash", opacity=0.5)
        fig.update_layout(template="plotly_dark", height=560,
                          xaxis_rangeslider_visible=False, margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)


def render_ml_optimizer(signals, hogs_df):
    st.header("ML Weight Optimizer")

    weight_cols = [c for c in signals.columns
                   if c in ["gb_weight", "ma_weight", "mr_weight"]]
    ens_col = "ensemble_signal"

    st.markdown("""
### How it Works

A **Random Forest Classifier** is trained on a rolling 252-day window of signal
features to predict whether the *next day's return* will be positive.
The predicted probability is then converted to a continuous signal in **[-1, +1]**.

**Walk-forward training:** The model is retrained every **63 trading days** (quarterly)
using all prior data, then deployed on the next quarter. Feature importances from each
training run determine the relative weight given to each strategy's raw signal.

This approach avoids look-ahead bias and tests whether any strategy contains
information about future returns that the others do not.
""")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Current Strategy Weights")
        if weight_cols and all(c in signals.columns for c in weight_cols):
            current_w = {c.replace("_weight", "").replace("_", " ").title():
                         float(signals[c].iloc[-1]) for c in weight_cols}
            # Normalize for display
            total = sum(abs(v) for v in current_w.values()) + 1e-8
            norm_w = {k: abs(v) / total for k, v in current_w.items()}
            fig_pie = go.Figure(go.Pie(
                labels=list(norm_w.keys()), values=list(norm_w.values()),
                hole=0.4, marker=dict(colors=["#00bfff", "#ffa500", "#00ff88"])))
            fig_pie.update_layout(template="plotly_dark", height=320, margin=dict(t=20))
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("ML weights not available — run with ML enabled.")

    with col2:
        st.subheader("Signal Correlation Matrix")
        core_sig_cols = [c for c in ["gb_signal", "ma_signal", "mr_signal", "seasonal_signal", "sd_signal"] if c in signals.columns]
        sig_subset = signals[core_sig_cols].dropna()
        corr = sig_subset.corr()
        rename = {c: c.replace("_signal", "").replace("_", " ").title() for c in corr.columns}
        corr = corr.rename(columns=rename, index=rename)
        fig_corr = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
            colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
            text=[[f"{v:.2f}" for v in row] for row in corr.values],
            texttemplate="%{text}"))
        fig_corr.update_layout(template="plotly_dark", height=320, margin=dict(t=20))
        st.plotly_chart(fig_corr, use_container_width=True)

    # Market regime detection
    st.subheader("Market Regime Detection")
    close = hogs_df["Close"]
    vol_20 = close.pct_change().rolling(20).std() * np.sqrt(252)
    vol_med = vol_20.median()
    trend_str = signals["ma_signal"].abs() if "ma_signal" in signals.columns else vol_20 * 0
    trend_med = trend_str.median()

    def regime(row):
        hi_v = row["vol"] > vol_med
        hi_t = row["trend"] > trend_med
        if hi_v and hi_t:    return "Breakout"
        if hi_v and not hi_t: return "Choppy"
        if not hi_v and hi_t: return "Trending"
        return "Range-Bound"

    regime_df = pd.DataFrame({"vol": vol_20, "trend": trend_str}).dropna()
    regime_df["regime"] = regime_df.apply(regime, axis=1)
    current_regime = regime_df["regime"].iloc[-1] if len(regime_df) else "Unknown"

    regime_color = {"Breakout": "#ff6600", "Choppy": "#ff4444",
                    "Trending": "#00ff88", "Range-Bound": "#00bfff"}
    counts = regime_df["regime"].value_counts()

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Current Regime", current_regime)
        best = {"Breakout": "Gaussian Breakout", "Trending": "Moving Average",
                "Range-Bound": "Mean Reversion", "Choppy": "Reduce Size"}
        st.metric("Preferred Strategy", best.get(current_regime, "Mixed"))
        fig_donut = go.Figure(go.Pie(
            labels=counts.index.tolist(), values=counts.values.tolist(), hole=0.5,
            marker=dict(colors=[regime_color.get(r, "#888888") for r in counts.index])))
        fig_donut.update_layout(template="plotly_dark", height=280, margin=dict(t=10))
        st.plotly_chart(fig_donut, use_container_width=True)

    with col2:
        # Regime timeline
        fig_reg = go.Figure()
        regime_num = {"Range-Bound": 1, "Trending": 2, "Choppy": 3, "Breakout": 4}
        r_num = regime_df["regime"].map(regime_num).fillna(1)
        fig_reg.add_trace(go.Scatter(
            x=regime_df.index[-252:], y=r_num.iloc[-252:],
            mode="markers",
            marker=dict(
                color=[regime_color.get(r, "#888888") for r in regime_df["regime"].iloc[-252:]],
                size=5,
            ),
            name="Regime",
        ))
        fig_reg.update_yaxes(
            tickvals=[1, 2, 3, 4],
            ticktext=["Range-Bound", "Trending", "Choppy", "Breakout"])
        fig_reg.update_layout(
            title="Market Regime (Last 252 Days)",
            template="plotly_dark", height=320, margin=dict(t=40))
        st.plotly_chart(fig_reg, use_container_width=True)


def render_backtest(hogs_df, signals, tc_bps, mc_n):
    st.header("Backtest & Monte Carlo Analysis")

    close = hogs_df["Close"]
    ens_col = "ensemble_signal" if "ensemble_signal" in signals.columns else "gb_signal"
    signal = signals[ens_col]

    with st.spinner("Running backtest…"):
        result = vectorized_backtest(signal, close, transaction_cost_bps=tc_bps)

    m = result["metrics"]
    if m:
        cols = st.columns(len(m))
        fmt_map = {
            "CAGR": ".1%", "Volatility": ".1%", "Sharpe": ".2f",
            "Sortino": ".2f", "Max Drawdown": ".1%", "Calmar": ".2f",
            "Win Rate": ".1%", "Profit Factor": ".2f", "B&H CAGR": ".1%",
        }
        for col, (key, val) in zip(cols, m.items()):
            formatted = f"{val:{fmt_map.get(key, '.2f')}}"
            col.metric(key, formatted)

    # Equity curve
    eq = result["equity_strategy"]
    bh = result["equity_buyhold"]
    cum = eq
    rolling_max = cum.expanding().max()
    dd = (cum - rolling_max) / rolling_max * 100

    fig_eq = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35],
                           vertical_spacing=0.05)
    fig_eq.add_trace(go.Scatter(x=bh.index, y=bh, name="Buy & Hold",
                                 line=dict(color="lightblue", dash="dot")), row=1, col=1)
    fig_eq.add_trace(go.Scatter(x=eq.index, y=eq, name="Strategy",
                                 line=dict(color="lime", width=2)), row=1, col=1)
    fig_eq.add_hline(y=1, row=1, col=1, line_color="gray", line_dash="dash", opacity=0.5)
    fig_eq.add_trace(go.Scatter(x=dd.index, y=dd, fill="tozeroy",
                                 name="Drawdown %", line=dict(color="red")), row=2, col=1)
    fig_eq.update_layout(template="plotly_dark", height=560, margin=dict(t=20))
    st.plotly_chart(fig_eq, use_container_width=True)

    # Monthly returns heatmap
    st.subheader("Monthly P&L Heatmap")
    daily_ret = result["daily_returns"]
    mo_ret = daily_ret.resample("ME").sum() * 100
    mo_pivot = mo_ret.groupby([mo_ret.index.year, mo_ret.index.month]).sum().unstack(1)
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    mo_pivot.columns = [month_names[c - 1] for c in mo_pivot.columns]
    fig_ph = go.Figure(go.Heatmap(
        z=mo_pivot.values, x=mo_pivot.columns.tolist(), y=mo_pivot.index.tolist(),
        colorscale="RdYlGn", zmid=0,
        text=[[f"{v:.1f}%" if not np.isnan(v) else "" for v in row] for row in mo_pivot.values],
        texttemplate="%{text}"))
    fig_ph.update_layout(title="Monthly Returns (%)", template="plotly_dark",
                         height=380, margin=dict(t=40))
    st.plotly_chart(fig_ph, use_container_width=True)

    # ── Monte Carlo Permutation Test ──
    st.subheader("Monte Carlo Permutation Test")
    st.markdown("""
**Purpose:** Determine if strategy returns are genuine alpha or a result of
data-mining / overfitting.

**Method:** Shuffle daily returns N times using *block bootstrap* (blocks of 20 days
to preserve autocorrelation). Compare the observed Sharpe ratio to the distribution
of shuffled Sharpe ratios. A p-value < 0.05 suggests the strategy is unlikely to be
pure luck.
""")
    with st.spinner(f"Running {mc_n:,} permutations…"):
        perm = permutation_test(daily_ret, n_permutations=mc_n)

    if perm:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Observed Sharpe", f"{perm['observed_sharpe']:.3f}")
            st.metric("Null Median Sharpe", f"{perm['null_median']:.3f}")
            st.metric("Null 95th Pctile", f"{perm['null_95th']:.3f}")
            st.metric("p-value", f"{perm['p_value']:.3f}")
            st.metric("Percentile Rank", f"{perm['percentile']:.1%}")
            if perm["significant"]:
                st.success("✅ **SIGNAL VALIDATED** — Beats 95%+ of random permutations.")
            elif perm["percentile"] > 0.80:
                st.warning("⚠️ **MODERATE SIGNAL** — Beats 80–95% of random permutations.")
            else:
                st.error("❌ **WEAK SIGNAL** — Cannot reliably distinguish from random entry.")
        with col2:
            null = perm["null_distribution"]
            fig_mc = go.Figure()
            fig_mc.add_trace(go.Histogram(
                x=null, nbinsx=60, name="Null Distribution",
                marker_color="steelblue", opacity=0.75))
            fig_mc.add_vline(x=perm["observed_sharpe"], line_color="lime",
                             line_width=2, line_dash="dash",
                             annotation_text=f"Observed: {perm['observed_sharpe']:.2f}",
                             annotation_font_color="lime")
            fig_mc.add_vline(x=perm["null_95th"], line_color="red",
                             line_width=1, line_dash="dot",
                             annotation_text="95th pctile", annotation_font_color="red")
            fig_mc.update_layout(
                title=f"Null Sharpe Distribution ({mc_n:,} permutations)",
                xaxis_title="Sharpe Ratio", yaxis_title="Frequency",
                template="plotly_dark", height=380, margin=dict(t=40))
            st.plotly_chart(fig_mc, use_container_width=True)

        # MC equity fan chart
        st.subheader("Monte Carlo Equity Fan Chart")
        with st.spinner("Generating equity paths…"):
            paths = monte_carlo_equity_paths(daily_ret, n_simulations=min(mc_n, 500))
        pct_bands = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)
        x_axis = list(range(paths.shape[1]))
        fig_fan = go.Figure()
        band_configs = [
            (0, 4, "#1a3a5c", "5th–95th %ile"),
            (1, 3, "#2a5a8c", "25th–75th %ile"),
        ]
        for lo, hi, color, name in band_configs:
            fig_fan.add_trace(go.Scatter(
                x=x_axis + x_axis[::-1],
                y=list(pct_bands[hi]) + list(pct_bands[lo])[::-1],
                fill="toself", fillcolor=color, line_color="rgba(0,0,0,0)",
                opacity=0.5, name=name))
        fig_fan.add_trace(go.Scatter(
            x=x_axis, y=pct_bands[2], name="Median Path",
            line=dict(color="white", width=2)))
        actual_eq = result["equity_strategy"].values
        fig_fan.add_trace(go.Scatter(
            x=list(range(len(actual_eq))), y=actual_eq,
            name="Actual Strategy", line=dict(color="lime", width=2.5)))
        fig_fan.add_hline(y=1.0, line_color="gray", line_dash="dash", opacity=0.6)
        fig_fan.update_layout(
            title="Block-Bootstrap MC Equity Paths (500 Simulations)",
            xaxis_title="Days", yaxis_title="Cumulative Return",
            template="plotly_dark", height=420, margin=dict(t=40))
        st.plotly_chart(fig_fan, use_container_width=True)


def render_curve(spot_price):
    st.header("Futures Curve & Spread Analysis")

    curve_df = synthetic_futures_curve(spot_price, n_contracts=8)
    spreads_df = compute_calendar_spreads(curve_df)

    # Curve shape classification
    prices = curve_df["price"].values
    slope = np.polyfit(range(len(prices)), prices, 1)[0]
    if abs(slope) < 0.15:
        shape, shape_col = "Flat", "#ffaa00"
    elif slope > 0:
        shape, shape_col = "Contango", "#ff4444"
    else:
        shape, shape_col = "Backwardation", "#00cc66"

    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("Forward Curve Term Structure")
        st.markdown(
            f"**Structure:** <span style='color:{shape_col}; font-size:1.2em'>{shape}</span> "
            f"| Slope: **{slope:+.3f} $/cwt per month**",
            unsafe_allow_html=True)
        fig_curve = go.Figure()
        fig_curve.add_trace(go.Bar(
            x=curve_df["contract"], y=curve_df["price"],
            marker=dict(
                color=curve_df["price"],
                colorscale="RdYlGn",
                showscale=True,
                colorbar=dict(title="$/cwt"),
            ),
            name="Forward Price",
        ))
        fig_curve.add_trace(go.Scatter(
            x=curve_df["contract"], y=curve_df["price"],
            mode="lines+markers", name="Curve",
            line=dict(color="white", width=2),
            marker=dict(size=8, color="white")))
        fig_curve.add_hline(
            y=spot_price, line_color="cyan", line_dash="dash",
            annotation_text=f"Spot ${spot_price:.2f}", annotation_font_color="cyan")
        fig_curve.update_layout(
            xaxis_title="Contract", yaxis_title="Price ($/cwt)",
            template="plotly_dark", height=420, margin=dict(t=20), showlegend=False)
        st.plotly_chart(fig_curve, use_container_width=True)

    with col2:
        st.subheader("Curve Interpretation")
        interp = {
            "Contango": """
🔴 **Contango** (upward sloping)
- Deferred months at premium to spot
- Market expects future supply tightening or seasonal demand peak
- **Long hedger** benefits: buy cheaper spot, sell expensive deferred
- **Roll cost:** negative for long futures holder (pays contango)
""",
            "Backwardation": """
🟢 **Backwardation** (downward sloping)
- Spot at premium to deferred months
- Current supply is tight; market expects future relief
- **Short hedger** (producer) benefits: lock in high forward prices
- **Roll yield:** positive for long futures holder (earns backwardation)
""",
            "Flat": """
🟡 **Flat Curve**
- No strong directional supply/demand expectation
- Trade the intra-curve spread volatility
- Watch for curve transition signals (contango → backwardation = bullish)
""",
        }
        st.markdown(interp[shape])

        roll_yield = (curve_df["price"].iloc[0] - curve_df["price"].iloc[1]) / curve_df["price"].iloc[1] * 12
        st.metric("Annualized Roll Yield", f"{roll_yield:.1%}",
                  help="Positive = earn roll (backwardation), Negative = pay roll (contango)")

    # Calendar Spreads
    st.subheader("Calendar Spread Analysis")
    col1, col2 = st.columns([2, 1])
    with col1:
        spread_colors = [
            "#00cc66" if s == "Backwardation" else ("#ff4444" if s == "Contango" else "#ffaa00")
            for s in spreads_df["structure"]
        ]
        fig_sp = go.Figure(go.Bar(
            x=spreads_df["name"], y=spreads_df["spread"],
            marker_color=spread_colors,
            text=[f"{v:+.2f}" for v in spreads_df["spread"]],
            textposition="auto",
        ))
        fig_sp.add_hline(y=0, line_color="white", line_dash="dash", opacity=0.4)
        fig_sp.update_layout(
            title="1-Month Calendar Spreads (Far − Near, $/cwt)",
            xaxis_tickangle=30, template="plotly_dark", height=380, margin=dict(t=40))
        st.plotly_chart(fig_sp, use_container_width=True)

    with col2:
        counts = spreads_df["structure"].value_counts()
        fig_pie = go.Figure(go.Pie(
            labels=counts.index.tolist(), values=counts.values.tolist(), hole=0.4,
            marker=dict(colors=["#00cc66", "#ff4444", "#ffaa00"][:len(counts)])))
        fig_pie.update_layout(title="Spread Structure Mix", template="plotly_dark",
                              height=300, margin=dict(t=40))
        st.plotly_chart(fig_pie, use_container_width=True)

    # Spread detail table
    st.subheader("Spread Detail Table")
    disp = spreads_df.copy()
    disp["near_price"] = disp["near_price"].map("${:.3f}".format)
    disp["far_price"] = disp["far_price"].map("${:.3f}".format)
    disp["spread"] = disp["spread"].map("{:+.3f}".format)
    disp["spread_pct"] = disp["spread_pct"].map("{:+.2f}%".format)
    st.dataframe(
        disp[["name", "near_price", "far_price", "spread", "spread_pct", "structure"]].rename(
            columns={
                "name": "Spread", "near_price": "Near", "far_price": "Far",
                "spread": "Spread ($/cwt)", "spread_pct": "Spread %",
                "structure": "Structure",
            }
        ),
        use_container_width=True, hide_index=True,
    )

    # Seasonal spread patterns
    st.subheader("Key Seasonal Spread Patterns")
    seasonal_patterns = pd.DataFrame([
        {"Spread": "Feb/Apr (G/J)", "Typical Direction": "Tightens into spring",
         "Driver": "Spring demand uptick as grilling season begins"},
        {"Spread": "Apr/Jun (J/M)", "Typical Direction": "Flattens or widens",
         "Driver": "Summer slaughter pace ramp-up; peak demand"},
        {"Spread": "Jun/Aug (M/Q)", "Typical Direction": "Narrows",
         "Driver": "Hog supplies start building post-summer"},
        {"Spread": "Aug/Oct (Q/V)", "Typical Direction": "Inverts to backwardation",
         "Driver": "Fall kill surge, export demand, hog supply peak"},
        {"Spread": "Oct/Dec (V/Z)", "Typical Direction": "Flat to contango",
         "Driver": "Holiday demand supports front months"},
    ])
    st.dataframe(seasonal_patterns, use_container_width=True, hide_index=True)


# ============================================================
# SECTION 10: MAIN APP ENTRY POINT
# ============================================================

def main():
    st.set_page_config(
        page_title="🐷 Lean Hogs Trading Model",
        page_icon="🐷",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🐷 Lean Hogs Futures Trading Model")
    st.caption("CME Lean Hogs (HE) | Quantitative Multi-Strategy Platform")

    # ── Sidebar ──
    with st.sidebar:
        st.header("⚙️ Configuration")
        lookback_days = st.select_slider(
            "Lookback Period",
            options=[252, 504, 756, 1260],
            value=756,
            format_func=lambda x: f"{x // 252}yr ({x}d)",
        )
        force_synthetic = st.checkbox("Use synthetic data (no network required)", value=False)
        use_ml = st.checkbox("Enable ML Weight Optimizer", value=True,
                              help="Walk-forward Random Forest weight optimizer. Adds ~5–15s compute.")

        st.divider()
        st.subheader("Strategy Parameters")
        gb_bandwidth = st.slider("Gaussian Bandwidth (days)", 5, 60, 20)
        gb_k = st.slider("Gaussian K (sigma)", 0.5, 3.0, 1.5, 0.1)
        ma_fast = st.slider("MA Fast Period", 5, 30, 10)
        ma_slow = st.slider("MA Slow Period", 20, 100, 40)
        mr_window = st.slider("MR Z-Score Window", 10, 60, 20)
        mr_z = st.slider("MR Entry Z-Threshold", 0.8, 3.0, 1.5, 0.1)

        st.divider()
        st.subheader("Backtest Settings")
        tc_bps = st.slider("Transaction Cost (bps)", 0, 50, 10,
                           help="Round-trip cost in basis points")
        mc_n = st.select_slider("Monte Carlo Permutations",
                                options=[100, 250, 500, 1000, 2000], value=500)

    # ── Load Data (cached) ──
    with st.spinner("Fetching market data…"):
        data = load_market_data(lookback_days=lookback_days, force_synthetic=force_synthetic)

    hogs_df = data["hogs"]
    corn_df = data["corn"]
    meal_df = data["meal"]
    close = hogs_df["Close"]
    corn_close = corn_df["Close"] if corn_df is not None else None
    meal_close = meal_df["Close"] if meal_df is not None else None

    # ── Compute Signals (cached via build) ──
    with st.spinner("Computing signals…"):
        signals = build_signal_matrix(
            close, corn_close, meal_close,
            gb_bandwidth=gb_bandwidth, gb_k=gb_k,
            ma_fast=ma_fast, ma_slow=ma_slow,
            mr_window=mr_window, mr_z=mr_z,
        )

    # ── ML Weights (optional) ──
    if use_ml:
        with st.spinner("Training ML walk-forward optimizer…"):
            returns = close.pct_change().fillna(0)
            aligned_signals = signals.reindex(returns.index).fillna(0)
            ml_result = walk_forward_optimize(
                signal_matrix_values=aligned_signals.values,
                signal_names=aligned_signals.columns.tolist(),
                signal_index=aligned_signals.index.tolist(),
                returns_values=returns.values,
            )
            signals = pd.concat([signals, ml_result], axis=1)

    # ── Fair Value ──
    fair_value = compute_fair_value(close.index, corn_close, meal_close)

    # ── Render Tabs ──
    tabs = st.tabs([
        "📊 Market Overview",
        "📈 Signals & Seasonality",
        "🎯 Trading Strategies",
        "🤖 ML Optimizer",
        "🔄 Backtest & Monte Carlo",
        "📉 Curve & Spreads",
    ])

    with tabs[0]:
        render_overview(hogs_df, corn_df, meal_df, signals, fair_value)
    with tabs[1]:
        render_signals(hogs_df, signals)
    with tabs[2]:
        render_strategies(hogs_df, signals, gb_bandwidth, gb_k, ma_fast, ma_slow, mr_window, mr_z)
    with tabs[3]:
        render_ml_optimizer(signals, hogs_df)
    with tabs[4]:
        render_backtest(hogs_df, signals, tc_bps, mc_n)
    with tabs[5]:
        render_curve(close.iloc[-1])


if __name__ == "__main__":
    main()
