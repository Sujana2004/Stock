"""Core library shared by ``stock.ipynb`` (training) and ``app.py`` (Streamlit UI).

Contents
--------
* :func:`download_stock_data` - Yahoo Finance download with retries and an on-disk cache
* :func:`prepare`             - turns a price table into the arrays the model needs
* :func:`build_dataset`       - builds (X, Y) training samples
* :func:`load_artifacts`      - loads the trained model and its metadata
* :func:`forecast_path`       - multi-day forecast with an 80% range
* :func:`walk_forward`        - out-of-sample backtest against naive / drift baselines
* :func:`recent_shock`        - flags unusually sharp recent moves (usually news-driven)

Modelling approach
------------------
The model never sees raw prices. Its input is ``LOOKBACK`` days of features built
from returns, volume and the daily high-low range, all divided by the window's
own volatility. Its output is, for each of the next ``HORIZON`` days, three
quantiles (10%, 50%, 90%) of the cumulative log-return, produced in one shot
(direct multi-step forecasting). Inputs and outputs are therefore scale-free:
the model keeps working when a price leaves the range seen in training, works
for any ticker or currency, and never feeds its own predictions back in.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
# IMPORT ORDER MATTERS: yfinance (via curl_cffi) must be imported before
# TensorFlow/Keras. TensorFlow ships its own libcurl; if it is loaded first,
# curl_cffi binds to it and every Yahoo request fails with "ImpersonateError".
# Keras is therefore only imported lazily inside load_artifacts(), and nothing
# else in this project may import it before this module.
# Guarded by tests/test_stock_utils.py::test_yahoo_client_works_after_tensorflow.
import yfinance as yf

logger = logging.getLogger(__name__)

LOOKBACK = 100  # trading days of history the model looks at
HORIZON = 100   # trading days it forecasts
N_FEATURES = 6
QUANTILES = (0.1, 0.5, 0.9)   # lower bound, median, upper bound of the 80% range
LONG_VOL_WINDOW = 500         # ~2 years: defines the stock's "normal" volatility
FEATURE_CLIP = 10.0           # bad ticks / extreme days cannot blow up the inputs
SHOCK_WINDOW = 10             # trading days checked for unusually sharp moves
SHOCK_DAILY_MULTIPLE = 3.5    # one day moving this many times the stock's normal daily move
SHOCK_WINDOW_MULTIPLE = 2.5   # or the whole window moving this many times its normal amount

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "data_cache"
MODEL_PATH = BASE_DIR / "models" / "stock_lstm.keras"
META_PATH = BASE_DIR / "models" / "model_meta.json"

PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
_TICKER_RE = re.compile(r"^\^?[A-Z0-9][A-Z0-9.\-=]{0,19}$")


class DataDownloadError(RuntimeError):
    """Raised when prices can be obtained neither from Yahoo nor from the cache."""


# ----------------------------------------------------------------------------
# 1. Data download
# ----------------------------------------------------------------------------
def normalize_ticker(ticker: str) -> str:
    """Upper-case and validate a ticker symbol (also keeps cache file names safe)."""
    cleaned = (ticker or "").strip().upper()
    if not _TICKER_RE.match(cleaned):
        raise ValueError(
            f"'{ticker}' is not a valid ticker symbol. Examples: GOOG, AAPL, TCS.NS, ^NSEI"
        )
    return cleaned


def _clean_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance's MultiIndex columns and return a tidy, date-sorted frame."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)  # ('Close', 'GOOG') -> 'Close'
    df = df.loc[:, ~df.columns.duplicated()]
    missing = [c for c in PRICE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Yahoo response is missing columns: {missing}")
    df = df[PRICE_COLUMNS].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df.index.name = "Date"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df.dropna(subset=["Close"])


def _fetch_from_yahoo(ticker: str, start: str, end: str | None) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False,
                     auto_adjust=True, threads=False)
    if df is None or df.empty:
        # The per-ticker endpoint frequently works when the bulk one is throttled.
        df = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
    if df is None or df.empty:
        raise ValueError("empty response (unknown symbol or rate limit)")
    return _clean_prices(df)


def download_stock_data(ticker: str, start: str = "2012-01-01", end: str | None = None,
                        max_retries: int = 3, backoff_seconds: float = 2.0) -> pd.DataFrame:
    """Return daily split/dividend-adjusted prices, indexed by date.

    ``end=None`` means "up to the latest trading day". Every successful download
    is written to ``data_cache/<ticker>.csv``; if Yahoo is unreachable or
    rate-limits the request, the cached copy is returned instead.

    ``df.attrs["source"]`` is ``"yahoo"`` or ``"cache"`` so callers can tell the
    user when the data is not live.
    """
    ticker = normalize_ticker(ticker)
    start = str(start)
    cache_file = CACHE_DIR / f"{ticker.replace('^', '_')}.csv"
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            df = _fetch_from_yahoo(ticker, start, end)
            try:
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                df.to_csv(cache_file)
            except OSError as exc:  # read-only file system: the cache is optional
                logger.warning("Could not write cache for %s: %s", ticker, exc)
            df.attrs["source"] = "yahoo"
            logger.info("%s: %d rows from Yahoo (%s to %s)", ticker, len(df),
                        df.index[0].date(), df.index[-1].date())
            return df
        except Exception as exc:  # yfinance raises many unrelated exception types
            last_error = exc
            logger.warning("%s: attempt %d/%d failed: %s", ticker, attempt, max_retries, exc)
            if attempt < max_retries:
                time.sleep(backoff_seconds * attempt)

    if cache_file.exists():
        df = pd.read_csv(cache_file, index_col="Date", parse_dates=True)
        df = df.loc[start:] if end is None else df.loc[start:str(end)]
        if not df.empty:
            df.attrs["source"] = "cache"
            logger.warning("%s: Yahoo unavailable, using cache (last date %s)",
                           ticker, df.index[-1].date())
            return df

    raise DataDownloadError(
        f"Could not download '{ticker}' from Yahoo Finance ({last_error}). "
        "Check the symbol; if it is correct, Yahoo may be rate-limiting - try again in a minute."
    )


# ----------------------------------------------------------------------------
# 2. Features
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class PriceArrays:
    """Per-day arrays derived once from a price table (see :func:`prepare`)."""
    index: pd.DatetimeIndex
    close: np.ndarray       # adjusted close
    logp: np.ndarray        # log(close)
    ret: np.ndarray         # daily log-return, ret[0] = 0
    log_volume: np.ndarray  # log(volume); NaN where volume is missing or zero
    hl_range: np.ndarray    # log(high / low); NaN where unavailable
    long_vol: np.ndarray    # rolling ~2-year volatility; NaN until enough history

    def __len__(self) -> int:
        return len(self.close)


def prepare(df: pd.DataFrame) -> PriceArrays:
    """Validate a price table (as returned by :func:`download_stock_data`) and derive arrays."""
    df = df.dropna(subset=["Close"])
    close = df["Close"].to_numpy(dtype="float64")
    if len(close) == 0 or (close <= 0).any():
        raise ValueError("Close prices must be positive and non-empty.")
    logp = np.log(close)
    ret = np.diff(logp, prepend=logp[0])

    def column(name: str) -> np.ndarray:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").to_numpy(dtype="float64")
        return np.full(len(close), np.nan)

    volume = column("Volume")
    with np.errstate(divide="ignore", invalid="ignore"):
        log_volume = np.where(volume > 0, np.log(volume), np.nan)
        hl_range = np.log(column("High") / column("Low"))
    hl_range = np.where(np.isfinite(hl_range) & (hl_range >= 0), hl_range, np.nan)
    long_vol = (pd.Series(ret).rolling(LONG_VOL_WINDOW, min_periods=LONG_VOL_WINDOW // 2)
                .std().to_numpy())
    return PriceArrays(index=pd.DatetimeIndex(df.index), close=close, logp=logp, ret=ret,
                       log_volume=log_volume, hl_range=hl_range, long_vol=long_vol)


def make_window(pa: PriceArrays, t: int, lookback: int = LOOKBACK) -> tuple[np.ndarray, float]:
    """Features for a forecast made on day ``t`` (the last known day).

    Returns ``(features[lookback, N_FEATURES], vol)``; ``vol`` is the window's daily
    volatility, used to scale both inputs and outputs. Only data up to day ``t`` is used.

    Features: 1 return, 2 shape of the price path, 3 size of each day's move (shocks),
    4 volume relative to the window average, 5 high-low range, 6 whether the stock is
    calmer or more turbulent than its own long-run normal.
    """
    seg = pa.logp[t - lookback: t + 1]                  # lookback + 1 prices
    returns = np.diff(seg)                              # lookback returns
    vol = float(returns.std()) + 1e-8
    days = slice(t - lookback + 1, t + 1)

    log_volume = pa.log_volume[days]
    volume_feature = (log_volume - log_volume.mean() if np.isfinite(log_volume).all()
                      else np.zeros(lookback))
    range_feature = np.nan_to_num(pa.hl_range[days] / vol - 1.5, nan=0.0)
    long_vol = pa.long_vol[t]
    regime = np.log(vol / long_vol) if np.isfinite(long_vol) and long_vol > 0 else 0.0

    features = np.stack([
        returns / vol,
        (seg[1:] - seg[-1]) / (vol * np.sqrt(lookback)),
        np.abs(returns) / vol - 0.8,
        volume_feature,
        range_feature,
        np.full(lookback, regime),
    ], axis=-1)
    return np.clip(features, -FEATURE_CLIP, FEATURE_CLIP).astype("float32"), vol


def build_dataset(pa: PriceArrays, lookback: int = LOOKBACK, horizon: int = HORIZON,
                  start_pos: int | None = None, end_pos: int | None = None, step: int = 1):
    """Build samples whose forecast origin ``t`` lies in ``[start_pos, end_pos)``.

    Returns ``X, Y, vols, origins`` where ``X[i]`` has shape ``(lookback, N_FEATURES)`` and
    ``Y[i]`` is the cumulative log-return from day ``t`` to ``t+1 .. t+horizon``,
    divided by the window volatility.
    """
    n = len(pa)
    lo = max(lookback, start_pos or 0)
    hi = min(n - horizon, end_pos if end_pos is not None else n)
    X, Y, V, T = [], [], [], []
    for t in range(lo, hi, step):
        features, vol = make_window(pa, t, lookback)
        X.append(features)
        Y.append((pa.logp[t + 1: t + horizon + 1] - pa.logp[t]) / vol)
        V.append(vol)
        T.append(t)
    if not X:
        return (np.zeros((0, lookback, N_FEATURES), "float32"), np.zeros((0, horizon), "float32"),
                np.zeros(0), np.zeros(0, dtype=int))
    return np.array(X), np.array(Y, dtype="float32"), np.array(V), np.array(T)


# ----------------------------------------------------------------------------
# 3. Model artifacts + forecasting
# ----------------------------------------------------------------------------
def adjust_quantiles(raw: np.ndarray, median_shrink: float, band_scale: float) -> np.ndarray:
    """Apply the two post-processing constants chosen on validation data.

    ``median_shrink`` (0..1) pulls the central forecast towards "no change";
    ``band_scale`` widens or narrows the range so that it covers 80% of outcomes.
    Input and output have shape ``(n, horizon, 3)`` = (lower, median, upper).
    """
    raw = np.sort(np.asarray(raw, dtype="float64"), axis=-1)   # quantiles must not cross
    median = median_shrink * raw[..., 1]
    lower = median - band_scale * (raw[..., 1] - raw[..., 0])
    upper = median + band_scale * (raw[..., 2] - raw[..., 1])
    return np.stack([lower, median, upper], axis=-1)


@dataclass(frozen=True)
class Artifacts:
    """The trained model plus the metadata written by ``train.py``."""
    model: object
    median_shrink: float = 1.0
    band_scale: float = 1.0
    lookback: int = LOOKBACK
    horizon: int = HORIZON
    trained_on: tuple[str, ...] = ()
    data_until: str = "unknown"

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Vol-scaled (lower, median, upper) cumulative log-returns, shape ``(n, horizon, 3)``."""
        raw = np.asarray(self.model.predict(X, verbose=0, batch_size=256))
        return adjust_quantiles(raw, self.median_shrink, self.band_scale)


def load_artifacts(model_path: Path = MODEL_PATH, meta_path: Path = META_PATH) -> Artifacts:
    """Load the Keras model and its metadata, validating that they match this code."""
    from keras.models import load_model  # lazy on purpose: see the import note at the top

    model_path, meta_path = Path(model_path), Path(meta_path)
    for path in (model_path, meta_path):
        if not path.exists():
            raise FileNotFoundError(f"{path} not found - run `python train.py` to train the model.")
    meta = json.loads(meta_path.read_text())
    model = load_model(model_path, compile=False)  # inference only: no optimizer needed

    expected_in, expected_out = (LOOKBACK, N_FEATURES), (HORIZON, len(QUANTILES))
    if tuple(model.input_shape[1:]) != expected_in or tuple(model.output_shape[1:]) != expected_out:
        raise ValueError(f"Model shapes {model.input_shape} -> {model.output_shape} do not match "
                         "stock_utils.py - retrain with `python train.py`.")
    return Artifacts(model=model,
                     median_shrink=float(meta["median_shrink"]),
                     band_scale=float(meta["band_scale"]),
                     trained_on=tuple(meta.get("trained_on", ())),
                     data_until=meta.get("data_until", "unknown"))


def _to_prices(scaled: np.ndarray, last_price, vol) -> np.ndarray:
    return last_price * np.exp(scaled * vol)


def future_dates(index: pd.DatetimeIndex, t: int, horizon: int) -> pd.DatetimeIndex:
    """Dates of the ``horizon`` trading days after position ``t``.

    Real dates are used where they exist (backtests); beyond the data, Monday-Friday
    business days are used. Exchange holidays are not excluded, so calendar dates far
    ahead are approximate.
    """
    dates = index[t + 1: t + 1 + horizon]
    if len(dates) < horizon:
        extra = pd.bdate_range(index[-1] + pd.Timedelta(days=1), periods=horizon - len(dates))
        dates = dates.append(extra)
    return dates


def forecast_path(art: Artifacts, pa: PriceArrays, horizon: int = HORIZON,
                  t: int | None = None) -> pd.DataFrame:
    """Forecast ``horizon`` trading days after position ``t``.

    ``t=None`` uses the last row, i.e. a genuine forecast of the unknown future.
    Columns: ``Predicted`` (median), ``Lower`` and ``Upper`` (80% range).
    """
    if t is None:
        t = len(pa) - 1
    if t < art.lookback:
        raise ValueError(f"Need at least {art.lookback + 1} days of history, got {t + 1}.")
    features, vol = make_window(pa, t, art.lookback)
    q = _to_prices(art.predict(features[None])[0][:horizon], pa.close[t], vol)
    out = pd.DataFrame({"Predicted": q[:, 1], "Lower": q[:, 0], "Upper": q[:, 2]},
                       index=future_dates(pa.index, t, horizon))
    out.index.name = "Date"
    return out


# ----------------------------------------------------------------------------
# 4. Backtesting
# ----------------------------------------------------------------------------
def walk_forward(art: Artifacts, pa: PriceArrays, start_pos: int, step: int = 10) -> dict | None:
    """Out-of-sample multi-step backtest.

    At many past dates, forecast the next ``HORIZON`` days using only data available
    on that date and compare with what happened. Two baselines are evaluated too:

    * naive - the price stays where it is
    * drift - the price keeps growing at its average rate up to that date

    Returns ``None`` if there is not enough data, otherwise a dict of arrays shaped
    ``(n_origins, horizon)``: actual, model, lower, upper, naive, drift.
    """
    X, _, V, T = build_dataset(pa, art.lookback, art.horizon, start_pos=start_pos, step=step)
    if len(X) == 0:
        return None
    horizon = art.horizon
    origin_price = pa.close[T][:, None]
    q = _to_prices(art.predict(X), origin_price[..., None], V[:, None, None])
    h = np.arange(1, horizon + 1)
    mu = np.array([(pa.logp[t] - pa.logp[0]) / t for t in T])  # average daily log-return so far
    return {
        "origins": T,
        "actual": np.stack([pa.close[t + 1: t + horizon + 1] for t in T]),
        "model": q[..., 1], "lower": q[..., 0], "upper": q[..., 2],
        "naive": np.repeat(origin_price, horizon, axis=1),
        "drift": origin_price * np.exp(mu[:, None] * h[None, :]),
    }


def recent_shock(pa: PriceArrays, window: int = SHOCK_WINDOW) -> dict | None:
    """Detect an unusually sharp move in the last ``window`` trading days.

    A move counts as unusual when a single day moved at least ``SHOCK_DAILY_MULTIPLE``
    times the stock's normal daily move, or the whole window moved at least
    ``SHOCK_WINDOW_MULTIPLE`` times what is normal for that many days. "Normal" is the
    stock's own ~2-year volatility. With these thresholds a typical stock is flagged on
    roughly 1 day in 12. Such moves are usually news-driven, which the model cannot see.

    Returns ``None`` if nothing unusual happened, otherwise the size and date of the
    largest daily move and the total move over the window (as fractions, e.g. -0.19).
    """
    if len(pa) <= window:
        return None
    normal = pa.long_vol[-1]
    if not (np.isfinite(normal) and normal > 0):
        normal = float(pa.ret[1:].std())
    if not normal > 0:
        return None
    recent = pa.ret[-window:]
    worst = int(np.abs(recent).argmax())
    total = float(recent.sum())
    if (abs(recent[worst]) < SHOCK_DAILY_MULTIPLE * normal
            and abs(total) < SHOCK_WINDOW_MULTIPLE * normal * np.sqrt(window)):
        return None
    return {
        "window": window,
        "total_move": float(np.expm1(total)),
        "largest_daily_move": float(np.expm1(recent[worst])),
        "largest_move_date": pa.index[len(pa) - window + worst],
    }


def mape_by_horizon(actual: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Mean absolute percentage error for each forecast distance (1..horizon)."""
    return np.mean(np.abs(pred - actual) / actual, axis=0) * 100


def coverage(bt: dict) -> float:
    """Share (%) of actual prices that fell inside the 80% range."""
    return float(((bt["actual"] >= bt["lower"]) & (bt["actual"] <= bt["upper"])).mean() * 100)


def summary_table(bt: dict, horizons=(1, 5, 10, 20, 50, 100)) -> pd.DataFrame:
    """MAPE (%) of the model and both baselines at selected horizons."""
    labels = {"model": "LSTM model", "naive": "Naive (no change)", "drift": "Drift (avg growth)"}
    horizons = [h for h in horizons if h <= bt["actual"].shape[1]]
    rows = {label: mape_by_horizon(bt["actual"], bt[key])[[h - 1 for h in horizons]]
            for key, label in labels.items()}
    return pd.DataFrame(rows, index=[f"{h}d ahead" for h in horizons]).T.round(2)
