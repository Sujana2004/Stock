"""Scenario simulation and the decision tools built on it.

The model forecasts three numbers per future day (10%, 50% and 90% quantiles). The
questions an investor asks - "what is the chance of reaching my target?", "will my
stop-loss be hit first?", "how much could I lose?" - need whole price paths.

:func:`simulate_paths` produces them in two steps:

1. Block bootstrap: stitch together randomly chosen multi-day blocks of the stock's
   own past returns. The paths therefore contain the stock's real crashes, jumps and
   bursts of turbulence, not an idealised bell curve.
2. Quantile matching: stretch the paths so that on every future day their 10%, 50%
   and 90% levels equal the model's forecast. The chart's range and every probability
   in the app then come from the same set of paths.

:func:`calibration_table` checks the result against history: when the tool said
"x% chance of finishing below this price", how often did that happen?
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stock_utils import Artifacts, PriceArrays, _to_prices, build_dataset

DEFAULT_PATHS = 5000
BLOCK_DAYS = 10            # length of each resampled block (keeps turbulence clustered)
MIN_HISTORY_DAYS = 60
BAD_CASE_LEVEL = 0.05      # "bad case" = the outcome only 1 in 20 scenarios is worse than
CALIBRATION_LEVELS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


def simulate_paths(pa: PriceArrays, forecast: np.ndarray, t: int | None = None,
                   n_paths: int = DEFAULT_PATHS, block: int = BLOCK_DAYS, seed: int = 0) -> np.ndarray:
    """Simulate price paths for the days after position ``t``.

    ``forecast`` has shape ``(horizon, 3)``: the model's (lower, median, upper) prices.
    Only returns up to day ``t`` are resampled, so backtests do not see the future.
    Returns prices with shape ``(n_paths, horizon)``. The same seed gives the same paths.
    """
    if t is None:
        t = len(pa) - 1
    forecast = np.asarray(forecast, dtype="float64")
    horizon = len(forecast)
    history = pa.ret[1: t + 1]
    if len(history) < max(MIN_HISTORY_DAYS, block):
        raise ValueError(f"Need at least {MIN_HISTORY_DAYS} days of history to simulate scenarios.")
    history = history - history.mean()       # direction comes from the model, not from the past

    rng = np.random.default_rng(seed)
    n_blocks = -(-horizon // block)
    starts = rng.integers(0, len(history) - block + 1, size=(n_paths, n_blocks))
    steps = history[starts[..., None] + np.arange(block)].reshape(n_paths, -1)[:, :horizon]
    sim = np.cumsum(steps, axis=1)           # cumulative log-returns

    s_lo, s_mid, s_hi = np.quantile(sim, [0.1, 0.5, 0.9], axis=0)
    m_lo, m_mid, m_hi = (np.log(forecast[:, i] / pa.close[t]) for i in range(3))
    up = (m_hi - m_mid) / np.maximum(s_hi - s_mid, 1e-12)
    down = (m_mid - m_lo) / np.maximum(s_mid - s_lo, 1e-12)
    matched = m_mid + (sim - s_mid) * np.where(sim >= s_mid, up, down)
    return pa.close[t] * np.exp(matched)


# ----------------------------------------------------------------------------
# Questions answered from the simulated paths
# ----------------------------------------------------------------------------
def prob_above(paths: np.ndarray, price: float, day: int) -> float:
    """Probability that the close is above ``price`` on trading day ``day`` (1-based)."""
    return float((paths[:, day - 1] > price).mean())


def prob_touch(paths: np.ndarray, price: float, day: int, last_price: float) -> float:
    """Probability that the close reaches ``price`` at any time within ``day`` trading days."""
    window = paths[:, :day]
    hit = window.max(axis=1) >= price if price >= last_price else window.min(axis=1) <= price
    return float(hit.mean())


def target_vs_stop(paths: np.ndarray, target: float, stop: float, day: int) -> dict[str, float]:
    """Which is reached first within ``day`` trading days: the target or the stop-loss?

    Uses daily closing prices, so a stop that is touched only during the day is not
    counted, and a real stop order can fill below the stop price after a gap.
    """
    if not stop < target:
        raise ValueError("The stop-loss must be below the target.")
    window = paths[:, :day]
    hit_target, hit_stop = window >= target, window <= stop
    never = day + 1
    first_target = np.where(hit_target.any(axis=1), hit_target.argmax(axis=1), never)
    first_stop = np.where(hit_stop.any(axis=1), hit_stop.argmax(axis=1), never)
    return {
        "target_first": float((first_target < first_stop).mean()),
        "stop_first": float((first_stop < first_target).mean()),
        "neither": float(((first_target == never) & (first_stop == never)).mean()),
    }


def outcome_summary(paths: np.ndarray, day: int, last_price: float, amount: float) -> dict[str, float]:
    """What ``amount`` invested today could be worth after ``day`` trading days."""
    growth = paths[:, day - 1] / last_price
    bad, typical, good = np.quantile(growth, [BAD_CASE_LEVEL, 0.5, 1 - BAD_CASE_LEVEL])
    return {
        "prob_loss": float((growth < 1).mean()),
        "bad_case_value": float(amount * bad),
        "typical_value": float(amount * typical),
        "good_case_value": float(amount * good),
        "bad_case_return": float(bad - 1),
    }


def max_position(paths: np.ndarray, day: int, last_price: float, max_loss: float) -> float | None:
    """Largest amount to invest so that the bad-case loss stays within ``max_loss``.

    Returns ``None`` when even the bad case is not a loss (no limit from this rule).
    """
    bad_return = float(np.quantile(paths[:, day - 1] / last_price, BAD_CASE_LEVEL) - 1)
    if bad_return >= 0:
        return None
    return float(max_loss / -bad_return)


# ----------------------------------------------------------------------------
# Track record
# ----------------------------------------------------------------------------
def calibration_table(art: Artifacts, pa: PriceArrays, start_pos: int, step: int = 10,
                      horizons: tuple[int, ...] = (20, 60, 100),
                      levels: tuple[float, ...] = CALIBRATION_LEVELS,
                      n_paths: int = 1000) -> tuple[pd.DataFrame, int] | None:
    """How often past probability statements came true.

    For each past forecast date (using only data up to that date) and each level x,
    find the price the tool gave an x% chance of finishing below, then check whether
    the real price did finish below it. A well-calibrated tool shows observed
    frequencies close to x.

    Returns ``(table, n_forecasts)``: rows are the stated chances, columns the horizons,
    values the observed frequency in %. ``None`` if there is not enough history.
    """
    X, _, V, T = build_dataset(pa, art.lookback, art.horizon, start_pos=start_pos, step=step)
    if len(X) == 0:
        return None
    horizons = tuple(h for h in horizons if h <= art.horizon)
    quantile_prices = _to_prices(art.predict(X), pa.close[T][:, None, None], V[:, None, None])
    below = np.zeros((len(T), len(levels), len(horizons)), dtype=bool)
    for i, t in enumerate(T):
        paths = simulate_paths(pa, quantile_prices[i], t=int(t), n_paths=n_paths, seed=int(t))
        for j, h in enumerate(horizons):
            thresholds = np.quantile(paths[:, h - 1], levels)
            below[i, :, j] = pa.close[t + h] <= thresholds
    table = pd.DataFrame(below.mean(axis=0) * 100,
                         index=[f"{level:.0%}" for level in levels],
                         columns=[f"{h} days" for h in horizons]).round(0)
    table.index.name = "Stated chance"
    return table, len(T)
