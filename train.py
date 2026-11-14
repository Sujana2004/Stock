"""Train the forecasting model and write everything the app needs into ``models/``.

Usage:  python train.py            (about 15 minutes on a CPU)

Steps
-----
1. Download the training universe from Yahoo Finance.
2. Split every stock by time: 70% train, 10% validation, 20% test, with gaps of
   ``HORIZON`` days so that no target overlaps the next block.
3. Train ``N_MEMBERS`` LSTMs that differ only in their random seed and average them
   (an ensemble is more stable than any single run).
4. On the validation block only, choose two constants: how far to shrink the central
   forecast towards "no change", and how much to scale the range so it covers 80%.
5. Evaluate once on the untouched test block and save the report.

Outputs: models/stock_lstm.keras, models/model_meta.json, models/metrics.json
"""
from __future__ import annotations

import json
import logging

import numpy as np

import stock_utils as su  # must be imported before keras (see the note in stock_utils)

import keras  # noqa: E402
from keras import layers  # noqa: E402

logger = logging.getLogger("train")

START = "2012-01-01"
UNIVERSE = [
    # US: technology, finance, energy, retail, industry, healthcare, media
    "GOOG", "AAPL", "MSFT", "AMZN", "TSLA", "NVDA", "INTC", "HDB", "JPM", "XOM",
    "WMT", "KO", "BA", "PFE", "DIS", "CAT",
    # India (NSE)
    "TCS.NS", "BAJAJFINSV.NS", "RELIANCE.NS", "INFY.NS", "HDFCBANK.NS", "ITC.NS",
    "SBIN.NS", "LT.NS",
]
TRAIN_FRACTION, VAL_FRACTION = 0.70, 0.80      # test = everything after VAL_FRACTION
N_MEMBERS = 3
SEEDS = (42, 43, 44)
TRAIN_STEP = 3                                 # neighbouring days are near-duplicates
EVAL_STEP = 10
SHRINK_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
REPORT_HORIZONS = (1, 5, 10, 20, 50, 100)
_Q = np.array(su.QUANTILES, dtype="float32")


def pinball_loss(y_true, y_pred):
    """Quantile loss: its minimum is the true 10% / 50% / 90% quantile, so large moves
    (outliers) shape the range instead of dragging a mean around."""
    error = keras.ops.expand_dims(y_true, -1) - y_pred
    return keras.ops.mean(keras.ops.maximum(_Q * error, (_Q - 1.0) * error))


def build_member() -> keras.Model:
    return keras.Sequential([
        layers.Input((su.LOOKBACK, su.N_FEATURES)),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dropout(0.2),
        layers.Dense(64, activation="relu"),
        layers.Dense(su.HORIZON * len(su.QUANTILES)),
        layers.Reshape((su.HORIZON, len(su.QUANTILES))),
    ])


def load_universe(tickers=UNIVERSE, start=START) -> dict[str, su.PriceArrays]:
    arrays = {}
    for ticker in tickers:
        try:
            arrays[ticker] = su.prepare(su.download_stock_data(ticker, start))
        except (su.DataDownloadError, ValueError) as exc:
            logger.warning("Skipping %s: %s", ticker, exc)
    if not arrays:
        raise RuntimeError("No training data could be downloaded.")
    return arrays


def split_positions(n: int) -> tuple[int, int]:
    return int(n * TRAIN_FRACTION), int(n * VAL_FRACTION)


def make_split(arrays: dict[str, su.PriceArrays], part: str, step: int):
    """Stack samples of one part ('train' | 'val' | 'test') across all tickers."""
    X, Y, V = [], [], []
    for pa in arrays.values():
        a, b = split_positions(len(pa))
        bounds = {"train": (None, a - su.HORIZON), "val": (a, b - su.HORIZON), "test": (b, None)}[part]
        x, y, v, _ = su.build_dataset(pa, start_pos=bounds[0], end_pos=bounds[1], step=step)
        X.append(x), Y.append(y), V.append(v)
    return np.concatenate(X), np.concatenate(Y), np.concatenate(V)


def train_member(seed: int, train, val) -> keras.Model:
    keras.utils.set_random_seed(seed)
    model = build_member()
    model.compile(keras.optimizers.Adam(1e-3), loss=pinball_loss)
    history = model.fit(
        train[0], train[1], validation_data=(val[0], val[1]),
        epochs=30, batch_size=128, verbose=0,
        callbacks=[keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True),
                   keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)])
    logger.info("seed %d: best validation loss %.4f after %d epochs", seed,
                min(history.history["val_loss"]), len(history.history["val_loss"]))
    return model


def build_ensemble(members: list[keras.Model]) -> keras.Model:
    """Average the members in one model that loads without any training-only code.

    The trained members carry a compile config that refers to ``pinball_loss``; a model
    saved with them could only be loaded where that function is importable. Copying the
    weights into fresh, uncompiled members keeps the saved file inference-only.
    """
    clean = []
    for member in members:
        clone = build_member()
        clone.set_weights(member.get_weights())
        clean.append(clone)
    inputs = layers.Input((su.LOOKBACK, su.N_FEATURES))
    outputs = layers.Average()([m(inputs) for m in clean]) if len(clean) > 1 else clean[0](inputs)
    return keras.Model(inputs, outputs, name="stock_lstm_ensemble")


def _mape(q_median, Y, V) -> float:
    """Mean absolute % price error of a vol-scaled median forecast, over all horizons."""
    return float(np.mean(np.abs(np.exp((q_median - Y) * V[:, None]) - 1)) * 100)


def choose_constants(raw: np.ndarray, Y: np.ndarray, V: np.ndarray) -> tuple[float, float]:
    """Pick median_shrink (lowest validation MAPE) and band_scale (80% validation coverage)."""
    shrink = min(SHRINK_GRID, key=lambda a: _mape(su.adjust_quantiles(raw, a, 1.0)[..., 1], Y, V))

    def cover(scale: float) -> float:
        q = su.adjust_quantiles(raw, shrink, scale)
        return float(((Y >= q[..., 0]) & (Y <= q[..., 2])).mean())

    scales = np.round(np.arange(0.70, 1.61, 0.01), 2)
    scale = float(min(scales, key=lambda s: abs(cover(s) - 0.80)))
    logger.info("validation: median_shrink=%.2f, band_scale=%.2f (coverage %.1f%%)",
                shrink, scale, cover(scale) * 100)
    return float(shrink), scale


def evaluate(art: su.Artifacts, arrays: dict[str, su.PriceArrays]) -> dict:
    """Walk-forward test on the held-out last block of every ticker."""
    idx = [h - 1 for h in REPORT_HORIZONS]
    per_ticker, pooled = {}, {k: [] for k in ("actual", "model", "naive", "drift", "lower", "upper")}
    for ticker, pa in arrays.items():
        bt = su.walk_forward(art, pa, start_pos=split_positions(len(pa))[1], step=EVAL_STEP)
        if bt is None:
            continue
        for key in pooled:
            pooled[key].append(bt[key])
        per_ticker[ticker] = {
            "forecasts": int(len(bt["origins"])),
            "coverage_80": round(su.coverage(bt), 1),
            **{name: su.mape_by_horizon(bt["actual"], bt[name])[idx].round(2).tolist()
               for name in ("model", "naive", "drift")},
        }
    pooled = {k: np.concatenate(v) for k, v in pooled.items()}
    overall = {
        "forecasts": int(len(pooled["actual"])),
        "coverage_80": round(su.coverage(pooled), 1),
        **{name: su.mape_by_horizon(pooled["actual"], pooled[name])[idx].round(2).tolist()
           for name in ("model", "naive", "drift")},
    }
    return {"horizons": list(REPORT_HORIZONS), "overall": overall, "per_ticker": per_ticker}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    arrays = load_universe()
    train, val = make_split(arrays, "train", TRAIN_STEP), make_split(arrays, "val", TRAIN_STEP)
    logger.info("train samples %s, validation samples %s", train[0].shape, val[0].shape)

    members = [train_member(seed, train, val) for seed in SEEDS[:N_MEMBERS]]
    ensemble = build_ensemble(members)
    raw_val = np.asarray(ensemble.predict(val[0], verbose=0, batch_size=256))
    shrink, scale = choose_constants(raw_val, val[1], val[2])

    data_until = max(str(pa.index[-1].date()) for pa in arrays.values())
    art = su.Artifacts(model=ensemble, median_shrink=shrink, band_scale=scale,
                       trained_on=tuple(arrays), data_until=data_until)
    metrics = evaluate(art, arrays)

    su.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    ensemble.save(su.MODEL_PATH)
    su.META_PATH.write_text(json.dumps({
        "lookback": su.LOOKBACK, "horizon": su.HORIZON, "n_features": su.N_FEATURES,
        "quantiles": list(su.QUANTILES), "median_shrink": shrink, "band_scale": scale,
        "ensemble_members": len(members), "trained_on": list(arrays), "data_until": data_until,
        "split": {"train": TRAIN_FRACTION, "validation_until": VAL_FRACTION},
    }, indent=2))
    (su.MODEL_PATH.parent / "metrics.json").write_text(json.dumps(metrics, indent=2))

    o = metrics["overall"]
    logger.info("TEST (%d forecasts): horizons %s", o["forecasts"], metrics["horizons"])
    for name in ("model", "naive", "drift"):
        logger.info("  %-6s MAPE %s", name, o[name])
    logger.info("  80%% range contained the real price %.1f%% of the time", o["coverage_80"])


if __name__ == "__main__":
    main()
