"""Offline tests: no network access is required."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scenarios as sc  # noqa: E402
import stock_utils as su  # noqa: E402


class FlatModel:
    """Stand-in for the Keras model: median 'no change', range of +/- sqrt(h) volatilities."""

    def predict(self, X, **_):
        spread = np.sqrt(np.arange(1, su.HORIZON + 1))
        one = np.stack([-spread, np.zeros(su.HORIZON), spread], axis=-1)
        return np.repeat(one[None], len(X), axis=0).astype("float32")


def make_frame(n=700, seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, n))) * scale
    spread = np.abs(rng.normal(0.01, 0.003, n))
    return pd.DataFrame({"Open": close, "High": close * (1 + spread), "Low": close * (1 - spread),
                         "Close": close, "Volume": rng.integers(1_000_000, 5_000_000, n)},
                        index=pd.bdate_range("2020-01-01", periods=n))


@pytest.fixture
def frame():
    return make_frame()


@pytest.fixture
def pa(frame):
    return su.prepare(frame)


@pytest.fixture
def art():
    return su.Artifacts(model=FlatModel())


# ---------------------------------------------------------------- data
@pytest.mark.parametrize("raw, expected", [(" goog ", "GOOG"), ("tcs.ns", "TCS.NS"), ("^nsei", "^NSEI")])
def test_normalize_ticker_accepts_valid(raw, expected):
    assert su.normalize_ticker(raw) == expected


@pytest.mark.parametrize("bad", ["", "../etc/passwd", "GOOG; rm", "GO^OG", "A" * 30])
def test_normalize_ticker_rejects_invalid(bad):
    with pytest.raises(ValueError):
        su.normalize_ticker(bad)


def test_clean_prices_flattens_multiindex():
    idx = pd.to_datetime(["2024-01-03", "2024-01-02", "2024-01-02"])
    cols = pd.MultiIndex.from_product([su.PRICE_COLUMNS, ["GOOG"]])
    raw = pd.DataFrame(np.arange(15, dtype=float).reshape(3, 5), index=idx, columns=cols)
    out = su._clean_prices(raw)
    assert list(out.columns) == su.PRICE_COLUMNS
    assert out.index.is_monotonic_increasing and out.index.is_unique


# ---------------------------------------------------------------- features
def test_build_dataset_shapes(pa):
    X, Y, V, T = su.build_dataset(pa)
    assert X.shape == (len(pa) - su.LOOKBACK - su.HORIZON, su.LOOKBACK, su.N_FEATURES)
    assert Y.shape == (len(X), su.HORIZON)
    assert np.isfinite(X).all() and np.isfinite(Y).all()


def test_features_do_not_look_ahead(frame, pa):
    X, _, _, T = su.build_dataset(pa)
    changed = frame.copy()
    changed.iloc[T[0] + 1:] *= 2          # alter everything after the first origin
    X2, _, _, _ = su.build_dataset(su.prepare(changed))
    np.testing.assert_allclose(X[0], X2[0])


def test_features_are_scale_free(pa):
    other = su.prepare(make_frame(scale=83.0))    # same stock quoted in another currency
    X1, Y1, _, _ = su.build_dataset(pa)
    X2, Y2, _, _ = su.build_dataset(other)
    np.testing.assert_allclose(X1, X2, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(Y1, Y2, rtol=1e-4, atol=1e-4)


def test_features_survive_missing_volume_and_extreme_days(frame):
    frame = frame.copy()
    frame["Volume"] = 0                                    # indices often have no volume
    frame.iloc[400, frame.columns.get_loc("Close")] *= 5   # a bad tick
    X, _, _, _ = su.build_dataset(su.prepare(frame))
    assert np.isfinite(X).all()
    assert np.abs(X).max() <= su.FEATURE_CLIP
    assert (X[..., 3] == 0).all()


def test_prepare_rejects_bad_prices(frame):
    frame = frame.copy()
    frame.iloc[5, frame.columns.get_loc("Close")] = -1
    with pytest.raises(ValueError):
        su.prepare(frame)


# ---------------------------------------------------------------- forecasting
def test_adjust_quantiles_orders_shrinks_and_scales():
    raw = np.array([[[2.0, 1.0, -1.0]]])                  # crossed on purpose
    q = su.adjust_quantiles(raw, median_shrink=0.5, band_scale=2.0)
    np.testing.assert_allclose(q, [[[-3.5, 0.5, 2.5]]])


def test_forecast_path_future_dates_and_range(art, pa):
    fc = su.forecast_path(art, pa)
    assert len(fc) == su.HORIZON
    assert fc.index[0] > pa.index[-1] and fc.index.is_monotonic_increasing
    assert (fc.index.dayofweek < 5).all()
    np.testing.assert_allclose(fc["Predicted"], pa.close[-1])
    assert (fc["Lower"] < fc["Predicted"]).all() and (fc["Predicted"] < fc["Upper"]).all()


def test_forecast_path_requires_enough_history(art, frame):
    with pytest.raises(ValueError):
        su.forecast_path(art, su.prepare(frame.iloc[:50]))


def test_walk_forward_flat_model_equals_naive(art, pa, frame):
    bt = su.walk_forward(art, pa, start_pos=300, step=20)
    np.testing.assert_allclose(bt["model"], bt["naive"])
    table = su.summary_table(bt)
    assert table.loc["LSTM model"].equals(table.loc["Naive (no change)"])
    assert 0 <= su.coverage(bt) <= 100
    assert su.walk_forward(art, su.prepare(frame.iloc[:150]), start_pos=120) is None


# ---------------------------------------------------------------- scenarios
@pytest.fixture
def paths(art, pa):
    fc = su.forecast_path(art, pa)
    return sc.simulate_paths(pa, fc[["Lower", "Predicted", "Upper"]].to_numpy(), n_paths=4000)


def test_simulated_paths_match_the_forecast_quantiles(art, pa, paths):
    fc = su.forecast_path(art, pa)
    assert paths.shape == (4000, su.HORIZON) and (paths > 0).all()
    lo, mid, hi = np.quantile(paths, [0.1, 0.5, 0.9], axis=0)
    np.testing.assert_allclose(lo, fc["Lower"], rtol=1e-4)
    np.testing.assert_allclose(mid, fc["Predicted"], rtol=1e-4)
    np.testing.assert_allclose(hi, fc["Upper"], rtol=1e-4)


def test_simulation_is_reproducible_and_uses_only_the_past(art, frame, pa):
    t = 500
    fc = su.forecast_path(art, pa, t=t)[["Lower", "Predicted", "Upper"]].to_numpy()
    first = sc.simulate_paths(pa, fc, t=t, n_paths=200, seed=1)
    np.testing.assert_array_equal(first, sc.simulate_paths(pa, fc, t=t, n_paths=200, seed=1))
    changed = frame.copy()
    changed.iloc[t + 1:] *= 3
    np.testing.assert_allclose(first, sc.simulate_paths(su.prepare(changed), fc, t=t, n_paths=200, seed=1))


def test_probabilities_are_consistent(pa, paths):
    last = pa.close[-1]
    assert sc.prob_above(paths, 0.0, 50) == 1.0
    assert sc.prob_above(paths, last * 100, 50) == 0.0
    assert abs(sc.prob_above(paths, last, 50) - 0.5) < 0.05            # flat median
    target = last * 1.1
    assert sc.prob_touch(paths, target, 50, last) >= sc.prob_above(paths, target, 50)

    race = sc.target_vs_stop(paths, target, last * 0.95, 50)
    assert abs(sum(race.values()) - 1.0) < 1e-9
    with pytest.raises(ValueError):
        sc.target_vs_stop(paths, last * 0.9, last * 0.95, 50)


def test_outcomes_and_position_size(pa, paths):
    last = pa.close[-1]
    out = sc.outcome_summary(paths, 60, last, 50_000)
    assert out["bad_case_value"] < out["typical_value"] < out["good_case_value"]
    limit = sc.max_position(paths, 60, last, max_loss=5_000)
    # investing exactly the limit makes the bad-case loss equal to the stated maximum
    assert sc.outcome_summary(paths, 60, last, limit)["bad_case_value"] == pytest.approx(limit - 5_000)
    assert sc.max_position(np.full((10, 60), last * 1.2), 60, last, 5_000) is None


def test_calibration_table_shape(art, pa):
    table, n = sc.calibration_table(art, pa, start_pos=300, step=40, n_paths=300)
    assert n > 0 and table.shape == (len(sc.CALIBRATION_LEVELS), 3)
    assert ((table >= 0) & (table <= 100)).all().all()
    assert (table.diff().dropna() >= 0).all().all()       # higher stated chance, higher frequency


# ---------------------------------------------------------------- saved model
def test_saved_model_loads_and_predicts(pa):
    pytest.importorskip("keras")
    art = su.load_artifacts()
    fc = su.forecast_path(art, pa)
    assert fc.notna().all().all()
    assert (fc["Lower"] <= fc["Predicted"]).all() and (fc["Predicted"] <= fc["Upper"]).all()


def test_yahoo_client_works_after_tensorflow():
    """TensorFlow bundles its own libcurl. If it is loaded before curl_cffi, Yahoo
    requests fail with ImpersonateError. Run in a fresh interpreter because the
    result depends on import order."""
    pytest.importorskip("keras")
    code = (
        "import stock_utils as su\n"
        "su.load_artifacts()\n"                      # loads TensorFlow
        "from curl_cffi import requests\n"
        "from curl_cffi.requests.exceptions import ImpersonateError\n"
        "try:\n"
        "    requests.Session(impersonate='chrome').get('http://127.0.0.1:9/')\n"  # closed port
        "except ImpersonateError:\n"
        "    raise\n"                                # the bug this test guards against
        "except Exception:\n"
        "    pass\n"                                 # connection refused is expected
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=su.BASE_DIR,
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr[-2000:]


def test_ensemble_saves_and_loads_without_training_code(tmp_path):
    """The saved model must load with plain ``load_model`` (the app never imports train.py)."""
    keras = pytest.importorskip("keras")
    import train

    members = []
    for _ in range(2):
        member = train.build_member()
        member.compile("adam", loss=train.pinball_loss)     # as after real training
        members.append(member)
    path = tmp_path / "ensemble.keras"
    ensemble = train.build_ensemble(members)
    ensemble.save(path)

    code = ("import sys, keras\n"
            "m = keras.models.load_model(sys.argv[1], compile=False)\n"
            "print(m.output_shape)\n")
    result = subprocess.run([sys.executable, "-c", code, str(path)], capture_output=True,
                            text=True, timeout=300)
    assert result.returncode == 0, result.stderr[-2000:]
    assert str((None, su.HORIZON, len(su.QUANTILES))) in result.stdout
    del keras
