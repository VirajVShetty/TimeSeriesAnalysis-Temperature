"""Smoke and sanity tests for the model zoo.

These do not chase accuracy — they assert the contract every forecaster must
honour, plus one substantive check: on data with real structure, the learned
models must beat naive persistence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tsa_temperature.config import CITY_COL, DATE_COL, TARGET_COL
from tsa_temperature.data import ClimatePanel
from tsa_temperature.models import (
    TORCH_AVAILABLE,
    ClimatologyForecaster,
    ClimatologyPlusPersistence,
    GlobalGBMForecaster,
    LSTMForecaster,
    NaivePersistence,
    QuantileGBMForecaster,
    SARIMAXFourier,
    SeasonalNaive,
    STLETSForecaster,
)
from tsa_temperature.simulate import simulate_climate_panel

HORIZON = 7


@pytest.fixture(scope="module")
def sim_panel() -> pd.DataFrame:
    raw = simulate_climate_panel(start="2024-01-01", end="2025-12-31", seed=13)
    rename = {
        "Date": "date",
        "City": "city",
        "State": "state",
        "Temperature_Max (°C)": "temp_max",
        "Temperature_Min (°C)": "temp_min",
        "Temperature_Avg (°C)": "temp_avg",
        "Humidity (%)": "humidity",
        "Rainfall (mm)": "rainfall",
        "Wind_Speed (km/h)": "wind_speed",
        "AQI": "aqi",
        "AQI_Category": "aqi_category",
        "Pressure (hPa)": "pressure",
        "Cloud_Cover (%)": "cloud_cover",
    }
    df = raw.rename(columns=rename)
    df["date"] = pd.to_datetime(df["date"])
    return ClimatePanel(df).fill_calendar().frame


@pytest.fixture(scope="module")
def split(sim_panel):
    origin = pd.Timestamp("2025-11-01")
    train = sim_panel[sim_panel[DATE_COL] <= origin]
    truth = sim_panel.rename(
        columns={DATE_COL: "target_date", TARGET_COL: "y_true"}
    )[[CITY_COL, "target_date", "y_true"]]
    return train, truth, origin


def _score(model, train, truth) -> float:
    preds = model.fit(train).predict(HORIZON).merge(
        truth, on=[CITY_COL, "target_date"], how="inner"
    )
    assert len(preds) > 0
    return float((preds["y_true"] - preds["y_pred"]).abs().mean())


@pytest.fixture(scope="module")
def backtest(sim_panel):
    """Multi-origin backtest of the models a single origin cannot separate.

    A single origin yields only ``n_cities`` points per horizon (10 here),
    which is far too few to rank models: sampling noise dominates. Accuracy
    claims in these tests are therefore made against a rolling-origin
    backtest.
    """
    from dataclasses import replace

    from tsa_temperature.backtest import walk_forward
    from tsa_temperature.config import FORECAST

    cfg = replace(
        FORECAST, horizon=14, n_backtest_origins=6, origin_stride=14, min_train_days=380
    )
    factories = {
        "naive": lambda h: NaivePersistence(),
        "climatology": lambda h: ClimatologyForecaster(),
        "clim+persistence": lambda h: ClimatologyPlusPersistence(),
        "gbm": lambda h: GlobalGBMForecaster(horizon=h),
    }
    return walk_forward(sim_panel, factories, cfg=cfg, verbose=False)


def _mae(result, model: str, horizon: int | None = None) -> float:
    d = result.predictions
    d = d[d["model"] == model]
    if horizon is not None:
        d = d[d["horizon"] == horizon]
    return float((d["y_true"] - d["y_pred"]).abs().mean())


FAST_MODELS = [
    NaivePersistence,
    SeasonalNaive,
    ClimatologyForecaster,
    ClimatologyPlusPersistence,
]


@pytest.mark.parametrize("cls", FAST_MODELS)
def test_baseline_contract(cls, split):
    train, truth, origin = split
    model = cls().fit(train)
    preds = model.predict(HORIZON)

    assert set(preds.columns) >= {"model", CITY_COL, "origin", "target_date", "horizon", "y_pred"}
    assert len(preds) == train[CITY_COL].nunique() * HORIZON
    assert preds["y_pred"].notna().all()
    assert (preds["origin"] == origin).all()
    # Every forecast must be strictly in the future.
    assert (preds["target_date"] > origin).all()
    assert (
        (preds["target_date"] - preds["origin"]).dt.days == preds["horizon"]
    ).all()


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        NaivePersistence().predict(HORIZON)


def test_fit_on_empty_panel_raises():
    with pytest.raises(ValueError):
        NaivePersistence().fit(pd.DataFrame(columns=[DATE_COL, CITY_COL, TARGET_COL]))


# --------------------------------------------------------------------------- #
# Accuracy claims, all measured over a rolling-origin backtest.
# --------------------------------------------------------------------------- #
def test_learned_models_beat_persistence_overall(backtest):
    """Pooled over all horizons, both learned models must beat a random walk."""
    naive = _mae(backtest, "naive")
    for model in ("clim+persistence", "gbm"):
        assert _mae(backtest, model) < naive, (
            f"{model} {_mae(backtest, model):.3f} should beat naive {naive:.3f}"
        )


def test_shrinkage_prevents_long_horizon_blowup(backtest, sim_panel):
    """The variance cap must keep the GBM competitive with climatology at day 14.

    Uncapped, the model's day-14 MAE was 2.33 °C against climatology's 1.73.
    """
    assert _mae(backtest, "gbm", 14) < 1.15 * _mae(backtest, "climatology", 14)


def test_shrinkage_schedule_is_monotone_and_bounded(sim_panel):
    origin = pd.Timestamp("2025-11-01")
    model = GlobalGBMForecaster(horizon=14).fit(sim_panel[sim_panel[DATE_COL] <= origin])
    table = model.shrinkage_table()

    assert len(table) == 14
    assert table["shrinkage"].between(0, 1).all()
    assert table["shrinkage"].is_monotonic_decreasing
    # Short lead times should retain most of the model's signal.
    assert table.loc[table["horizon"] == 1, "shrinkage"].iloc[0] > 0.5
    # Long lead times should have largely reverted to climatology.
    assert table.loc[table["horizon"] == 14, "shrinkage"].iloc[0] < 0.3
    assert 0.0 <= model.anomaly_rho_ <= 0.98


def test_persistence_decays_with_lead_time(backtest):
    """Sanity check on the DGP: a random walk must degrade as the horizon grows."""
    assert _mae(backtest, "naive", 1) < _mae(backtest, "naive", 14)


def test_climatology_overtakes_persistence_at_long_horizons(backtest):
    """Seasonal normals should win once the persistence signal has decayed."""
    assert _mae(backtest, "climatology", 14) < _mae(backtest, "naive", 14)
    assert _mae(backtest, "clim+persistence", 14) < _mae(backtest, "naive", 14)


def test_gbm_is_strongest_at_short_lead_times(backtest):
    """The ML model's edge comes from short-range dynamics, not the seasonal cycle."""
    assert _mae(backtest, "gbm", 1) < _mae(backtest, "climatology", 1)


def test_gbm_does_not_overfit(sim_panel):
    """In-sample and out-of-sample error must stay within the same ballpark.

    Before ``doy``/``time_idx``/``dayofweek`` were removed from the feature
    set, the trees memorised individual dates: in-sample MAE sat at a flat
    0.38 across every horizon while out-of-sample error was ~4x worse. This
    test guards that regression.
    """
    origin = pd.Timestamp("2025-11-01")
    train = sim_panel[sim_panel[DATE_COL] <= origin]
    model = GlobalGBMForecaster(horizon=HORIZON).fit(train)

    resid = model.in_sample_residuals()
    in_sample = float(resid["residual"].abs().mean())

    truth = sim_panel.rename(
        columns={DATE_COL: "target_date", TARGET_COL: "y_true"}
    )[[CITY_COL, "target_date", "y_true"]]
    out_sample = _score(GlobalGBMForecaster(horizon=HORIZON), train, truth)

    assert in_sample > 0.5, f"in-sample MAE {in_sample:.3f} is implausibly low"
    assert out_sample < 3 * in_sample


def test_no_calendar_identifier_features_leak_in(sim_panel):
    """``doy``/``time_idx``/``dayofweek`` must stay out of the default feature set."""
    origin = pd.Timestamp("2025-11-01")
    train = sim_panel[sim_panel[DATE_COL] <= origin]
    model = GlobalGBMForecaster(horizon=HORIZON).fit(train)
    banned = {"doy", "time_idx", "dayofweek"}
    assert banned.isdisjoint(set(model.feature_names_))


def test_gbm_feature_importance_is_populated(split):
    train, _, _ = split
    model = GlobalGBMForecaster(horizon=HORIZON).fit(train)
    imp = model.feature_importance(top_n=10)
    assert len(imp) == 10
    assert imp["gain_pct"].sum() > 0
    assert imp["gain"].is_monotonic_decreasing


def test_quantile_gbm_intervals_are_ordered(split):
    train, _, _ = split
    model = QuantileGBMForecaster(horizon=HORIZON, coverage=0.9).fit(train)
    preds = model.predict(HORIZON)
    assert (preds["lower"] <= preds["upper"]).all()
    assert (preds["lower"] <= preds["y_pred"] + 1e-6).all()
    assert (preds["y_pred"] <= preds["upper"] + 1e-6).all()


def test_sarimax_produces_finite_forecasts(split):
    train, truth, _ = split
    score = _score(SARIMAXFourier(), train, truth)
    assert np.isfinite(score)


def test_stl_ets_produces_finite_forecasts(split):
    train, truth, _ = split
    score = _score(STLETSForecaster(), train, truth)
    assert np.isfinite(score)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
def test_lstm_trains_and_produces_sane_forecasts(split, sim_panel):
    train, truth, _ = split
    model = LSTMForecaster(horizon=HORIZON, epochs=15, patience=4).fit(train)
    preds = model.predict(HORIZON)

    assert np.isfinite(preds["y_pred"]).all()
    # Forecasts must lie inside the observed climate envelope.
    lo, hi = sim_panel[TARGET_COL].min() - 10, sim_panel[TARGET_COL].max() + 10
    assert preds["y_pred"].between(lo, hi).all()
    # And the training loss must actually have moved.
    curve = model.training_curve()
    assert len(curve) > 1
    assert curve["train"].iloc[-1] < curve["train"].iloc[0]


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
def test_lstm_is_reproducible(split):
    train, truth, _ = split
    a = _score(LSTMForecaster(horizon=HORIZON, epochs=5, patience=3), train, truth)
    b = _score(LSTMForecaster(horizon=HORIZON, epochs=5, patience=3), train, truth)
    assert a == pytest.approx(b, rel=1e-6)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
def test_lstm_rejects_horizon_beyond_training(split):
    train, _, _ = split
    model = LSTMForecaster(horizon=3, epochs=2, patience=1).fit(train)
    with pytest.raises(ValueError):
        model.predict(10)
