"""Tests for metrics, conformal calibration and the forecastability audit."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tsa_temperature import diagnostics
from tsa_temperature.conformal import SplitConformal
from tsa_temperature.evaluation import (
    coverage,
    diebold_mariano,
    leaderboard,
    mae,
    mase,
    metrics_by_horizon,
    rmse,
    smape,
)
from tsa_temperature.simulate import simulate_climate_panel


# --------------------------------------------------------------------------- #
# Point metrics
# --------------------------------------------------------------------------- #
def test_perfect_forecast_scores_zero():
    y = np.array([1.0, 2.0, 3.0])
    assert mae(y, y) == 0.0
    assert rmse(y, y) == 0.0
    assert smape(y, y) == 0.0


def test_rmse_penalises_large_errors_more_than_mae():
    y = np.zeros(4)
    spread = np.array([0.0, 0.0, 0.0, 4.0])
    even = np.array([1.0, 1.0, 1.0, 1.0])
    assert mae(y, spread) == mae(y, even)
    assert rmse(y, spread) > rmse(y, even)


def test_mase_equals_one_for_the_naive_benchmark():
    rng = np.random.default_rng(0)
    insample = np.cumsum(rng.normal(size=300))
    y_true = insample[1:]
    y_pred = insample[:-1]  # one-step naive
    assert mase(y_true, y_pred, insample, seasonality=1) == pytest.approx(1.0, rel=0.05)


def test_mase_below_one_means_better_than_naive():
    rng = np.random.default_rng(1)
    insample = np.cumsum(rng.normal(size=300))
    y_true = insample[1:]
    y_pred = 0.5 * insample[:-1] + 0.5 * y_true  # oracle-ish
    assert mase(y_true, y_pred, insample, seasonality=1) < 1.0


def test_smape_is_bounded():
    assert smape(np.array([1.0]), np.array([-1.0])) <= 200.0


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def _results_frame() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    rows = []
    for model, noise in [("good", 0.5), ("bad", 3.0)]:
        for h in range(1, 5):
            y = rng.normal(25, 5, 40)
            rows.append(
                pd.DataFrame(
                    {
                        "model": model,
                        "horizon": h,
                        "y_true": y,
                        "y_pred": y + rng.normal(0, noise, 40),
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def test_leaderboard_ranks_the_better_model_first():
    board = leaderboard(_results_frame())
    assert board.index[0] == "good"
    assert board.loc["good", "MAE"] < board.loc["bad", "MAE"]


def test_metrics_by_horizon_covers_every_combination():
    by_h = metrics_by_horizon(_results_frame())
    assert len(by_h) == 8
    assert set(by_h["horizon"]) == {1, 2, 3, 4}


def test_diebold_mariano_detects_a_real_difference():
    rng = np.random.default_rng(7)
    y = rng.normal(0, 1, 400)
    good = y + rng.normal(0, 0.2, 400)
    bad = y + rng.normal(0, 2.0, 400)
    dm = diebold_mariano(y, good, bad, h=1)
    assert dm["statistic"] < 0
    assert dm["p_value"] < 0.05


def test_diebold_mariano_finds_no_difference_between_identical_models():
    rng = np.random.default_rng(8)
    y = rng.normal(0, 1, 400)
    a = y + rng.normal(0, 1, 400)
    dm = diebold_mariano(y, a, a, h=1)
    assert dm["p_value"] > 0.99 or np.isnan(dm["p_value"])


# --------------------------------------------------------------------------- #
# Conformal
# --------------------------------------------------------------------------- #
def test_conformal_achieves_nominal_coverage():
    rng = np.random.default_rng(11)
    cal = rng.normal(0, 2, 3000)
    test_resid = rng.normal(0, 2, 3000)
    sc = SplitConformal(coverage=0.9).fit(cal)
    lo, hi = sc.predict_interval(np.zeros(3000))
    assert coverage(test_resid, lo, hi) == pytest.approx(0.9, abs=0.03)


def test_conformal_widens_with_horizon():
    rng = np.random.default_rng(12)
    horizons = np.repeat([1, 7, 14], 800)
    scale = np.where(horizons == 1, 1.0, np.where(horizons == 7, 2.0, 4.0))
    resid = rng.normal(0, 1, len(horizons)) * scale
    sc = SplitConformal(coverage=0.9).fit(resid, horizons)
    assert sc.quantiles_[1] < sc.quantiles_[7] < sc.quantiles_[14]


def test_conformal_rejects_empty_calibration():
    with pytest.raises(ValueError):
        SplitConformal().fit(np.array([np.nan, np.nan]))


# --------------------------------------------------------------------------- #
# Forecastability audit
# --------------------------------------------------------------------------- #
def _white_noise_panel(n_days: int = 731, n_cities: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(21)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    rows = []
    for i in range(n_cities):
        rows.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "city": f"C{i}",
                    "temp_avg": rng.uniform(18, 42, n_days),
                    "humidity": rng.uniform(30, 95, n_days),
                    "rainfall": rng.uniform(0, 20, n_days),
                    "wind_speed": rng.uniform(1, 20, n_days),
                    "aqi": rng.uniform(20, 400, n_days),
                    "pressure": rng.uniform(995, 1030, n_days),
                    "cloud_cover": rng.uniform(0, 100, n_days),
                    "temp_max": rng.uniform(25, 45, n_days),
                    "temp_min": rng.uniform(10, 25, n_days),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def test_audit_flags_white_noise_as_not_forecastable():
    report = diagnostics.audit(_white_noise_panel())
    assert report.verdict.startswith("NOT FORECASTABLE")
    assert not report.ljung_box["rejects_white_noise"].all()
    assert not report.seasonality["has_seasonality"].any()


def test_audit_flags_structured_data_as_forecastable():
    from tsa_temperature.data import ClimatePanel

    raw = simulate_climate_panel(seed=5)
    panel = ClimatePanel(
        raw.rename(
            columns={
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
        ).assign(date=lambda d: pd.to_datetime(d["date"]))
    )
    report = diagnostics.audit(panel.frame)
    assert report.verdict.startswith("FORECASTABLE")
    assert report.seasonality["has_seasonality"].all()
    assert report.cross_series.loc["mean_pairwise_corr", "value"] > 0.3


def test_simulated_panel_has_realistic_seasonal_amplitude():
    raw = simulate_climate_panel(seed=2)
    delhi = raw[raw["City"] == "Delhi"]
    monthly = delhi.groupby(pd.DatetimeIndex(delhi["Date"]).month)["Temperature_Avg (°C)"].mean()
    assert monthly.max() - monthly.min() > 8.0
