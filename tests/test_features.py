"""Tests for feature engineering — the leakage checks are the important ones."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tsa_temperature.config import CITY_COL, DATE_COL, TARGET_COL, FeatureConfig
from tsa_temperature.features import (
    DayOfYearClimatology,
    build_history_features,
    build_supervised_dataset,
    calendar_features,
    fourier_terms,
)


@pytest.fixture
def toy_panel() -> pd.DataFrame:
    """Two cities, 500 days, a clean sinusoid plus a city offset."""
    dates = pd.date_range("2024-01-01", periods=500, freq="D")
    rows = []
    for i, city in enumerate(["Alpha", "Beta"]):
        t = np.arange(len(dates))
        temp = 25 + 5 * i + 8 * np.sin(2 * np.pi * t / 365)
        rows.append(
            pd.DataFrame(
                {
                    DATE_COL: dates,
                    CITY_COL: city,
                    TARGET_COL: temp,
                    "temp_max": temp + 4,
                    "temp_min": temp - 4,
                    "humidity": 60 + 5 * np.cos(2 * np.pi * t / 365),
                    "rainfall": 0.0,
                    "wind_speed": 8.0,
                    "aqi": 100.0,
                    "pressure": 1010.0,
                    "cloud_cover": 40.0,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------- #
# Fourier / calendar
# --------------------------------------------------------------------------- #
def test_fourier_terms_shape_and_range():
    dates = pd.date_range("2024-01-01", periods=365, freq="D")
    f = fourier_terms(dates, order=3)
    assert f.shape == (365, 6)
    assert f.abs().max().max() <= 1.0 + 1e-9


def test_fourier_terms_are_annually_periodic():
    a = fourier_terms(pd.DatetimeIndex(["2023-03-15"]), order=2)
    b = fourier_terms(pd.DatetimeIndex(["2025-03-15"]), order=2)
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), atol=1e-6)


def test_calendar_features_seasons():
    cal = calendar_features(pd.DatetimeIndex(["2024-01-15", "2024-04-15", "2024-07-15", "2024-11-15"]))
    assert cal["imd_season"].tolist() == [0, 1, 2, 3]


# --------------------------------------------------------------------------- #
# Leakage
# --------------------------------------------------------------------------- #
def test_history_features_use_no_future_information(toy_panel):
    """Truncating the series must not change features at earlier origins."""
    g = (
        toy_panel[toy_panel[CITY_COL] == "Alpha"]
        .sort_values(DATE_COL)
        .set_index(DATE_COL)
    )
    full = build_history_features(g)
    truncated = build_history_features(g.iloc[:400])

    common = truncated.index
    pd.testing.assert_frame_equal(
        full.loc[common], truncated.loc[common], check_exact=False, atol=1e-9
    )


def test_y_lag1_is_the_observation_at_the_origin(toy_panel):
    g = (
        toy_panel[toy_panel[CITY_COL] == "Alpha"]
        .sort_values(DATE_COL)
        .set_index(DATE_COL)
    )
    feats = build_history_features(g)
    np.testing.assert_allclose(
        feats["y_lag1"].to_numpy(), g[TARGET_COL].to_numpy(), atol=1e-9
    )


def test_supervised_target_matches_origin_plus_horizon(toy_panel):
    ds = build_supervised_dataset(toy_panel, horizon=5)
    lookup = toy_panel.set_index([CITY_COL, DATE_COL])[TARGET_COL]
    sample = ds.meta.sample(50, random_state=0)
    for idx, row in sample.iterrows():
        expected = lookup.loc[(row[CITY_COL], row["target_date"])]
        assert np.isclose(ds.y.loc[idx], expected, atol=1e-9)


def test_target_date_equals_origin_plus_horizon(toy_panel):
    ds = build_supervised_dataset(toy_panel, horizon=7)
    delta = (ds.meta["target_date"] - ds.meta["origin"]).dt.days
    assert (delta == ds.meta["horizon"]).all()


def test_no_feature_correlates_perfectly_with_the_target(toy_panel):
    """A feature that reproduces the target exactly would signal leakage."""
    ds = build_supervised_dataset(toy_panel, horizon=3)
    numeric = ds.X.select_dtypes(include="number")
    for col in numeric.columns:
        v = numeric[col].to_numpy(dtype=float)
        if np.nanstd(v) < 1e-9:
            continue
        r = np.corrcoef(np.nan_to_num(v), ds.y.to_numpy())[0, 1]
        assert abs(r) < 0.9999, f"{col} is perfectly correlated with the target"


# --------------------------------------------------------------------------- #
# Climatology
# --------------------------------------------------------------------------- #
def test_climatology_is_fit_only_on_supplied_data(toy_panel):
    train = toy_panel[toy_panel[DATE_COL] <= "2024-12-31"]
    clim = DayOfYearClimatology().fit(train)
    assert clim.table_ is not None
    # Values must come from the training window only.
    assert clim.global_mean_ == pytest.approx(train[TARGET_COL].mean(), rel=1e-9)


def test_climatology_transform_returns_finite_values_for_unseen_dates(toy_panel):
    clim = DayOfYearClimatology().fit(toy_panel)
    out = clim.transform(
        pd.Series(["Alpha", "Beta"]), pd.Series(pd.to_datetime(["2027-06-30", "2027-12-31"]))
    )
    assert np.isfinite(out).all()


def test_climatology_tracks_the_seasonal_signal(toy_panel):
    clim = DayOfYearClimatology().fit(toy_panel)
    alpha = toy_panel[toy_panel[CITY_COL] == "Alpha"]
    pred = clim.transform(alpha[CITY_COL], alpha[DATE_COL])
    assert np.corrcoef(pred, alpha[TARGET_COL])[0, 1] > 0.95


# --------------------------------------------------------------------------- #
# Config plumbing
# --------------------------------------------------------------------------- #
def test_feature_config_controls_lag_columns(toy_panel):
    cfg = FeatureConfig(target_lags=(1, 5), rolling_windows=(7,), exog_lags=(1,), use_exog=False)
    g = toy_panel[toy_panel[CITY_COL] == "Alpha"].set_index(DATE_COL)
    feats = build_history_features(g, cfg)
    lag_cols = sorted(c for c in feats.columns if c.startswith("y_lag"))
    assert lag_cols == ["y_lag1", "y_lag5"]
    assert not any(c.startswith("humidity") for c in feats.columns)
