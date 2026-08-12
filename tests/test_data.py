"""Tests for loading, validation and chronological splitting."""
from __future__ import annotations

import pandas as pd
import pytest

from tsa_temperature.config import CITY_COL, DATE_COL, RAW_CSV, TARGET_COL
from tsa_temperature.data import (
    ClimatePanel,
    DataValidationError,
    rolling_origins,
    train_test_split_by_date,
)


@pytest.fixture(scope="module")
def panel() -> ClimatePanel:
    if not RAW_CSV.exists():
        pytest.skip("raw dataset not available")
    return ClimatePanel.from_csv(RAW_CSV)


def test_loads_and_validates(panel):
    assert len(panel.frame) > 0
    assert TARGET_COL in panel.frame.columns
    assert not panel.frame.duplicated([CITY_COL, DATE_COL]).any()


def test_city_series_is_daily_and_gapless(panel):
    s = panel.city_series(panel.cities[0])
    assert isinstance(s.index, pd.DatetimeIndex)
    assert s.index.freqstr == "D"
    assert not s.isna().any()


def test_wide_matrix_shape(panel):
    w = panel.wide()
    assert w.shape[1] == len(panel.cities)
    assert not w.isna().any().any()


def test_validation_rejects_impossible_temperature():
    df = pd.DataFrame(
        {
            DATE_COL: pd.date_range("2024-01-01", periods=3),
            CITY_COL: "X",
            TARGET_COL: [25.0, 999.0, 26.0],
            "temp_min": [20.0, 20.0, 20.0],
            "temp_max": [30.0, 30.0, 30.0],
        }
    )
    with pytest.raises(DataValidationError):
        ClimatePanel(df).validate()


def test_validation_rejects_min_above_max():
    df = pd.DataFrame(
        {
            DATE_COL: pd.date_range("2024-01-01", periods=2),
            CITY_COL: "X",
            TARGET_COL: [25.0, 26.0],
            "temp_min": [30.0, 20.0],
            "temp_max": [28.0, 30.0],
        }
    )
    with pytest.raises(DataValidationError):
        ClimatePanel(df).validate()


def test_validation_rejects_duplicates():
    df = pd.DataFrame(
        {
            DATE_COL: pd.to_datetime(["2024-01-01", "2024-01-01"]),
            CITY_COL: "X",
            TARGET_COL: [25.0, 26.0],
            "temp_min": [20.0, 20.0],
            "temp_max": [30.0, 30.0],
        }
    )
    with pytest.raises(DataValidationError):
        ClimatePanel(df).validate()


# --------------------------------------------------------------------------- #
def test_split_is_chronological():
    df = pd.DataFrame(
        {DATE_COL: pd.date_range("2024-01-01", periods=100), "v": range(100)}
    )
    train, test = train_test_split_by_date(df, pd.Timestamp("2024-02-01"))
    assert train[DATE_COL].max() <= pd.Timestamp("2024-02-01")
    assert test[DATE_COL].min() > pd.Timestamp("2024-02-01")
    assert len(train) + len(test) == len(df)


def test_rolling_origins_leave_room_for_the_horizon():
    dates = pd.date_range("2024-01-01", periods=731)
    origins = rolling_origins(dates, horizon=14, n_origins=5, stride=14, min_train_days=400)
    assert len(origins) == 5
    assert origins == sorted(origins)
    for o in origins:
        remaining = (dates[-1] - o).days
        assert remaining >= 14
        assert (o - dates[0]).days >= 400


def test_rolling_origins_raises_when_history_too_short():
    dates = pd.date_range("2024-01-01", periods=100)
    with pytest.raises(ValueError):
        rolling_origins(dates, horizon=14, n_origins=5, stride=14, min_train_days=400)


def test_fill_calendar_produces_complete_index():
    dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-05"])
    df = pd.DataFrame(
        {
            DATE_COL: list(dates) * 1,
            CITY_COL: "X",
            TARGET_COL: [25.0, 26.0, 27.0],
            "temp_min": [20.0, 21.0, 22.0],
            "temp_max": [30.0, 31.0, 32.0],
        }
    )
    filled = ClimatePanel(df).fill_calendar()
    assert len(filled.frame) == 5
    assert filled.frame[TARGET_COL].notna().all()
