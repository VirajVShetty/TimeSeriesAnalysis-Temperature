"""Tests for the anomaly-detection ensemble."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tsa_temperature.anomaly import (
    brutlag_bands,
    detect_anomalies,
    isolation_forest_anomalies,
    matrix_profile_anomalies,
    stl_residual_anomalies,
)


@pytest.fixture
def spiked_series():
    """A clean seasonal series with five known injected spikes."""
    dates = pd.date_range("2024-01-01", periods=731, freq="D")
    t = np.arange(len(dates))
    rng = np.random.default_rng(0)
    values = 27 + 9 * np.sin(2 * np.pi * t / 365) + rng.normal(0, 0.8, len(t))
    spike_positions = [90, 200, 365, 500, 650]
    values[spike_positions] += 14.0
    return pd.Series(values, index=dates, name="temp_avg"), spike_positions


def test_stl_detector_finds_injected_spikes(spiked_series):
    s, spikes = spiked_series
    res = stl_residual_anomalies(s, period=365, z_threshold=3.0)
    flagged = np.flatnonzero(res["flag_stl"].to_numpy())
    assert set(spikes).issubset(set(flagged.tolist()))


def test_stl_detector_is_not_trigger_happy(spiked_series):
    s, _ = spiked_series
    res = stl_residual_anomalies(s, period=365, z_threshold=3.0)
    assert res["flag_stl"].mean() < 0.05


def test_brutlag_bands_bracket_the_series(spiked_series):
    s, _ = spiked_series
    res = brutlag_bands(s, period=365)
    assert (res["upper"] >= res["lower"]).all()
    inside = ((s >= res["lower"]) & (s <= res["upper"])).mean()
    assert inside > 0.8


def test_matrix_profile_returns_one_score_per_timestamp(spiked_series):
    s, _ = spiked_series
    res = matrix_profile_anomalies(s, window=14, top_k=10)
    assert len(res) == len(s)
    assert res["flag_mp"].sum() > 0
    assert res["flag_mp"].sum() < len(s) * 0.1


def test_matrix_profile_handles_short_series():
    s = pd.Series(np.arange(20.0), index=pd.date_range("2024-01-01", periods=20))
    res = matrix_profile_anomalies(s, window=14)
    assert not res["flag_mp"].any()


def test_isolation_forest_flags_roughly_the_contamination_rate():
    rng = np.random.default_rng(4)
    n = 800
    df = pd.DataFrame(
        {
            "temp_avg": rng.normal(28, 3, n),
            "humidity": rng.normal(60, 10, n),
            "pressure": rng.normal(1010, 4, n),
        },
        index=pd.date_range("2024-01-01", periods=n),
    )
    res = isolation_forest_anomalies(
        df, ["temp_avg", "humidity", "pressure"], contamination=0.02
    )
    assert res["flag_iforest"].mean() == pytest.approx(0.02, abs=0.015)


def test_ensemble_requires_agreement(spiked_series):
    s, spikes = spiked_series
    frame = pd.DataFrame({"temp_avg": s, "humidity": 60.0, "pressure": 1010.0})
    res = detect_anomalies(frame, target_col="temp_avg")

    assert "flag_ensemble" in res.frame
    assert len(res.flag_columns) >= 3
    # The ensemble must be at least as conservative as its loosest member.
    loosest = max(res.frame[c].sum() for c in res.flag_columns if c != "flag_ensemble")
    assert res.frame["flag_ensemble"].sum() <= loosest
    # Large injected spikes should still survive the vote.
    flagged = set(np.flatnonzero(res.frame["flag_ensemble"].to_numpy()).tolist())
    assert len(flagged & set(spikes)) >= 3


def test_ensemble_summary_and_agreement_shapes(spiked_series):
    s, _ = spiked_series
    frame = pd.DataFrame({"temp_avg": s, "humidity": 60.0})
    res = detect_anomalies(frame, target_col="temp_avg")
    summary = res.summary()
    agreement = res.agreement()
    assert len(summary) == len(res.flag_columns)
    assert agreement.shape == (len(res.flag_columns), len(res.flag_columns))
    assert (np.diag(agreement) == summary.reindex(agreement.index).to_numpy()).all()


def test_anomalies_accessor_rejects_unknown_column(spiked_series):
    s, _ = spiked_series
    res = detect_anomalies(pd.DataFrame({"temp_avg": s}), target_col="temp_avg")
    with pytest.raises(KeyError):
        res.anomalies("flag_does_not_exist")
