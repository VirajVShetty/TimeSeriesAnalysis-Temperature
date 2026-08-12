"""Tests for the descriptive analysis module.

Wherever possible these check results against **hand-computable** values on a
tiny constructed panel, rather than just asserting "it returns a DataFrame".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tsa_temperature import analysis as an
from tsa_temperature.config import CITY_COL, DATE_COL, STATE_COL, TARGET_COL


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def toy_panel() -> pd.DataFrame:
    """Two cities, two full years, a known sinusoid plus a city offset.

    ``Hot`` has a 10 °C seasonal half-amplitude (20 °C peak-to-trough) and mean
    30; ``Mild`` has a 2 °C half-amplitude and mean 25. Both peak in July.
    """
    dates = pd.date_range("2024-01-01", "2025-12-31", freq="D")
    doy = dates.dayofyear.to_numpy()
    rng = np.random.default_rng(7)
    frames = []
    for city, mean_t, amp in [("Hot", 30.0, 10.0), ("Mild", 25.0, 2.0)]:
        seasonal = amp * np.sin(2 * np.pi * (doy - 105) / 365.0)
        temp = mean_t + seasonal
        n = len(dates)
        frames.append(
            pd.DataFrame(
                {
                    DATE_COL: dates,
                    CITY_COL: city,
                    STATE_COL: f"State-{city}",
                    TARGET_COL: temp,
                    "temp_max": temp + 5,
                    "temp_min": temp - 5,
                    # humidity and aqi track temperature exactly (used to check
                    # that correlation detection works); the rest carry
                    # independent noise so no column is constant — a
                    # zero-variance column makes correlation undefined.
                    "humidity": 60 + 0.5 * seasonal,
                    "aqi": 150 - 2 * seasonal,
                    "rainfall": rng.gamma(2.0, 3.0, n),
                    "wind_speed": 10 + rng.normal(0, 2, n),
                    "pressure": 1010 + rng.normal(0, 3, n),
                    "cloud_cover": 50 + rng.normal(0, 10, n),
                    "aqi_category": "Moderate",
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def uniform_panel() -> pd.DataFrame:
    """Three cities of pure uniform noise — no seasonality, no memory."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-01", "2025-12-31", freq="D")
    frames = []
    for city in ["A", "B", "C"]:
        temp = rng.uniform(18, 42, len(dates))
        frames.append(
            pd.DataFrame(
                {
                    DATE_COL: dates,
                    CITY_COL: city,
                    STATE_COL: "S",
                    TARGET_COL: temp,
                    "temp_max": temp + 5,
                    "temp_min": temp - 5,
                    "humidity": rng.uniform(30, 95, len(dates)),
                    "rainfall": rng.uniform(0, 20, len(dates)),
                    "wind_speed": rng.uniform(2, 25, len(dates)),
                    "aqi": rng.uniform(40, 350, len(dates)),
                    "pressure": rng.uniform(990, 1025, len(dates)),
                    "cloud_cover": rng.uniform(5, 100, len(dates)),
                    "aqi_category": "Moderate",
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# Calendar helpers
# --------------------------------------------------------------------------- #
def test_season_assignment_follows_imd_definitions():
    df = pd.DataFrame(
        {
            DATE_COL: pd.to_datetime(
                ["2024-01-15", "2024-04-15", "2024-07-15", "2024-11-15", "2024-12-15"]
            ),
            CITY_COL: "X",
            TARGET_COL: 25.0,
        }
    )
    out = an.add_calendar_columns(df)
    assert out["season"].tolist() == [
        "Winter", "Pre-monsoon", "Monsoon", "Post-monsoon", "Winter",
    ]
    assert out["month_name"].tolist() == ["Jan", "Apr", "Jul", "Nov", "Dec"]


def test_every_month_maps_to_exactly_one_season():
    assigned = [m for months in an.IMD_SEASONS.values() for m in months]
    assert sorted(assigned) == list(range(1, 13))


def test_month_names_are_ordered_categorical(toy_panel):
    out = an.add_calendar_columns(toy_panel)
    assert out["month_name"].cat.ordered
    assert list(out["month_name"].cat.categories) == an.MONTH_NAMES


# --------------------------------------------------------------------------- #
# Dataset profile
# --------------------------------------------------------------------------- #
def test_dataset_profile_counts(toy_panel):
    prof = an.dataset_profile(toy_panel)
    assert prof["rows"] == 731 * 2
    assert prof["cities"] == 2
    assert prof["days_spanned"] == 731
    assert prof["days_per_city"] == 731
    assert prof["completeness_pct"] == 100.0
    assert prof["duplicate_city_date_rows"] == 0
    assert prof["total_missing_cells"] == 0


def test_dataset_profile_detects_incompleteness(toy_panel):
    trimmed = toy_panel.drop(toy_panel.index[:50])
    assert an.dataset_profile(trimmed)["completeness_pct"] < 100.0


def test_column_inventory_reports_missing(toy_panel):
    df = toy_panel.copy()
    df.loc[df.index[:10], TARGET_COL] = np.nan
    inv = an.column_inventory(df)
    assert inv.loc[TARGET_COL, "n_missing"] == 10
    assert inv.loc[CITY_COL, "n_unique"] == 2


# --------------------------------------------------------------------------- #
# Distributions
# --------------------------------------------------------------------------- #
def test_numeric_summary_matches_pandas(toy_panel):
    summary = an.numeric_summary(toy_panel, [TARGET_COL])
    s = toy_panel[TARGET_COL]
    assert summary.loc[TARGET_COL, "mean"] == pytest.approx(s.mean(), abs=1e-3)
    assert summary.loc[TARGET_COL, "median"] == pytest.approx(s.median(), abs=1e-3)
    assert summary.loc[TARGET_COL, "iqr"] == pytest.approx(
        s.quantile(0.75) - s.quantile(0.25), abs=1e-3
    )


def test_distribution_shape_identifies_uniform():
    rng = np.random.default_rng(1)
    out = an.distribution_shape(pd.Series(rng.uniform(0, 1, 4000)))
    assert out["closest_shape"] == "uniform"
    # Uniform has excess kurtosis of -1.2 in the limit.
    assert out["excess_kurtosis"] == pytest.approx(-1.2, abs=0.2)


def test_distribution_shape_identifies_normal():
    rng = np.random.default_rng(2)
    out = an.distribution_shape(pd.Series(rng.normal(25, 5, 4000)))
    assert out["closest_shape"] == "normal-ish"
    assert out["excess_kurtosis"] == pytest.approx(0.0, abs=0.25)


def test_distribution_shape_by_city_covers_all(uniform_panel):
    out = an.distribution_shape_by_city(uniform_panel)
    assert len(out) == 3
    assert (out["closest_shape"] == "uniform").all()


# --------------------------------------------------------------------------- #
# City profiles
# --------------------------------------------------------------------------- #
def test_city_profile_recovers_known_amplitude(toy_panel):
    prof = an.city_profile(toy_panel)
    # 10 degC half-amplitude -> ~20 degC between hottest and coldest monthly mean.
    assert prof.loc["Hot", "seasonal_amplitude"] == pytest.approx(20, abs=1.5)
    assert prof.loc["Mild", "seasonal_amplitude"] == pytest.approx(4, abs=0.6)
    assert prof.loc["Hot", "mean_temp"] == pytest.approx(30, abs=0.3)
    assert prof.loc["Mild", "mean_temp"] == pytest.approx(25, abs=0.3)


def test_city_profile_finds_the_right_extreme_months(toy_panel):
    prof = an.city_profile(toy_panel)
    assert prof.loc["Hot", "hottest_month"] == "Jul"
    assert prof.loc["Hot", "coldest_month"] == "Jan"


def test_city_profile_diurnal_range(toy_panel):
    # temp_max = t + 5, temp_min = t - 5 -> range is exactly 10.
    assert an.city_profile(toy_panel)["mean_diurnal_range"].tolist() == [10.0, 10.0]


def test_rank_cities_orders_descending(toy_panel):
    ranked = an.rank_cities(toy_panel, "mean_temp")
    assert ranked.index[0] == "Hot"
    assert ranked["rank"].tolist() == [1, 2]


def test_rank_cities_rejects_unknown_metric(toy_panel):
    with pytest.raises(KeyError):
        an.rank_cities(toy_panel, "not_a_metric")


def test_climatology_plausibility_flags_flat_seasonality(uniform_panel):
    df = uniform_panel.replace({"A": "Delhi", "B": "Mumbai", "C": "Jaipur"})
    out = an.climatology_plausibility(df)
    assert len(out) == 3
    # Uniform noise has no annual cycle, so nothing should look plausible.
    assert not out["plausible"].any()
    # Judge only the strongly-seasonal references. Mumbai's real amplitude is
    # just 6 degC, so sampling noise in 12 monthly means can land near half of
    # it by chance; Delhi and Jaipur (19-20 degC) leave no such ambiguity.
    assert out.loc["Delhi", "amplitude_ratio"] < 0.35
    assert out.loc["Jaipur", "amplitude_ratio"] < 0.35


# --------------------------------------------------------------------------- #
# Temporal breakdowns
# --------------------------------------------------------------------------- #
def test_monthly_profile_has_twelve_rows(toy_panel):
    out = an.monthly_profile(toy_panel, city="Hot")
    assert len(out) == 12
    assert out["mean"].idxmax() == "Jul"


def test_monthly_profile_rejects_unknown_city(toy_panel):
    with pytest.raises(KeyError):
        an.monthly_profile(toy_panel, city="Atlantis")


def test_seasonal_profile_covers_four_seasons(toy_panel):
    out = an.seasonal_profile(toy_panel)
    assert len(out) == 4
    assert out["n"].sum() == len(toy_panel)


def test_city_month_matrix_shape(toy_panel):
    mat = an.city_month_matrix(toy_panel)
    assert mat.shape == (2, 12)
    assert list(mat.columns) == an.MONTH_NAMES


def test_monthly_variation_separates_seasonal_from_flat(toy_panel, uniform_panel):
    seasonal = an.monthly_variation_summary(toy_panel)
    flat = an.monthly_variation_summary(uniform_panel)
    # A clean sinusoid is almost entirely explained by month-of-year.
    assert seasonal.loc["Hot", "eta_squared"] > 0.9
    # Uniform noise is not.
    assert (flat["eta_squared"] < 0.1).all()


# --------------------------------------------------------------------------- #
# Extremes
# --------------------------------------------------------------------------- #
def test_longest_run_basic_cases():
    assert an.longest_run(pd.Series([False, False])) == 0
    assert an.longest_run(pd.Series([True, True, False, True])) == 2
    assert an.longest_run(pd.Series([True] * 5)) == 5


def test_extreme_day_counts_are_consistent(toy_panel):
    out = an.extreme_day_counts(toy_panel, hot_threshold=35.0, cold_threshold=15.0)
    hot_col = "days_max_ge_35"
    expected = int((toy_panel.query("city == 'Hot'")["temp_max"] >= 35).sum())
    assert out.loc["Hot", hot_col] == expected
    assert out.loc["Hot", "pct_hot"] == pytest.approx(100 * expected / 731, abs=0.1)


def test_heat_spells_cluster_in_seasonal_data(toy_panel):
    """A real seasonal cycle produces long consecutive hot runs."""
    out = an.heat_spell_summary(toy_panel, hot_threshold=35.0)
    assert out.loc["Hot", "longest_hot_run"] > 30
    assert out.loc["Hot", "clustering"] > 2.0


def test_heat_spells_do_not_cluster_in_noise(uniform_panel):
    """Independent days give runs close to the random-chance expectation."""
    out = an.heat_spell_summary(uniform_panel, hot_threshold=35.0)
    assert (out["clustering"].dropna() < 2.0).all()


def test_expected_longest_run_is_sane():
    # 700 trials at p=0.5 -> longest run around log2(350) ~ 8-9.
    assert 7 <= an._expected_longest_run(700, 0.5) <= 10


def test_diurnal_range_profile(toy_panel):
    out = an.diurnal_range_profile(toy_panel)
    assert out["mean"].round(2).tolist() == [10.0, 10.0]
    assert {"Winter", "Monsoon"} <= set(out.columns)


def test_diurnal_range_requires_min_max(toy_panel):
    with pytest.raises(KeyError):
        an.diurnal_range_profile(toy_panel.drop(columns=["temp_max"]))


# --------------------------------------------------------------------------- #
# Relationships
# --------------------------------------------------------------------------- #
def test_correlation_with_significance_detects_real_link(toy_panel):
    # Single city: pooling two cities with different seasonal amplitudes but
    # the same humidity mean dilutes the correlation (to ~0.90), which is
    # correct behaviour but not what this test is checking.
    out = an.correlation_with_significance(toy_panel.query("city == 'Hot'"))
    # humidity was built as 60 + 0.5 * seasonal, so it tracks temperature exactly.
    assert out.loc["humidity", "pearson_r"] == pytest.approx(1.0, abs=0.01)
    assert bool(out.loc["humidity", "significant"])
    # aqi was built as 150 - 2 * seasonal -> perfect negative correlation.
    assert out.loc["aqi", "pearson_r"] == pytest.approx(-1.0, abs=0.01)


def test_plausibility_requires_more_than_amplitude(uniform_panel):
    """Noise must not pass the check just by matching a low-amplitude city.

    Mumbai's real seasonal amplitude is only 6 degC, and twelve monthly means
    of uniform noise spread ~3 degC by chance — enough to clear a
    ratio-only test. The eta-squared requirement is what rejects it.
    """
    df = uniform_panel.replace({"A": "Delhi", "B": "Mumbai", "C": "Jaipur"})
    out = an.climatology_plausibility(df)
    assert (out["eta_squared"] < 0.1).all()
    assert not out["plausible"].any()


def test_correlation_finds_nothing_in_independent_noise(uniform_panel):
    out = an.correlation_with_significance(uniform_panel)
    real = out.drop(index=[c for c in ("temp_max", "temp_min") if c in out.index])
    assert not real["significant"].any()


def test_holm_thresholds_are_ordered(uniform_panel):
    out = an.correlation_with_significance(uniform_panel)
    # Holm thresholds must be non-decreasing as p-value rank increases.
    assert out["holm_threshold"].is_monotonic_increasing


def test_cross_city_correlation_is_symmetric_with_unit_diagonal(uniform_panel):
    corr = an.cross_city_correlation(uniform_panel)
    assert corr.shape == (3, 3)
    assert np.allclose(np.diag(corr), 1.0)
    assert np.allclose(corr.to_numpy(), corr.to_numpy().T)


def test_channel_correlation_matrix_is_square(toy_panel):
    m = an.channel_correlation_matrix(toy_panel)
    assert m.shape[0] == m.shape[1]
    assert np.allclose(np.diag(m), 1.0)


# --------------------------------------------------------------------------- #
# Air quality
# --------------------------------------------------------------------------- #
def test_aqi_category_burden_rows_sum_to_100(toy_panel):
    out = an.aqi_category_burden(toy_panel)
    assert np.allclose(out.sum(axis=1), 100.0, atol=0.1)


def test_aqi_seasonality_has_season_columns(toy_panel):
    out = an.aqi_seasonality(toy_panel)
    assert {"Winter", "Monsoon"} <= set(out.columns)
    # aqi = 150 - 2 * seasonal, and seasonal peaks in July -> winter AQI is higher.
    assert (out["winter_minus_monsoon"] > 0).all()


def test_aqi_helpers_require_their_columns(toy_panel):
    with pytest.raises(KeyError):
        an.aqi_category_burden(toy_panel.drop(columns=["aqi_category"]))
    with pytest.raises(KeyError):
        an.aqi_seasonality(toy_panel.drop(columns=["aqi"]))
