"""Descriptive analysis of the climate panel.

This module is deliberately **model-free**. Nothing here fits, forecasts or
predicts — it describes what is in the data: how values are distributed, how
cities and months differ, how the weather channels relate to each other, and
how often extremes occur.

Companion to :mod:`tsa_temperature.diagnostics`, which asks a narrower
question ("is there forecastable signal?"). This module answers the broader
one a analyst asks first: "what does this dataset actually contain?"

A note on interpretation
------------------------
Descriptive statistics are computed identically whether the underlying data is
real or synthetic — the arithmetic never fails. Several helpers here therefore
return a ``plausible`` or ``flag`` column that compares the observed value
against what Indian climatology would give, so an implausible result is
labelled rather than quietly reported as a finding. See
:func:`climatology_plausibility`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .config import CITY_COL, DATE_COL, EXOG_COLS, STATE_COL, TARGET_COL

MONTH_NAMES = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]

#: Indian Meteorological Department season definitions.
IMD_SEASONS: dict[str, tuple[int, ...]] = {
    "Winter": (12, 1, 2),
    "Pre-monsoon": (3, 4, 5),
    "Monsoon": (6, 7, 8, 9),
    "Post-monsoon": (10, 11),
}

#: Rough real-world reference values for major Indian cities, used only to
#: sanity-check whether an observed statistic is physically plausible.
#: (annual mean °C, annual range between hottest and coldest monthly mean °C)
REFERENCE_CLIMATE: dict[str, tuple[float, float]] = {
    "Mumbai": (27.5, 6.0),
    "Delhi": (25.0, 20.0),
    "Bengaluru": (24.0, 6.0),
    "Chennai": (28.5, 7.0),
    "Kolkata": (26.5, 14.0),
    "Hyderabad": (26.5, 10.0),
    "Ahmedabad": (27.5, 16.0),
    "Jaipur": (25.5, 19.0),
    "Lucknow": (25.5, 20.0),
    "Bhopal": (25.0, 16.0),
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def add_calendar_columns(panel: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with ``year``, ``month``, ``month_name`` and ``season``."""
    df = panel.copy()
    idx = pd.DatetimeIndex(df[DATE_COL])
    df["year"] = idx.year
    df["month"] = idx.month
    df["month_name"] = pd.Categorical(
        [MONTH_NAMES[m - 1] for m in idx.month], categories=MONTH_NAMES, ordered=True
    )
    lookup = {m: name for name, months in IMD_SEASONS.items() for m in months}
    df["season"] = pd.Categorical(
        [lookup[m] for m in idx.month], categories=list(IMD_SEASONS), ordered=True
    )
    return df


def _numeric_columns(panel: pd.DataFrame) -> list[str]:
    return [c for c in [TARGET_COL, *EXOG_COLS] if c in panel.columns]


# --------------------------------------------------------------------------- #
# 1. Dataset-level profile
# --------------------------------------------------------------------------- #
def dataset_profile(panel: pd.DataFrame) -> pd.Series:
    """One-glance description of the panel: size, span, completeness."""
    idx = pd.DatetimeIndex(panel[DATE_COL])
    start, end = idx.min(), idx.max()
    n_cities = panel[CITY_COL].nunique()
    expected_days = (end - start).days + 1

    return pd.Series(
        {
            "rows": len(panel),
            "columns": panel.shape[1],
            "cities": n_cities,
            "states": panel[STATE_COL].nunique() if STATE_COL in panel else np.nan,
            "start": start.date().isoformat(),
            "end": end.date().isoformat(),
            "days_spanned": expected_days,
            "days_per_city": int(round(len(panel) / n_cities)) if n_cities else 0,
            "expected_rows": expected_days * n_cities,
            "completeness_pct": round(100 * len(panel) / (expected_days * n_cities), 2),
            "duplicate_city_date_rows": int(panel.duplicated([CITY_COL, DATE_COL]).sum()),
            "total_missing_cells": int(panel.isna().sum().sum()),
            "memory_mb": round(panel.memory_usage(deep=True).sum() / 1024**2, 2),
        },
        name="value",
    )


def column_inventory(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-column dtype, missingness, cardinality and range."""
    rows = []
    for col in panel.columns:
        s = panel[col]
        row = {
            "column": col,
            "dtype": str(s.dtype),
            "n_missing": int(s.isna().sum()),
            "pct_missing": round(100 * s.isna().mean(), 2),
            "n_unique": int(s.nunique()),
        }
        if pd.api.types.is_numeric_dtype(s):
            row["min"], row["max"] = round(float(s.min()), 2), round(float(s.max()), 2)
        else:
            row["min"] = row["max"] = ""
        rows.append(row)
    return pd.DataFrame(rows).set_index("column")


# --------------------------------------------------------------------------- #
# 2. Distributions
# --------------------------------------------------------------------------- #
def numeric_summary(panel: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Extended describe(): adds IQR, CV, skewness and excess kurtosis.

    Shape statistics are the useful part. Skew near 0 with excess kurtosis near
    -1.2 indicates a *uniform* distribution; a natural temperature series is
    closer to skew ~0 and kurtosis ~-0.5 to -1.0 with visible seasonal modes.
    """
    columns = columns or _numeric_columns(panel)
    rows = []
    for col in columns:
        s = panel[col].dropna().astype(float)
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        rows.append(
            {
                "variable": col,
                "n": len(s),
                "mean": s.mean(),
                "std": s.std(),
                "min": s.min(),
                "q25": q1,
                "median": s.median(),
                "q75": q3,
                "max": s.max(),
                "iqr": q3 - q1,
                "cv_pct": 100 * s.std() / s.mean() if s.mean() else np.nan,
                "skew": stats.skew(s),
                "excess_kurtosis": stats.kurtosis(s),
            }
        )
    return pd.DataFrame(rows).set_index("variable").round(3)


def distribution_shape(series: pd.Series, sample: int = 5000) -> pd.Series:
    """Compare a series against normal and uniform reference distributions.

    Both tests use the same sample. A large uniform p-value combined with a
    tiny normal p-value and excess kurtosis near -1.2 is the signature of
    values drawn from ``uniform(min, max)`` rather than measured.
    """
    s = series.dropna().astype(float)
    if len(s) > sample:
        s = s.sample(sample, random_state=0)
    x = s.to_numpy()

    _, p_normal = stats.normaltest(x)
    scaled = (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else x
    _, p_uniform = stats.kstest(scaled, "uniform")

    return pd.Series(
        {
            "n": len(x),
            "mean": round(float(np.mean(x)), 3),
            "std": round(float(np.std(x, ddof=1)), 3),
            "skew": round(float(stats.skew(x)), 3),
            "excess_kurtosis": round(float(stats.kurtosis(x)), 3),
            "p_normal": round(float(p_normal), 4),
            "p_uniform": round(float(p_uniform), 4),
            "closest_shape": "uniform" if p_uniform > p_normal else "normal-ish",
        }
    )


def distribution_shape_by_city(panel: pd.DataFrame, column: str = TARGET_COL) -> pd.DataFrame:
    """Run :func:`distribution_shape` for every city."""
    return pd.DataFrame(
        {city: distribution_shape(grp[column]) for city, grp in panel.groupby(CITY_COL)}
    ).T


# --------------------------------------------------------------------------- #
# 3. City profiles
# --------------------------------------------------------------------------- #
def city_profile(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-city temperature characteristics an analyst would quote first."""
    df = add_calendar_columns(panel)
    rows = []
    for city, grp in df.groupby(CITY_COL):
        monthly = grp.groupby("month", observed=True)[TARGET_COL].mean()
        hottest, coldest = monthly.idxmax(), monthly.idxmin()
        diurnal = (
            (grp["temp_max"] - grp["temp_min"]).mean()
            if {"temp_max", "temp_min"} <= set(grp.columns)
            else np.nan
        )
        rows.append(
            {
                "city": city,
                "state": grp[STATE_COL].iloc[0] if STATE_COL in grp else "",
                "mean_temp": grp[TARGET_COL].mean(),
                "std_temp": grp[TARGET_COL].std(),
                "min_temp": grp[TARGET_COL].min(),
                "max_temp": grp[TARGET_COL].max(),
                "observed_range": grp[TARGET_COL].max() - grp[TARGET_COL].min(),
                "seasonal_amplitude": monthly.max() - monthly.min(),
                "hottest_month": MONTH_NAMES[hottest - 1],
                "coldest_month": MONTH_NAMES[coldest - 1],
                "mean_diurnal_range": diurnal,
            }
        )
    return pd.DataFrame(rows).set_index("city").round(2).sort_values(
        "mean_temp", ascending=False
    )


def climatology_plausibility(
    panel: pd.DataFrame, min_eta_squared: float = 0.25
) -> pd.DataFrame:
    """Compare observed city statistics against real-world reference values.

    ``seasonal_amplitude`` is the difference between the hottest and coldest
    monthly mean. Delhi's is about 20 °C in reality; Mumbai's about 6 °C.

    Amplitude alone is not enough to judge plausibility, because it is a range
    of twelve sample means and therefore inflated by noise. Twelve monthly
    means of pure ``uniform(18, 42)`` noise spread about 3 °C by chance — which
    is half of Mumbai's genuine 6 °C amplitude, so a noise series can
    accidentally "match" a low-amplitude coastal city.

    The verdict therefore also requires ``eta_squared`` — the share of variance
    explained by month-of-year — to clear ``min_eta_squared``. That statistic
    is noise-aware: it is ~0.02 for random data and 0.5-0.85 for real Indian
    temperature, so it separates the two cases cleanly regardless of amplitude.
    """
    prof = city_profile(panel)
    variation = monthly_variation_summary(panel)
    rows = []
    for city, row in prof.iterrows():
        ref = REFERENCE_CLIMATE.get(city)
        if ref is None:
            continue
        ref_mean, ref_amp = ref
        ratio = row["seasonal_amplitude"] / ref_amp
        eta_sq = float(variation.loc[city, "eta_squared"])
        rows.append(
            {
                "city": city,
                "observed_mean": row["mean_temp"],
                "reference_mean": ref_mean,
                "mean_gap": round(row["mean_temp"] - ref_mean, 2),
                "observed_amplitude": row["seasonal_amplitude"],
                "reference_amplitude": ref_amp,
                "amplitude_ratio": round(ratio, 3),
                "eta_squared": eta_sq,
                "plausible": bool(
                    abs(row["mean_temp"] - ref_mean) < 4
                    and 0.5 <= ratio <= 1.8
                    and eta_sq >= min_eta_squared
                ),
            }
        )
    return pd.DataFrame(rows).set_index("city")


def rank_cities(panel: pd.DataFrame, metric: str = "mean_temp", top: int = 10) -> pd.DataFrame:
    """Rank cities by any column produced by :func:`city_profile`."""
    prof = city_profile(panel)
    if metric not in prof.columns:
        raise KeyError(f"{metric!r} not available. Choose from {list(prof.columns)}")
    out = prof[[metric]].sort_values(metric, ascending=False).head(top)
    out["rank"] = range(1, len(out) + 1)
    return out


# --------------------------------------------------------------------------- #
# 4. Temporal breakdowns
# --------------------------------------------------------------------------- #
def monthly_profile(
    panel: pd.DataFrame, column: str = TARGET_COL, city: str | None = None
) -> pd.DataFrame:
    """Month-by-month statistics, optionally for a single city."""
    df = add_calendar_columns(panel)
    if city is not None:
        df = df.loc[df[CITY_COL] == city]
        if df.empty:
            raise KeyError(f"Unknown city {city!r}")
    g = df.groupby("month_name", observed=True)[column]
    out = pd.DataFrame(
        {
            "n": g.size(),
            "mean": g.mean(),
            "std": g.std(),
            "min": g.min(),
            "median": g.median(),
            "max": g.max(),
        }
    ).round(2)
    out["spread_from_annual_mean"] = (out["mean"] - df[column].mean()).round(2)
    return out


def seasonal_profile(panel: pd.DataFrame, column: str = TARGET_COL) -> pd.DataFrame:
    """Statistics grouped by IMD season."""
    df = add_calendar_columns(panel)
    g = df.groupby("season", observed=True)[column]
    return pd.DataFrame(
        {"n": g.size(), "mean": g.mean(), "std": g.std(), "min": g.min(), "max": g.max()}
    ).round(2)


def city_month_matrix(panel: pd.DataFrame, column: str = TARGET_COL) -> pd.DataFrame:
    """``city x month`` mean matrix — the compact view of seasonal structure."""
    df = add_calendar_columns(panel)
    mat = df.pivot_table(
        index=CITY_COL, columns="month_name", values=column, aggfunc="mean", observed=True
    )
    return mat.reindex(columns=MONTH_NAMES).round(2)


def monthly_variation_summary(panel: pd.DataFrame, column: str = TARGET_COL) -> pd.DataFrame:
    """How much of each city's variance is explained by month-of-year.

    ``eta_squared`` is the one-way ANOVA effect size. Indian cities should
    score well above 0.5 for temperature: month is the dominant driver.
    """
    df = add_calendar_columns(panel)
    rows = []
    for city, grp in df.groupby(CITY_COL):
        groups = [g[column].to_numpy() for _, g in grp.groupby("month", observed=True)]
        grand = grp[column].mean()
        ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
        ss_total = float(((grp[column] - grand) ** 2).sum())
        eta_sq = ss_between / ss_total if ss_total else np.nan
        monthly = grp.groupby("month", observed=True)[column].mean()
        rows.append(
            {
                "city": city,
                "monthly_mean_range": round(float(monthly.max() - monthly.min()), 2),
                "eta_squared": round(float(eta_sq), 4),
                "variance_explained_pct": round(100 * float(eta_sq), 1),
            }
        )
    return pd.DataFrame(rows).set_index("city").sort_values(
        "eta_squared", ascending=False
    )


# --------------------------------------------------------------------------- #
# 5. Extremes and comfort
# --------------------------------------------------------------------------- #
def extreme_day_counts(
    panel: pd.DataFrame, hot_threshold: float = 35.0, cold_threshold: float = 15.0
) -> pd.DataFrame:
    """Count hot and cold days per city, on the daily maximum and minimum."""
    df = add_calendar_columns(panel)
    rows = []
    for city, grp in df.groupby(CITY_COL):
        n = len(grp)
        hot = (grp.get("temp_max", grp[TARGET_COL]) >= hot_threshold).sum()
        cold = (grp.get("temp_min", grp[TARGET_COL]) <= cold_threshold).sum()
        rows.append(
            {
                "city": city,
                "n_days": n,
                f"days_max_ge_{hot_threshold:g}": int(hot),
                "pct_hot": round(100 * hot / n, 1),
                f"days_min_le_{cold_threshold:g}": int(cold),
                "pct_cold": round(100 * cold / n, 1),
            }
        )
    return pd.DataFrame(rows).set_index("city").sort_values("pct_hot", ascending=False)


def longest_run(mask: pd.Series) -> int:
    """Length of the longest consecutive ``True`` run in a boolean series."""
    arr = np.asarray(mask, dtype=bool)
    if not arr.any():
        return 0
    best = run = 0
    for v in arr:
        run = run + 1 if v else 0
        best = max(best, run)
    return int(best)


def heat_spell_summary(panel: pd.DataFrame, hot_threshold: float = 35.0) -> pd.DataFrame:
    """Longest consecutive stretch of hot days per city.

    Real heatwaves cluster — hot days arrive in multi-day blocks. If the
    longest run across two years is only 1-2 days, consecutive days are
    independent, which is itself a strong statement about the data.
    """
    rows = []
    for city, grp in panel.groupby(CITY_COL):
        g = grp.sort_values(DATE_COL)
        series = g.get("temp_max", g[TARGET_COL])
        hot = series >= hot_threshold
        n_hot = int(hot.sum())
        run = longest_run(hot)
        expected = _expected_longest_run(len(hot), hot.mean())
        rows.append(
            {
                "city": city,
                "n_hot_days": n_hot,
                "longest_hot_run": run,
                "expected_run_if_independent": expected,
                "clustering": round(run / expected, 2) if expected else np.nan,
            }
        )
    return pd.DataFrame(rows).set_index("city").sort_values(
        "longest_hot_run", ascending=False
    )


def _expected_longest_run(n: int, p: float) -> float:
    """Expected longest success run in ``n`` independent Bernoulli(p) trials."""
    if not 0 < p < 1 or n <= 1:
        return 0.0
    return round(float(np.log(n * (1 - p)) / np.log(1 / p)), 2)


def diurnal_range_profile(panel: pd.DataFrame) -> pd.DataFrame:
    """Daily temperature range (max - min) by city and season.

    Physically, diurnal range narrows under cloud and humidity — clear desert
    air swings far more than a humid monsoon day. That relationship is a good
    integrity check on any climate dataset.
    """
    if not {"temp_max", "temp_min"} <= set(panel.columns):
        raise KeyError("diurnal range needs temp_max and temp_min")
    df = add_calendar_columns(panel)
    df["diurnal_range"] = df["temp_max"] - df["temp_min"]

    by_city = df.groupby(CITY_COL)["diurnal_range"].agg(["mean", "std", "min", "max"])
    by_season = df.pivot_table(
        index=CITY_COL, columns="season", values="diurnal_range",
        aggfunc="mean", observed=True,
    )
    out = by_city.join(by_season).round(2)

    if "cloud_cover" in df.columns:
        corr = df.groupby(CITY_COL).apply(
            lambda g: g["diurnal_range"].corr(g["cloud_cover"]), include_groups=False
        )
        out["corr_with_cloud"] = corr.round(3)
    return out.sort_values("mean", ascending=False)


# --------------------------------------------------------------------------- #
# 6. Relationships between channels
# --------------------------------------------------------------------------- #
def correlation_with_significance(
    panel: pd.DataFrame, target: str = TARGET_COL, columns: list[str] | None = None
) -> pd.DataFrame:
    """Pearson and Spearman correlation of each channel with the target.

    Includes p-values and a Holm-Bonferroni corrected significance flag, since
    testing eight channels at once inflates the false-positive rate.
    """
    columns = columns or [c for c in EXOG_COLS if c in panel.columns]
    rows = []
    for col in columns:
        sub = panel[[target, col]].dropna()
        r, p_r = stats.pearsonr(sub[target], sub[col])
        rho, p_rho = stats.spearmanr(sub[target], sub[col])
        rows.append(
            {
                "variable": col,
                "pearson_r": round(float(r), 4),
                "pearson_p": float(p_r),
                "spearman_rho": round(float(rho), 4),
                "spearman_p": float(p_rho),
                "abs_r": abs(round(float(r), 4)),
                "n": len(sub),
            }
        )
    out = pd.DataFrame(rows).sort_values("abs_r", ascending=False).reset_index(drop=True)

    # Holm-Bonferroni step-down correction.
    m = len(out)
    order = out["pearson_p"].rank(method="first").astype(int)
    out["holm_threshold"] = 0.05 / (m - order + 1)
    out["significant"] = out["pearson_p"] < out["holm_threshold"]
    out["pearson_p"] = out["pearson_p"].round(5)
    out["spearman_p"] = out["spearman_p"].round(5)
    return out.set_index("variable").drop(columns="abs_r")


def channel_correlation_matrix(
    panel: pd.DataFrame, columns: list[str] | None = None, method: str = "pearson"
) -> pd.DataFrame:
    """Full correlation matrix across the numeric weather channels."""
    columns = columns or _numeric_columns(panel)
    return panel[columns].corr(method=method).round(3)


def cross_city_correlation(panel: pd.DataFrame, column: str = TARGET_COL) -> pd.DataFrame:
    """Correlation between cities on the same day."""
    wide = panel.pivot_table(index=DATE_COL, columns=CITY_COL, values=column, observed=True)
    return wide.corr().round(3)


# --------------------------------------------------------------------------- #
# 7. Categorical / air quality
# --------------------------------------------------------------------------- #
def aqi_category_burden(panel: pd.DataFrame) -> pd.DataFrame:
    """Share of days in each AQI category, per city (percentages)."""
    if "aqi_category" not in panel.columns:
        raise KeyError("panel has no aqi_category column")
    ct = pd.crosstab(panel[CITY_COL], panel["aqi_category"], normalize="index") * 100
    order = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
    ct = ct.reindex(columns=[c for c in order if c in ct.columns])
    return ct.round(1)


def aqi_seasonality(panel: pd.DataFrame) -> pd.DataFrame:
    """Mean AQI by city and season.

    In reality north-Indian AQI peaks sharply in winter, when cool stagnant air
    traps particulates, and is scrubbed by the monsoon.
    """
    if "aqi" not in panel.columns:
        raise KeyError("panel has no aqi column")
    df = add_calendar_columns(panel)
    out = df.pivot_table(
        index=CITY_COL, columns="season", values="aqi", aggfunc="mean", observed=True
    ).round(1)
    if {"Winter", "Monsoon"} <= set(out.columns):
        out["winter_minus_monsoon"] = (out["Winter"] - out["Monsoon"]).round(1)
    return out.sort_values(out.columns[0], ascending=False)


__all__ = [
    "IMD_SEASONS",
    "MONTH_NAMES",
    "REFERENCE_CLIMATE",
    "add_calendar_columns",
    "aqi_category_burden",
    "aqi_seasonality",
    "channel_correlation_matrix",
    "city_month_matrix",
    "city_profile",
    "climatology_plausibility",
    "column_inventory",
    "correlation_with_significance",
    "cross_city_correlation",
    "dataset_profile",
    "distribution_shape",
    "distribution_shape_by_city",
    "diurnal_range_profile",
    "extreme_day_counts",
    "heat_spell_summary",
    "longest_run",
    "monthly_profile",
    "monthly_variation_summary",
    "numeric_summary",
    "rank_cities",
    "seasonal_profile",
]
