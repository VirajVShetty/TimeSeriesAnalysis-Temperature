"""Anomaly detection for daily temperature series.

The original project used a single detector — Brutlag confidence bands around
a Holt-Winters fit. That approach is retained here for continuity and
comparison, but three complementary modern detectors are added and combined
into a voting ensemble:

============================  ==========================================
Detector                      What it is good at
============================  ==========================================
``brutlag``                   Seasonally-varying deviation bands (legacy)
``stl_residual``              Robust decomposition; ignores trend/season
``isolation_forest``          Multivariate context (humidity, AQI, ...)
``matrix_profile``            *Shape* anomalies — unusual sub-sequences
============================  ==========================================

Single detectors disagree often. The ensemble flags a day only when at least
``ANOMALY.ensemble_min_votes`` methods agree, which sharply reduces false
positives.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from .config import ANOMALY, RANDOM_SEED, AnomalyConfig
from .eda import decompose


@dataclass
class AnomalyResult:
    """Per-timestamp anomaly flags and scores from one or more detectors."""

    frame: pd.DataFrame  # index: date; columns: value, score columns, flag columns

    @property
    def flag_columns(self) -> list[str]:
        return [c for c in self.frame.columns if c.startswith("flag_")]

    def anomalies(self, column: str = "flag_ensemble") -> pd.DataFrame:
        if column not in self.frame:
            raise KeyError(f"{column} not in results. Available: {self.flag_columns}")
        return self.frame.loc[self.frame[column]]

    def agreement(self) -> pd.DataFrame:
        """How many days each pair of detectors agrees on."""
        flags = self.frame[self.flag_columns].astype(int)
        return flags.T @ flags

    def summary(self) -> pd.Series:
        flags = self.frame[self.flag_columns]
        return pd.Series(
            {c: int(flags[c].sum()) for c in self.flag_columns}, name="n_flagged"
        )


# --------------------------------------------------------------------------- #
# 1. Brutlag (legacy, retained for comparison)
# --------------------------------------------------------------------------- #
def brutlag_bands(
    series: pd.Series,
    period: int = 365,
    gamma: float = ANOMALY.brutlag_gamma,
    scaling_factor: float = ANOMALY.brutlag_scaling_factor,
) -> pd.DataFrame:
    """Brutlag's seasonally-adaptive deviation bands around a smoothed fit.

    ``d_t = gamma * |y_t - yhat_t| + (1 - gamma) * d_{t-m}`` tracks how much
    deviation is *normal* for this point in the cycle, so bands widen in
    volatile seasons and tighten in stable ones.

    The baseline comes from :func:`tsa_temperature.eda.decompose`, which picks
    STL or harmonic regression depending on how many cycles the data supports.
    The original implementation used a Holt-Winters fit whose seasonal period
    equalled the data length, which made the bands collapse onto the series.
    """
    s = series.astype(float).dropna()
    if len(s) < 2 * period:
        period = max(7, len(s) // 3)

    dec = decompose(s, period=period, method="auto")
    fitted = dec.trend + dec.seasonal

    diff = (s - fitted).to_numpy()
    abs_diff = np.abs(diff)

    # Initialisation fix. The recursion d_t = g|e_t| + (1-g) d_{t-m} has the
    # steady state d = E|e|, but the legacy implementation seeded the first
    # season with d_t = g|e_t|, which is g times too small (g ~ 0.37 here).
    # The bands were therefore roughly a third of their intended width and
    # flagged most of the first year as anomalous. Seeding with an expanding
    # mean of |e| starts the recursion at the correct scale.
    dt = np.zeros(len(s))
    running = np.cumsum(abs_diff) / np.arange(1, len(s) + 1)
    for i in range(len(s)):
        if i < period:
            dt[i] = gamma * abs_diff[i] + (1 - gamma) * running[i]
        else:
            dt[i] = gamma * abs_diff[i] + (1 - gamma) * dt[i - period]

    upper = fitted.to_numpy() + scaling_factor * dt
    lower = fitted.to_numpy() - scaling_factor * dt
    return pd.DataFrame(
        {
            "value": s,
            "predicted": fitted,
            "deviation": dt,
            "upper": upper,
            "lower": lower,
            "flag_brutlag": (s.to_numpy() > upper) | (s.to_numpy() < lower),
        },
        index=s.index,
    )


# --------------------------------------------------------------------------- #
# 2. STL residual z-score
# --------------------------------------------------------------------------- #
def stl_residual_anomalies(
    series: pd.Series,
    period: int = ANOMALY.stl_period,
    z_threshold: float = ANOMALY.residual_z_threshold,
    robust: bool = ANOMALY.stl_robust,
) -> pd.DataFrame:
    """Flag points whose decomposition remainder is extreme.

    Uses the median absolute deviation rather than the standard deviation so a
    handful of large outliers cannot inflate the threshold and mask themselves.
    """
    s = series.astype(float).dropna()
    if len(s) < 2 * period:
        period = max(7, len(s) // 3)
    dec = decompose(s, period=period, method="auto", robust=robust)
    resid = dec.resid

    med = float(np.median(resid))
    mad = float(np.median(np.abs(resid - med)))
    scale = 1.4826 * mad if mad > 1e-9 else float(resid.std()) or 1.0
    z = (resid - med) / scale

    return pd.DataFrame(
        {
            "value": s,
            "stl_resid": resid,
            "stl_z": z,
            "flag_stl": z.abs() > z_threshold,
        },
        index=s.index,
    )


# --------------------------------------------------------------------------- #
# 3. Isolation Forest on multivariate context
# --------------------------------------------------------------------------- #
def isolation_forest_anomalies(
    frame: pd.DataFrame,
    feature_cols: list[str],
    contamination: float = ANOMALY.isolation_forest_contamination,
    random_state: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Detect days that are unusual given the *joint* weather state.

    A 34 °C day in Delhi is unremarkable in May; a 34 °C day with 95% humidity,
    1030 hPa pressure and zero cloud cover is not. Univariate detectors cannot
    see that — an isolation forest can.
    """
    X = frame[feature_cols].astype(float)
    X = X.interpolate().ffill().bfill()
    Xs = StandardScaler().fit_transform(X)

    iso = IsolationForest(
        n_estimators=400,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    ).fit(Xs)

    # Higher score = more anomalous (sklearn's score_samples is inverted).
    score = -iso.score_samples(Xs)
    return pd.DataFrame(
        {
            "iforest_score": score,
            "flag_iforest": iso.predict(Xs) == -1,
        },
        index=frame.index,
    )


# --------------------------------------------------------------------------- #
# 4. Matrix profile (shape-based)
# --------------------------------------------------------------------------- #
def matrix_profile_anomalies(
    series: pd.Series, window: int = 14, top_k: int = 10
) -> pd.DataFrame:
    """Self-join matrix profile — finds *discords*, i.e. unusual sub-sequences.

    Each window is compared to its nearest non-trivial neighbour elsewhere in
    the series using z-normalised Euclidean distance. Windows whose nearest
    neighbour is still far away are shape anomalies: an unprecedented pattern
    of change rather than a single extreme reading.

    Implemented directly with NumPy to avoid an extra dependency; the series
    here is short enough that the O(n^2) sliding-dot-product is instant.
    """
    s = series.astype(float).dropna()
    x = s.to_numpy()
    n = len(x)
    m = window
    if n < 3 * m:
        return pd.DataFrame(
            {"mp_distance": np.nan, "flag_mp": False}, index=s.index
        )

    n_sub = n - m + 1
    subs = np.lib.stride_tricks.sliding_window_view(x, m)
    mu = subs.mean(axis=1, keepdims=True)
    sigma = subs.std(axis=1, keepdims=True)
    sigma[sigma < 1e-8] = 1.0
    z = (subs - mu) / sigma

    profile = np.full(n_sub, np.inf)
    exclusion = max(1, m // 2)
    # Chunked to keep the pairwise matrix small in memory.
    chunk = 512
    for start in range(0, n_sub, chunk):
        end = min(start + chunk, n_sub)
        block = z[start:end]
        d = np.sqrt(
            np.maximum(
                2 * m * (1 - (block @ z.T) / m),
                0.0,
            )
        )
        rows = np.arange(start, end)
        for i, r in enumerate(rows):
            lo = max(0, r - exclusion)
            hi = min(n_sub, r + exclusion + 1)
            d[i, lo:hi] = np.inf
        profile[start:end] = d.min(axis=1)

    # Map sub-sequence scores back to their starting timestamp.
    dist = pd.Series(np.nan, index=s.index)
    dist.iloc[:n_sub] = profile
    threshold = np.nanpercentile(profile, 100 * (1 - top_k / max(n_sub, 1)))
    flag = dist >= threshold
    return pd.DataFrame({"mp_distance": dist, "flag_mp": flag.fillna(False)}, index=s.index)


# --------------------------------------------------------------------------- #
# Ensemble
# --------------------------------------------------------------------------- #
def detect_anomalies(
    city_frame: pd.DataFrame,
    target_col: str = "temp_avg",
    context_cols: list[str] | None = None,
    cfg: AnomalyConfig = ANOMALY,
    use_matrix_profile: bool = True,
) -> AnomalyResult:
    """Run all detectors on one city and combine them by majority vote.

    Parameters
    ----------
    city_frame:
        ``DatetimeIndex``ed daily frame for a single city.
    context_cols:
        Extra channels handed to the isolation forest.
    """
    series = city_frame[target_col].astype(float)
    parts = [pd.DataFrame({"value": series}, index=series.index)]

    parts.append(brutlag_bands(series, period=cfg.stl_period)[["flag_brutlag", "upper", "lower"]])
    parts.append(
        stl_residual_anomalies(
            series, period=cfg.stl_period, z_threshold=cfg.residual_z_threshold
        )[["stl_resid", "stl_z", "flag_stl"]]
    )

    if context_cols is None:
        context_cols = [
            c for c in city_frame.columns if c != target_col and city_frame[c].dtype.kind in "fi"
        ]
    if context_cols:
        parts.append(
            isolation_forest_anomalies(
                city_frame.assign(**{target_col: series}),
                feature_cols=[target_col, *context_cols],
                contamination=cfg.isolation_forest_contamination,
            )
        )

    if use_matrix_profile:
        parts.append(matrix_profile_anomalies(series))

    out = pd.concat(parts, axis=1)
    out = out.loc[:, ~out.columns.duplicated()]

    flag_cols = [c for c in out.columns if c.startswith("flag_")]
    votes = out[flag_cols].fillna(False).astype(int).sum(axis=1)
    out["n_votes"] = votes
    out["flag_ensemble"] = votes >= cfg.ensemble_min_votes
    return AnomalyResult(out)


def detect_anomalies_panel(
    panel: pd.DataFrame,
    city_col: str = "city",
    date_col: str = "date",
    target_col: str = "temp_avg",
    cfg: AnomalyConfig = ANOMALY,
) -> pd.DataFrame:
    """Run the ensemble detector across every city and stack the results."""
    frames = []
    for city, grp in panel.groupby(city_col):
        g = grp.sort_values(date_col).set_index(date_col)
        numeric = g.select_dtypes(include="number")
        res = detect_anomalies(numeric, target_col=target_col, cfg=cfg)
        f = res.frame.copy()
        f[city_col] = city
        frames.append(f.reset_index().rename(columns={"index": date_col}))
    return pd.concat(frames, ignore_index=True)


__all__ = [
    "AnomalyResult",
    "brutlag_bands",
    "detect_anomalies",
    "detect_anomalies_panel",
    "isolation_forest_anomalies",
    "matrix_profile_anomalies",
    "stl_residual_anomalies",
]
