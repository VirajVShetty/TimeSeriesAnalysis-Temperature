"""Leakage-safe feature engineering for direct multi-horizon forecasting.

Convention used throughout this module
--------------------------------------
Features are indexed by the **forecast origin** ``t`` — the last date whose
observations are known. Targets are ``y[t + h]`` for horizons ``h = 1..H``.

* ``y_lag{k}`` is ``y[t - k + 1]``, so ``y_lag1`` is the most recent *observed*
  value. Nothing dated after ``t`` ever enters a feature.
* Calendar / Fourier terms are evaluated on the **target date** ``t + h``,
  which is legitimate because the calendar is known arbitrarily far ahead.
* Any statistic estimated from the target (e.g. day-of-year climatology) is
  fit on the training split only and applied to the test split.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import CITY_COL, DATE_COL, EXOG_COLS, FEATURES, TARGET_COL, FeatureConfig


# --------------------------------------------------------------------------- #
# Calendar / Fourier
# --------------------------------------------------------------------------- #
def day_of_year(dates: pd.DatetimeIndex) -> np.ndarray:
    """Leap-year-safe day of year normalised to [0, 1)."""
    doy = dates.dayofyear.to_numpy(dtype=float)
    year_len = np.where(dates.is_leap_year, 366.0, 365.0)
    return doy / year_len


def fourier_terms(
    dates: pd.DatetimeIndex, order: int = 4, prefix: str = "annual"
) -> pd.DataFrame:
    """Sin/cos harmonics encoding the annual cycle.

    Fourier terms let a model represent a 365-day seasonality with ``2*order``
    parameters instead of 364 dummy variables — essential when only two annual
    cycles of data are available.
    """
    frac = day_of_year(dates)
    out = {}
    for k in range(1, order + 1):
        out[f"{prefix}_sin{k}"] = np.sin(2 * np.pi * k * frac)
        out[f"{prefix}_cos{k}"] = np.cos(2 * np.pi * k * frac)
    return pd.DataFrame(out, index=dates)


#: Calendar columns that are safe for a tree model to see.
#:
#: Deliberately excluded:
#:
#: ``doy`` / ``time_idx``
#:     A day-of-year or absolute time index uniquely identifies a training row
#:     once combined with ``city_code``. Trees then memorise individual dates
#:     instead of learning dynamics — in-sample error collapses to a flat value
#:     across all horizons while out-of-sample error degrades. ``time_idx`` is
#:     doubly harmful: at prediction time it takes values never seen in
#:     training, so every forecast falls into the same extreme leaf.
#: ``dayofweek``
#:     There is no weekly cycle in temperature. Including it hands the model a
#:     7-way split that can only fit noise; it was picking up 4.4% of total
#:     gain on data generated with no weekly component at all.
#:
#: The annual position is supplied smoothly by the Fourier terms instead, which
#: cannot isolate a single day.
SAFE_CALENDAR_COLS: tuple[str, ...] = ("month", "imd_season")


def calendar_features(
    dates: pd.DatetimeIndex, include: tuple[str, ...] = SAFE_CALENDAR_COLS
) -> pd.DataFrame:
    """Deterministic calendar attributes of the target date.

    See :data:`SAFE_CALENDAR_COLS` for why the high-cardinality columns are not
    included by default.
    """
    idx = pd.DatetimeIndex(dates)
    month = idx.month
    # Indian Meteorological Department seasons: winter, pre-monsoon, monsoon,
    # post-monsoon.
    season = np.select(
        [np.isin(month, [12, 1, 2]), np.isin(month, [3, 4, 5]), np.isin(month, [6, 7, 8, 9])],
        [0, 1, 2],
        default=3,
    )
    available = {
        "month": month.astype(int),
        "imd_season": season.astype(int),
        "doy": idx.dayofyear.astype(int),
        "dayofweek": idx.dayofweek.astype(int),
        "time_idx": (idx - idx.min()).days.astype(int),
    }
    unknown = set(include) - set(available)
    if unknown:
        raise ValueError(f"Unknown calendar features: {sorted(unknown)}")
    return pd.DataFrame({c: available[c] for c in include}, index=idx)


# --------------------------------------------------------------------------- #
# Day-of-year climatology (fit on train only)
# --------------------------------------------------------------------------- #
@dataclass
class DayOfYearClimatology:
    """Smoothed per-city seasonal normal, estimated from training data only.

    Acts as a strong, physically meaningful prior: the expected temperature in
    a given city on a given day of the year. Because it is estimated from the
    target it must never see test observations.
    """

    window: int = 15
    table_: pd.DataFrame | None = None
    global_mean_: float = 0.0

    def fit(self, df: pd.DataFrame) -> DayOfYearClimatology:
        work = df[[CITY_COL, DATE_COL, TARGET_COL]].copy()
        work["doy"] = pd.DatetimeIndex(work[DATE_COL]).dayofyear
        raw = work.groupby([CITY_COL, "doy"])[TARGET_COL].mean().unstack("doy")
        raw = raw.reindex(columns=range(1, 367))
        # Circular smoothing so 31 Dec and 1 Jan are neighbours.
        tripled = pd.concat([raw, raw, raw], axis=1)
        smoothed = (
            tripled.T.interpolate(limit_direction="both")
            .rolling(self.window, center=True, min_periods=1)
            .mean()
            .T
        )
        n = raw.shape[1]
        smoothed = smoothed.iloc[:, n : 2 * n]
        smoothed.columns = raw.columns
        self.table_ = smoothed
        self.global_mean_ = float(work[TARGET_COL].mean())
        return self

    def transform(self, cities: pd.Series, dates: pd.Series) -> pd.Series:
        if self.table_ is None:
            raise RuntimeError("DayOfYearClimatology must be fit before transform.")
        doy = pd.DatetimeIndex(dates).dayofyear
        # ``future_stack`` only exists on pandas >= 2.1; fall back cleanly so the
        # package still works on older pandas rather than raising TypeError.
        try:
            stacked = self.table_.stack(future_stack=True)
        except TypeError:  # pragma: no cover - depends on installed pandas
            stacked = self.table_.stack(dropna=False)
        keys = pd.MultiIndex.from_arrays([cities.to_numpy(), doy])
        vals = stacked.reindex(keys).to_numpy(dtype=float)
        vals = np.where(np.isnan(vals), self.global_mean_, vals)
        return pd.Series(vals, index=cities.index, name="climatology")

    def fit_transform(self, df: pd.DataFrame) -> pd.Series:
        return self.fit(df).transform(df[CITY_COL], df[DATE_COL])


# --------------------------------------------------------------------------- #
# Origin-indexed history features
# --------------------------------------------------------------------------- #
def build_history_features(
    city_frame: pd.DataFrame, cfg: FeatureConfig = FEATURES
) -> pd.DataFrame:
    """Features known at each forecast origin for a single city.

    Parameters
    ----------
    city_frame:
        Daily frame for one city, ``DatetimeIndex``ed, containing the target
        and the exogenous channels.
    """
    y = city_frame[TARGET_COL].astype(float)
    feats: dict[str, pd.Series] = {}

    # --- target lags: y_lag1 is the most recent observation ----------------
    for lag in cfg.target_lags:
        feats[f"y_lag{lag}"] = y.shift(lag - 1)

    # --- rolling statistics over the observed window ----------------------
    for w in cfg.rolling_windows:
        roll = y.rolling(w, min_periods=max(2, w // 3))
        feats[f"y_rmean{w}"] = roll.mean()
        feats[f"y_rstd{w}"] = roll.std()
        feats[f"y_rmin{w}"] = roll.min()
        feats[f"y_rmax{w}"] = roll.max()

    # --- momentum / short-term dynamics -----------------------------------
    feats["y_diff1"] = y.diff()
    feats["y_diff7"] = y.diff(7)
    feats["y_trend7"] = y.rolling(7, min_periods=3).mean() - y.rolling(
        28, min_periods=7
    ).mean()
    feats["y_ewm7"] = y.ewm(span=7, adjust=False).mean()
    feats["y_ewm30"] = y.ewm(span=30, adjust=False).mean()

    # --- exogenous channels, lagged only ----------------------------------
    if cfg.use_exog:
        for col in EXOG_COLS:
            if col not in city_frame:
                continue
            s = city_frame[col].astype(float)
            for lag in cfg.exog_lags:
                feats[f"{col}_lag{lag}"] = s.shift(lag - 1)
            feats[f"{col}_rmean7"] = s.rolling(7, min_periods=3).mean()
        if {"temp_max", "temp_min"} <= set(city_frame.columns):
            rng = city_frame["temp_max"].astype(float) - city_frame["temp_min"].astype(float)
            feats["diurnal_range"] = rng
            feats["diurnal_range_rmean7"] = rng.rolling(7, min_periods=3).mean()

    out = pd.DataFrame(feats, index=city_frame.index)
    out.index.name = "origin"
    return out


# --------------------------------------------------------------------------- #
# Supervised dataset assembly
# --------------------------------------------------------------------------- #
@dataclass
class SupervisedDataset:
    """Stacked (origin, horizon) design matrix for direct multi-horizon models."""

    X: pd.DataFrame
    y: pd.Series
    meta: pd.DataFrame  # origin, target_date, city, horizon

    def __len__(self) -> int:
        return len(self.X)

    @property
    def feature_names(self) -> list[str]:
        return list(self.X.columns)

    def filter(self, mask: np.ndarray | pd.Series) -> SupervisedDataset:
        mask = np.asarray(mask)
        return SupervisedDataset(
            X=self.X.loc[mask].reset_index(drop=True),
            y=self.y.loc[mask].reset_index(drop=True),
            meta=self.meta.loc[mask].reset_index(drop=True),
        )


def build_supervised_dataset(
    panel: pd.DataFrame,
    horizon: int,
    cfg: FeatureConfig = FEATURES,
    drop_incomplete: bool = True,
) -> SupervisedDataset:
    """Turn the long climate panel into a direct multi-horizon training set.

    One row per ``(city, origin, horizon)`` triple. The horizon itself is a
    feature, so a single global model covers all lead times — far more
    sample-efficient than fitting ``H`` separate models on two years of data.
    """
    frames_X, frames_y, frames_meta = [], [], []

    for city, grp in panel.groupby(CITY_COL, sort=True):
        g = grp.sort_values(DATE_COL).set_index(DATE_COL)
        hist = build_history_features(g, cfg)
        y = g[TARGET_COL].astype(float)

        for h in range(1, horizon + 1):
            target = y.shift(-h)
            target_date = pd.Series(g.index, index=g.index) + pd.Timedelta(days=h)

            block = hist.copy()
            block["horizon"] = h
            # Calendar features refer to the *target* date (known in advance).
            cal = calendar_features(
                pd.DatetimeIndex(target_date.to_numpy()), include=cfg.calendar_cols
            )
            fou = fourier_terms(
                pd.DatetimeIndex(target_date.to_numpy()), order=cfg.fourier_order
            )
            cal.index = block.index
            fou.index = block.index
            block = pd.concat([block, cal, fou], axis=1)

            meta = pd.DataFrame(
                {
                    "origin": g.index,
                    "target_date": target_date.to_numpy(),
                    CITY_COL: city,
                    "horizon": h,
                },
                index=g.index,
            )

            frames_X.append(block)
            frames_y.append(target)
            frames_meta.append(meta)

    X = pd.concat(frames_X, axis=0).reset_index(drop=True)
    y = pd.concat(frames_y, axis=0).reset_index(drop=True)
    meta = pd.concat(frames_meta, axis=0).reset_index(drop=True)

    if drop_incomplete:
        # Require the target and the short lags; long lags (e.g. 365) are
        # allowed to be NaN because LightGBM handles missing values natively.
        essential = [c for c in X.columns if c.startswith("y_lag") and _lag_of(c) <= 28]
        keep = y.notna() & X[essential].notna().all(axis=1)
        X, y, meta = X.loc[keep], y.loc[keep], meta.loc[keep]

    order = np.lexsort((meta["horizon"].to_numpy(), meta["origin"].to_numpy()))
    X = X.iloc[order].reset_index(drop=True)
    y = y.iloc[order].reset_index(drop=True).rename(TARGET_COL)
    meta = meta.iloc[order].reset_index(drop=True)
    return SupervisedDataset(X=X, y=y, meta=meta)


def _lag_of(name: str) -> int:
    try:
        return int(name.replace("y_lag", ""))
    except ValueError:
        return 10**9


def add_city_encodings(
    ds: SupervisedDataset, cities: list[str], climatology: DayOfYearClimatology | None
) -> SupervisedDataset:
    """Attach city identity and (optionally) the training-fit climatology prior."""
    X = ds.X.copy()
    code_map = {c: i for i, c in enumerate(sorted(cities))}
    X["city_code"] = ds.meta[CITY_COL].map(code_map).astype("category")
    if climatology is not None:
        clim = climatology.transform(ds.meta[CITY_COL], ds.meta["target_date"])
        X["climatology"] = clim.to_numpy()
        # Anomaly relative to the seasonal normal is usually more learnable
        # than the raw level.
        X["clim_gap_lag1"] = X["y_lag1"].to_numpy() - clim.to_numpy()
    return SupervisedDataset(X=X, y=ds.y, meta=ds.meta)


__all__ = [
    "DayOfYearClimatology",
    "SupervisedDataset",
    "add_city_encodings",
    "build_history_features",
    "build_supervised_dataset",
    "calendar_features",
    "day_of_year",
    "fourier_terms",
    "SAFE_CALENDAR_COLS",
]
