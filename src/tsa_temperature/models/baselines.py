"""Reference baselines.

Any serious forecasting study must state what it is beating. For daily
temperature the bar is high: persistence is strong at short lead times and the
seasonal climatology is strong at long ones.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import CITY_COL, DATE_COL, TARGET_COL
from ..features import DayOfYearClimatology
from .base import BaseForecaster


class NaivePersistence(BaseForecaster):
    """Random walk: every future day equals the last observed day."""

    name = "Naive (persistence)"

    def _fit(self, train_panel: pd.DataFrame) -> None:
        last = train_panel.sort_values(DATE_COL).groupby(CITY_COL)[TARGET_COL].last()
        self.last_ = last.to_dict()

    def _predict(self, horizon: int) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {CITY_COL: c, "horizon": h, "y_pred": self.last_[c]}
                for c in self.cities_
                for h in range(1, horizon + 1)
            ]
        )


class SeasonalNaive(BaseForecaster):
    """Value observed one seasonal period ago.

    With only two annual cycles the 365-day lag may be unavailable early in the
    sample, so the model degrades gracefully to the trailing 7-day mean.
    """

    name = "Seasonal naive (m=365)"

    def __init__(self, period: int = 365) -> None:
        super().__init__()
        self.period = period

    def _fit(self, train_panel: pd.DataFrame) -> None:
        self.series_ = {
            city: grp.sort_values(DATE_COL).set_index(DATE_COL)[TARGET_COL].astype(float)
            for city, grp in train_panel.groupby(CITY_COL)
        }

    def _predict(self, horizon: int) -> pd.DataFrame:
        rows = []
        for city, s in self.series_.items():
            fallback = float(s.iloc[-7:].mean())
            for h in range(1, horizon + 1):
                ref_date = self.origin_ + pd.Timedelta(days=h - self.period)
                val = s.get(ref_date, np.nan)
                rows.append(
                    {
                        CITY_COL: city,
                        "horizon": h,
                        "y_pred": float(val) if np.isfinite(val) else fallback,
                    }
                )
        return pd.DataFrame(rows)


class ClimatologyForecaster(BaseForecaster):
    """Smoothed day-of-year normal — the meteorologist's default.

    Estimated only from the training window, then optionally nudged by the
    recent anomaly (how far the last week sat above or below normal), which
    decays towards zero as the lead time grows.
    """

    name = "Climatology"

    def __init__(self, window: int = 15, anomaly_decay: float = 0.0) -> None:
        super().__init__()
        self.window = window
        self.anomaly_decay = anomaly_decay

    def _fit(self, train_panel: pd.DataFrame) -> None:
        self.clim_ = DayOfYearClimatology(window=self.window).fit(train_panel)
        recent = train_panel.sort_values(DATE_COL).groupby(CITY_COL).tail(7)
        base = self.clim_.transform(recent[CITY_COL], recent[DATE_COL])
        recent = recent.assign(_clim=base.to_numpy())
        self.recent_anomaly_ = (
            (recent[TARGET_COL] - recent["_clim"]).groupby(recent[CITY_COL]).mean().to_dict()
        )

    def _predict(self, horizon: int) -> pd.DataFrame:
        rows = []
        for city in self.cities_:
            for h in range(1, horizon + 1):
                target_date = self.origin_ + pd.Timedelta(days=h)
                base = float(
                    self.clim_.transform(
                        pd.Series([city]), pd.Series([target_date])
                    ).iloc[0]
                )
                adj = self.recent_anomaly_.get(city, 0.0) * (self.anomaly_decay**h)
                rows.append({CITY_COL: city, "horizon": h, "y_pred": base + adj})
        return pd.DataFrame(rows)


class ClimatologyPlusPersistence(ClimatologyForecaster):
    """Climatology with an exponentially decaying persistence correction.

    A surprisingly hard baseline: it captures the seasonal cycle *and* the fact
    that a warm spell tends to continue for a few days.
    """

    name = "Climatology + damped persistence"

    def __init__(self, window: int = 15, anomaly_decay: float = 0.85) -> None:
        super().__init__(window=window, anomaly_decay=anomaly_decay)


__all__ = [
    "ClimatologyForecaster",
    "ClimatologyPlusPersistence",
    "NaivePersistence",
    "SeasonalNaive",
]
