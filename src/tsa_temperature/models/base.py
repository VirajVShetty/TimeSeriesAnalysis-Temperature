"""Common forecaster interface.

Every model in this project implements the same tiny contract:

``fit(train_panel)``   Learn from a long-format panel whose last date is the
                       forecast origin.
``predict(horizon)``   Return one row per ``(city, horizon)`` for the ``horizon``
                       days immediately following that origin.

This uniformity is what lets the walk-forward harness treat a seasonal-naive
rule, a SARIMAX, a gradient-boosted tree ensemble and an LSTM identically.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from ..config import CITY_COL, DATE_COL, TARGET_COL


class BaseForecaster(ABC):
    """Abstract base class for all forecasters."""

    name: str = "base"
    #: Set to True by models that produce their own uncertainty estimates.
    provides_intervals: bool = False

    def __init__(self) -> None:
        self.origin_: pd.Timestamp | None = None
        self.cities_: list[str] = []
        self.is_fitted_: bool = False

    # ------------------------------------------------------------------ #
    @abstractmethod
    def _fit(self, train_panel: pd.DataFrame) -> None:
        """Model-specific fitting logic."""

    @abstractmethod
    def _predict(self, horizon: int) -> pd.DataFrame:
        """Return columns ``[city, horizon, y_pred]``."""

    # ------------------------------------------------------------------ #
    def fit(self, train_panel: pd.DataFrame) -> BaseForecaster:
        if train_panel.empty:
            raise ValueError(f"{self.name}: empty training panel.")
        self.origin_ = pd.Timestamp(train_panel[DATE_COL].max())
        self.cities_ = sorted(train_panel[CITY_COL].unique().tolist())
        self._fit(train_panel)
        self.is_fitted_ = True
        return self

    def predict(self, horizon: int) -> pd.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError(f"{self.name}: call fit() before predict().")
        out = self._predict(horizon)
        missing = {CITY_COL, "horizon", "y_pred"} - set(out.columns)
        if missing:
            raise RuntimeError(f"{self.name}: _predict must return {missing}")
        out = out.copy()
        out["target_date"] = self.origin_ + pd.to_timedelta(out["horizon"], unit="D")
        out["origin"] = self.origin_
        out["model"] = self.name

        core = ["model", CITY_COL, "origin", "target_date", "horizon", "y_pred"]
        # Preserve anything extra a model chose to emit (e.g. lower/upper bounds
        # from SARIMAX or the quantile GBM) rather than silently dropping it.
        extra = [c for c in out.columns if c not in core]
        return out[core + extra]

    # ------------------------------------------------------------------ #
    @staticmethod
    def _as_wide(train_panel: pd.DataFrame, column: str = TARGET_COL) -> pd.DataFrame:
        wide = train_panel.pivot(index=DATE_COL, columns=CITY_COL, values=column)
        return wide.asfreq("D").interpolate("time").ffill().bfill()

    def _empty_predictions(self, horizon: int, value: float = np.nan) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {CITY_COL: c, "horizon": h, "y_pred": value}
                for c in self.cities_
                for h in range(1, horizon + 1)
            ]
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{type(self).__name__} name={self.name!r} fitted={self.is_fitted_}>"


__all__ = ["BaseForecaster"]
