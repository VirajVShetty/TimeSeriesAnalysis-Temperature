"""Classical statistical forecasters, modernised for daily data.

Why the original Holt-Winters setup was replaced
------------------------------------------------
The previous version of this project fit ``ExponentialSmoothing`` with
``seasonal_periods=12`` on *monthly* data. The current dataset is **daily**,
which would require ``seasonal_periods=365`` — 365 seasonal parameters
estimated from ~700 observations. That is unidentifiable and numerically
unstable.

Two standard remedies are implemented instead:

1. **STL + ETS** — decompose with STL, apply exponential smoothing to the
   seasonally adjusted series, then add the seasonal component back.
2. **SARIMAX + Fourier** — represent the long annual cycle with a handful of
   harmonics supplied as exogenous regressors, leaving ARIMA to model the
   short-memory dynamics. This is Hyndman's recommended approach for
   high-frequency seasonality.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from ..config import CITY_COL, DATE_COL, MODELS, TARGET_COL
from ..eda import decompose
from ..features import fourier_terms
from .base import BaseForecaster

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")


class STLETSForecaster(BaseForecaster):
    """Seasonal decomposition followed by ETS on the seasonally adjusted series.

    The decomposition is delegated to :func:`tsa_temperature.eda.decompose`, so
    it automatically degrades from STL to robust harmonic regression when the
    training window holds fewer than three annual cycles.
    """

    name = "STL + ETS"

    def __init__(self, period: int = 365, seasonal_smoother: int = 91, damped: bool = True):
        super().__init__()
        self.period = period
        self.seasonal_smoother = (
            seasonal_smoother if seasonal_smoother % 2 else seasonal_smoother + 1
        )
        self.damped = damped

    def _fit(self, train_panel: pd.DataFrame) -> None:
        self.fitted_: dict[str, dict] = {}
        for city, grp in train_panel.groupby(CITY_COL):
            s = grp.sort_values(DATE_COL).set_index(DATE_COL)[TARGET_COL].astype(float)
            s = s.asfreq("D").interpolate("time").ffill().bfill()

            period = self.period if len(s) >= 2 * self.period else max(7, len(s) // 3)
            try:
                dec = decompose(
                    s,
                    period=period,
                    method="auto",
                    robust=True,
                    seasonal=self.seasonal_smoother,
                )
                seasonal = dec.seasonal
                adjusted = s - seasonal
            except Exception:
                seasonal = pd.Series(0.0, index=s.index)
                adjusted = s

            try:
                ets = ExponentialSmoothing(
                    adjusted,
                    trend="add",
                    damped_trend=self.damped,
                    seasonal=None,
                    initialization_method="estimated",
                ).fit(optimized=True)
            except Exception:
                ets = None

            self.fitted_[city] = {
                "ets": ets,
                "seasonal": seasonal,
                "last_adjusted": float(adjusted.iloc[-1]),
                "index": s.index,
            }

    def _seasonal_at(self, seasonal: pd.Series, target_date: pd.Timestamp) -> float:
        """Project the STL seasonal component forward one full year."""
        ref = target_date
        while ref > seasonal.index[-1]:
            ref -= pd.Timedelta(days=365)
        if ref in seasonal.index:
            return float(seasonal.loc[ref])
        pos = seasonal.index.get_indexer([ref], method="nearest")[0]
        return float(seasonal.iloc[pos])

    def _predict(self, horizon: int) -> pd.DataFrame:
        rows = []
        for city, state in self.fitted_.items():
            ets = state["ets"]
            if ets is not None:
                adj_fc = np.asarray(ets.forecast(horizon), dtype=float)
            else:
                adj_fc = np.full(horizon, state["last_adjusted"])
            for h in range(1, horizon + 1):
                target_date = self.origin_ + pd.Timedelta(days=h)
                seas = self._seasonal_at(state["seasonal"], target_date)
                rows.append(
                    {CITY_COL: city, "horizon": h, "y_pred": float(adj_fc[h - 1] + seas)}
                )
        return pd.DataFrame(rows)


class SARIMAXFourier(BaseForecaster):
    """SARIMAX with Fourier terms encoding the annual cycle.

    ``order`` governs short-memory dynamics; the ``2 * fourier_order``
    harmonics carry the seasonality, which keeps the parameter count small
    enough for two years of daily data.
    """

    name = "SARIMAX + Fourier"
    provides_intervals = True

    def __init__(
        self,
        order: tuple[int, int, int] = MODELS.sarimax_order,
        fourier_order: int = MODELS.sarimax_fourier_order,
        trend: str = "c",
    ):
        super().__init__()
        self.order = order
        self.fourier_order = fourier_order
        self.trend = trend

    def _fit(self, train_panel: pd.DataFrame) -> None:
        self.fitted_: dict[str, dict] = {}
        for city, grp in train_panel.groupby(CITY_COL):
            s = grp.sort_values(DATE_COL).set_index(DATE_COL)[TARGET_COL].astype(float)
            s = s.asfreq("D").interpolate("time").ffill().bfill()
            exog = fourier_terms(s.index, order=self.fourier_order)
            try:
                res = SARIMAX(
                    s,
                    exog=exog,
                    order=self.order,
                    trend=self.trend,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False, maxiter=200)
            except Exception:
                res = None
            self.fitted_[city] = {"res": res, "last": float(s.iloc[-1])}

    def _predict(self, horizon: int) -> pd.DataFrame:
        future = pd.date_range(
            self.origin_ + pd.Timedelta(days=1), periods=horizon, freq="D"
        )
        exog_future = fourier_terms(future, order=self.fourier_order)
        rows = []
        for city, state in self.fitted_.items():
            res = state["res"]
            if res is None:
                preds = np.full(horizon, state["last"])
                lo = hi = preds
            else:
                fc = res.get_forecast(steps=horizon, exog=exog_future)
                preds = np.asarray(fc.predicted_mean, dtype=float)
                ci = fc.conf_int(alpha=0.1)
                lo = np.asarray(ci.iloc[:, 0], dtype=float)
                hi = np.asarray(ci.iloc[:, 1], dtype=float)
            for h in range(1, horizon + 1):
                rows.append(
                    {
                        CITY_COL: city,
                        "horizon": h,
                        "y_pred": float(preds[h - 1]),
                        "lower": float(lo[h - 1]),
                        "upper": float(hi[h - 1]),
                    }
                )
        return pd.DataFrame(rows)


class SARIMAXExog(SARIMAXFourier):
    """SARIMAX with Fourier terms *and* lagged weather covariates.

    Only lag-1 exogenous values are used and they are held constant across the
    forecast window, mirroring what would actually be available in production.
    """

    name = "SARIMAX + Fourier + exog"

    def __init__(self, exog_cols: tuple[str, ...] = ("humidity", "pressure", "cloud_cover"), **kw):
        super().__init__(**kw)
        self.exog_cols = exog_cols

    def _fit(self, train_panel: pd.DataFrame) -> None:
        self.fitted_ = {}
        for city, grp in train_panel.groupby(CITY_COL):
            g = grp.sort_values(DATE_COL).set_index(DATE_COL).asfreq("D")
            s = g[TARGET_COL].astype(float).interpolate("time").ffill().bfill()
            extra = (
                g[list(self.exog_cols)]
                .astype(float)
                .shift(1)
                .interpolate("time")
                .ffill()
                .bfill()
            )
            exog = pd.concat([fourier_terms(s.index, order=self.fourier_order), extra], axis=1)
            try:
                res = SARIMAX(
                    s,
                    exog=exog,
                    order=self.order,
                    trend=self.trend,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False, maxiter=200)
            except Exception:
                res = None
            self.fitted_[city] = {
                "res": res,
                "last": float(s.iloc[-1]),
                "last_exog": extra.iloc[-1],
            }

    def _predict(self, horizon: int) -> pd.DataFrame:
        future = pd.date_range(
            self.origin_ + pd.Timedelta(days=1), periods=horizon, freq="D"
        )
        base_exog = fourier_terms(future, order=self.fourier_order)
        rows = []
        for city, state in self.fitted_.items():
            res = state["res"]
            if res is None:
                preds = np.full(horizon, state["last"])
                lo = hi = preds
            else:
                held = pd.DataFrame(
                    np.tile(state["last_exog"].to_numpy(), (horizon, 1)),
                    index=future,
                    columns=list(self.exog_cols),
                )
                fc = res.get_forecast(
                    steps=horizon, exog=pd.concat([base_exog, held], axis=1)
                )
                preds = np.asarray(fc.predicted_mean, dtype=float)
                ci = fc.conf_int(alpha=0.1)
                lo = np.asarray(ci.iloc[:, 0], dtype=float)
                hi = np.asarray(ci.iloc[:, 1], dtype=float)
            for h in range(1, horizon + 1):
                rows.append(
                    {
                        CITY_COL: city,
                        "horizon": h,
                        "y_pred": float(preds[h - 1]),
                        "lower": float(lo[h - 1]),
                        "upper": float(hi[h - 1]),
                    }
                )
        return pd.DataFrame(rows)


__all__ = ["SARIMAXExog", "SARIMAXFourier", "STLETSForecaster"]
