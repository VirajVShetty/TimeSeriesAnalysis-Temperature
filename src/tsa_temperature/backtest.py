"""Rolling-origin (walk-forward) backtesting harness.

Why not a single train/test split
---------------------------------
The original project split the series once at 70/80%. With two years of daily
data that gives a single, noisy estimate of accuracy that depends entirely on
which fortnight happened to land in the test set. Rolling-origin evaluation
refits the model at several successive cutoffs and averages the out-of-sample
error, which is the standard protocol for time-series model selection.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import CITY_COL, DATE_COL, FORECAST, TARGET_COL, ForecastConfig
from .conformal import SplitConformal
from .data import rolling_origins
from .models.base import BaseForecaster

logger = logging.getLogger(__name__)

#: A factory takes the horizon and returns a *fresh* unfitted forecaster.
ForecasterFactory = Callable[[int], BaseForecaster]


@dataclass
class BacktestResult:
    """Container for the outcome of a walk-forward experiment."""

    predictions: pd.DataFrame
    """Long frame: model, city, origin, target_date, horizon, y_pred, y_true."""

    timings: pd.DataFrame
    origins: list[pd.Timestamp] = field(default_factory=list)
    config: ForecastConfig = FORECAST

    @property
    def models(self) -> list[str]:
        return sorted(self.predictions["model"].unique().tolist())

    def for_model(self, model: str) -> pd.DataFrame:
        return self.predictions.loc[self.predictions["model"] == model]

    def pivot_predictions(self) -> pd.DataFrame:
        """Wide frame with one column per model — convenient for DM tests."""
        keys = [CITY_COL, "origin", "target_date", "horizon"]
        wide = self.predictions.pivot_table(
            index=keys, columns="model", values="y_pred"
        )
        truth = self.predictions.drop_duplicates(keys).set_index(keys)["y_true"]
        return wide.join(truth).reset_index()


def walk_forward(
    panel: pd.DataFrame,
    factories: dict[str, ForecasterFactory],
    cfg: ForecastConfig = FORECAST,
    origins: Sequence[pd.Timestamp] | None = None,
    verbose: bool = True,
) -> BacktestResult:
    """Run every model over every rolling origin and collect the forecasts.

    Parameters
    ----------
    panel:
        Full long-format climate panel.
    factories:
        ``{display_name: lambda horizon -> BaseForecaster}``. A factory is used
        rather than an instance so each origin gets a genuinely refit model.
    """
    dates = pd.DatetimeIndex(sorted(panel[DATE_COL].unique()))
    if origins is None:
        origins = rolling_origins(
            dates,
            horizon=cfg.horizon,
            n_origins=cfg.n_backtest_origins,
            stride=cfg.origin_stride,
            min_train_days=cfg.min_train_days,
        )

    truth = panel[[CITY_COL, DATE_COL, TARGET_COL]].rename(
        columns={DATE_COL: "target_date", TARGET_COL: "y_true"}
    )

    all_preds, timings = [], []
    for origin in origins:
        train = panel.loc[panel[DATE_COL] <= origin]
        for label, factory in factories.items():
            t0 = time.perf_counter()
            try:
                model = factory(cfg.horizon)
                model.fit(train)
                preds = model.predict(cfg.horizon)
                preds["model"] = label
                all_preds.append(preds)
                status = "ok"
            except Exception as exc:  # keep the sweep alive on one bad model
                logger.warning("%s failed at origin %s: %s", label, origin.date(), exc)
                logger.debug("traceback for %s", label, exc_info=True)
                status = f"error: {exc}"
            elapsed = time.perf_counter() - t0
            timings.append(
                {
                    "model": label,
                    "origin": origin,
                    "seconds": round(elapsed, 3),
                    "status": status,
                }
            )
            if verbose:
                print(f"  [{origin.date()}] {label:<32s} {elapsed:6.2f}s  {status}")

    predictions = pd.concat(all_preds, ignore_index=True)
    predictions = predictions.merge(truth, on=[CITY_COL, "target_date"], how="left")
    predictions = predictions.dropna(subset=["y_true"]).reset_index(drop=True)

    return BacktestResult(
        predictions=predictions,
        timings=pd.DataFrame(timings),
        origins=list(origins),
        config=cfg,
    )


def calibrate_conformal(
    panel: pd.DataFrame,
    factory: ForecasterFactory,
    cfg: ForecastConfig = FORECAST,
    n_calibration_origins: int = 6,
) -> SplitConformal:
    """Fit a conformal calibrator on origins that precede the test window.

    Residuals come from genuine out-of-sample forecasts made *before* the
    evaluation period, so the resulting intervals carry no look-ahead bias.
    """
    dates = pd.DatetimeIndex(sorted(panel[DATE_COL].unique()))
    first_test_origin_idx = len(dates) - cfg.horizon - 1 - (cfg.n_backtest_origins - 1) * cfg.origin_stride
    cal_idxs = [
        first_test_origin_idx - (k + 1) * cfg.horizon for k in range(n_calibration_origins)
    ]
    cal_idxs = [i for i in cal_idxs if i >= cfg.min_train_days - cfg.calibration_days]
    if not cal_idxs:
        raise ValueError("Not enough history to calibrate conformal intervals.")

    truth = panel[[CITY_COL, DATE_COL, TARGET_COL]].rename(
        columns={DATE_COL: "target_date", TARGET_COL: "y_true"}
    )
    residual_frames = []
    for i in sorted(cal_idxs):
        origin = dates[i]
        train = panel.loc[panel[DATE_COL] <= origin]
        model = factory(cfg.horizon)
        model.fit(train)
        preds = model.predict(cfg.horizon).merge(
            truth, on=[CITY_COL, "target_date"], how="left"
        )
        preds = preds.dropna(subset=["y_true"])
        preds["residual"] = preds["y_true"] - preds["y_pred"]
        residual_frames.append(preds)

    residuals = pd.concat(residual_frames, ignore_index=True)
    return SplitConformal(coverage=cfg.coverage).fit(
        residuals["residual"], residuals["horizon"]
    )


def seasonal_naive_insample(panel: pd.DataFrame, seasonality: int = 1) -> np.ndarray:
    """Concatenated per-city in-sample series used as the MASE scale."""
    pieces: list[np.ndarray] = []
    for _, grp in panel.groupby(CITY_COL):
        pieces.append(grp.sort_values(DATE_COL)[TARGET_COL].to_numpy(dtype=float))
    return np.concatenate(pieces)


__all__ = [
    "BacktestResult",
    "ForecasterFactory",
    "calibrate_conformal",
    "seasonal_naive_insample",
    "walk_forward",
]
