"""Forecast accuracy metrics, statistical comparison and interval scoring."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# --------------------------------------------------------------------------- #
# Point-forecast metrics
# --------------------------------------------------------------------------- #
def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    d = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.sqrt(np.mean(d**2)))


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_pred) - np.asarray(y_true)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true)
    mask = denom > 1e-8
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / denom[mask])) * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric MAPE — bounded at 200%, safer than MAPE near zero."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denom > 1e-8
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)


def mase(
    y_true: np.ndarray, y_pred: np.ndarray, insample: np.ndarray, seasonality: int = 1
) -> float:
    """Mean Absolute Scaled Error (Hyndman & Koehler, 2006).

    Scale is the in-sample MAE of a seasonal-naive forecast, so MASE < 1 means
    the model beats seasonal naive. Scale-free and defined for zero values,
    which makes it the right headline metric for temperature in degrees Celsius.
    """
    insample = np.asarray(insample, dtype=float)
    if len(insample) <= seasonality:
        return float("nan")
    scale = np.mean(np.abs(insample[seasonality:] - insample[:-seasonality]))
    if scale <= 1e-12:
        return float("nan")
    return float(mae(y_true, y_pred) / scale)


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def evaluate_point(
    y_true, y_pred, insample=None, seasonality: int = 1
) -> dict[str, float]:
    out = {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
        "Bias": bias(y_true, y_pred),
        "R2": r2(y_true, y_pred),
    }
    if insample is not None:
        out["MASE"] = mase(y_true, y_pred, insample, seasonality)
    return {k: round(v, 4) for k, v in out.items()}


# --------------------------------------------------------------------------- #
# Interval metrics
# --------------------------------------------------------------------------- #
def coverage(y_true, lower, upper) -> float:
    y_true = np.asarray(y_true, dtype=float)
    inside = (y_true >= np.asarray(lower)) & (y_true <= np.asarray(upper))
    return float(np.mean(inside))


def mean_interval_width(lower, upper) -> float:
    return float(np.mean(np.asarray(upper) - np.asarray(lower)))


def interval_score(y_true, lower, upper, alpha: float = 0.1) -> float:
    """Winkler / interval score — rewards narrow intervals, penalises misses."""
    y = np.asarray(y_true, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    width = hi - lo
    penalty_lo = (2 / alpha) * np.clip(lo - y, 0, None)
    penalty_hi = (2 / alpha) * np.clip(y - hi, 0, None)
    return float(np.mean(width + penalty_lo + penalty_hi))


def evaluate_intervals(y_true, lower, upper, alpha: float = 0.1) -> dict[str, float]:
    return {
        "Coverage": round(coverage(y_true, lower, upper), 4),
        "NominalCoverage": round(1 - alpha, 4),
        "MeanWidth": round(mean_interval_width(lower, upper), 4),
        "IntervalScore": round(interval_score(y_true, lower, upper, alpha), 4),
    }


# --------------------------------------------------------------------------- #
# Model comparison
# --------------------------------------------------------------------------- #
def diebold_mariano(
    y_true, pred_a, pred_b, h: int = 1, loss: str = "mae"
) -> dict[str, float]:
    """Diebold-Mariano test of equal predictive accuracy.

    Uses the Harvey-Leybourne-Newbold small-sample correction and a
    Newey-West long-run variance to account for the ``h``-step overlap.

    H0: the two forecasts have equal expected loss.
    A negative statistic with a small p-value favours model A.
    """
    y = np.asarray(y_true, dtype=float)
    a = np.asarray(pred_a, dtype=float)
    b = np.asarray(pred_b, dtype=float)
    if loss == "mae":
        d = np.abs(y - a) - np.abs(y - b)
    elif loss == "mse":
        d = (y - a) ** 2 - (y - b) ** 2
    else:
        raise ValueError("loss must be 'mae' or 'mse'")

    n = len(d)
    if n < 10:
        return {"statistic": float("nan"), "p_value": float("nan"), "n": n}

    d_bar = d.mean()
    gamma0 = np.mean((d - d_bar) ** 2)
    lrv = gamma0
    for lag in range(1, h):
        cov = np.mean((d[lag:] - d_bar) * (d[:-lag] - d_bar))
        lrv += 2 * (1 - lag / h) * cov
    lrv = max(lrv, 1e-12)

    dm = d_bar / np.sqrt(lrv / n)
    correction = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_star = dm * correction
    p = 2 * (1 - stats.t.cdf(abs(dm_star), df=n - 1))
    return {
        "statistic": round(float(dm_star), 4),
        "p_value": round(float(p), 4),
        "mean_loss_diff": round(float(d_bar), 4),
        "n": int(n),
    }


def metrics_by_horizon(results: pd.DataFrame, model_col: str = "model") -> pd.DataFrame:
    """MAE / RMSE broken out by forecast lead time.

    ``results`` must contain ``[model, horizon, y_true, y_pred]``.
    """
    rows = []
    for (model, h), grp in results.groupby([model_col, "horizon"]):
        rows.append(
            {
                model_col: model,
                "horizon": h,
                "MAE": mae(grp["y_true"], grp["y_pred"]),
                "RMSE": rmse(grp["y_true"], grp["y_pred"]),
                "n": len(grp),
            }
        )
    return pd.DataFrame(rows).sort_values([model_col, "horizon"]).reset_index(drop=True)


def leaderboard(
    results: pd.DataFrame,
    insample: np.ndarray | None = None,
    seasonality: int = 1,
    model_col: str = "model",
) -> pd.DataFrame:
    """Aggregate metrics for every model, ranked by MAE."""
    rows = []
    for model, grp in results.groupby(model_col):
        m = evaluate_point(
            grp["y_true"].to_numpy(),
            grp["y_pred"].to_numpy(),
            insample=insample,
            seasonality=seasonality,
        )
        m[model_col] = model
        m["n"] = len(grp)
        rows.append(m)
    out = pd.DataFrame(rows).set_index(model_col).sort_values("MAE")
    cols = ["MAE", "RMSE", "MASE", "sMAPE", "MAPE", "Bias", "R2", "n"]
    return out[[c for c in cols if c in out.columns]]


__all__ = [
    "bias",
    "coverage",
    "diebold_mariano",
    "evaluate_intervals",
    "evaluate_point",
    "interval_score",
    "leaderboard",
    "mae",
    "mape",
    "mase",
    "mean_interval_width",
    "metrics_by_horizon",
    "r2",
    "rmse",
    "smape",
]
