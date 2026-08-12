"""Reusable result visualisations."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PALETTE = [
    "#22333b",
    "#c1440e",
    "#0b6e4f",
    "#4f8fc0",
    "#8f4f9e",
    "#b58900",
    "#7a7a7a",
    "#d1495b",
]


def _save(fig, savepath):
    if savepath:
        fig.savefig(savepath, dpi=140, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig


def plot_leaderboard(board: pd.DataFrame, metric: str = "MAE", savepath=None):
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(board) + 2))
    vals = board[metric].to_numpy()
    order = np.argsort(vals)
    names = board.index.to_numpy()[order]
    ax.barh(range(len(vals)), vals[order], color=PALETTE[0], alpha=0.85)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    for i, v in enumerate(vals[order]):
        ax.text(v, i, f"  {v:.3f}", va="center", fontsize=8)
    ax.set_xlabel(f"{metric} (°C)" if metric in {"MAE", "RMSE"} else metric)
    ax.set_title(f"Model comparison — {metric} (lower is better)")
    ax.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    return _save(fig, savepath)


def plot_error_by_horizon(by_h: pd.DataFrame, metric: str = "MAE", savepath=None):
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (model, grp) in enumerate(by_h.groupby("model")):
        ax.plot(
            grp["horizon"],
            grp[metric],
            marker="o",
            ms=4,
            lw=1.6,
            color=PALETTE[i % len(PALETTE)],
            label=model,
        )
    ax.set_xlabel("Forecast lead time (days)")
    ax.set_ylabel(f"{metric} (°C)")
    ax.set_title(f"Error growth with lead time — {metric}")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, savepath)


def plot_forecast_example(
    history: pd.Series,
    actual: pd.Series,
    forecasts: dict[str, pd.Series],
    lower: pd.Series | None = None,
    upper: pd.Series | None = None,
    title: str = "",
    context_days: int = 90,
    savepath=None,
):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    hist = history.iloc[-context_days:]
    ax.plot(hist.index, hist.to_numpy(), color="#22333b", lw=1.3, label="History")
    ax.plot(actual.index, actual.to_numpy(), color="black", lw=2.2, marker="o", ms=4, label="Actual")

    for i, (name, fc) in enumerate(forecasts.items()):
        ax.plot(
            fc.index,
            fc.to_numpy(),
            lw=1.8,
            ls="--",
            marker="s",
            ms=3.5,
            color=PALETTE[(i + 1) % len(PALETTE)],
            label=name,
        )
    if lower is not None and upper is not None:
        ax.fill_between(
            lower.index,
            lower.to_numpy(),
            upper.to_numpy(),
            color=PALETTE[1],
            alpha=0.16,
            label="90% conformal interval",
        )
    ax.axvline(hist.index[-1], color="grey", ls=":", lw=1.2)
    ax.set_title(title or "Forecast vs actual")
    ax.set_xlabel("Date")
    ax.set_ylabel("Temperature (°C)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.autofmt_xdate()
    fig.tight_layout()
    return _save(fig, savepath)


def plot_feature_importance(imp: pd.DataFrame, top_n: int = 20, savepath=None):
    d = imp.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.36 * len(d) + 1.5))
    ax.barh(d["feature"], d["gain_pct"], color=PALETTE[2], alpha=0.85)
    ax.set_xlabel("Gain (% of total)")
    ax.set_title(f"LightGBM feature importance (top {top_n})")
    ax.grid(alpha=0.25, axis="x")
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    return _save(fig, savepath)


def plot_residual_diagnostics(residuals: np.ndarray, savepath=None):
    r = np.asarray(residuals, dtype=float)
    r = r[np.isfinite(r)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    axes[0].hist(r, bins=45, color=PALETTE[0], alpha=0.8)
    axes[0].axvline(0, color=PALETTE[1], lw=1.4)
    axes[0].set_title(f"Residual distribution (mean={r.mean():.3f})")
    axes[0].set_xlabel("Residual (°C)")

    qs = np.linspace(0.001, 0.999, min(len(r), 500))
    from scipy import stats as _st

    theo = _st.norm.ppf(qs, loc=r.mean(), scale=r.std() or 1.0)
    axes[1].plot(theo, np.quantile(r, qs), ".", ms=3, color=PALETTE[0])
    lims = [min(theo.min(), r.min()), max(theo.max(), r.max())]
    axes[1].plot(lims, lims, color=PALETTE[1], lw=1.2)
    axes[1].set_title("Q-Q plot vs normal")
    axes[1].set_xlabel("Theoretical")
    axes[1].set_ylabel("Empirical")

    nlags = min(40, len(r) // 3)
    from statsmodels.tsa.stattools import acf as _acf

    a = _acf(r, nlags=nlags, fft=True)
    ci = 1.96 / np.sqrt(len(r))
    axes[2].vlines(range(len(a)), 0, a, color=PALETTE[0])
    axes[2].axhspan(-ci, ci, color=PALETTE[3], alpha=0.2)
    axes[2].axhline(0, color="black", lw=0.7)
    axes[2].set_title("Residual ACF")
    axes[2].set_xlabel("Lag")

    for ax in axes:
        ax.grid(alpha=0.22)
    fig.tight_layout()
    return _save(fig, savepath)


def plot_anomalies(
    result_frame: pd.DataFrame,
    value_col: str = "value",
    flag_col: str = "flag_ensemble",
    title: str = "",
    savepath=None,
):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    idx = result_frame.index
    ax.plot(idx, result_frame[value_col], color="#22333b", lw=1.0, label="Temperature")

    if {"upper", "lower"} <= set(result_frame.columns):
        ax.fill_between(
            idx,
            result_frame["lower"],
            result_frame["upper"],
            color="#4f8fc0",
            alpha=0.15,
            label="Brutlag band",
        )
    flagged = result_frame.loc[result_frame[flag_col].fillna(False)]
    ax.plot(
        flagged.index,
        flagged[value_col],
        "o",
        ms=6,
        color="#d1495b",
        label=f"Anomaly ({len(flagged)})",
    )
    ax.set_title(title or "Detected temperature anomalies")
    ax.set_xlabel("Date")
    ax.set_ylabel("Temperature (°C)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    return _save(fig, savepath)


def plot_coverage_by_horizon(cov_table: pd.DataFrame, nominal: float = 0.9, savepath=None):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(
        cov_table["horizon"], cov_table["Coverage"], marker="o", color=PALETTE[0], lw=1.8
    )
    ax.axhline(nominal, color=PALETTE[1], ls="--", lw=1.4, label=f"Nominal {nominal:.0%}")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Forecast lead time (days)")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Conformal interval coverage by horizon")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return _save(fig, savepath)


__all__ = [
    "plot_anomalies",
    "plot_coverage_by_horizon",
    "plot_error_by_horizon",
    "plot_feature_importance",
    "plot_forecast_example",
    "plot_leaderboard",
    "plot_residual_diagnostics",
]
