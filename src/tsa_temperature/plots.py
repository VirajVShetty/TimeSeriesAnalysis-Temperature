"""Reusable result visualisations."""
from __future__ import annotations

import sys

import matplotlib

# Only force the headless Agg backend when running outside an interactive
# session. Calling ``matplotlib.use("Agg")`` unconditionally at import time
# overrides a notebook's inline backend, which suppresses figure display.
if not (
    "IPython" in sys.modules
    and getattr(sys.modules["IPython"], "get_ipython", lambda: None)() is not None
):
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


# --------------------------------------------------------------------------- #
# Descriptive / EDA plots (see tsa_temperature.analysis)
# --------------------------------------------------------------------------- #
def plot_distribution_grid(
    panel: pd.DataFrame,
    column: str = "temp_avg",
    city_col: str = "city",
    bins: int = 30,
    reference: bool = True,
    savepath=None,
):
    """Histogram per city with normal and uniform reference curves overlaid.

    The reference curves make distribution *shape* legible at a glance: a
    histogram that hugs the flat uniform line rather than the bell curve is a
    strong hint that the values were drawn, not measured.
    """
    cities = sorted(panel[city_col].unique())
    ncols = 5
    nrows = int(np.ceil(len(cities) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.5 * nrows), squeeze=False)

    for i, city in enumerate(cities):
        ax = axes[i // ncols][i % ncols]
        vals = panel.loc[panel[city_col] == city, column].dropna().to_numpy()
        ax.hist(vals, bins=bins, density=True, color=PALETTE[0], alpha=0.75)

        if reference and len(vals) > 2:
            lo, hi = vals.min(), vals.max()
            xs = np.linspace(lo, hi, 200)
            mu, sd = vals.mean(), vals.std()
            if sd > 0:
                ax.plot(
                    xs,
                    np.exp(-0.5 * ((xs - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi)),
                    color=PALETTE[1], lw=1.4, label="normal",
                )
            if hi > lo:
                ax.plot(xs, np.full_like(xs, 1 / (hi - lo)),
                        color=PALETTE[3], lw=1.4, ls="--", label="uniform")
        ax.set_title(city, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2)

    for j in range(len(cities), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9)
    fig.suptitle(f"Distribution of {column} by city", fontsize=12)
    fig.tight_layout(rect=(0, 0.05 if handles else 0, 1, 0.97))
    return _save(fig, savepath)


def plot_monthly_boxplots(
    panel: pd.DataFrame,
    column: str = "temp_avg",
    cities: list[str] | None = None,
    city_col: str = "city",
    date_col: str = "date",
    savepath=None,
):
    """Month-of-year box plots for a handful of cities, on a shared y-axis.

    The shared axis is the point: real Indian cities produce visibly different
    seasonal envelopes, and a flat band across all twelve months means no
    annual cycle.
    """
    cities = cities or sorted(panel[city_col].unique())[:4]
    fig, axes = plt.subplots(1, len(cities), figsize=(3.3 * len(cities), 4), sharey=True)
    axes = np.atleast_1d(axes)
    months = pd.DatetimeIndex(panel[date_col]).month

    for ax, city in zip(axes, cities):
        mask = (panel[city_col] == city).to_numpy()
        data = [panel.loc[mask & (months == m), column].dropna().to_numpy() for m in range(1, 13)]
        bp = ax.boxplot(data, patch_artist=True, widths=0.65, showfliers=False)
        for patch in bp["boxes"]:
            patch.set_facecolor("#cfe3d4")
            patch.set_edgecolor(PALETTE[2])
        for med in bp["medians"]:
            med.set_color(PALETTE[1])
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"],
                           fontsize=8)
        ax.set_title(city, fontsize=10)
        ax.grid(alpha=0.22, axis="y")
    axes[0].set_ylabel(f"{column} (°C)")
    fig.suptitle("Monthly distribution by city (shared scale)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, savepath)


def plot_city_month_heatmap(matrix: pd.DataFrame, title: str = "", savepath=None):
    """Heatmap of a ``city x month`` matrix, centred on the grand mean."""
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(matrix) + 2.2))
    vals = matrix.to_numpy(dtype=float)
    centre = np.nanmean(vals)
    span = max(np.nanmax(np.abs(vals - centre)), 1e-6)
    im = ax.imshow(vals, cmap="RdYlBu_r", aspect="auto",
                   vmin=centre - span, vmax=centre + span)

    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, fontsize=9)
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels(matrix.index, fontsize=9)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{vals[i, j]:.1f}", ha="center", va="center", fontsize=7)
    ax.set_title(title or "Mean by city and month (°C)")
    fig.colorbar(im, ax=ax, shrink=0.85, label="°C")
    fig.tight_layout()
    return _save(fig, savepath)


def plot_ecdf(
    panel: pd.DataFrame,
    column: str = "temp_avg",
    city_col: str = "city",
    cities: list[str] | None = None,
    savepath=None,
):
    """Empirical CDFs overlaid. Straight diagonals indicate uniform data."""
    cities = cities or sorted(panel[city_col].unique())
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, city in enumerate(cities):
        v = np.sort(panel.loc[panel[city_col] == city, column].dropna().to_numpy())
        ax.plot(v, np.arange(1, len(v) + 1) / len(v), lw=1.3,
                color=PALETTE[i % len(PALETTE)], alpha=0.85, label=city)
    ax.set_xlabel(f"{column} (°C)")
    ax.set_ylabel("cumulative probability")
    ax.set_title(f"Empirical CDF of {column} by city")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    return _save(fig, savepath)


def plot_ranking(
    table: pd.DataFrame, column: str, title: str = "", xlabel: str = "", savepath=None
):
    """Horizontal bar chart of a ranked statistic, with optional reference bars."""
    d = table.sort_values(column)
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(d) + 2))
    ax.barh(d.index.astype(str), d[column], color=PALETTE[0], alpha=0.85)
    for i, v in enumerate(d[column]):
        ax.text(v, i, f"  {v:.2f}", va="center", fontsize=8)
    ax.set_xlabel(xlabel or column)
    ax.set_title(title or f"Cities ranked by {column}")
    ax.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    return _save(fig, savepath)


def plot_observed_vs_reference(plaus: pd.DataFrame, savepath=None):
    """Observed seasonal amplitude against real-world reference amplitude."""
    d = plaus.sort_values("reference_amplitude")
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(d) + 2))
    ax.barh(y - 0.2, d["reference_amplitude"], height=0.4,
            color=PALETTE[3], alpha=0.85, label="real-world reference")
    ax.barh(y + 0.2, d["observed_amplitude"], height=0.4,
            color=PALETTE[1], alpha=0.9, label="observed in dataset")
    ax.set_yticks(y)
    ax.set_yticklabels(d.index, fontsize=9)
    ax.set_xlabel("Seasonal amplitude — hottest minus coldest monthly mean (°C)")
    ax.set_title("Observed seasonal amplitude vs Indian climatology")
    ax.grid(alpha=0.25, axis="x")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return _save(fig, savepath)


def plot_scatter_relationship(
    panel: pd.DataFrame, x: str, y: str, sample: int = 3000, savepath=None
):
    """Scatter of two channels with a least-squares line and the correlation."""
    d = panel[[x, y]].dropna()
    if len(d) > sample:
        d = d.sample(sample, random_state=0)
    xv, yv = d[x].to_numpy(float), d[y].to_numpy(float)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(xv, yv, s=7, alpha=0.32, color=PALETTE[0], edgecolors="none")
    if len(d) > 2 and np.std(xv) > 0:
        slope, intercept = np.polyfit(xv, yv, 1)
        xs = np.linspace(xv.min(), xv.max(), 100)
        ax.plot(xs, slope * xs + intercept, color=PALETTE[1], lw=2)
        r = float(np.corrcoef(xv, yv)[0, 1])
        ax.set_title(f"{y} vs {x}   (r = {r:+.3f})")
    else:
        ax.set_title(f"{y} vs {x}")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return _save(fig, savepath)


def plot_calendar_heatmap(
    series: pd.Series, title: str = "", cmap: str = "RdYlBu_r", savepath=None
):
    """Year x day-of-year heatmap — makes seasonal banding obvious or absent."""
    s = series.dropna()
    idx = pd.DatetimeIndex(s.index)
    years = sorted(idx.year.unique())
    grid = np.full((len(years), 366), np.nan)
    for yi, yr in enumerate(years):
        m = idx.year == yr
        grid[yi, idx[m].dayofyear - 1] = s[m].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(13, 1.3 * len(years) + 1.8))
    im = ax.imshow(grid, aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_yticks(range(len(years)))
    ax.set_yticklabels(years)
    starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    ax.set_xticks([s_ - 1 for s_ in starts])
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_title(title or "Daily values by day of year")
    fig.colorbar(im, ax=ax, shrink=0.85, label="°C")
    fig.tight_layout()
    return _save(fig, savepath)


def plot_stacked_shares(
    table: pd.DataFrame, title: str = "", xlabel: str = "share of days (%)", savepath=None
):
    """Horizontal stacked bars for categorical shares (e.g. AQI categories)."""
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(table) + 2.2))
    left = np.zeros(len(table))
    cmap = plt.get_cmap("RdYlGn_r")
    colors = cmap(np.linspace(0.1, 0.9, table.shape[1]))
    for j, col in enumerate(table.columns):
        vals = table[col].to_numpy(dtype=float)
        ax.barh(table.index.astype(str), vals, left=left, label=str(col), color=colors[j])
        left += vals
    ax.set_xlabel(xlabel)
    ax.set_title(title or "Category shares by city")
    ax.legend(fontsize=8, ncol=min(table.shape[1], 6), loc="lower center",
              bbox_to_anchor=(0.5, -0.28))
    ax.grid(alpha=0.2, axis="x")
    fig.tight_layout()
    return _save(fig, savepath)


__all__ = [
    "plot_anomalies",
    "plot_calendar_heatmap",
    "plot_city_month_heatmap",
    "plot_coverage_by_horizon",
    "plot_distribution_grid",
    "plot_ecdf",
    "plot_error_by_horizon",
    "plot_feature_importance",
    "plot_forecast_example",
    "plot_leaderboard",
    "plot_monthly_boxplots",
    "plot_observed_vs_reference",
    "plot_ranking",
    "plot_residual_diagnostics",
    "plot_scatter_relationship",
    "plot_stacked_shares",
]
