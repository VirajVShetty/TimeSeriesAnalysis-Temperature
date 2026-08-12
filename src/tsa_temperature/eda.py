"""Exploratory analysis: decomposition, stationarity testing and diagnostics.

Replaces the original ``seasonal_decompose`` workflow with STL (Seasonal-Trend
decomposition using LOESS), which handles a long seasonal period and outliers
far better than classical moving-average decomposition.

A caveat that matters here
--------------------------
STL needs several full cycles to separate season from noise. With daily data
and ``period=365`` this dataset supplies only **two** cycles — two observations
per cycle-subseries — so STL's seasonal smoother interpolates them exactly and
drives the residual to zero. Empirically, on a series built as
``27 + 9*sin(annual) + N(0, 0.8)``, STL returns a residual standard deviation
of **0.001** instead of 0.8: the entire noise process is misattributed to
seasonality.

:func:`decompose` therefore checks the cycle count and falls back to a robust
**harmonic regression** (Huber-fit trend + Fourier seasonality) whenever fewer
than :data:`MIN_CYCLES_FOR_STL` cycles are available. Harmonic regression
spends ``2K + 2`` parameters instead of one per cycle-subseries, so it cannot
overfit the noise.
"""
from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass

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
from sklearn.linear_model import HuberRegressor
from statsmodels.tsa.seasonal import MSTL, STL
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

#: STL needs at least this many full seasonal cycles to be identifiable.
MIN_CYCLES_FOR_STL = 3


# --------------------------------------------------------------------------- #
# Decomposition
# --------------------------------------------------------------------------- #
@dataclass
class DecompositionResult:
    observed: pd.Series
    trend: pd.Series
    seasonal: pd.Series
    resid: pd.Series
    method: str

    @property
    def seasonal_strength(self) -> float:
        """F_S from Wang, Smith & Hyndman (2006): 1 - Var(R) / Var(S + R)."""
        return _strength(self.resid, self.seasonal + self.resid)

    @property
    def trend_strength(self) -> float:
        """F_T: 1 - Var(R) / Var(T + R)."""
        return _strength(self.resid, self.trend + self.resid)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "observed": self.observed,
                "trend": self.trend,
                "seasonal": self.seasonal,
                "resid": self.resid,
            }
        )

    def plot(self, title: str = "", savepath=None):
        frame = self.to_frame()
        fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
        # NB: no ``strict=`` kwarg — it is Python 3.10+ only and raises
        # "TypeError: zip() takes no keyword arguments" on 3.9. All three
        # iterables here are fixed at length 4 anyway.
        for ax, col, color in zip(
            axes,
            ["observed", "trend", "seasonal", "resid"],
            ["#22333b", "#c1440e", "#0b6e4f", "#7a7a7a"],
        ):
            if col == "resid":
                ax.axhline(0, color="black", lw=0.6, alpha=0.5)
                ax.plot(frame.index, frame[col], ".", ms=2.5, color=color)
            else:
                ax.plot(frame.index, frame[col], lw=1.1, color=color)
            ax.set_ylabel(col, fontsize=9)
            ax.grid(alpha=0.25)
        axes[0].set_title(
            f"{self.method} decomposition {title}\n"
            f"trend strength={self.trend_strength:.3f}  "
            f"seasonal strength={self.seasonal_strength:.3f}",
            fontsize=11,
        )
        fig.autofmt_xdate()
        fig.tight_layout()
        if savepath:
            fig.savefig(savepath, dpi=140, bbox_inches="tight")
            plt.close(fig)
            return None
        return fig


def _strength(resid: pd.Series, combined: pd.Series) -> float:
    var_r = float(np.nanvar(resid))
    var_c = float(np.nanvar(combined))
    if var_c <= 0:
        return 0.0
    return float(max(0.0, 1.0 - var_r / var_c))


def harmonic_decompose(
    series: pd.Series,
    period: int = 365,
    fourier_order: int = 4,
    trend_degree: int = 2,
    robust: bool = True,
) -> DecompositionResult:
    """Trend + Fourier-seasonality decomposition via robust regression.

    Fits ``y ~ poly(t, trend_degree) + sum_k [sin, cos](2*pi*k*t/period)`` with
    a Huber loss so that heatwaves and cold snaps do not drag the fit towards
    themselves. Because the seasonal shape costs only ``2 * fourier_order``
    parameters, it stays identifiable when STL would overfit.
    """
    s = series.astype(float).dropna()
    n = len(s)
    t = np.arange(n, dtype=float)

    trend_cols = [((t - t.mean()) / max(t.std(), 1e-9)) ** d for d in range(1, trend_degree + 1)]
    seas_cols = []
    for k in range(1, fourier_order + 1):
        seas_cols.append(np.sin(2 * np.pi * k * t / period))
        seas_cols.append(np.cos(2 * np.pi * k * t / period))

    X = np.column_stack(trend_cols + seas_cols)
    y = s.to_numpy()

    if robust:
        try:
            model = HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=500).fit(X, y)
            coef, intercept = model.coef_, model.intercept_
        except Exception:
            coef, *_ = np.linalg.lstsq(np.column_stack([np.ones(n), X]), y, rcond=None)
            intercept, coef = coef[0], coef[1:]
    else:
        beta, *_ = np.linalg.lstsq(np.column_stack([np.ones(n), X]), y, rcond=None)
        intercept, coef = beta[0], beta[1:]

    n_trend = len(trend_cols)
    trend = intercept + X[:, :n_trend] @ coef[:n_trend]
    seasonal = X[:, n_trend:] @ coef[n_trend:]
    resid = y - trend - seasonal

    return DecompositionResult(
        observed=s,
        trend=pd.Series(trend, index=s.index),
        seasonal=pd.Series(seasonal, index=s.index),
        resid=pd.Series(resid, index=s.index),
        method=f"Harmonic (K={fourier_order})",
    )


def decompose(
    series: pd.Series,
    period: int = 365,
    method: str = "auto",
    robust: bool = True,
    seasonal: int = 91,
    fourier_order: int = 4,
) -> DecompositionResult:
    """Decompose a series, choosing a method that the data can actually support.

    ``method='auto'`` uses STL when at least :data:`MIN_CYCLES_FOR_STL` full
    cycles are present and robust harmonic regression otherwise. Pass
    ``'stl'`` or ``'harmonic'`` to force a choice.
    """
    s = series.astype(float).dropna()
    n_cycles = len(s) / period

    if method == "auto":
        method = "stl" if n_cycles >= MIN_CYCLES_FOR_STL else "harmonic"
        if method == "harmonic":
            warnings.warn(
                f"Only {n_cycles:.1f} seasonal cycles available (period={period}); "
                f"STL would overfit, using robust harmonic regression instead.",
                stacklevel=2,
            )

    if method == "harmonic":
        return harmonic_decompose(
            s, period=period, fourier_order=fourier_order, robust=robust
        )
    if method != "stl":
        raise ValueError(f"Unknown method {method!r}; use 'auto', 'stl' or 'harmonic'.")

    if seasonal % 2 == 0:
        seasonal += 1
    res = STL(s, period=period, seasonal=seasonal, robust=robust).fit()
    return DecompositionResult(
        observed=s,
        trend=pd.Series(res.trend, index=s.index),
        seasonal=pd.Series(res.seasonal, index=s.index),
        resid=pd.Series(res.resid, index=s.index),
        method="STL",
    )


def stl_decompose(
    series: pd.Series, period: int = 365, robust: bool = True, seasonal: int = 91
) -> DecompositionResult:
    """Backwards-compatible wrapper around :func:`decompose` in ``auto`` mode."""
    return decompose(
        series, period=period, method="auto", robust=robust, seasonal=seasonal
    )


def mstl_decompose(
    series: pd.Series, periods: tuple[int, ...] = (7, 365)
) -> DecompositionResult:
    """Multi-seasonal STL — separates weekly and annual cycles simultaneously."""
    usable = tuple(p for p in periods if len(series) >= 2 * p)
    if not usable:
        usable = (7,)
    res = MSTL(series, periods=usable).fit()
    seasonal = res.seasonal
    seasonal_total = seasonal.sum(axis=1) if isinstance(seasonal, pd.DataFrame) else seasonal
    return DecompositionResult(
        observed=series,
        trend=pd.Series(res.trend, index=series.index),
        seasonal=pd.Series(seasonal_total, index=series.index),
        resid=pd.Series(res.resid, index=series.index),
        method=f"MSTL{usable}",
    )


# --------------------------------------------------------------------------- #
# Stationarity
# --------------------------------------------------------------------------- #
def stationarity_report(series: pd.Series, alpha: float = 0.05) -> pd.Series:
    """Run ADF and KPSS and reconcile their (opposite) null hypotheses.

    ADF  H0: unit root  (non-stationary)
    KPSS H0: stationary around a deterministic trend
    """
    s = series.dropna()
    adf_stat, adf_p, *_ = adfuller(s, autolag="AIC")
    kpss_stat, kpss_p, *_ = kpss(s, regression="c", nlags="auto")

    adf_stationary = adf_p < alpha
    kpss_stationary = kpss_p > alpha
    if adf_stationary and kpss_stationary:
        verdict = "stationary"
    elif not adf_stationary and not kpss_stationary:
        verdict = "non-stationary (differencing recommended)"
    elif adf_stationary and not kpss_stationary:
        verdict = "trend-stationary (detrend rather than difference)"
    else:
        verdict = "difference-stationary"

    return pd.Series(
        {
            "adf_stat": round(adf_stat, 4),
            "adf_pvalue": round(adf_p, 4),
            "kpss_stat": round(kpss_stat, 4),
            "kpss_pvalue": round(kpss_p, 4),
            "verdict": verdict,
        }
    )


def acf_pacf_table(series: pd.Series, nlags: int = 60) -> pd.DataFrame:
    s = series.dropna()
    nlags = min(nlags, len(s) // 2 - 1)
    a = acf(s, nlags=nlags, fft=True)
    p = pacf(s, nlags=nlags)
    return pd.DataFrame({"lag": range(len(a)), "acf": a, "pacf": p}).set_index("lag")


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_city_overview(wide: pd.DataFrame, savepath=None):
    """Overlay every city's temperature series plus the cross-city mean."""
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for col in wide.columns:
        ax.plot(wide.index, wide[col], lw=0.8, alpha=0.55, label=col)
    ax.plot(wide.index, wide.mean(axis=1), lw=2.2, color="black", label="All-city mean")
    ax.set_title("Daily average temperature by city (2024-2025)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Temperature (°C)")
    ax.grid(alpha=0.25)
    ax.legend(ncol=4, fontsize=8, loc="lower center")
    fig.autofmt_xdate()
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=140, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig


def plot_seasonal_profile(series: pd.Series, savepath=None):
    """Month-of-year distribution — the classic seasonal subseries view."""
    df = pd.DataFrame({"value": series})
    df["month"] = df.index.month
    fig, ax = plt.subplots(figsize=(9, 4.5))
    data = [df.loc[df.month == m, "value"].dropna().values for m in range(1, 13)]
    bp = ax.boxplot(data, patch_artist=True, widths=0.6)
    for patch in bp["boxes"]:
        patch.set_facecolor("#cfe3d4")
        patch.set_edgecolor("#0b6e4f")
    for med in bp["medians"]:
        med.set_color("#c1440e")
    ax.set_xticklabels(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    )
    ax.set_title(f"Monthly temperature distribution — {series.name}")
    ax.set_ylabel("Temperature (°C)")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=140, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig


def plot_acf_pacf(series: pd.Series, nlags: int = 60, savepath=None):
    tbl = acf_pacf_table(series, nlags=nlags)
    n = len(series.dropna())
    ci = 1.96 / np.sqrt(n)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, col, title in zip(axes, ["acf", "pacf"], ["ACF", "PACF"]):
        ax.vlines(tbl.index, 0, tbl[col], color="#22333b", lw=1.4)
        ax.axhline(0, color="black", lw=0.7)
        ax.axhspan(-ci, ci, color="#4f8fc0", alpha=0.18)
        ax.set_title(f"{title} — {series.name}")
        ax.set_xlabel("Lag (days)")
        ax.grid(alpha=0.2)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=140, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig


def plot_correlation_heatmap(wide: pd.DataFrame, savepath=None):
    corr = wide.corr()
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(corr.values, cmap="RdYlBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr.index, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(
                j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7
            )
    ax.set_title("Cross-city temperature correlation")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=140, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig


__all__ = [
    "MIN_CYCLES_FOR_STL",
    "DecompositionResult",
    "acf_pacf_table",
    "decompose",
    "harmonic_decompose",
    "mstl_decompose",
    "plot_acf_pacf",
    "plot_city_overview",
    "plot_correlation_heatmap",
    "plot_seasonal_profile",
    "stationarity_report",
    "stl_decompose",
]
