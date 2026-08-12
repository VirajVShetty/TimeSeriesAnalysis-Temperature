"""Forecastability audit — is there any signal to model in the first place?

This module exists because of a hard lesson from this project. Every model in
the zoo can be fit, scored and ranked on any dataset; a leaderboard will always
be produced. What a leaderboard *cannot* tell you is whether the winner is
capturing real structure or just estimating a mean.

Before comparing models, run these tests. They ask:

1. **Is there memory?**            Ljung-Box, significant ACF lags
2. **Is there seasonality?**       year-over-year reproducibility, monthly spread
3. **Do related series co-move?**  cross-series correlation
4. **Do the covariates matter?**   correlation and mutual information with the target
5. **How much is predictable?**    a variance-decomposition ceiling on R^2

A series that fails all five is white noise. On white noise the optimal
forecast is the unconditional mean, every model converges to it, and reported
differences between models are sampling noise.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf

from .config import CITY_COL, DATE_COL, EXOG_COLS, TARGET_COL


@dataclass
class ForecastabilityReport:
    """Structured verdict on whether a panel contains forecastable signal."""

    autocorrelation: pd.DataFrame
    ljung_box: pd.DataFrame
    seasonality: pd.DataFrame
    cross_series: pd.DataFrame
    exogenous: pd.DataFrame
    ceiling: pd.DataFrame
    verdict: str = ""
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "evidence": self.evidence,
            "autocorrelation": self.autocorrelation.to_dict(),
            "ljung_box": self.ljung_box.to_dict(),
            "seasonality": self.seasonality.to_dict(),
            "cross_series": self.cross_series.to_dict(),
            "exogenous": self.exogenous.to_dict(),
            "ceiling": self.ceiling.to_dict(),
        }

    def print_report(self) -> None:
        print("=" * 74)
        print("FORECASTABILITY AUDIT")
        print("=" * 74)
        print("\n1. Short-lag autocorrelation (|rho| above the 95% noise band?)")
        print(self.autocorrelation.to_string())
        print("\n2. Ljung-Box test for any serial dependence (H0: white noise)")
        print(self.ljung_box.to_string())
        print("\n3. Seasonality: does the annual pattern repeat across years?")
        print(self.seasonality.to_string())
        print("\n4. Cross-series co-movement")
        print(self.cross_series.to_string())
        print("\n5. Exogenous covariate informativeness")
        print(self.exogenous.to_string())
        print("\n6. Predictability ceiling")
        print(self.ceiling.to_string())
        print("\n" + "-" * 74)
        print(f"VERDICT: {self.verdict}")
        for line in self.evidence:
            print(f"  - {line}")
        print("-" * 74)


# --------------------------------------------------------------------------- #
def autocorrelation_test(wide: pd.DataFrame, lags: int = 7) -> pd.DataFrame:
    """Per-series ACF at short lags against the white-noise confidence band."""
    rows = []
    for col in wide.columns:
        s = wide[col].dropna()
        band = 1.96 / np.sqrt(len(s))
        a = acf(s, nlags=lags, fft=True)[1:]
        rows.append(
            {
                "series": col,
                **{f"acf_lag{i+1}": round(float(v), 4) for i, v in enumerate(a)},
                "noise_band": round(float(band), 4),
                "n_significant": int(np.sum(np.abs(a) > band)),
            }
        )
    return pd.DataFrame(rows).set_index("series")


def ljung_box_test(wide: pd.DataFrame, lags: tuple[int, ...] = (7, 14, 30)) -> pd.DataFrame:
    """H0: the series is white noise up to the given lag."""
    rows = []
    for col in wide.columns:
        s = wide[col].dropna()
        lb = acorr_ljungbox(s, lags=list(lags), return_df=True)
        row = {"series": col}
        for lag in lags:
            row[f"p_lag{lag}"] = round(float(lb.loc[lag, "lb_pvalue"]), 4)
        row["rejects_white_noise"] = bool((lb["lb_pvalue"] < 0.05).any())
        rows.append(row)
    return pd.DataFrame(rows).set_index("series")


def seasonality_test(wide: pd.DataFrame) -> pd.DataFrame:
    """Two independent checks of an annual cycle.

    ``yoy_corr``
        Correlation between the same calendar day in consecutive years. A real
        climate series scores 0.8+; noise scores ~0.
    ``month_eta_sq``
        Share of total variance explained by month-of-year (one-way ANOVA
        effect size). Indian cities should exceed 0.5.
    """
    rows = []
    for col in wide.columns:
        s = wide[col].dropna()
        years = sorted(s.index.year.unique())
        yoy = np.nan
        if len(years) >= 2:
            a = s[s.index.year == years[0]]
            b = s[s.index.year == years[1]]
            common = np.intersect1d(a.index.dayofyear, b.index.dayofyear)
            if len(common) > 30:
                av = a[a.index.dayofyear.isin(common)].to_numpy()
                bv = b[b.index.dayofyear.isin(common)].to_numpy()
                n = min(len(av), len(bv))
                yoy = float(np.corrcoef(av[:n], bv[:n])[0, 1])

        groups = [g.to_numpy() for _, g in s.groupby(s.index.month)]
        f_stat, p_val = stats.f_oneway(*groups)
        grand = s.mean()
        ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
        ss_total = float(((s - grand) ** 2).sum())
        eta_sq = ss_between / ss_total if ss_total > 0 else np.nan

        rows.append(
            {
                "series": col,
                "yoy_corr": round(yoy, 4) if np.isfinite(yoy) else np.nan,
                "monthly_range_degC": round(
                    float(s.groupby(s.index.month).mean().max() - s.groupby(s.index.month).mean().min()),
                    3,
                ),
                "month_eta_sq": round(float(eta_sq), 4),
                "anova_p": round(float(p_val), 4),
                "has_seasonality": bool(eta_sq > 0.15 and p_val < 0.05),
            }
        )
    return pd.DataFrame(rows).set_index("series")


def cross_series_test(wide: pd.DataFrame) -> pd.DataFrame:
    """Do geographically related series move together?"""
    corr = wide.corr()
    off = corr.values[np.triu_indices(len(corr), 1)]
    return pd.DataFrame(
        {
            "metric": [
                "mean_pairwise_corr",
                "median_pairwise_corr",
                "max_pairwise_corr",
                "min_pairwise_corr",
                "share_above_0.5",
            ],
            "value": [
                round(float(np.mean(off)), 4),
                round(float(np.median(off)), 4),
                round(float(np.max(off)), 4),
                round(float(np.min(off)), 4),
                round(float(np.mean(off > 0.5)), 4),
            ],
        }
    ).set_index("metric")


def exogenous_test(
    panel: pd.DataFrame,
    target: str = TARGET_COL,
    exog: list[str] | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Linear correlation and (nonlinear) mutual information with the target.

    Mutual information catches relationships a correlation coefficient misses,
    so a variable scoring ~0 on both is genuinely uninformative.
    """
    exog = exog or [c for c in EXOG_COLS if c in panel.columns]
    df = panel[[target, *exog]].dropna()
    y = df[target].to_numpy()
    X = df[exog].to_numpy()
    mi = mutual_info_regression(X, y, random_state=random_state)
    rows = []
    for i, col in enumerate(exog):
        r = float(np.corrcoef(df[col], y)[0, 1])
        rows.append(
            {
                "variable": col,
                "pearson_r": round(r, 4),
                "abs_r": round(abs(r), 4),
                "mutual_info": round(float(mi[i]), 4),
                "informative": bool(abs(r) > 0.15 or mi[i] > 0.05),
            }
        )
    return pd.DataFrame(rows).set_index("variable").sort_values("abs_r", ascending=False)


def predictability_ceiling(wide: pd.DataFrame) -> pd.DataFrame:
    """Upper bound on achievable R^2 from an oracle day-of-year + AR(1) model.

    Both components are computed *in sample*, so the numbers are optimistic by
    construction. If even this optimistic ceiling is near zero, no model can do
    better than predicting the mean.
    """
    rows = []
    for col in wide.columns:
        s = wide[col].dropna()
        var_total = float(s.var())

        doy_means = s.groupby(s.index.dayofyear).transform("mean")
        r2_season = 1 - float(((s - doy_means) ** 2).mean()) / var_total

        lag1 = s.shift(1).dropna()
        cur = s.loc[lag1.index]
        rho = float(np.corrcoef(cur, lag1)[0, 1])
        r2_ar1 = rho**2

        rows.append(
            {
                "series": col,
                "var_total": round(var_total, 3),
                "oracle_R2_seasonal": round(r2_season, 4),
                "R2_AR1": round(r2_ar1, 4),
                "combined_ceiling": round(min(1.0, max(r2_season, 0) + max(r2_ar1, 0)), 4),
                "irreducible_noise_share": round(
                    1 - min(1.0, max(r2_season, 0) + max(r2_ar1, 0)), 4
                ),
            }
        )
    return pd.DataFrame(rows).set_index("series")


# --------------------------------------------------------------------------- #
def audit(
    panel: pd.DataFrame,
    target: str = TARGET_COL,
    city_col: str = CITY_COL,
    date_col: str = DATE_COL,
) -> ForecastabilityReport:
    """Run the full forecastability audit on a long-format panel."""
    wide = panel.pivot(index=date_col, columns=city_col, values=target)
    wide = wide.asfreq("D").interpolate("time").ffill().bfill()
    wide.columns.name = None

    ac = autocorrelation_test(wide)
    lb = ljung_box_test(wide)
    seas = seasonality_test(wide)
    cross = cross_series_test(wide)
    exo = exogenous_test(panel, target=target)
    ceil = predictability_ceiling(wide)

    evidence, fails = [], 0

    frac_sig = float((ac["n_significant"] > 1).mean())
    if frac_sig < 0.3:
        fails += 1
        evidence.append(
            f"No usable short-term memory: only {frac_sig:.0%} of series have >1 "
            f"significant ACF lag among the first 7."
        )
    else:
        evidence.append(f"{frac_sig:.0%} of series show significant short-lag memory.")

    frac_reject = float(lb["rejects_white_noise"].mean())
    if frac_reject < 0.5:
        fails += 1
        evidence.append(
            f"Ljung-Box fails to reject white noise for {1 - frac_reject:.0%} of series."
        )
    else:
        evidence.append(f"Ljung-Box rejects white noise for {frac_reject:.0%} of series.")

    frac_seasonal = float(seas["has_seasonality"].mean())
    mean_yoy = float(seas["yoy_corr"].mean(skipna=True))
    if frac_seasonal < 0.5:
        fails += 1
        evidence.append(
            f"No reproducible annual cycle: {frac_seasonal:.0%} of series pass the "
            f"seasonality test, mean year-over-year correlation {mean_yoy:.3f} "
            f"(a real temperature series scores >0.8)."
        )
    else:
        evidence.append(
            f"Annual cycle present in {frac_seasonal:.0%} of series "
            f"(mean YoY correlation {mean_yoy:.2f})."
        )

    mean_cross = float(cross.loc["mean_pairwise_corr", "value"])
    if abs(mean_cross) < 0.2:
        fails += 1
        evidence.append(
            f"Series are mutually independent (mean pairwise correlation "
            f"{mean_cross:.3f}); real cities within one country co-move strongly."
        )
    else:
        evidence.append(f"Series co-move (mean pairwise correlation {mean_cross:.2f}).")

    n_informative = int(exo["informative"].sum())
    if n_informative == 0:
        fails += 1
        evidence.append("No exogenous covariate carries information about the target.")
    else:
        evidence.append(f"{n_informative} exogenous covariate(s) carry signal.")

    mean_ceiling = float(ceil["combined_ceiling"].mean())
    evidence.append(
        f"Optimistic in-sample predictability ceiling: R^2 = {mean_ceiling:.3f} "
        f"({1 - mean_ceiling:.0%} of variance is irreducible noise)."
    )

    if fails >= 4:
        verdict = (
            "NOT FORECASTABLE — the target is statistically indistinguishable from "
            "white noise. Model comparison on this data is meaningless; the optimal "
            "forecast is the unconditional mean."
        )
    elif fails >= 2:
        verdict = (
            "WEAKLY FORECASTABLE — some structure exists but most variance is noise. "
            "Expect small, possibly insignificant gaps between models."
        )
    else:
        verdict = "FORECASTABLE — the series contains exploitable structure."

    return ForecastabilityReport(
        autocorrelation=ac,
        ljung_box=lb,
        seasonality=seas,
        cross_series=cross,
        exogenous=exo,
        ceiling=ceil,
        verdict=verdict,
        evidence=evidence,
    )


__all__ = [
    "ForecastabilityReport",
    "audit",
    "autocorrelation_test",
    "cross_series_test",
    "exogenous_test",
    "ljung_box_test",
    "predictability_ceiling",
    "seasonality_test",
]
