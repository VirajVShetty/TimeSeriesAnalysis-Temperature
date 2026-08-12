"""End-to-end pipeline: audit -> EDA -> benchmark -> intervals -> anomalies.

The **audit step runs first and is not optional**. Model leaderboards are only
meaningful on data that contains signal; see :mod:`tsa_temperature.diagnostics`.

Usage
-----
::

    python -m tsa_temperature.pipeline                    # bundled CSV
    python -m tsa_temperature.pipeline --simulated        # reference panel with known signal
    python -m tsa_temperature.pipeline --both             # run both and compare
    python -m tsa_temperature.pipeline --no-deep --no-slow --horizon 7
"""
from __future__ import annotations

import argparse
import json
import logging
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from . import diagnostics, eda, plots
from .anomaly import detect_anomalies
from .backtest import calibrate_conformal, seasonal_naive_insample, walk_forward
from .config import (
    ANOMALY,
    CITY_COL,
    DATE_COL,
    FIGURES_DIR,
    FORECAST,
    METRICS_DIR,
    RAW_CSV,
    RAW_DATA_DIR,
    TARGET_COL,
)
from .data import ClimatePanel, load_panel
from .evaluation import (
    diebold_mariano,
    evaluate_intervals,
    leaderboard,
    metrics_by_horizon,
)
from .models import (
    TORCH_AVAILABLE,
    ClimatologyForecaster,
    ClimatologyPlusPersistence,
    GlobalGBMForecaster,
    GRUForecaster,
    LSTMForecaster,
    NaivePersistence,
    SARIMAXFourier,
    SeasonalNaive,
    STLETSForecaster,
)
from .simulate import write_reference_csv

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

GBM_LABEL = "LightGBM (global, direct)"


# --------------------------------------------------------------------------- #
# Output path helpers — a tag keeps the two datasets' artefacts separate.
# --------------------------------------------------------------------------- #
def fig_path(name: str, tag: str) -> Path:
    stem, ext = name.rsplit(".", 1)
    return FIGURES_DIR / (f"{stem}__{tag}.{ext}" if tag else name)


def met_path(name: str, tag: str) -> Path:
    stem, ext = name.rsplit(".", 1)
    return METRICS_DIR / (f"{stem}__{tag}.{ext}" if tag else name)


def header(text: str) -> None:
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


# --------------------------------------------------------------------------- #
def build_factories(include_deep: bool, include_slow: bool) -> dict:
    factories = {
        "Naive (persistence)": lambda h: NaivePersistence(),
        "Seasonal naive (m=365)": lambda h: SeasonalNaive(period=365),
        "Climatology": lambda h: ClimatologyForecaster(),
        "Climatology + damped persistence": lambda h: ClimatologyPlusPersistence(),
        GBM_LABEL: lambda h: GlobalGBMForecaster(horizon=h),
    }
    if include_slow:
        factories["STL + ETS"] = lambda h: STLETSForecaster()
        factories["SARIMAX + Fourier"] = lambda h: SARIMAXFourier()
    if include_deep and TORCH_AVAILABLE:
        factories["LSTM (global, direct)"] = lambda h: LSTMForecaster(horizon=h)
        factories["GRU (global, direct)"] = lambda h: GRUForecaster(horizon=h)
    return factories


# --------------------------------------------------------------------------- #
def run_audit(panel_df: pd.DataFrame, tag: str) -> diagnostics.ForecastabilityReport:
    """Establish whether the data contains forecastable signal at all.

    Runs *before* any modelling. If the verdict is "not forecastable" then the
    leaderboard downstream is measuring sampling noise, not model skill.
    """
    report = diagnostics.audit(panel_df)
    report.print_report()
    report.autocorrelation.to_csv(met_path("audit_autocorrelation.csv", tag))
    report.ljung_box.to_csv(met_path("audit_ljung_box.csv", tag))
    report.seasonality.to_csv(met_path("audit_seasonality.csv", tag))
    report.cross_series.to_csv(met_path("audit_cross_series.csv", tag))
    report.exogenous.to_csv(met_path("audit_exogenous.csv", tag))
    report.ceiling.to_csv(met_path("audit_ceiling.csv", tag))
    with open(met_path("audit_verdict.json", tag), "w") as f:
        json.dump({"verdict": report.verdict, "evidence": report.evidence}, f, indent=2)
    return report


# --------------------------------------------------------------------------- #
def run_eda(panel: ClimatePanel, wide: pd.DataFrame, tag: str) -> str:
    header("Exploratory analysis")
    summary = panel.summary()
    summary.to_csv(met_path("city_summary.csv", tag))
    print(summary.to_string())

    eda.plot_city_overview(wide, savepath=fig_path("01_city_overview.png", tag))
    eda.plot_correlation_heatmap(wide, savepath=fig_path("02_city_correlation.png", tag))

    focus = "Delhi" if "Delhi" in wide.columns else wide.columns[0]
    s = wide[focus].rename(focus)
    dec = eda.stl_decompose(s, period=365)
    dec.plot(title=f"— {focus}", savepath=fig_path("03_stl_decomposition.png", tag))
    eda.plot_seasonal_profile(s, savepath=fig_path("04_seasonal_profile.png", tag))
    eda.plot_acf_pacf(s, nlags=60, savepath=fig_path("05_acf_pacf.png", tag))

    stat = pd.DataFrame({c: eda.stationarity_report(wide[c]) for c in wide.columns}).T
    stat.to_csv(met_path("stationarity.csv", tag))
    print("\nStationarity tests (ADF / KPSS):")
    print(stat.to_string())

    strengths = {}
    for c in wide.columns:
        d = eda.stl_decompose(wide[c])
        strengths[c] = {
            "trend_strength": round(d.trend_strength, 3),
            "seasonal_strength": round(d.seasonal_strength, 3),
        }
    strength = pd.DataFrame(strengths).T
    strength.to_csv(met_path("decomposition_strength.csv", tag))
    print("\nSTL component strengths:")
    print(strength.to_string())
    return focus


# --------------------------------------------------------------------------- #
def run_benchmark(panel_df, cfg, tag, include_deep, include_slow):
    header("Walk-forward benchmark")
    factories = build_factories(include_deep, include_slow)
    print(f"Models ({len(factories)}): {list(factories)}\n")
    result = walk_forward(panel_df, factories, cfg=cfg, verbose=True)

    insample = seasonal_naive_insample(panel_df)
    board = leaderboard(result.predictions, insample=insample, seasonality=1)
    board.to_csv(met_path("leaderboard.csv", tag))
    print("\nLeaderboard (pooled over origins x cities x horizons):")
    print(board.to_string())

    by_h = metrics_by_horizon(result.predictions)
    by_h.to_csv(met_path("metrics_by_horizon.csv", tag), index=False)

    by_city = (
        result.predictions.assign(abs_err=lambda d: (d.y_true - d.y_pred).abs())
        .groupby(["model", CITY_COL])["abs_err"]
        .mean()
        .unstack(CITY_COL)
        .round(3)
    )
    by_city.to_csv(met_path("mae_by_city.csv", tag))
    print("\nMAE by city (°C):")
    print(by_city.to_string())

    result.timings.to_csv(met_path("timings.csv", tag), index=False)
    fit_time = result.timings.groupby("model")["seconds"].mean().round(2)
    print("\nMean fit+predict time per origin (s):")
    print(fit_time.to_string())

    plots.plot_leaderboard(board, "MAE", savepath=fig_path("06_leaderboard.png", tag))
    plots.plot_error_by_horizon(
        by_h, "MAE", savepath=fig_path("07_error_by_horizon.png", tag)
    )
    return result, board, by_h


# --------------------------------------------------------------------------- #
def run_significance(result, board, tag) -> pd.DataFrame:
    header("Diebold-Mariano tests (best model vs each challenger)")
    wide = result.pivot_predictions().dropna()
    best = board.index[0]
    reserved = {best, CITY_COL, "origin", "target_date", "horizon", "y_true"}
    rows = []
    for model in wide.columns:
        if model in reserved or model not in board.index:
            continue
        dm = diebold_mariano(
            wide["y_true"], wide[best], wide[model], h=result.config.horizon, loss="mae"
        )
        dm.update({"best_model": best, "challenger": model})
        rows.append(dm)
    dm_tbl = pd.DataFrame(rows)
    if not dm_tbl.empty:
        dm_tbl["verdict"] = np.where(
            dm_tbl["p_value"] < 0.05,
            np.where(dm_tbl["statistic"] < 0, "best significantly better", "challenger better"),
            "no significant difference",
        )
        dm_tbl = dm_tbl[
            ["best_model", "challenger", "statistic", "p_value", "mean_loss_diff", "n", "verdict"]
        ]
        dm_tbl.to_csv(met_path("diebold_mariano.csv", tag), index=False)
        print(dm_tbl.to_string(index=False))
    return dm_tbl


# --------------------------------------------------------------------------- #
def run_intervals(panel_df, result, cfg, tag) -> pd.DataFrame:
    header("Conformal prediction intervals")
    calibrator = calibrate_conformal(
        panel_df, lambda h: GlobalGBMForecaster(horizon=h), cfg=cfg
    )
    widths = calibrator.width_table()
    widths.to_csv(met_path("conformal_widths.csv", tag), index=False)
    print("Calibrated half-widths by horizon (°C):")
    print(widths.to_string(index=False))

    gbm = result.for_model(GBM_LABEL).copy()
    lo, hi = calibrator.predict_interval(gbm["y_pred"], gbm["horizon"])
    gbm["lower"], gbm["upper"] = lo, hi

    overall = evaluate_intervals(
        gbm["y_true"], gbm["lower"], gbm["upper"], alpha=1 - cfg.coverage
    )
    print(f"\nOverall: {overall}")

    rows = []
    for h, grp in gbm.groupby("horizon"):
        m = evaluate_intervals(
            grp["y_true"], grp["lower"], grp["upper"], alpha=1 - cfg.coverage
        )
        m["horizon"] = h
        rows.append(m)
    cov = pd.DataFrame(rows)[["horizon", "Coverage", "MeanWidth", "IntervalScore"]]
    cov.to_csv(met_path("interval_coverage_by_horizon.csv", tag), index=False)
    plots.plot_coverage_by_horizon(
        cov, nominal=cfg.coverage, savepath=fig_path("08_interval_coverage.png", tag)
    )
    with open(met_path("interval_summary.json", tag), "w") as f:
        json.dump(overall, f, indent=2)
    return gbm


# --------------------------------------------------------------------------- #
def run_forecast_example(panel_df, result, gbm_iv, focus_city, tag) -> None:
    last_origin = max(result.origins)
    sub = result.predictions.loc[
        (result.predictions["origin"] == last_origin)
        & (result.predictions[CITY_COL] == focus_city)
    ]
    if sub.empty:
        return

    hist = (
        panel_df.loc[
            (panel_df[CITY_COL] == focus_city) & (panel_df[DATE_COL] <= last_origin)
        ]
        .set_index(DATE_COL)[TARGET_COL]
        .astype(float)
    )
    actual = sub.drop_duplicates("target_date").set_index("target_date")["y_true"].sort_index()
    keep = {GBM_LABEL, "Climatology + damped persistence", "SARIMAX + Fourier", "LSTM (global, direct)"}
    forecasts = {
        m: g.set_index("target_date")["y_pred"].sort_index()
        for m, g in sub.groupby("model")
        if m in keep
    }
    band = (
        gbm_iv.loc[
            (gbm_iv["origin"] == last_origin) & (gbm_iv[CITY_COL] == focus_city)
        ]
        .set_index("target_date")
        .sort_index()
    )
    plots.plot_forecast_example(
        history=hist,
        actual=actual,
        forecasts=forecasts,
        lower=band["lower"] if len(band) else None,
        upper=band["upper"] if len(band) else None,
        title=f"{focus_city}: {result.config.horizon}-day forecast from {last_origin.date()}",
        savepath=fig_path("09_forecast_example.png", tag),
    )
    gbm_res = result.for_model(GBM_LABEL)
    plots.plot_residual_diagnostics(
        (gbm_res["y_true"] - gbm_res["y_pred"]).to_numpy(),
        savepath=fig_path("10_residual_diagnostics.png", tag),
    )


# --------------------------------------------------------------------------- #
def run_importance(panel_df, cfg, tag) -> pd.DataFrame:
    header("Feature importance (LightGBM, fit on the full sample)")
    model = GlobalGBMForecaster(horizon=cfg.horizon).fit(panel_df)
    imp = model.feature_importance(top_n=30)
    imp.to_csv(met_path("feature_importance.csv", tag), index=False)
    print(imp.head(15).to_string(index=False))
    plots.plot_feature_importance(imp, savepath=fig_path("11_feature_importance.png", tag))
    return imp


# --------------------------------------------------------------------------- #
def run_anomalies(panel_df, focus_city, tag) -> pd.DataFrame:
    header("Anomaly detection (4-detector voting ensemble)")
    rows = []
    for city, grp in panel_df.groupby(CITY_COL):
        g = grp.sort_values(DATE_COL).set_index(DATE_COL).select_dtypes("number")
        res = detect_anomalies(g, target_col=TARGET_COL, cfg=ANOMALY)
        s = res.summary()
        s["city"] = city
        rows.append(s)
        if city == focus_city:
            plots.plot_anomalies(
                res.frame,
                title=f"Temperature anomalies — {city}",
                savepath=fig_path("12_anomalies.png", tag),
            )
            res.frame.to_csv(met_path(f"anomaly_detail_{city}.csv", tag))
            print(f"\nDetector agreement matrix ({city}):")
            print(res.agreement().to_string())
            flagged = res.anomalies()
            print(f"\nEnsemble-flagged days in {city}: {len(flagged)}")
            if len(flagged):
                print(flagged[["value", "n_votes"]].round(2).head(12).to_string())
    tbl = pd.DataFrame(rows).set_index("city")
    tbl.to_csv(met_path("anomaly_counts.csv", tag))
    print("\nFlag counts by detector and city:")
    print(tbl.to_string())
    return tbl


# --------------------------------------------------------------------------- #
def run_dataset(panel: ClimatePanel, cfg, tag: str, args) -> dict:
    panel_df = panel.frame
    start, end = panel.date_range
    header(f"DATASET: {tag or 'bundled CSV'}")
    print(
        f"{len(panel_df):,} rows | {len(panel.cities)} cities | "
        f"{start.date()} .. {end.date()}"
    )
    print(
        f"Protocol: horizon={cfg.horizon}d, {cfg.n_backtest_origins} rolling origins, "
        f"stride={cfg.origin_stride}d, nominal coverage={cfg.coverage:.0%}"
    )

    report = run_audit(panel_df, tag)

    wide = panel.wide()
    focus = "Delhi" if "Delhi" in wide.columns else wide.columns[0]
    if not args.skip_eda:
        focus = run_eda(panel, wide, tag)

    result, board, by_h = run_benchmark(
        panel_df, cfg, tag, include_deep=not args.no_deep, include_slow=not args.no_slow
    )
    run_significance(result, board, tag)
    gbm_iv = run_intervals(panel_df, result, cfg, tag)
    run_forecast_example(panel_df, result, gbm_iv, focus, tag)
    run_importance(panel_df, cfg, tag)
    run_anomalies(panel_df, focus, tag)
    return {"report": report, "board": board, "result": result}


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Indian temperature time-series pipeline")
    p.add_argument("--data", type=Path, default=RAW_CSV)
    p.add_argument("--horizon", type=int, default=FORECAST.horizon)
    p.add_argument("--origins", type=int, default=FORECAST.n_backtest_origins)
    p.add_argument("--stride", type=int, default=FORECAST.origin_stride)
    p.add_argument("--simulated", action="store_true", help="use the simulated reference panel")
    p.add_argument("--both", action="store_true", help="run bundled CSV and simulated panel")
    p.add_argument("--no-deep", action="store_true", help="skip LSTM/GRU")
    p.add_argument("--no-slow", action="store_true", help="skip SARIMAX / STL-ETS")
    p.add_argument("--skip-eda", action="store_true")
    args = p.parse_args(argv)

    cfg = replace(
        FORECAST,
        horizon=args.horizon,
        n_backtest_origins=args.origins,
        origin_stride=args.stride,
    )

    header("Indian City Temperature — Time Series Analysis & Forecasting")

    sim_csv = RAW_DATA_DIR / "simulated_reference_climate.csv"

    def simulated_panel() -> ClimatePanel:
        if not sim_csv.exists():
            write_reference_csv(sim_csv)
        return ClimatePanel.from_csv(sim_csv).fill_calendar()

    outcomes = {}
    if args.both:
        outcomes["actual"] = run_dataset(load_panel(args.data), cfg, "actual", args)
        outcomes["simulated"] = run_dataset(simulated_panel(), cfg, "simulated", args)
    elif args.simulated:
        outcomes["simulated"] = run_dataset(simulated_panel(), cfg, "simulated", args)
    else:
        outcomes["actual"] = run_dataset(load_panel(args.data), cfg, "actual", args)

    if len(outcomes) > 1:
        header("CROSS-DATASET COMPARISON")
        comp = pd.concat(
            {k: v["board"]["MAE"] for k, v in outcomes.items()}, axis=1
        ).round(3)
        comp["MAE_ratio_sim_vs_actual"] = (comp["simulated"] / comp["actual"]).round(3)
        comp.to_csv(METRICS_DIR / "cross_dataset_mae.csv")
        print(comp.to_string())
        print("\nVerdicts:")
        for k, v in outcomes.items():
            print(f"  {k:10s} {v['report'].verdict.split(' — ')[0]}")

    header("Done")
    print(f"Figures  -> {FIGURES_DIR}")
    print(f"Metrics  -> {METRICS_DIR}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
