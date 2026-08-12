# Temperature-TSA

Time-series analysis, forecasting and anomaly detection for daily temperature
across ten Indian cities.

> **Headline finding — read this before the model results.**
> The bundled `Indian_Climate_Dataset_2024_2025.csv` is **statistically
> indistinguishable from white noise**. It has no autocorrelation, no annual
> cycle, no correlation between cities, and no relationship between temperature
> and humidity/pressure/AQI. No forecasting model can beat a constant on it.
> The project therefore ships a **forecastability audit** that proves this
> before any model is fit, plus a **physically realistic simulator** so the
> model zoo can be validated on data that actually contains signal.

---

## What changed from version 1

The original project fit a Holt-Winters model with `seasonal_periods=12` to a
**monthly** series (`dataset/average_temp_india.csv`, 2000–2018) and detected
anomalies with Brutlag confidence bands. That file no longer exists in the
repository, and the dataset that replaced it is **daily**, **multi-city**, and
carries eight extra weather channels — so none of the old scripts ran.

| Area | v1 | v2 |
|---|---|---|
| Structure | 3 loose scripts at repo root | installable `src/` package, CLI, test suite |
| Data | monthly, 1 series | daily panel, 10 cities, 9 channels, schema validation |
| Decomposition | `seasonal_decompose(period=12)` | STL / MSTL with an automatic harmonic fallback |
| Seasonality | 12 monthly dummies | Fourier harmonics (2K parameters, not 365) |
| Models | Holt-Winters only | 4 baselines + STL-ETS + 2 SARIMAX + LightGBM + LSTM/GRU |
| Validation | single 70/30 split | rolling-origin walk-forward, Diebold–Mariano tests |
| Uncertainty | none | split-conformal intervals, per horizon |
| Anomalies | Brutlag only | 4-detector voting ensemble |
| Data quality | assumed | explicit forecastability audit |

---

## Quick start

```bash
pip install -e ".[all]"          # or: pip install -r requirements.txt

# Audit + benchmark on the bundled CSV
python -m tsa_temperature.pipeline

# Same, on the simulated panel that has real structure
python -m tsa_temperature.pipeline --simulated

# Both, with a side-by-side comparison
python -m tsa_temperature.pipeline --both

# Faster: skip the neural nets and the per-city statistical models
python -m tsa_temperature.pipeline --no-deep --no-slow --horizon 7

pytest                            # run the test suite
```

Outputs land in `reports/figures/` (PNG) and `reports/metrics/` (CSV/JSON).

```python
from tsa_temperature import load_panel, diagnostics

panel = load_panel()
diagnostics.audit(panel.frame).print_report()
```

---

## Repository layout

```
src/tsa_temperature/
├── config.py        # every tunable: paths, schema, protocol, hyper-parameters
├── data.py          # loading, schema + physical validation, calendar filling, splits
├── diagnostics.py   # forecastability audit (run this first)
├── simulate.py      # realistic reference panel for pipeline validation
├── eda.py           # STL / MSTL / harmonic decomposition, ADF + KPSS, ACF/PACF
├── features.py      # leakage-safe lags, rolling stats, Fourier terms, climatology
├── models/
│   ├── base.py      # the fit/predict contract every model implements
│   ├── baselines.py # persistence, seasonal naive, climatology (+damped persistence)
│   ├── classical.py # STL+ETS, SARIMAX+Fourier, SARIMAX+Fourier+exog
│   ├── ml.py        # global LightGBM, direct multi-horizon, quantile variant
│   └── deep.py      # LSTM / GRU with city embeddings
├── backtest.py      # rolling-origin harness + conformal calibration
├── conformal.py     # split-conformal prediction intervals
├── evaluation.py    # MAE/RMSE/MAPE/sMAPE/MASE, interval scores, Diebold–Mariano
├── anomaly.py       # Brutlag, STL residual, Isolation Forest, matrix profile
├── plots.py         # result visualisations
└── pipeline.py      # end-to-end CLI
tests/               # 60+ tests: leakage, contracts, metrics, detectors
legacy/              # the original v1 scripts, kept for reference
```

---

## 1. The forecastability audit

`diagnostics.audit()` runs five independent checks before any model is fit.

| Check | Method | Bundled CSV | Simulated reference |
|---|---|---|---|
| Short-term memory | ACF lags 1–7 vs 95% band | **0%** of cities have >1 significant lag | 100% |
| Serial dependence | Ljung–Box (lags 7/14/30) | **fails to reject white noise for 80%** | rejects for 100% |
| Annual cycle | year-over-year correlation of same day-of-year | **0.02** | 0.77 |
| Spatial structure | mean pairwise city correlation | **−0.006** | 0.85 |
| Covariates | Pearson *r* + mutual information | humidity 0.004, pressure −0.004, AQI 0.012 | 8 informative |

Delhi's monthly means in the bundled file run 28.3–30.9 °C across *all twelve
months*. A real Delhi series swings from roughly 14 °C in January to 34 °C in
June. The distribution is near-uniform (excess kurtosis −1.04), which is what
you get from `uniform(18, 42)` — not from a physical process.

**Verdict: NOT FORECASTABLE.** On this data the optimal forecast is the
unconditional mean, every model converges to it, and any gap between models on
a leaderboard is sampling noise.

This is why the audit exists as a pipeline stage rather than a footnote: a
leaderboard can always be produced, and it will always look meaningful.

---

## 2. Models

All models implement one contract — `fit(train_panel)` then
`predict(horizon)` — so the backtest harness treats a naive rule and an LSTM
identically.

**Baselines.** Persistence (random walk), seasonal naive (m=365), day-of-year
climatology, and climatology + damped persistence. The last one is a genuinely
hard baseline: it captures the seasonal cycle *and* the fact that a warm spell
persists for a few days.

**Classical.** `STL + ETS` (decompose, smooth the seasonally-adjusted series,
add the season back) and `SARIMAX + Fourier`, optionally with lagged weather
covariates.

> Holt-Winters with `seasonal_periods=365` — the direct translation of the v1
> approach to daily data — would estimate 365 seasonal parameters from ~700
> observations. It is unidentifiable. Fourier harmonics encode the same annual
> cycle in `2K` parameters instead.

**Machine learning.** One **global** LightGBM across all cities, with city
identity as a categorical feature and lead time as a numeric feature, so a
single model serves every horizon (**direct** multi-horizon — no recursive
error compounding). It predicts the *departure from climatology*, not the raw
level.

**Deep learning.** A global LSTM/GRU encoder over a 60-day multivariate window
with learned city embeddings and a direct 14-step output head, trained on
anomalies with early stopping and deterministic seeding.

**Uncertainty.** Split-conformal intervals calibrated **per horizon** on
genuine out-of-sample residuals from origins preceding the test window —
distribution-free, with finite-sample coverage guarantees.

---

## 3. Evaluation protocol

Rolling-origin (walk-forward) backtesting: the model is refit at each of
several successive cutoffs and scored on the following 14 days.

A single 70/30 split — the v1 approach — gives one noisy accuracy estimate that
depends entirely on which fortnight landed in the test set. On one origin each
horizon has only `n_cities` = 10 points; differences of 0.5 °C between models
are pure chance. Every accuracy claim in this repository is pooled over
multiple origins.

Metrics: MAE, RMSE, MAPE, sMAPE, **MASE** (scale-free, the headline metric),
bias, R². Intervals are scored on empirical coverage, mean width and the
Winkler interval score. Model pairs are compared with a **Diebold–Mariano**
test using the Harvey–Leybourne–Newbold small-sample correction.

---

## 4. Results on the simulated reference panel

Six rolling origins, 14-day horizon, 10 cities — 840 forecast/actual pairs.
These numbers validate that the code is correct; they are **not** claims about
Indian climate.

| Model | MAE (°C) | RMSE (°C) | R² | MAE @ h=1 | MAE @ h=14 |
|---|---|---|---|---|---|
| Climatology + damped persistence | **1.546** | **1.984** | 0.758 | 1.19 | 1.76 |
| LightGBM (global, direct) | 1.566 | 2.032 | 0.746 | 1.22 | 1.73 |
| Climatology | 1.622 | 2.091 | 0.731 | 1.58 | 1.73 |
| Naive (persistence) | 1.707 | 2.349 | 0.661 | **0.64** | 2.79 |
| LightGBM, variance cap disabled | 1.733 | 2.231 | 0.694 | 1.22 | 2.33 |

Three things worth reading off this table:

1. **A correctly-specified simple model wins.** The simulator generates
   climatology + damped AR(1) anomalies, and `Climatology + damped persistence`
   is exactly that model. Gradient boosting has to *estimate* the same
   structure from two years of data and pays for it in variance. This is the
   expected outcome, and a useful reminder that ML is not automatically better.
2. **Skill is horizon-dependent.** Persistence is unbeatable at day 1 (0.64 °C)
   and worst by day 14 (2.79 °C). Climatology is the reverse. Pooled numbers
   hide this — always look at the per-horizon breakdown.
3. **The variance cap matters.** Without it the GBM is better at short range
   but blows up at day 14 (2.33 vs 1.73 °C), ending up worse overall than doing
   nothing.

---

## 5. Anomaly detection

Four detectors vote; a day is flagged only when at least two agree.

| Detector | Catches |
|---|---|
| Brutlag bands (v1, retained) | seasonally-varying deviation from a smooth baseline |
| STL/harmonic residual z-score | extreme remainders, using MAD so outliers can't mask themselves |
| Isolation Forest | days that are odd given the *joint* weather state |
| Matrix profile | *shape* anomalies — unusual sub-sequences, not just extreme readings |

The matrix profile is a self-join over z-normalised 14-day windows,
implemented directly in NumPy. It finds an unprecedented *pattern of change*,
which the other three cannot: a day can be unremarkable in isolation while the
fortnight around it is unlike anything else in the record.

---

## 6. Bugs found and fixed along the way

These are documented because each one silently produced plausible-looking but
wrong output.

**STL is degenerate with two seasonal cycles.** `STL(period=365)` on 731 daily
observations has two points per cycle-subseries and interpolates them exactly.
On a series built as `27 + 9·sin(annual) + N(0, 0.8)`, it returned a residual
standard deviation of **0.001** instead of 0.8 — the entire noise process was
absorbed into "seasonality", and both trend and seasonal strength reported a
perfect 1.000. `eda.decompose()` now counts cycles and falls back to robust
harmonic regression below three, recovering 0.792.

**Brutlag bands were ~3× too narrow.** The v1 recursion seeded the first season
with `d_t = γ·|e_t|`, but the steady state of `d_t = γ|e_t| + (1−γ)d_{t−m}` is
`E|e|`. With γ ≈ 0.37 the bands started at roughly a third of their intended
width and flagged most of the first year as anomalous — empirical coverage was
30% against a nominal 95%. Seeding with an expanding mean of `|e|` fixes it.

**The tree model was memorising dates.** With `doy`, `time_idx` and
`dayofweek` in the feature set, `city_code` + `doy` uniquely identifies a
training row. In-sample MAE was a suspiciously flat 0.38 °C across *every*
horizon while out-of-sample error was four times worse, and `dayofweek` was
collecting 4.4% of total gain on data with no weekly cycle at all. `time_idx`
is worse still: at prediction time it takes values never seen in training, so
every forecast falls into the same extreme leaf. Annual position is now carried
by Fourier terms only, which cannot isolate a single day.

**Interval bounds were being discarded.** `BaseForecaster.predict()` returned a
fixed column list, silently dropping the `lower`/`upper` columns that SARIMAX
and the quantile GBM emit.

**Early stopping collapsed the model to a constant.** With a
climatology-residual target the validation loss is nearly flat from iteration
one, so early stopping halted at iteration 1 and the model's forecasts became
plain climatology. It is now off by default; capacity is controlled through
hyper-parameters instead.

**Held-out shrinkage calibration was measuring the wrong thing.** The first
attempt at long-horizon shrinkage estimated `a_h = cov(r, r̂)/var(r̂)` on a
chronological tail. Every estimate for h ≥ 3 came out *negative*. Two causes:
`DayOfYearClimatology` averages all available years, so with only two years the
2024 and 2025 residuals for the same day-of-year are two halves of one
deviation and mechanically anti-correlated; and the calibration model forecast
up to 150 days past its training end while at inference it only ever forecasts
14 days past a full-length history. The regimes aren't comparable. The shipped
solution uses no held-out window at all — it caps the predicted anomaly's
standard deviation at the AR(1) theoretical maximum `σ·ρ^h`, with `ρ` and `σ`
estimated from the training series alone.

---

## 7. Guarding against leakage

Features are indexed by **forecast origin** `t`; targets are `y[t+h]`.

- `y_lag1` is `y[t]`, the most recent *observed* value. Nothing dated after `t`
  enters a feature. A test truncates the series and asserts that features at
  earlier origins are byte-identical.
- Calendar and Fourier terms are evaluated on the **target** date `t+h`, which
  is legitimate — the calendar is known arbitrarily far ahead.
- Exogenous weather channels are used in **lagged form only**. SARIMAX holds
  the last observed covariate constant across the forecast window, mirroring
  what is actually available in production.
- `DayOfYearClimatology` is estimated from the target, so it is fit on the
  training split and applied to the test split.
- A test asserts no feature correlates with the target at |r| > 0.9999.

---

## 8. Limitations

- **Two years of data.** Two annual cycles cannot identify a climate trend,
  support STL at `period=365`, or give the seasonal-naive baseline a full
  reference year. The horizon is deliberately capped at 14 days.
- **The bundled dataset carries no signal.** Every accuracy number in this
  README comes from the simulated reference panel.
- **The simulator is a test fixture, not a climate model.** It exists to
  validate the pipeline against data with known properties. It must never be
  presented as observations.
- **`temp_avg` is exactly `(temp_max + temp_min)/2`** in the bundled file, so
  those two channels carry no independent information about the target.

To run this on real data, drop a CSV with the same schema into `data/raw/` and
point `--data` at it. `config.COLUMN_RENAME` maps source headers to internal
names.

---

## License

See `LICENSE`.
