"""Central configuration for the temperature time-series project.

Every tunable lives here so experiments stay reproducible and the rest of the
code base contains no magic numbers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"

RAW_CSV = RAW_DATA_DIR / "Indian_Climate_Dataset_2024_2025.csv"

for _d in (PROCESSED_DATA_DIR, FIGURES_DIR, METRICS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Column schema of the raw dataset
# --------------------------------------------------------------------------- #
DATE_COL = "date"
CITY_COL = "city"
STATE_COL = "state"
TARGET_COL = "temp_avg"

#: Raw CSV header -> internal snake_case name.
COLUMN_RENAME: dict[str, str] = {
    "Date": DATE_COL,
    "City": CITY_COL,
    "State": STATE_COL,
    "Temperature_Max (°C)": "temp_max",
    "Temperature_Min (°C)": "temp_min",
    "Temperature_Avg (°C)": TARGET_COL,
    "Humidity (%)": "humidity",
    "Rainfall (mm)": "rainfall",
    "Wind_Speed (km/h)": "wind_speed",
    "AQI": "aqi",
    "AQI_Category": "aqi_category",
    "Pressure (hPa)": "pressure",
    "Cloud_Cover (%)": "cloud_cover",
}

#: Numeric weather channels available besides the target. These are only ever
#: consumed in *lagged* form so that no future information leaks into a forecast.
EXOG_COLS: list[str] = [
    "temp_max",
    "temp_min",
    "humidity",
    "rainfall",
    "wind_speed",
    "aqi",
    "pressure",
    "cloud_cover",
]


# --------------------------------------------------------------------------- #
# Experiment configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ForecastConfig:
    """Forecasting protocol.

    The dataset spans 2024-01-01 .. 2025-12-31 (731 daily observations per
    city), i.e. only *two* annual cycles. That is far too little to identify a
    long-horizon climate trend, so the project deliberately targets
    short-to-medium range forecasting evaluated with rolling-origin
    (walk-forward) backtesting.
    """

    horizon: int = 14
    """Forecast horizon in days."""

    n_backtest_origins: int = 8
    """Number of rolling-origin evaluation windows."""

    origin_stride: int = 14
    """Days between consecutive backtest origins."""

    min_train_days: int = 400
    """Minimum training history required before the first origin."""

    calibration_days: int = 90
    """Tail of the training window held out to calibrate conformal intervals."""

    seasonal_period: int = 365
    """Annual seasonality of daily data (used by MASE and the naive baseline)."""

    coverage: float = 0.9
    """Nominal coverage of the prediction intervals."""


@dataclass(frozen=True)
class FeatureConfig:
    """Feature-engineering knobs for the ML / DL models."""

    target_lags: tuple[int, ...] = (1, 2, 3, 7, 14, 21, 28, 365)
    exog_lags: tuple[int, ...] = (1, 7, 14)
    rolling_windows: tuple[int, ...] = (7, 14, 28, 90)
    fourier_order: int = 4
    """Number of sin/cos pairs used to encode the annual cycle."""

    calendar_cols: tuple[str, ...] = ("month", "imd_season")
    """Calendar columns exposed to the model.

    Kept deliberately low-cardinality. ``doy``, ``dayofweek`` and ``time_idx``
    are available but excluded by default — see
    :data:`tsa_temperature.features.SAFE_CALENDAR_COLS`.
    """

    use_exog: bool = True
    use_city_features: bool = True


@dataclass(frozen=True)
class ModelConfig:
    """Hyper-parameters for the individual model families."""

    # LightGBM global model.
    #
    # Capacity is deliberately modest. The panel contains ~100k rows but only
    # ~700 genuinely independent time points, so a 900-tree / 63-leaf model
    # memorises rather than generalises. These settings were selected on a
    # 10-origin walk-forward backtest of the simulated reference panel.
    lgbm_params: dict = field(
        default_factory=lambda: {
            "objective": "regression_l1",
            "n_estimators": 400,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 40,
            "subsample": 0.85,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "verbosity": -1,
            "n_jobs": -1,
        }
    )

    # SARIMAX with Fourier terms
    sarimax_order: tuple[int, int, int] = (2, 0, 2)
    sarimax_fourier_order: int = 3

    # Holt-Winters / ETS
    ets_seasonal_periods: int = 365

    # LSTM
    lstm_lookback: int = 60
    lstm_hidden: int = 96
    lstm_layers: int = 2
    lstm_dropout: float = 0.15
    lstm_embed_dim: int = 8
    lstm_epochs: int = 60
    lstm_batch_size: int = 128
    lstm_lr: float = 3e-3
    lstm_patience: int = 8


@dataclass(frozen=True)
class AnomalyConfig:
    """Anomaly-detection thresholds."""

    stl_period: int = 365
    stl_robust: bool = True
    residual_z_threshold: float = 3.0
    isolation_forest_contamination: float = 0.02
    brutlag_gamma: float = 0.3684211
    brutlag_scaling_factor: float = 1.96
    ensemble_min_votes: int = 2
    """A point is flagged only when at least this many detectors agree."""


RANDOM_SEED = 42

FORECAST = ForecastConfig()
FEATURES = FeatureConfig()
MODELS = ModelConfig()
ANOMALY = AnomalyConfig()
