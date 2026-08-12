"""Time-series analysis and forecasting of Indian city temperatures.

Quick start
-----------
>>> from tsa_temperature import load_panel
>>> panel = load_panel()
>>> panel.summary().head()

Run the full benchmark from the command line::

    python -m tsa_temperature.pipeline --horizon 14 --origins 8
"""
from . import diagnostics
from .anomaly import detect_anomalies, detect_anomalies_panel
from .backtest import BacktestResult, calibrate_conformal, walk_forward
from .conformal import SplitConformal
from .data import ClimatePanel, load_panel
from .eda import decompose, harmonic_decompose, stationarity_report, stl_decompose
from .evaluation import evaluate_point, leaderboard, metrics_by_horizon
from .simulate import simulate_climate_panel

__version__ = "2.0.0"

__all__ = [
    "BacktestResult",
    "ClimatePanel",
    "SplitConformal",
    "__version__",
    "calibrate_conformal",
    "detect_anomalies",
    "detect_anomalies_panel",
    "diagnostics",
    "evaluate_point",
    "leaderboard",
    "load_panel",
    "simulate_climate_panel",
    "metrics_by_horizon",
    "stationarity_report",
    "decompose",
    "harmonic_decompose",
    "stl_decompose",
    "walk_forward",
]
