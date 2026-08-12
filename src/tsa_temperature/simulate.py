"""Physically plausible synthetic Indian climate panel.

Purpose
-------
The bundled ``Indian_Climate_Dataset_2024_2025.csv`` turns out to be white
noise (see :mod:`tsa_temperature.diagnostics`). That makes it useless for
*validating* a forecasting pipeline: every model scores the same, so a passing
benchmark proves nothing about whether the code is correct.

This module generates a reference panel with the structure real daily
temperature actually has, so the model zoo can be exercised against a known
ground truth:

* a sinusoidal annual cycle with city-specific amplitude and phase,
* AR(1) synoptic-scale persistence (warm spells last several days),
* a spatially correlated shock field, so nearby cities co-move,
* a monsoon window that suppresses temperature and drives rainfall/humidity,
* covariates causally linked to temperature rather than drawn independently,
* occasional heatwave and cold-wave events for anomaly detection to find.

The generator is *not* a climate model and should never be presented as real
observations. It exists purely as a test fixture with known properties.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RANDOM_SEED

#: (city, state, annual mean °C, annual half-amplitude °C, phase shift days,
#:  daily noise σ, AR(1) persistence, monsoon suppression °C)
CITY_PROFILES: list[tuple[str, str, float, float, int, float, float, float]] = [
    ("Mumbai", "Maharashtra", 27.8, 3.2, 15, 1.1, 0.80, 3.0),
    ("Delhi", "Delhi", 25.6, 9.5, 8, 2.2, 0.78, 4.5),
    ("Bengaluru", "Karnataka", 24.2, 3.0, 12, 1.3, 0.74, 2.5),
    ("Chennai", "Tamil Nadu", 28.9, 3.8, 25, 1.2, 0.77, 2.0),
    ("Kolkata", "West Bengal", 27.1, 7.0, 10, 1.7, 0.79, 3.5),
    ("Hyderabad", "Telangana", 27.0, 5.2, 10, 1.6, 0.76, 3.0),
    ("Ahmedabad", "Gujarat", 28.2, 8.0, 6, 2.1, 0.78, 4.0),
    ("Jaipur", "Rajasthan", 26.3, 9.8, 5, 2.4, 0.79, 4.5),
    ("Lucknow", "Uttar Pradesh", 26.0, 9.9, 8, 2.3, 0.80, 4.5),
    ("Bhopal", "Madhya Pradesh", 25.8, 7.4, 9, 1.9, 0.77, 4.0),
]


def _ar1(n: int, rho: float, sigma: float, rng: np.random.Generator, shocks=None):
    """Simulate an AR(1) path with optional externally supplied innovations."""
    if shocks is None:
        shocks = rng.normal(0, 1, n)
    x = np.zeros(n)
    scale = sigma * np.sqrt(1 - rho**2)
    for t in range(1, n):
        x[t] = rho * x[t - 1] + scale * shocks[t]
    return x


def simulate_climate_panel(
    start: str = "2024-01-01",
    end: str = "2025-12-31",
    seed: int = RANDOM_SEED,
    spatial_corr: float = 0.55,
    n_heatwaves: int = 6,
    n_coldwaves: int = 4,
) -> pd.DataFrame:
    """Generate a realistic multi-city daily climate panel.

    Returns a long-format frame with the same schema as
    :func:`tsa_temperature.data.load_panel`, so it is a drop-in replacement
    anywhere in the pipeline.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    n, n_cities = len(dates), len(CITY_PROFILES)
    doy = dates.dayofyear.to_numpy()
    year_len = np.where(dates.is_leap_year, 366.0, 365.0)
    frac = doy / year_len

    # Shared synoptic shock field: a common component plus idiosyncratic noise
    # produces realistic cross-city correlation without perfect co-movement.
    common = rng.normal(0, 1, n)
    shocks = np.sqrt(spatial_corr) * common[:, None] + np.sqrt(
        1 - spatial_corr
    ) * rng.normal(0, 1, (n, n_cities))

    # Monsoon: roughly June-September, smooth onset and withdrawal.
    monsoon = np.exp(-0.5 * ((doy - 205) / 38.0) ** 2)

    # A mild warming trend across the two years (~0.4 degC/decade equivalent).
    trend = np.linspace(0, 0.08, n)

    records = []
    for j, (city, state, mean_t, amp, phase, sigma, rho, mons_effect) in enumerate(
        CITY_PROFILES
    ):
        seasonal = amp * np.sin(2 * np.pi * (frac - (phase + 100) / 365.0))
        weather = _ar1(n, rho, sigma, rng, shocks[:, j])
        temp_avg = mean_t + seasonal - mons_effect * monsoon + weather + trend

        # Extreme events, so anomaly detectors have something real to find.
        for _ in range(rng.poisson(n_heatwaves / n_cities * 2)):
            start_i = rng.integers(0, n - 8)
            length = rng.integers(3, 8)
            temp_avg[start_i : start_i + length] += rng.uniform(3.5, 6.5)
        for _ in range(rng.poisson(n_coldwaves / n_cities * 2)):
            start_i = rng.integers(0, n - 6)
            length = rng.integers(2, 6)
            temp_avg[start_i : start_i + length] -= rng.uniform(3.0, 5.5)

        # Diurnal range shrinks under cloud and monsoon conditions.
        cloud = np.clip(
            18 + 62 * monsoon + 14 * rng.normal(0, 1, n) - 0.6 * weather, 0, 100
        )
        diurnal = np.clip(13.5 - 0.07 * cloud - 2.4 * monsoon + rng.normal(0, 1.0, n), 3, 20)
        temp_max = temp_avg + diurnal / 2
        temp_min = temp_avg - diurnal / 2

        humidity = np.clip(
            42 + 38 * monsoon + 0.22 * cloud - 0.75 * (temp_avg - mean_t) + rng.normal(0, 5, n),
            15,
            100,
        )
        rain_prob = np.clip(0.04 + 0.62 * monsoon + 0.004 * (humidity - 60), 0, 0.95)
        rainfall = np.where(
            rng.random(n) < rain_prob, rng.gamma(1.7, 6.5 * (0.35 + monsoon), n), 0.0
        )
        wind = np.clip(6 + 5 * monsoon + rng.gamma(2.0, 1.6, n), 0.5, 45)
        pressure = np.clip(
            1013 + 9 * np.cos(2 * np.pi * frac) - 5 * monsoon - 0.28 * (temp_avg - mean_t)
            + rng.normal(0, 2.0, n),
            985,
            1040,
        )
        # AQI: worst in cool, still, dry winter air; scrubbed by rain.
        aqi = np.clip(
            120
            + 95 * np.cos(2 * np.pi * frac)
            - 2.6 * wind
            - 0.9 * np.minimum(rainfall, 40)
            + 22 * rng.normal(0, 1, n),
            15,
            480,
        )

        records.append(
            pd.DataFrame(
                {
                    "Date": dates,
                    "City": city,
                    "State": state,
                    "Temperature_Max (°C)": np.round(temp_max, 1),
                    "Temperature_Min (°C)": np.round(temp_min, 1),
                    "Temperature_Avg (°C)": np.round(temp_avg, 1),
                    "Humidity (%)": np.round(humidity, 1),
                    "Rainfall (mm)": np.round(rainfall, 1),
                    "Wind_Speed (km/h)": np.round(wind, 1),
                    "AQI": np.round(aqi).astype(int),
                    "AQI_Category": pd.cut(
                        aqi,
                        [-1, 50, 100, 200, 300, 400, 10_000],
                        labels=[
                            "Good",
                            "Satisfactory",
                            "Moderate",
                            "Poor",
                            "Very Poor",
                            "Severe",
                        ],
                    ).astype(str),
                    "Pressure (hPa)": np.round(pressure, 1),
                    "Cloud_Cover (%)": np.round(cloud, 1),
                }
            )
        )

    return pd.concat(records, ignore_index=True)


def write_reference_csv(path, **kwargs) -> pd.DataFrame:
    """Generate the reference panel and write it in the raw CSV schema."""
    df = simulate_climate_panel(**kwargs)
    df.to_csv(path, index=False)
    return df


__all__ = ["CITY_PROFILES", "simulate_climate_panel", "write_reference_csv"]
