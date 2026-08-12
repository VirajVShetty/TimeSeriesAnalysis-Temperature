"""Data loading, validation and reshaping for the Indian climate dataset."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import (
    CITY_COL,
    COLUMN_RENAME,
    DATE_COL,
    EXOG_COLS,
    RAW_CSV,
    TARGET_COL,
)

logger = logging.getLogger(__name__)


class DataValidationError(ValueError):
    """Raised when the loaded frame violates the expected schema."""


@dataclass
class ClimatePanel:
    """A tidy, validated daily panel of Indian city climate observations.

    Attributes
    ----------
    frame:
        Long-format frame indexed by a ``RangeIndex`` with ``date``/``city``
        columns, sorted by ``(city, date)``.
    """

    frame: pd.DataFrame

    # ------------------------------------------------------------------ #
    # Constructors
    # ------------------------------------------------------------------ #
    @classmethod
    def from_csv(cls, path: str | Path = RAW_CSV) -> ClimatePanel:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {path}. Expected the raw CSV under data/raw/."
            )
        raw = pd.read_csv(path)
        missing = set(COLUMN_RENAME) - set(raw.columns)
        if missing:
            raise DataValidationError(f"Missing expected columns: {sorted(missing)}")

        df = raw.rename(columns=COLUMN_RENAME)
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])
        df = df.sort_values([CITY_COL, DATE_COL]).reset_index(drop=True)
        panel = cls(df)
        panel.validate()
        return panel

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    def validate(self) -> ClimatePanel:
        df = self.frame
        if df.duplicated([CITY_COL, DATE_COL]).any():
            dupes = df[df.duplicated([CITY_COL, DATE_COL], keep=False)]
            raise DataValidationError(
                f"Duplicate (city, date) rows detected:\n{dupes.head()}"
            )

        gaps: dict[str, int] = {}
        for city, grp in df.groupby(CITY_COL):
            expected = pd.date_range(grp[DATE_COL].min(), grp[DATE_COL].max(), freq="D")
            n_missing = len(expected) - grp[DATE_COL].nunique()
            if n_missing:
                gaps[city] = n_missing
        if gaps:
            logger.warning("Calendar gaps found (city -> missing days): %s", gaps)

        # Physical plausibility of the temperature channels.
        bad = df[(df[TARGET_COL] < -40) | (df[TARGET_COL] > 60)]
        if len(bad):
            raise DataValidationError(
                f"{len(bad)} implausible temperature values outside [-40, 60] degC."
            )
        if (df["temp_min"] > df["temp_max"]).any():
            raise DataValidationError("Found rows where temp_min exceeds temp_max.")
        return self

    # ------------------------------------------------------------------ #
    # Properties / accessors
    # ------------------------------------------------------------------ #
    @property
    def cities(self) -> list[str]:
        return sorted(self.frame[CITY_COL].unique().tolist())

    @property
    def date_range(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        return self.frame[DATE_COL].min(), self.frame[DATE_COL].max()

    def city_series(self, city: str, column: str = TARGET_COL) -> pd.Series:
        """Return one city's channel as a ``DatetimeIndex``ed series at daily freq."""
        grp = self.frame.loc[self.frame[CITY_COL] == city]
        if grp.empty:
            raise KeyError(f"Unknown city {city!r}. Available: {self.cities}")
        s = grp.set_index(DATE_COL)[column].astype(float)
        s = s.asfreq("D")
        if s.isna().any():
            s = s.interpolate("time").ffill().bfill()
        s.name = f"{city}:{column}"
        return s

    def wide(self, column: str = TARGET_COL) -> pd.DataFrame:
        """Pivot to a ``date x city`` matrix for a single channel."""
        wide = self.frame.pivot(index=DATE_COL, columns=CITY_COL, values=column)
        wide = wide.asfreq("D").interpolate("time").ffill().bfill()
        wide.columns.name = None
        return wide

    # ------------------------------------------------------------------ #
    # Cleaning helpers
    # ------------------------------------------------------------------ #
    def fill_calendar(self) -> ClimatePanel:
        """Reindex every city onto a complete daily calendar, interpolating gaps."""
        start, end = self.date_range
        full = pd.date_range(start, end, freq="D")
        pieces = []
        numeric = [TARGET_COL, *EXOG_COLS]
        for city, grp in self.frame.groupby(CITY_COL):
            g = grp.set_index(DATE_COL).reindex(full)
            g[CITY_COL] = city
            for col in g.columns:
                if col in numeric:
                    g[col] = g[col].interpolate("time").ffill().bfill()
                elif col != CITY_COL:
                    g[col] = g[col].ffill().bfill()
            g.index.name = DATE_COL
            pieces.append(g.reset_index())
        out = pd.concat(pieces, ignore_index=True).sort_values([CITY_COL, DATE_COL])
        return ClimatePanel(out.reset_index(drop=True))

    def summary(self) -> pd.DataFrame:
        """Per-city descriptive statistics of the target channel."""
        g = self.frame.groupby(CITY_COL)[TARGET_COL]
        out = pd.DataFrame(
            {
                "n_obs": g.size(),
                "mean": g.mean(),
                "std": g.std(),
                "min": g.min(),
                "p05": g.quantile(0.05),
                "median": g.median(),
                "p95": g.quantile(0.95),
                "max": g.max(),
                "annual_range": g.max() - g.min(),
            }
        )
        return out.round(2).sort_values("mean", ascending=False)


# ---------------------------------------------------------------------- #
# Splitting utilities
# ---------------------------------------------------------------------- #
def train_test_split_by_date(
    df: pd.DataFrame, cutoff: pd.Timestamp, date_col: str = DATE_COL
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split. Everything strictly after ``cutoff`` is test."""
    cutoff = pd.Timestamp(cutoff)
    train = df.loc[df[date_col] <= cutoff].copy()
    test = df.loc[df[date_col] > cutoff].copy()
    return train, test


def rolling_origins(
    dates: pd.DatetimeIndex,
    horizon: int,
    n_origins: int,
    stride: int,
    min_train_days: int,
) -> list[pd.Timestamp]:
    """Compute cutoff dates for rolling-origin (walk-forward) backtesting.

    Origins are laid out backwards from the end of the sample so that the last
    origin leaves exactly ``horizon`` days of data to score against.
    """
    dates = pd.DatetimeIndex(sorted(pd.DatetimeIndex(dates).unique()))
    last_possible = len(dates) - horizon - 1
    if last_possible < min_train_days:
        raise ValueError(
            f"Not enough history: need >= {min_train_days + horizon + 1} days, "
            f"got {len(dates)}."
        )
    idxs = [last_possible - k * stride for k in range(n_origins)]
    idxs = [i for i in idxs if i >= min_train_days]
    if not idxs:
        raise ValueError("No valid backtest origins for the given configuration.")
    return [dates[i] for i in sorted(idxs)]


def load_panel(path: str | Path = RAW_CSV) -> ClimatePanel:
    """Convenience loader: read, validate and calendar-complete the dataset."""
    return ClimatePanel.from_csv(path).fill_calendar()


__all__ = [
    "ClimatePanel",
    "DataValidationError",
    "load_panel",
    "rolling_origins",
    "train_test_split_by_date",
]
