"""Split-conformal prediction intervals.

Distribution-free uncertainty quantification: given a calibration set that the
model never trained on, the empirical quantile of absolute residuals yields
intervals with finite-sample marginal coverage guarantees — no Gaussian or
homoscedasticity assumption required.

Because forecast uncertainty grows with lead time, quantiles are estimated
*per horizon* rather than pooled.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class SplitConformal:
    """Per-horizon split-conformal calibrator.

    Parameters
    ----------
    coverage:
        Target marginal coverage, e.g. 0.9 for a 90% interval.
    """

    coverage: float = 0.9
    quantiles_: dict[int, float] = field(default_factory=dict)
    global_quantile_: float = float("nan")

    @property
    def alpha(self) -> float:
        return 1 - self.coverage

    def fit(
        self,
        residuals: np.ndarray | pd.Series,
        horizons: np.ndarray | pd.Series | None = None,
    ) -> SplitConformal:
        r = np.abs(np.asarray(residuals, dtype=float))
        r = r[np.isfinite(r)]
        if len(r) == 0:
            raise ValueError("No finite residuals supplied for calibration.")
        self.global_quantile_ = float(_conformal_quantile(r, self.alpha))

        if horizons is not None:
            h = np.asarray(horizons)
            finite = np.isfinite(np.asarray(residuals, dtype=float))
            h = h[finite]
            r_all = np.abs(np.asarray(residuals, dtype=float))[finite]
            for hz in np.unique(h):
                sub = r_all[h == hz]
                # Fall back to the pooled quantile when a horizon is thin.
                if len(sub) >= max(10, int(np.ceil(1 / self.alpha))):
                    self.quantiles_[int(hz)] = float(
                        _conformal_quantile(sub, self.alpha)
                    )
                else:
                    self.quantiles_[int(hz)] = self.global_quantile_
        return self

    def predict_interval(
        self, y_pred: np.ndarray | pd.Series, horizons: np.ndarray | pd.Series | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        y_pred = np.asarray(y_pred, dtype=float)
        if horizons is None or not self.quantiles_:
            q = np.full(len(y_pred), self.global_quantile_)
        else:
            h = np.asarray(horizons)
            q = np.array(
                [self.quantiles_.get(int(x), self.global_quantile_) for x in h],
                dtype=float,
            )
        return y_pred - q, y_pred + q

    def width_table(self) -> pd.DataFrame:
        if not self.quantiles_:
            return pd.DataFrame(
                {"horizon": ["all"], "half_width": [self.global_quantile_]}
            )
        return (
            pd.DataFrame(
                {
                    "horizon": list(self.quantiles_),
                    "half_width": [round(v, 3) for v in self.quantiles_.values()],
                }
            )
            .sort_values("horizon")
            .reset_index(drop=True)
        )


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """The finite-sample-corrected ``(1-alpha)`` quantile of conformity scores."""
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


__all__ = ["SplitConformal"]
