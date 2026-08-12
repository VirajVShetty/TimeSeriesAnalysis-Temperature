"""Sequence-to-vector deep learning forecaster (LSTM / GRU with city embeddings).

Design notes
------------
* **Target = anomaly, not level.** The network predicts the departure from the
  day-of-year climatology. Removing the dominant deterministic annual cycle is
  what makes a neural net trainable on ~7k daily observations.
* **Global model with city embeddings.** All cities share one encoder; a small
  learned embedding lets the model specialise per location.
* **Direct multi-horizon head.** A single forward pass emits all ``H`` days at
  once, avoiding recursive error accumulation.
* Deterministic seeding plus early stopping on a chronological validation tail
  keeps results reproducible and honest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import CITY_COL, DATE_COL, EXOG_COLS, MODELS, RANDOM_SEED, TARGET_COL
from ..features import DayOfYearClimatology, fourier_terms
from .base import BaseForecaster

try:  # torch is an optional heavy dependency
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    TORCH_AVAILABLE = False


class _SeqEncoder(nn.Module if TORCH_AVAILABLE else object):  # type: ignore[misc]
    """LSTM/GRU encoder + city embedding + direct multi-horizon head."""

    def __init__(
        self,
        n_features: int,
        n_cities: int,
        horizon: int,
        hidden: int = 96,
        layers: int = 2,
        dropout: float = 0.15,
        embed_dim: int = 8,
        cell: str = "lstm",
    ) -> None:
        super().__init__()
        rnn_cls = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.embed = nn.Embedding(n_cities, embed_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden + embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, horizon),
        )

    def forward(self, x, city_idx):
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        z = torch.cat([last, self.embed(city_idx)], dim=1)
        return self.head(z)


class LSTMForecaster(BaseForecaster):
    """Global recurrent network over the multi-city climate panel."""

    name = "LSTM (global, direct)"

    def __init__(
        self,
        horizon: int,
        lookback: int = MODELS.lstm_lookback,
        hidden: int = MODELS.lstm_hidden,
        layers: int = MODELS.lstm_layers,
        dropout: float = MODELS.lstm_dropout,
        embed_dim: int = MODELS.lstm_embed_dim,
        epochs: int = MODELS.lstm_epochs,
        batch_size: int = MODELS.lstm_batch_size,
        lr: float = MODELS.lstm_lr,
        patience: int = MODELS.lstm_patience,
        cell: str = "lstm",
        val_fraction: float = 0.15,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for LSTMForecaster. Install with `pip install torch`."
            )
        self.horizon = horizon
        self.lookback = lookback
        self.hidden = hidden
        self.layers = layers
        self.dropout = dropout
        self.embed_dim = embed_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.cell = cell
        self.val_fraction = val_fraction
        self.verbose = verbose
        self.history_: list[dict] = []

    # ------------------------------------------------------------------ #
    # Tensor construction
    # ------------------------------------------------------------------ #
    def _channel_frame(self, g: pd.DataFrame, clim: np.ndarray) -> np.ndarray:
        """Per-timestep input channels for one city (already anomaly-centred)."""
        anomaly = g[TARGET_COL].to_numpy(dtype=float) - clim
        cols = [anomaly]
        for col in EXOG_COLS:
            if col in g.columns:
                cols.append(g[col].to_numpy(dtype=float))
        fou = fourier_terms(pd.DatetimeIndex(g[DATE_COL]), order=2).to_numpy()
        return np.column_stack(cols + [fou])

    def _prepare(self, panel: pd.DataFrame):
        self.clim_ = DayOfYearClimatology().fit(panel)
        self.city_index_ = {c: i for i, c in enumerate(sorted(panel[CITY_COL].unique()))}

        raw_blocks = {}
        for city, grp in panel.groupby(CITY_COL):
            g = grp.sort_values(DATE_COL).reset_index(drop=True)
            clim = self.clim_.transform(g[CITY_COL], g[DATE_COL]).to_numpy(dtype=float)
            raw_blocks[city] = {
                "channels": self._channel_frame(g, clim),
                "anomaly": g[TARGET_COL].to_numpy(dtype=float) - clim,
                "dates": pd.DatetimeIndex(g[DATE_COL]),
            }

        stacked = np.concatenate([b["channels"] for b in raw_blocks.values()], axis=0)
        self.mu_ = np.nanmean(stacked, axis=0)
        self.sigma_ = np.nanstd(stacked, axis=0)
        self.sigma_[self.sigma_ < 1e-6] = 1.0
        self.target_sigma_ = float(
            np.nanstd(np.concatenate([b["anomaly"] for b in raw_blocks.values()]))
        ) or 1.0

        for b in raw_blocks.values():
            b["channels"] = (b["channels"] - self.mu_) / self.sigma_
            b["channels"] = np.nan_to_num(b["channels"])
        self.blocks_ = raw_blocks

    def _windows(self):
        X, Y, C = [], [], []
        for city, b in self.blocks_.items():
            ch, an = b["channels"], b["anomaly"]
            n = len(ch)
            for t in range(self.lookback - 1, n - self.horizon):
                X.append(ch[t - self.lookback + 1 : t + 1])
                Y.append(an[t + 1 : t + 1 + self.horizon] / self.target_sigma_)
                C.append(self.city_index_[city])
        if not X:
            raise ValueError(
                f"Not enough history for lookback={self.lookback} and horizon={self.horizon}."
            )
        return (
            np.asarray(X, dtype=np.float32),
            np.asarray(Y, dtype=np.float32),
            np.asarray(C, dtype=np.int64),
        )

    # ------------------------------------------------------------------ #
    def _fit(self, train_panel: pd.DataFrame) -> None:
        torch.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

        self._prepare(train_panel)
        X, Y, C = self._windows()

        # Chronological split: windows are ordered by city then time, so split
        # inside each city's block to keep validation strictly "later".
        n_val = max(1, int(len(X) * self.val_fraction))
        order = np.arange(len(X))
        val_idx = order[-n_val:]
        train_idx = order[:-n_val]

        device = torch.device("cpu")
        self.device_ = device
        self.model_ = _SeqEncoder(
            n_features=X.shape[2],
            n_cities=len(self.city_index_),
            horizon=self.horizon,
            hidden=self.hidden,
            layers=self.layers,
            dropout=self.dropout,
            embed_dim=self.embed_dim,
            cell=self.cell,
        ).to(device)

        ds_tr = TensorDataset(
            torch.from_numpy(X[train_idx]),
            torch.from_numpy(C[train_idx]),
            torch.from_numpy(Y[train_idx]),
        )
        ds_va = TensorDataset(
            torch.from_numpy(X[val_idx]),
            torch.from_numpy(C[val_idx]),
            torch.from_numpy(Y[val_idx]),
        )
        dl_tr = DataLoader(ds_tr, batch_size=self.batch_size, shuffle=True, drop_last=False)
        dl_va = DataLoader(ds_va, batch_size=512, shuffle=False)

        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
        loss_fn = nn.HuberLoss(delta=1.0)

        best, best_state, bad = float("inf"), None, 0
        for epoch in range(self.epochs):
            self.model_.train()
            tr_loss = 0.0
            for xb, cb, yb in dl_tr:
                opt.zero_grad()
                loss = loss_fn(self.model_(xb, cb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                opt.step()
                tr_loss += loss.item() * len(xb)
            tr_loss /= max(1, len(ds_tr))

            self.model_.eval()
            va_loss = 0.0
            with torch.no_grad():
                for xb, cb, yb in dl_va:
                    va_loss += loss_fn(self.model_(xb, cb), yb).item() * len(xb)
            va_loss /= max(1, len(ds_va))
            sched.step(va_loss)
            self.history_.append({"epoch": epoch, "train": tr_loss, "val": va_loss})
            if self.verbose:
                print(f"epoch {epoch:03d}  train={tr_loss:.4f}  val={va_loss:.4f}")

            if va_loss < best - 1e-5:
                best, bad = va_loss, 0
                best_state = {k: v.clone() for k, v in self.model_.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.model_.eval()

    # ------------------------------------------------------------------ #
    def _predict(self, horizon: int) -> pd.DataFrame:
        if horizon > self.horizon:
            raise ValueError(
                f"{self.name} was trained for horizon={self.horizon}, got {horizon}."
            )
        rows = []
        with torch.no_grad():
            for city, b in self.blocks_.items():
                window = b["channels"][-self.lookback :]
                if len(window) < self.lookback:
                    pad = np.repeat(window[:1], self.lookback - len(window), axis=0)
                    window = np.vstack([pad, window])
                xb = torch.from_numpy(window[None, ...].astype(np.float32))
                cb = torch.tensor([self.city_index_[city]], dtype=torch.long)
                anom = self.model_(xb, cb).numpy().ravel() * self.target_sigma_
                for h in range(1, horizon + 1):
                    target_date = self.origin_ + pd.Timedelta(days=h)
                    base = float(
                        self.clim_.transform(
                            pd.Series([city]), pd.Series([target_date])
                        ).iloc[0]
                    )
                    rows.append(
                        {
                            CITY_COL: city,
                            "horizon": h,
                            "y_pred": float(base + anom[h - 1]),
                        }
                    )
        return pd.DataFrame(rows)

    def training_curve(self) -> pd.DataFrame:
        return pd.DataFrame(self.history_)


class GRUForecaster(LSTMForecaster):
    """Same architecture with GRU cells — cheaper, often equally accurate."""

    name = "GRU (global, direct)"

    def __init__(self, horizon: int, **kw) -> None:
        kw.setdefault("cell", "gru")
        super().__init__(horizon=horizon, **kw)


__all__ = ["GRUForecaster", "LSTMForecaster", "TORCH_AVAILABLE"]
