# src/0_benchmarks/after.py
from __future__ import annotations
import numpy as np
import pandas as pd

def _normalize(w: np.ndarray) -> np.ndarray:
    w = np.maximum(w, 1e-12)
    return w / w.sum()

def after_fit_weights(
    y: pd.Series,
    base_one_step_preds: dict[str, pd.Series],
    burn_in: int = 35,
    eta: float = 0.5,
) -> pd.Series:
    names = list(base_one_step_preds.keys())
    w = np.ones(len(names)) / len(names)

    idx = y.index
    for t in range(burn_in, len(idx)):
        yt = y.iloc[t]
        preds_t = np.array([base_one_step_preds[n].iloc[t] for n in names], dtype=float)
        err2 = (yt - preds_t) ** 2
        w = _normalize(w * np.exp(-eta * err2))
    return pd.Series(w, index=names, name="after_weights")

def after_forecast(
    base_forecasts: dict[str, pd.Series],
    weights: pd.Series,
) -> pd.Series:
    names = list(base_forecasts.keys())
    w = weights.reindex(names).fillna(0.0).values.astype(float)
    w = w / w.sum()

    df = pd.concat([base_forecasts[n].rename(n) for n in names], axis=1)
    fc = df.values @ w
    return pd.Series(fc, index=df.index, name="after_fc")