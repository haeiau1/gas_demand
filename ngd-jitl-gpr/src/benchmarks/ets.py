# src/0_benchmarks/ets.py
from __future__ import annotations
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def fit_predict_ets(train: pd.Series, steps: int):
    model = ExponentialSmoothing(
        train,
        trend="add",
        seasonal="add",
        seasonal_periods=12,
        initialization_method="estimated",
    ).fit(optimized=True)

    fc = model.forecast(steps)
    return model, fc