# src/0_benchmarks/stl.py
from __future__ import annotations
import numpy as np
import pandas as pd
from statsmodels.tsa.forecasting.stl import STLForecast
from statsmodels.tsa.arima.model import ARIMA

def fit_predict_stl(train: pd.Series, steps: int):
    stlf = STLForecast(
        train,
        model=ARIMA,
        model_kwargs={"order": (0, 0, 0)},
        period=12,
        robust=True,
        seasonal=13,  
    )
    res = stlf.fit()
    fc = res.forecast(steps=steps)
    return res, fc