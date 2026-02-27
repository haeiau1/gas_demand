# src/0_benchmarks/sarima.py
from __future__ import annotations
import pandas as pd
import pmdarima as pm

def fit_predict_sarima(train: pd.Series, steps: int, seasonal_period: int = 12):
    model = pm.auto_arima(
        train,
        seasonal=True,
        m=seasonal_period,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        max_p=3, max_q=3, max_P=3, max_Q=3,
        d=None, D=None,
        information_criterion="aic",
    )
    fc = model.predict(n_periods=steps)
    idx = pd.date_range(train.index[-1] + pd.offsets.MonthBegin(1), periods=steps, freq="MS")
    fc = pd.Series(fc, index=idx, name="sarima_fc")
    return model, fc