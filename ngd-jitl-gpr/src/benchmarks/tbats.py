# src/0_benchmarks/tbats.py
from __future__ import annotations
import pandas as pd
from tbats import TBATS

def fit_predict_tbats(train: pd.Series, steps: int):
    estimator = TBATS(
        seasonal_periods=[12],
        use_box_cox=True,
        use_trend=True,
        use_damped_trend=False,
        use_arma_errors=True,
    )
    model = estimator.fit(train.values)
    fc_vals = model.forecast(steps=steps)
    idx = pd.date_range(train.index[-1] + pd.offsets.MonthBegin(1), periods=steps, freq="MS")
    fc = pd.Series(fc_vals, index=idx, name="tbats_fc")
    return model, fc