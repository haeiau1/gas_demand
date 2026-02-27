from __future__ import annotations
import pandas as pd

from src.common import load_series, train_test_split_by_date
from src.benchmarks.stl import fit_predict_stl
from src.benchmarks.sarima import fit_predict_sarima
from src.benchmarks.ets import fit_predict_ets
from src.benchmarks.tbats import fit_predict_tbats

CSV_PATH = "/Users/hayret/Desktop/gas_demand/ngd-jitl-gpr/data/processed/turkiye_ngd_2016_2023_model_ready.csv"
TRAIN_END = "2024-12-01"  
Y_COL = "log_ngd"

def main():
    s = load_series(CSV_PATH, y_col=Y_COL)
    train, test = train_test_split_by_date(s, train_end=TRAIN_END)
    steps = len(test)

    # --- base models ---
    _, fc_stl = fit_predict_stl(train, steps)
    _, fc_sar = fit_predict_sarima(train, steps)
    _, fc_ets = fit_predict_ets(train, steps)
    _, fc_tb  = fit_predict_tbats(train, steps)

    # --- results ---
    out = pd.DataFrame({
        "y_true": test,
        "STL": fc_stl.reindex(test.index),
        "SARIMA": fc_sar.reindex(test.index),
        "ETS": fc_ets.reindex(test.index),
        "TBATS": fc_tb.reindex(test.index),
    })

    out_path = "/Users/hayret/Desktop/gas_demand/ngd-jitl-gpr/data/processed/benchmarks_test_forecasts.csv"
    out.to_csv(out_path, index=True)
    print(f"Saved forecasts -> {out_path}")

if __name__ == "__main__":
    main()