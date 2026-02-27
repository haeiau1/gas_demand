import argparse
import os
import pandas as pd

from src.jitl_gpr.io import load_dataset
from src.jitl_gpr.config import JITLGPRConfig
from src.jitl_gpr.forecast import (
    split_train_test_by_year,
    forecast_one_year_recursive,
    summarize_metrics
)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out_details", required=True)
    p.add_argument("--out_metrics", required=True)

    p.add_argument("--test_year", type=int, default=2025)
    p.add_argument("--target", default="log_ngd", choices=["ngd", "log_ngd"])

    # No tuning: manual set
    p.add_argument("--Wy", type=int, default=5)
    p.add_argument("--Wm", type=int, default=3)

    # Kernel init
    p.add_argument("--sigma_f", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--noise", type=float, default=1e-2)

    args = p.parse_args()

    df = load_dataset(args.data)
    df_train, df_test = split_train_test_by_year(df, test_year=args.test_year)

    cfg = JITLGPRConfig(
        Wy=args.Wy,
        Wm=args.Wm,
        target_mode=args.target,
        sigma_f=args.sigma_f,
        beta=args.beta,
        alpha=args.alpha,
        noise=args.noise,
        n_restarts_optimizer=2
    )

    res = forecast_one_year_recursive(df_train, df_test, cfg, test_year=args.test_year)
    metrics = summarize_metrics(res)

    os.makedirs(os.path.dirname(args.out_details), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_metrics), exist_ok=True)

    res.to_csv(args.out_details, index=False)
    metrics.to_csv(args.out_metrics, index=False)

    print(f"Wrote details: {args.out_details}")
    print(f"Wrote metrics: {args.out_metrics}")
    print(metrics.to_string(index=False))

if __name__ == "__main__":
    main()