import argparse
import os
import pandas as pd

from src.jitl_gpr.io import load_dataset
from src.jitl_gpr.config import JITLGPRConfig
from src.jitl_gpr.forecast import forecast_recursive

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to processed CSV")
    p.add_argument("--out", required=True, help="Output forecast CSV path")
    p.add_argument("--target", default="log_ngd", choices=["ngd", "log_ngd"])

    # No tuning: you set these manually
    p.add_argument("--Wy", type=int, default=5)
    p.add_argument("--Wm", type=int, default=3)
    p.add_argument("--H", type=int, default=12)

    # Kernel init
    p.add_argument("--sigma_f", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--noise", type=float, default=1e-2)

    args = p.parse_args()

    df = load_dataset(args.data)

    cfg = JITLGPRConfig(
        Wy=args.Wy,
        Wm=args.Wm,
        horizon_months=args.H,
        target_mode=args.target,
        sigma_f=args.sigma_f,
        beta=args.beta,
        alpha=args.alpha,
        noise=args.noise,
        n_restarts_optimizer=2,
    )

    fc = forecast_recursive(df, cfg)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fc.to_csv(args.out, index=False)
    print(f"Wrote forecasts to: {args.out}")

if __name__ == "__main__":
    main()