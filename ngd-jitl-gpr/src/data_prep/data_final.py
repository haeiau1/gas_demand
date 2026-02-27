import os
import numpy as np
import pandas as pd

INPUT_PATH = r"/Users/hayret/Desktop/gas_demand/ngd-jitl-gpr/data/processed/turkiye_ngd_2016_2023.csv"

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "year", "month", "ngd"}
    missing = required - set(df.columns.str.lower())
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Expected at least: {required}")

    rename_map = {c: c.lower() for c in df.columns}
    df = df.rename(columns=rename_map)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        bad = df.loc[df["date"].isna()].head(10)
        raise ValueError(
            "Some 'date' values could not be parsed. Example rows:\n"
            f"{bad.to_string(index=False)}"
        )

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["month"] = pd.to_numeric(df["month"], errors="coerce").astype("Int64")
    df["ngd"] = pd.to_numeric(df["ngd"], errors="coerce")

    if df["ngd"].isna().any():
        bad = df.loc[df["ngd"].isna()].head(10)
        raise ValueError(
            "Some 'ngd' values are not numeric. Example rows:\n"
            f"{bad.to_string(index=False)}"
        )

    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)

    df["t"] = np.arange(1, len(df) + 1, dtype=int)

    if (df["ngd"] <= 0).any():
        bad = df.loc[df["ngd"] <= 0, ["date", "ngd"]].head(10)
        raise ValueError(
            "Found non-positive ngd values; cannot take log. Example rows:\n"
            f"{bad.to_string(index=False)}"
        )
    df["log_ngd"] = np.log(df["ngd"])

    # --- Month cyclic encoding ---
    # month is 1..12
    m = df["month"].astype(int)
    df["month_sin"] = np.sin(2 * np.pi * m / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * m / 12.0)

    # --- Winter flag (Turkey heating season; tweak if you want) ---
    df["is_winter"] = m.isin([11, 12, 1, 2, 3]).astype(int)

    # --- Lags (use log_ngd for stability; also keep raw lags if you want) ---
    df["lag1_log"] = df["log_ngd"].shift(1)
    df["lag12_log"] = df["log_ngd"].shift(12)

    df["lag1"] = df["ngd"].shift(1)
    df["lag12"] = df["ngd"].shift(12)

    # --- Rolling means (shifted to avoid leakage) ---
    df["roll3_log"] = df["log_ngd"].shift(1).rolling(window=3, min_periods=3).mean()
    df["roll12_log"] = df["log_ngd"].shift(1).rolling(window=12, min_periods=12).mean()

    df["roll3"] = df["ngd"].shift(1).rolling(window=3, min_periods=3).mean()
    df["roll12"] = df["ngd"].shift(1).rolling(window=12, min_periods=12).mean()

    # Optional: month & year sanity check vs date
    # If your year/month columns sometimes mismatch, uncomment to override them:
    # df["year"] = df["date"].dt.year
    # df["month"] = df["date"].dt.month

    return df

def main():
    df = pd.read_csv(INPUT_PATH)
    out = build_features(df)

    base, ext = os.path.splitext(INPUT_PATH)
    output_path = f"{base}_model_ready.csv"


    out.to_csv(output_path, index=False)

    dropped = out.dropna(subset=["lag1_log", "lag12_log", "roll3_log", "roll12_log"]).reset_index(drop=True)
    output_path_dropped = f"{base}_model_ready_droppedNA.csv"
    dropped.to_csv(output_path_dropped, index=False)

    print("Wrote:")
    print(" -", output_path)
    print(" -", output_path_dropped)

if __name__ == "__main__":
    main()