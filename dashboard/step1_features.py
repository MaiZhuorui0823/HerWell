import pandas as pd
import numpy as np
from pathlib import Path
xi  # !/usr/bin/env python3
"""Step 1: build MVP feature list for dashboard cards."""


BASE_DIR = Path(__file__).parent
INPUT_FILE = BASE_DIR / "data_processed" / "dashboard_ready.csv"
OUTPUT_FILE = BASE_DIR / "data_processed" / "dashboard_kpi.csv"


def to_numeric_score(series: pd.Series) -> pd.Series:
    """Convert mixed symptom text/numeric values into numeric severity."""
    mapping = {
        "not at all": 0,
        "very low/little": 1,
        "low": 2,
        "moderate": 3,
        "high": 4,
        "very high": 5,
    }
    lower = series.astype(str).str.strip().str.lower()
    mapped = lower.map(mapping)
    numeric = pd.to_numeric(series, errors="coerce")
    return mapped.fillna(numeric)


def heavy_flow_count(series: pd.Series) -> int:
    """Count heavy flow days from text labels."""
    lower = series.astype(str).str.strip().str.lower()
    heavy_keywords = ["heavy", "very heavy", "high"]
    return int(lower.apply(lambda x: any(k in x for k in heavy_keywords)).sum())


def build_user_kpis(user_df: pd.DataFrame) -> dict:
    user_df = user_df.sort_values("day_in_study")
    today = user_df["day_in_study"].max()

    d60 = user_df[user_df["day_in_study"] >= today - 60]
    d30 = user_df[user_df["day_in_study"] >= today - 30]
    d14 = user_df[user_df["day_in_study"] >= today - 14]
    d7 = user_df[user_df["day_in_study"] >= today - 7]
    p7 = user_df[user_df["day_in_study"].between(today - 14, today - 7)]

    cramps_30 = to_numeric_score(d30.get("cramps", pd.Series(dtype=float)))
    mood_30 = to_numeric_score(d30.get("moodswing", pd.Series(dtype=float)))

    flow_days = user_df[
        user_df.get("flow_volume", pd.Series(dtype=float)).notna()
        & (user_df.get("flow_volume", pd.Series(dtype=float)).astype(str).str.lower() != "not at all")
    ]

    sleep7 = pd.to_numeric(
        d7.get("overall_score", pd.Series(dtype=float)), errors="coerce")
    sleep_prev7 = pd.to_numeric(
        p7.get("overall_score", pd.Series(dtype=float)), errors="coerce")
    steps7 = pd.to_numeric(
        d7.get("steps", pd.Series(dtype=float)), errors="coerce")
    steps_prev7 = pd.to_numeric(
        p7.get("steps", pd.Series(dtype=float)), errors="coerce")

    return {
        "id": int(user_df["id"].iloc[0]),
        "data_completeness_60d": int(len(d60)),
        "cramps_mean_30d": float(cramps_30.mean()) if len(cramps_30) else np.nan,
        "cramps_max_30d": float(cramps_30.max()) if len(cramps_30) else np.nan,
        "moodswing_mean_30d": float(mood_30.mean()) if len(mood_30) else np.nan,
        "moodswing_std_30d": float(mood_30.std()) if len(mood_30) else np.nan,
        "flow_heavy_count_30d": heavy_flow_count(d30.get("flow_volume", pd.Series(dtype=str))),
        "latest_flow_day": float(flow_days["day_in_study"].max()) if len(flow_days) else np.nan,
        "sleep_score_mean_14d": float(pd.to_numeric(d14.get("overall_score", pd.Series(dtype=float)), errors="coerce").mean()),
        "steps_mean_7d": float(steps7.mean()),
        "active_min_mean_7d": float(pd.to_numeric(d7.get("very", pd.Series(dtype=float)), errors="coerce").mean()),
        "rhr_mean_14d": float(pd.to_numeric(d14.get("resting_heart_rate", pd.Series(dtype=float)), errors="coerce").mean()),
        "stress_mean_14d": float(pd.to_numeric(d14.get("stress_score", pd.Series(dtype=float)), errors="coerce").mean()),
        "sleep_score_delta_7d": float(sleep7.mean() - sleep_prev7.mean()) if len(sleep_prev7) else np.nan,
        "steps_delta_7d": float(steps7.mean() - steps_prev7.mean()) if len(steps_prev7) else np.nan,
    }


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)
    required = {"id", "day_in_study"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    kpis = [build_user_kpis(g) for _, g in df.groupby("id")]
    kpi_df = pd.DataFrame(kpis).sort_values("id").reset_index(drop=True)
    kpi_df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved: {OUTPUT_FILE}")
    print(f"Shape: {kpi_df.shape[0]} users x {kpi_df.shape[1]} columns")
    print("Columns:")
    for c in kpi_df.columns:
        print(f"- {c}")


if __name__ == "__main__":
    main()
