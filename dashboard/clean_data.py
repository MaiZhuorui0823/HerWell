#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_RAW = BASE_DIR / "data_raw"
DATA_OUT = BASE_DIR / "data_processed"

# 简洁的数据清洗
tables = [
    ("resting_heart_rate.csv", "first"),
    ("sleep_score.csv", "first"),
    ("stress_score.csv", "first"),
    ("steps.csv", "sum"),
    ("active_minutes.csv", "sum"),
    ("hormones_and_selfreport.csv", "first"),
]

print("开始data处理...")
dfs = []

for fname, agg_method in tables:
    fpath = DATA_RAW / fname
    if not fpath.exists():
        continue

    df = pd.read_csv(fpath)
    print(f"✓ {fname}: {len(df)} rows")

    # 删除问题列
    cols_drop = [c for c in ["study_interval",
                             "is_weekend", "timestamp"] if c in df.columns]
    df = df.drop(columns=cols_drop, errors='ignore')

    # 聚合
    metric_cols = [c for c in df.columns if c not in ["id", "day_in_study"]]
    if agg_method == "sum":
        df = df.groupby(["id", "day_in_study"])[
            metric_cols].sum().reset_index()
    else:
        df = df.groupby(["id", "day_in_study"])[
            metric_cols].first().reset_index()

    print(f"  → {len(df)} rows after agg")
    dfs.append(df)

# 合并
print("\n合并...")
result = dfs[0]
for i, df in enumerate(dfs[1:], 1):
    # 删除重复列
    dup_cols = [c for c in df.columns if c in result.columns and c not in [
        "id", "day_in_study"]]
    df = df.drop(columns=dup_cols, errors='ignore')

    result = result.merge(df, on=["id", "day_in_study"], how="outer")
    print(f"  merge {i+1}: {result.shape[1]} cols")

result = result.sort_values(["id", "day_in_study"]).reset_index(drop=True)

# 保存
out = DATA_OUT / "dashboard_ready.csv"
result.to_csv(out, index=False)
print(f"\n✓ 完成: {result.shape[0]} rows × {result.shape[1]} cols")
print(f"  保存到: {out}")
