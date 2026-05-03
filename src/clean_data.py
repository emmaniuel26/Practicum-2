"""
Clean raw experimental runs into a usable modeling dataset.

Cleaning removes:
- failed runs
- missing important metrics
- invalid values
- duplicate run IDs
"""

import pandas as pd

from config import RAW_RUNS_CSV, CLEAN_RUNS_CSV


def main() -> None:
    """
    Load raw data, apply cleaning steps,
    save cleaned dataset, and print summary statistics.
    """
    # Load raw experimental results
    
    df = pd.read_csv(RAW_RUNS_CSV)
    raw_count = len(df)

    # Remove failed runs
    
    failed_removed = df["status"].ne("success").sum()
    df = df[df["status"] == "success"].copy()

    # Remove rows with missing critical runtime metrics
    
    missing_metrics_mask = (
        df["input_tokens"].isna()
        | df["output_tokens"].isna()
        | df["latency_sec"].isna()
        | df["total_runtime_sec"].isna()
        | df["energy_joules"].isna()
    )

    missing_removed = int(missing_metrics_mask.sum())
    df = df[~missing_metrics_mask].copy()

    # Remove rows containing invalid numerical values
    
    invalid_mask = (
        (df["input_tokens"] <= 0)
        | (df["output_tokens"] < 0)
        | (df["latency_sec"] <= 0)
        | (df["total_runtime_sec"] <= 0)
        | (df["energy_joules"] < 0)
    )

    invalid_removed = int(invalid_mask.sum())
    df = df[~invalid_mask].copy()

    # Remove duplicate run IDs
    
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["run_id"])
    duplicate_removed = before_dedup - len(df)

    # Standardize categorical labels for consistency
    
    df["prompt_style"] = (
        df["prompt_style"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["task_type"] = (
        df["task_type"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["dataset_name"] = (
        df["dataset_name"]
        .astype(str)
        .str.strip()
    )

    # Save cleaned dataset
    
    df.to_csv(CLEAN_RUNS_CSV, index=False)

    # Print cleaning summary
    
    print("=== CLEANING SUMMARY ===")
    print(f"Raw rows: {raw_count}")
    print(f"Failed rows removed: {failed_removed}")
    print(f"Missing metric rows removed: {missing_removed}")
    print(f"Invalid metric rows removed: {invalid_removed}")
    print(f"Duplicate rows removed: {duplicate_removed}")
    print(f"Final cleaned rows: {len(df)}")
    print(f"Saved: {CLEAN_RUNS_CSV}")


if __name__ == "__main__":
    main()