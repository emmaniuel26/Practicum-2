"""
Create before-cleaning and after-cleaning audit outputs.

Outputs:
- raw_preview.csv
- clean_preview.csv
- cleaning_summary.csv
- cleaning_counts.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from config import RAW_RUNS_CSV, CLEAN_RUNS_CSV, OUTPUTS_DIR, PLOTS_DIR


# Directory for audit output files

AUDIT_DIR = OUTPUTS_DIR / "audit"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)


def save_preview(df: pd.DataFrame, out_path: Path, n: int = 10) -> None:
    """
    Save the first n rows of a DataFrame for quick review.
    """
    # Save a small preview file for inspection
    
    df.head(n).to_csv(out_path, index=False)


def build_cleaning_summary(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reproduce cleaning logic and count removed rows at each stage.
    """
    # Count total raw rows before cleaning
    
    raw_count = len(raw_df)

    # Step 1: count and remove failed runs
    
    failed_mask = raw_df["status"] != "success"
    failed_removed = int(failed_mask.sum())
    df = raw_df[~failed_mask].copy()

    # Step 2: count and remove rows with missing key metrics
    
    missing_metrics_mask = (
        df["input_tokens"].isna()
        | df["output_tokens"].isna()
        | df["latency_sec"].isna()
        | df["total_runtime_sec"].isna()
        | df["energy_joules"].isna()
    )
    missing_removed = int(missing_metrics_mask.sum())
    df = df[~missing_metrics_mask].copy()

    # Step 3: count and remove invalid numerical values
    
    invalid_mask = (
        (df["input_tokens"] <= 0)
        | (df["output_tokens"] < 0)
        | (df["latency_sec"] <= 0)
        | (df["total_runtime_sec"] <= 0)
        | (df["energy_joules"] < 0)
    )
    invalid_removed = int(invalid_mask.sum())
    df = df[~invalid_mask].copy()

    # Step 4: count and remove duplicate run IDs
    
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["run_id"])
    duplicate_removed = before_dedup - len(df)

    # Count final cleaned rows
    
    final_cleaned = len(df)

    # Store audit summary as a DataFrame
    
    summary = pd.DataFrame([
        {"stage": "Raw collected", "rows": raw_count},
        {"stage": "Failed runs removed", "rows": failed_removed},
        {"stage": "Missing metric rows removed", "rows": missing_removed},
        {"stage": "Invalid metric rows removed", "rows": invalid_removed},
        {"stage": "Duplicate rows removed", "rows": duplicate_removed},
        {"stage": "Final cleaned rows", "rows": final_cleaned},
    ])

    return summary


def plot_cleaning_summary(summary_df: pd.DataFrame, out_path: Path) -> None:
    """
    Create and save a simple cleaning summary bar chart.
    """
    # Plot row counts for each cleaning stage
    
    plt.figure(figsize=(8, 4))
    plt.bar(summary_df["stage"], summary_df["rows"])

    # Improve label readability
    
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Row Count")
    plt.title("Data Cleaning Summary")

    # Save plot
    
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    """
    Generate audit artifacts for raw and cleaned data.
    """
    # Load raw and cleaned datasets
    
    raw_df = pd.read_csv(RAW_RUNS_CSV)
    clean_df = pd.read_csv(CLEAN_RUNS_CSV)

    # Save preview files
    
    save_preview(raw_df, AUDIT_DIR / "raw_preview.csv", n=10)
    save_preview(clean_df, AUDIT_DIR / "clean_preview.csv", n=10)

    # Build and save cleaning summary
    
    summary_df = build_cleaning_summary(raw_df)
    summary_df.to_csv(AUDIT_DIR / "cleaning_summary.csv", index=False)

    # Create cleaning summary plot
    
    plot_cleaning_summary(summary_df, PLOTS_DIR / "cleaning_counts.png")

    # Print output paths
    
    print("=== DATA AUDIT OUTPUTS CREATED ===")
    print(AUDIT_DIR / "raw_preview.csv")
    print(AUDIT_DIR / "clean_preview.csv")
    print(AUDIT_DIR / "cleaning_summary.csv")
    print(PLOTS_DIR / "cleaning_counts.png")


if __name__ == "__main__":
    main()