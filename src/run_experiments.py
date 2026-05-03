"""
Main experiment runner.

For every dataset sample and prompt style:
- build prompt
- measure latency
- measure total runtime
- log power samples
- compute energy
- save one CSV row
"""

import csv
import traceback
from pathlib import Path

import pandas as pd

from config import (
    DEVICE_NAME,
    MODEL_NAME,
    PROMPT_STYLES,
    RAW_RUNS_CSV,
    SAMPLES_DIR,
)
from energy_logger import TegraStatsLogger
from model_backend import HFBackend
from prompt_templates import build_prompt
from utils import make_run_id, now_ts, safe_write_header_if_needed

# CSV columns for raw experimental runs

HEADER = [
    "run_id",
    "timestamp",
    "device_name",
    "model_name",
    "dataset_name",
    "dataset_year",
    "task_type",
    "sample_id",
    "prompt_style",
    "prompt_template_version",
    "input_text",
    "reference_output",
    "input_tokens",
    "output_text",
    "output_tokens",
    "latency_sec",
    "total_runtime_sec",
    "energy_joules",
    "avg_power_watts",
    "status",
    "error_message",
]


def load_sample_files() -> list[Path]:
    """
    Return all sampled dataset CSV files.
    """
    # Load all CSVs created during dataset sampling
    
    return sorted(SAMPLES_DIR.glob("*_sampled.csv"))


def existing_run_ids(csv_path: Path) -> set[str]:
    """
    Load existing run IDs so we can resume without duplicating runs.
    """
    # Return empty set if raw results file does not exist yet
    
    if not csv_path.exists():
        return set()

    # Read only run_id column for faster resume checking
    
    df = pd.read_csv(csv_path, usecols=["run_id"])
    return set(df["run_id"].astype(str).tolist())


def append_row(csv_path: Path, row: dict) -> None:
    """
    Append one row to the raw runs CSV.
    """
    # Write CSV header only once
    
    safe_write_header_if_needed(csv_path, HEADER)

    # Append result row
    
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        writer.writerow(row)


def main() -> None:
    """
    Run all experiments across:
    - sampled dataset rows
    - 4 prompt styles
    """
    # Load model backend and power logger
    
    backend = HFBackend()
    power_logger = TegraStatsLogger()

    # Track previously completed runs for resume support
    
    seen = existing_run_ids(RAW_RUNS_CSV)

    # Loop through each sampled dataset file
    
    for sample_csv in load_sample_files():
        df = pd.read_csv(sample_csv)

        # Loop through each dataset row and prompt style
        
        for _, r in df.iterrows():
            for prompt_style in PROMPT_STYLES:
                
                # Deterministic ID so reruns do not duplicate work
                
                key = {
                    "dataset_name": r["dataset_name"],
                    "sample_id": int(r["sample_id"]),
                    "prompt_style": prompt_style,
                    "model_name": MODEL_NAME,
                    "device_name": DEVICE_NAME,
                }
                run_id = make_run_id(key)

                # Skip already completed/saved run IDs
                
                if run_id in seen:
                    continue

                # Build final prompt for selected task and style
                
                prompt = build_prompt(
                    task_type=r["task_type"],
                    prompt_style=prompt_style,
                    input_text=r["input_text"],
                )

                # Base row structure
                
                row = {
                    "run_id": run_id,
                    "timestamp": now_ts(),
                    "device_name": DEVICE_NAME,
                    "model_name": MODEL_NAME,
                    "dataset_name": r["dataset_name"],
                    "dataset_year": r["dataset_year"],
                    "task_type": r["task_type"],
                    "sample_id": int(r["sample_id"]),
                    "prompt_style": prompt_style,
                    "prompt_template_version": "v1",
                    "input_text": r["input_text"],
                    "reference_output": r["reference_output"],
                    "input_tokens": None,
                    "output_text": "",
                    "output_tokens": None,
                    "latency_sec": None,
                    "total_runtime_sec": None,
                    "energy_joules": None,
                    "avg_power_watts": None,
                    "status": "failed",
                    "error_message": "",
                }

                try:
                    # Measure total pipeline time
                    
                    total_start = now_ts()

                    # Start power logger before inference
                    
                    power_logger.start()

                    # Measure model inference latency
                    
                    infer_start = now_ts()
                    result = backend.generate(prompt)
                    infer_end = now_ts()

                    # Stop power logger after inference
                    
                    power_logger.stop()

                    # End total runtime timer
                    
                    total_end = now_ts()

                    # Calculate runtime metrics
                    
                    latency_sec = infer_end - infer_start
                    total_runtime_sec = total_end - total_start

                    # Calculate total energy from logged power samples
                    
                    energy_joules = power_logger.energy_joules(total_start, total_end)
                    
                    # Calculate average power across the run
                    
                    avg_power_watts = (
                        energy_joules / total_runtime_sec if total_runtime_sec > 0 else 0.0
                    )

                    # Store successful run metrics
                    
                    row.update({
                        "input_tokens": result.input_tokens,
                        "output_text": result.output_text,
                        "output_tokens": result.output_tokens,
                        "latency_sec": latency_sec,
                        "total_runtime_sec": total_runtime_sec,
                        "energy_joules": energy_joules,
                        "avg_power_watts": avg_power_watts,
                        "status": "success",
                    })

                except Exception as e:
                     # Stop power logger safely if run fails
                    try:
                        power_logger.stop()
                    except Exception:
                        pass

                    # Save error details for debugging
                    
                    row["error_message"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

                # Save run result to CSV
                
                append_row(RAW_RUNS_CSV, row)
                seen.add(run_id)

                print(
                    f"Saved run {run_id} | "
                    f"{row['dataset_name']} | {prompt_style} | {row['status']}"
                )


if __name__ == "__main__":
    main()
