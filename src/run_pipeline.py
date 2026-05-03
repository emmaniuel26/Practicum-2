"""
Master workflow runner for the edge prompt energy project.

Usage examples:
    python run_pipeline.py --stage sample
    python run_pipeline.py --stage run
    python run_pipeline.py --stage clean
    python run_pipeline.py --stage audit
    python run_pipeline.py --stage train
    python run_pipeline.py --stage all

This script allows the full experimental pipeline
to be executed from one central command.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_step(step_name: str, command: list[str]) -> None:
    """
    Run one pipeline step and stop immediately if it fails.
    """
    # Display current stage
    
    print(f"\n=== RUNNING STEP: {step_name} ===")
    print("Command:", " ".join(command))

     # Run subprocess command
    
    result = subprocess.run(command)

     # Stop pipeline if stage fails
    
    if result.returncode != 0:
        print(f"\n❌ Step failed: {step_name}")
        sys.exit(result.returncode)

    # Confirm successful completion
    
    print(f"✅ Step completed: {step_name}")


def main() -> None:
    """
    Parse command-line arguments and run the requested stage(s).
    """
    # Create argument parser
    
    parser = argparse.ArgumentParser(description="Run edge prompt energy pipeline")
    
    # Select pipeline stage
    
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=["sample", "run", "clean", "audit", "train", "all"],
        help="Which stage of the pipeline to run",
    )

    args = parser.parse_args()

     # Run dataset sampling only
    
    if args.stage == "sample":
        run_step("sample datasets", [sys.executable, "dataset_adapters.py"])

     # Run experiment collection only
    
    elif args.stage == "run":
        run_step("run experiments", [sys.executable, "run_experiments.py"])

     # Run cleaning stage only
    
    elif args.stage == "clean":
        run_step("clean data", [sys.executable, "clean_data.py"])

     # Run audit stage only
    
    elif args.stage == "audit":
        run_step("audit data", [sys.executable, "data_audit.py"])

    # Run regression training only
    
    elif args.stage == "train":
        run_step("train models", [sys.executable, "train_models.py"])

    # Run full end-to-end pipeline
    
    elif args.stage == "all":
        run_step("sample datasets", [sys.executable, "dataset_adapters.py"])
        run_step("run experiments", [sys.executable, "run_experiments.py"])
        run_step("clean data", [sys.executable, "clean_data.py"])
        run_step("audit data", [sys.executable, "data_audit.py"])
        run_step("train models", [sys.executable, "train_models.py"])


if __name__ == "__main__":
    main()
