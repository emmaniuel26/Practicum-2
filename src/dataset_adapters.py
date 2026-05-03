"""
Dataset loaders + samplers.

Current working setup:
- REPLIQA -> QA
- MultiHopQA -> reasoning
- CCSum -> summarization
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from datasets import concatenate_datasets, get_dataset_split_names, load_dataset

from config import SAMPLES_DIR, SAMPLES_PER_DATASET, SEED


def _save_df(df: pd.DataFrame, name: str) -> Path:
    """Save sampled DataFrame to CSV."""

    # Create output path for sampled dataset
    
    out = SAMPLES_DIR / f"{name}_sampled.csv"

    # Save sampled rows
    
    df.to_csv(out, index=False)
    return out


def _first_nonempty(ex: dict, keys: list[str], default: str = "") -> str:
    """Return first non-empty value among candidate keys."""

    # Search through possible field names
    
    for k in keys:
        if k in ex and ex[k] is not None:
            value = ex[k]

            # Return non-empty string value
            
            if isinstance(value, str):
                if value.strip():
                    return value
                    
            # Convert non-string values to JSON text
            
            else:
                return json.dumps(value)
    return default


def load_repliqa_sample(n: int = SAMPLES_PER_DATASET) -> pd.DataFrame:
    """Load and sample REPLIQA."""
    dataset_name = "ServiceNow/repliqa"

    # Get available dataset splits
    
    split_names = get_dataset_split_names(dataset_name)
    print(f"[INFO] REPLIQA available splits: {split_names}")

    # Select REPLIQA-specific splits
    
    target_splits = [s for s in split_names if s.startswith("repliqa_")]
    if not target_splits:
        raise RuntimeError(f"No REPLIQA splits found in {dataset_name}. Found: {split_names}")

    # Load and combine selected splits
    
    split_datasets = [load_dataset(dataset_name, split=s) for s in target_splits]
    ds = concatenate_datasets(split_datasets)

    rows = []

    # Shuffle with fixed seed for reproducible sampling
    
    sampled = ds.shuffle(seed=SEED).select(range(min(n, len(ds))))

    for i, ex in enumerate(sampled):
        if i == 0:
            print("[INFO] REPLIQA example keys:", list(ex.keys()))

        # Extract question, context, and answer fields
        
        question = _first_nonempty(ex, ["question", "query", "prompt"], "")
        context = _first_nonempty(
            ex,
            ["document_extracted", "document", "context", "passage", "article", "text"],
            ""
        )
        answer = _first_nonempty(ex, ["answer", "long_answer", "answers", "reference"], "")

        # Combine context and question into one input field
        
        input_text = f"{context}\n\nQuestion: {question}".strip()

        # Store standardized sample row
        
        rows.append({
            "dataset_name": "REPLIQA",
            "dataset_year": "2024-2025",
            "task_type": "qa",
            "sample_id": i,
            "input_text": input_text,
            "reference_output": answer,
        })

    return pd.DataFrame(rows)


def load_multihopqa_sample(n: int = SAMPLES_PER_DATASET) -> pd.DataFrame:
    """Load and sample MultiHopQA."""

    # Load HotpotQA configuration from MultiHopQA
    
    ds = load_dataset("corag/multihopqa", "hotpotqa", split="train")

    rows = []

    # Shuffle with fixed seed for reproducible sampling
    
    sampled = ds.shuffle(seed=SEED).select(range(min(n, len(ds))))

    for i, ex in enumerate(sampled):
        if i == 0:
            print("[INFO] MultiHopQA example keys:", list(ex.keys()))

        # Extract question field
        
        question = _first_nonempty(ex, ["query", "question", "prompt"], "")

        # Extract first available answer
        
        answers = ex.get("answers", [])
        answer = answers[0] if isinstance(answers, list) and len(answers) > 0 else _first_nonempty(
            ex, ["answer", "reference"], ""
        )

        input_text = question.strip()

        # Store standardized sample row
        
        rows.append({
            "dataset_name": "MultiHopQA",
            "dataset_year": "2025",
            "task_type": "reasoning",
            "sample_id": i,
            "input_text": input_text,
            "reference_output": answer,
        })

    return pd.DataFrame(rows)


def load_ccsum_sample(n: int = SAMPLES_PER_DATASET) -> pd.DataFrame:
    """
    Load and sample CCSum.

    IMPORTANT:
    ccsum_summary_only does not provide the full article body.
    So we build input_text from the best available text fields.
    """
    # Load CCSum dataset
    
    ds = load_dataset("ccsum/ccsum_summary_only", split="train")

    rows = []

    # Shuffle with fixed seed for reproducible sampling
    
    sampled = ds.shuffle(seed=SEED).select(range(min(n, len(ds))))

    for i, ex in enumerate(sampled):
        if i == 0:
            print("[INFO] CCSum example keys:", list(ex.keys()))

        # Try multiple fields in order of usefulness
        article = _first_nonempty(
            ex,
            [
                "article",
                "document",
                "text",
                "body",
                "content",
                "article_title",
                "url",
            ],
            ""
        )

        # Extract reference summary
        
        summary = _first_nonempty(
            ex,
            ["summary", "highlights", "reference", "target"],
            ""
        )

        # Store standardized sample row
        
        rows.append({
            "dataset_name": "CCSum",
            "dataset_year": "2024",
            "task_type": "summarization",
            "sample_id": i,
            "input_text": article,
            "reference_output": summary,
        })

    return pd.DataFrame(rows)


def build_all_samples() -> list[Path]:
    """Build and save all sampled datasets."""
    outputs = []

    # Create sampled CSV for each benchmark dataset
    
    outputs.append(_save_df(load_repliqa_sample(), "REPLIQA"))
    outputs.append(_save_df(load_multihopqa_sample(), "MultiHopQA"))
    outputs.append(_save_df(load_ccsum_sample(), "CCSum"))
    return outputs


if __name__ == "__main__":
    
    # Run dataset sampling when script is executed directly
    
    paths = build_all_samples()

    # Print saved file paths
    
    for p in paths:
        print(f"Saved: {p}")
