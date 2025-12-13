"""
Data Loaders Module.

This module handles loading datasets from various sources (Wikipedia, AIDA, Files).
"""

import pandas as pd
from datasets import load_dataset
from itertools import islice


def load_wikipedia_dataset(num_samples: int = 1000) -> pd.DataFrame:
    """
    Load Wikipedia dataset.

    Args:
    Args:
        num_samples: Number of samples to load.

    Returns:
        DataFrame with a 'text' column.
    """
    print(f"Loading Wikipedia dataset ({num_samples} samples)...")

    if num_samples == 0:
        return pd.DataFrame([])

    ds_stream = load_dataset(
        "wikimedia/wikipedia", "20231101.en", split="train", streaming=True
    )
    df = pd.DataFrame(list(islice(ds_stream, num_samples)))
    if "text" in df.columns:
        df = df[["text"]].copy()
        df["text"] = df["text"].fillna("")
        return df
    return pd.DataFrame(columns=["text"])


def load_aida_dataset(num_samples: int = 1000) -> pd.DataFrame:
    """
    Load AIDA dataset.

    Args:
        num_samples: Number of samples to load.

    Returns:
        DataFrame with a 'text' column.
    """
    print(f"Loading AIDA dataset ({num_samples} samples)...")

    if num_samples == 0:
        return pd.DataFrame([])

    try:
        ds_stream = load_dataset(
            "json", data_files="dataset/aida_dev.json", split="train", streaming=True
        )
    except Exception as e:
        print(f"Warning: Could not load AIDA dataset: {e}")
        return pd.DataFrame([])

    seen = set()
    unique_texts = []

    try:
        for example in ds_stream:
            text = example.get("text", "")
            if text and (text not in seen):
                seen.add(text)
                unique_texts.append(example)
            if len(unique_texts) >= num_samples:
                break
    except Exception as e:
        print(f"Error reading AIDA stream: {e}")

    df = pd.DataFrame(unique_texts)
    if not df.empty and "text" in df.columns:
        df = df[["text"]].copy()
        df["text"] = df["text"].fillna("")
        return df

    return pd.DataFrame(columns=["text"])


def load_aida_conll_parquet(num_samples: int = 1000) -> pd.DataFrame:
    """
    Load AIDA CoNLL parquet dataset from Hugging Face.

    Args:
        num_samples: Number of samples to load.

    Returns:
        DataFrame with a 'text' column.
    """
    print(f"Loading AIDA CoNLL parquet dataset ({num_samples} samples)...")

    if num_samples == 0:
        return pd.DataFrame([])

    try:
        ds_stream = load_dataset(
            "cyanic-selkie/aida-conll-yago-wikidata",
            split="test",
            streaming=True,
        )
        rows = list(islice(ds_stream, num_samples))
        df = pd.DataFrame(rows)

        # Limit to num_samples if specified
        if num_samples > 0 and len(df) > num_samples:
            df = df.head(num_samples)

        df = df[["text"]].copy()
        df["text"] = df["text"].fillna("")
        return df

    except Exception as e:
        print(f"Error loading AIDA CoNLL parquet dataset: {e}")
        return pd.DataFrame(columns=["text"])


def load_text_from_file(filepath: str) -> pd.DataFrame:
    """
    Load text from a text file.

    Args:
        filepath: Path to the text file.

    Returns:
        DataFrame with a single 'text' row.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        return pd.DataFrame([{"text": text}])
    except Exception as e:
        print(f"Error loading file {filepath}: {e}")
        return pd.DataFrame(columns=["text"])
