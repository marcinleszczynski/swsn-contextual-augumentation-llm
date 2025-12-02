"""
Main entry point for the Contextual Augmentation Agent.

This script demonstrates the usage of the ContextualAugmentation agent
to extract, expand, and describe entities from a sample text.

As well as the usage of the pipeline to process datasets
and generate NER and Entity Resolution datasets.
"""

import os
from dotenv import load_dotenv
import pandas as pd
from datasets import load_dataset
from itertools import islice

from src.agent.contextual_augmentation import ContextualAugmentation
from src.pipeline.pipeline import DatasetPipeline


def load_wikipedia_dataset(num_samples: int = 1000) -> pd.DataFrame:
    """Load Wikipedia dataset."""
    print(f"Loading Wikipedia dataset ({num_samples} samples)...")

    if num_samples == 0:
        return pd.DataFrame([])

    ds_stream = load_dataset(
        "wikimedia/wikipedia", "20231101.en", split="train", streaming=True
    )
    df = pd.DataFrame(list(islice(ds_stream, num_samples)))
    df = df[["text"]].copy()
    df["text"] = df["text"].fillna("")
    return df


def load_aida_dataset(num_samples: int = 1000) -> pd.DataFrame:
    """Load AIDA dataset."""
    print(f"Loading AIDA dataset ({num_samples} samples)...")

    if num_samples == 0:
        return pd.DataFrame([])

    ds_stream = load_dataset(
        "json",
        data_files="dataset/aida_dev.json",
        split="train",
        streaming=True
    )

    seen = set()
    unique_texts = []
    for example in ds_stream:
        text = example.get("text", "")
        if text and (text not in seen):
            seen.add(text)
            unique_texts.append(example)
        if len(unique_texts) >= num_samples:
            break

    df = pd.DataFrame(unique_texts)
    df = df[["text"]].copy()
    df["text"] = df["text"].fillna("")
    return df


def demo_single_example():
    """
    Run a single example to demonstrate the agent functionality.
    """
    print("="*80)
    print("DEMO: Single Example Processing")
    print("="*80)

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Please set GEMINI_API_KEY environment variable.")
        return

    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    try:
        augmentor = ContextualAugmentation(
            model_name=model_name, api_key=api_key
        )
        print(f"Initialized with model: {model_name}")
    except Exception as e:
        print(f"Error initializing agent: {e}")
        return

    text = "Angelina met her partner Brad and her father Jon in AK."
    print(f"\nOriginal Text: {text}\n")

    # Step 1: Extract Entities
    print("--- Step 1: Extracting Entities ---")
    try:
        entities = augmentor.extract_entities(text)
        print(f"Extracted Entities: {entities}\n")
    except Exception as e:
        print(f"Error in extraction: {e}")
        return

    # Step 2: Expand Entities
    print("--- Step 2: Expanding Entities ---")
    try:
        expanded_entities = augmentor.expand_entities(text, entities)
        for entity in expanded_entities:
            print(
                f"Original: {entity.original} -> Expanded: {entity.expanded}"
            )
        print()
    except Exception as e:
        print(f"Error in expansion: {e}")
        return

    # Step 3: Describe Entities
    print("--- Step 3: Describing Entities ---")
    try:
        descriptions = augmentor.describe_entities(expanded_entities)
        for item in descriptions:
            print(f"Name: {item.name}")
            print(f"Description: {item.description}\n")
    except Exception as e:
        print(f"Error in description: {e}")
        return

    # Step 4: Link Entitites to DBpedia
    print("--- Step 4: Linking Entitites to DBpedia ---")
    try:
        pipeline = DatasetPipeline(augmentor)
        for item in descriptions:
            print(f"Name: {item.name}")
            print(f"Link: {pipeline._find_dbpedia_link(item.name)}\n")
    except Exception as e:
        print(f"Error in description: {e}")
        return


def run_pipeline(
    wikipedia_samples: int = 100,
    aida_samples: int = 100,
    output_dir: str = "datasets"
):
    """
    Run the full pipeline on datasets.

    Args:
        wikipedia_samples: Number of Wikipedia samples to process
        aida_samples: Number of AIDA samples to process
        output_dir: Directory to save output datasets
    """
    print("\n" + "="*80)
    print("RUNNING FULL PIPELINE")
    print("="*80 + "\n")

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Please set GEMINI_API_KEY environment variable.")
        return

    # Initialize agent
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    try:
        augmentor = ContextualAugmentation(
            model_name=model_name, api_key=api_key
        )
        print(f"Initialized with model: {model_name}\n")
    except Exception as e:
        print(f"Error initializing agent: {e}")
        return

    # Initialize pipeline
    pipeline = DatasetPipeline(augmentor, output_dir=output_dir)

    # Process Wikipedia dataset
    try:
        wiki_df = load_wikipedia_dataset(wikipedia_samples)
        pipeline.process_dataframe(wiki_df, dataset_name="wikipedia")
    except Exception as e:
        print(f"Error processing Wikipedia dataset: {e}")

    # Process AIDA dataset
    try:
        aida_df = load_aida_dataset(aida_samples)
        pipeline.process_dataframe(aida_df, dataset_name="aida")
    except Exception as e:
        print(f"Error processing AIDA dataset: {e}")

    # Save datasets
    pipeline.save_datasets()

    # Print statistics
    print("\n" + "="*80)
    print("DATASET STATISTICS")
    print("="*80)
    stats = pipeline.get_statistics()
    print(f"NER Documents: {stats['ner_documents']}")
    print(f"ER Documents: {stats['er_documents']}")
    print(f"Total Entity Mentions: {stats['total_entities']}")
    print(f"Unique Entities in KB: {stats['unique_entities']}")


def main():
    """
    Main entry point.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Process datasets to create NER and ER datasets"
    )
    parser.add_argument(
        '--mode',
        choices=['demo', 'pipeline'],
        default='demo',
        help='Run mode: demo (single example) or pipeline (full processing)'
    )
    parser.add_argument(
        '--wikipedia-samples',
        type=int,
        default=100,
        help='Number of Wikipedia samples to process (default: 100)'
    )
    parser.add_argument(
        '--aida-samples',
        type=int,
        default=100,
        help='Number of AIDA samples to process (default: 100)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='datasets',
        help='Output directory for datasets (default: datasets)'
    )

    args = parser.parse_args()

    if args.mode == 'demo':
        demo_single_example()
    else:
        run_pipeline(
            wikipedia_samples=args.wikipedia_samples,
            aida_samples=args.aida_samples,
            output_dir=args.output_dir
        )


if __name__ == "__main__":
    main()
