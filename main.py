"""
Main entry point for the Contextual Augmentation Agent.

This script demonstrates the usage of the ContextualAugmentation agent
to extract, expand, and describe entities from a sample text.
"""

import os
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.agent.contextual_augmentation import ContextualAugmentation  # noqa: E402, E501


def main():
    """
    Run the main execution flow of the agent.

    1. Initialize the agent with API key and model name.
    2. Extract entities from sample text.
    3. Expand the extracted entities.
    4. Generate descriptions for the expanded entities.
    """
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Please set GEMINI_API_KEY environment variable.")
        return

    # Initialize the agent
    # You can pass a specific model name if needed, e.g., 'gemini-2.5-flash'
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    try:
        augmentor = ContextualAugmentation(
            model_name=model_name, api_key=api_key
        )
        print(f"Initialized with model: {model_name}")
    except Exception as e:
        print(f"Error initializing agent: {e}")
        return

    text = "Angelina met her partner Brad and her partner Jon in AK."
    print(f"Original Text: {text}\n")

    # Step 1: Extract Entities
    print("--- Step 1: Extracting Entities ---")
    try:
        entities = augmentor.extract_entities(text)
        print(f"Extracted Entities: {entities}\n")
    except Exception:
        import traceback
        traceback.print_exc()
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


if __name__ == "__main__":
    main()
