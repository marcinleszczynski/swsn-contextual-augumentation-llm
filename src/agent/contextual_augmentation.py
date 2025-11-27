"""
Contextual Augmentation Agent.

This module defines the ContextualAugmentation class, which uses
a Gemini model to extract, expand, and describe entities from text.
"""

import os
from typing import List, Optional
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel

from src.models.schemas import (
    NamedEntityList,
    ExpandedEntityList,
    EntityWithDescriptionList,
    ExpandedEntity,
    EntityWithDescription,
)


class ContextualAugmentation:
    """
    Agent for augmenting text with contextual information about entities.

    Uses a Gemini model to perform Named Entity Recognition (NER),
    Entity Linking (Expansion), and Entity Description.
    """

    def __init__(
        self,
        model_name: str = 'gemini-2.5-flash',
        api_key: Optional[str] = None
    ):
        """
        Initialize the ContextualAugmentation agent.

        Args:
            model_name: The name of the Gemini model to use.
            api_key: The Google API key. If None, reads from env
                GEMINI_API_KEY.

        Raises:
            ValueError: If GEMINI_API_KEY is not set.
        """
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set and no model provided."
            )
        os.environ["GEMINI_API_KEY"] = api_key
        self.model = GeminiModel(model_name)

    def extract_entities(self, text: str) -> List[str]:
        """
        Extract named entities from the given text.

        Args:
            text: The input text.

        Returns:
            A list of extracted entity names.
        """
        agent = Agent(
            self.model,
            output_type=NamedEntityList,
            system_prompt=(
                "You are an expert Named Entity Recognition system. Extract "
                "all named entities (people, organizations, locations, etc.) "
                "from the provided text."
            )
        )
        result = agent.run_sync(f"Text: {text}")
        return result.output.entities

    def expand_entities(
        self, text: str, entities: List[str]
    ) -> List[ExpandedEntity]:
        """
        Expand entities to their full names based on context.

        Args:
            text: The original text context.
            entities: List of entity names to expand.

        Returns:
            A list of ExpandedEntity objects containing original and
            expanded names.
        """
        agent = Agent(
            self.model,
            output_type=ExpandedEntityList,
            system_prompt=(
                "You are an expert entity linker. Given the original text and "
                "a list of extracted entities, expand each entity to its most "
                "likely full name based on the context. Return the original "
                "name and the expanded name."
            )
        )
        input_data = (
            f"Original Text: {text}\n"
            f"Extracted Entities: {', '.join(entities)}"
        )
        result = agent.run_sync(input_data)
        return result.output.entities

    def describe_entities(
        self, expanded_entities: List[ExpandedEntity]
    ) -> List[EntityWithDescription]:
        """
        Generate descriptions for a list of entities.

        Args:
            expanded_entities: List of ExpandedEntity objects.

        Returns:
            A list of EntityWithDescription objects.
        """
        agent = Agent(
            self.model,
            output_type=EntityWithDescriptionList,
            system_prompt=(
                "You are a knowledge graph expert. Given a list of entities, "
                "provide a brief, one-sentence description for each one."
            )
        )
        # We only need the expanded names for description
        names = [e.expanded for e in expanded_entities]
        input_data = f"Entities: {', '.join(names)}"
        result = agent.run_sync(input_data)
        return result.output.entities
