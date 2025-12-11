"""
Contextual Augmentation Agent.

This module defines the ContextualAugmentation class, which uses
a Gemini model to extract, expand, and describe entities from text.
"""

import os
from typing import List, Optional
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.settings import ModelSettings

from src.models.schemas import (
    DetectedEntityList,
    InternalExpandedEntityList,
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
        api_key: Optional[str] = None,
        temperature: float = 0.0,
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
        self.model_settings = ModelSettings(temperature=temperature)

    def extract_entities(self, text: str) -> List[str]:
        """
        Extract named entities from the given text.
        
        Uses a rich schema (Category/Reasoning) for better recall,
        then returns a simple list of strings.

        Args:
            text: The input text.

        Returns:
            A list of extracted entity names.
        """
        # Define the strict schema for the LLM to follow
        entity_categories = (
            "- PERSON: People, including fictional.\n"
            "- NORP: Nationalities or religious or political groups.\n"
            "- GPE: Countries, cities, states.\n"
            "- ORG: Companies, agencies, institutions.\n"
            "- EVENT: Named hurricanes, battles, wars, sports events.\n"
        )

        agent = Agent(
            self.model,
            # We use the richer model here for the LLM interaction
            output_type=DetectedEntityList,
            system_prompt=(
                "You are a strict Named Entity Recognition system. "
                "Your goal is to extract every entity that fits the following schema:\n\n"
                f"{entity_categories}\n"
                "Guidelines:\n"
                "1. Analyze the sentence structure to identify boundaries.\n"
                "2. Provide reasoning for every extraction to ensure accuracy.\n"
                "3. Extract the exact span of text from the input.\n"
                "4. Do not include determiners (like 'The') unless they are part of the official name.\n"
            )
        )
        
        result = agent.run_sync(f"Text: {text}", model_settings=self.model_settings)
        
        # We strip the metadata (reasoning/category) here to keep
        # the rest of the pipeline working as expected.
        return [entity.name for entity in result.output.entities]

    def expand_entities(
        self, text: str, entities: List[str]
    ) -> List[ExpandedEntity]:
        """
        Expand entities to their full names based on context.
        
        Uses an internal schema with reasoning for accuracy, but returns
        clean ExpandedEntity objects.

        Args:
            text: The original text context.
            entities: List of entity names to expand.

        Returns:
            A list of ExpandedEntity objects (without reasoning).
        """
        agent = Agent(
            self.model,
            output_type=InternalExpandedEntityList,  # Use the Reasoning-aware schema here
            system_prompt=(
                "You are an Expert Entity Canonicalizer. Your goal is to transform "
                "ambiguous, partial, or acronym-based mentions into their **precise, "
                "full Canonical Names** (similar to Wikipedia page titles).\n\n"
                "**The Golden Rule:** Use the provided text to *identify* who/what the entity is, "
                "but use your internal knowledge to provide their *full legal/official name*.\n\n"
                "**Instructions:**\n"
                "1. **Analyze Domain:** Determine the topic of the text (e.g., Physics, Basketball, mythological).\n"
                "2. **Disambiguate:** Resolve ambiguity using the domain.\n"
                "   - Example: In a Physics text, 'Bohr' -> 'Niels Bohr'.\n"
                "   - Example: In a Tech text, 'Apple' -> 'Apple Inc.'.\n"
                "3. **Expand Acronyms:** Always output the full organization/concept name.\n"
                "   - Example: 'UN' -> 'United Nations'.\n"
                "4. **Complete Names:** If a person is mentioned by surname only, output their full name.\n"
                "   - Example: 'Curie' -> 'Marie Curie' (if context implies her).\n"
                "5. **Strict Output:** If the expanded name is not 100% certain from context, keep the original."
            )
        )
        
        input_data = (
            f"Context Text: {text}\n"
            f"Entities to Canonicalize: {', '.join(entities)}"
        )
        
        # Run the agent (returns InternalExpandedEntity objects)
        result = agent.run_sync(input_data, model_settings=self.model_settings)
        
        # Convert Internal entities -> Clean Public entities
        # This strips out the 'reasoning' field so the rest of your app doesn't see it.
        clean_entities = [
            ExpandedEntity(original=e.original, expanded=e.expanded)
            for e in result.output.entities
        ]
        
        return clean_entities

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
                "You are an Encyclopedic Context Resolver. "
                "Your task is to provide a brief, one-sentence "
                "definitive description for each entity in the provided list."
            )
        )

        formatted_list = "\n".join([f"- {e.expanded}" for e in expanded_entities])
        
        result = agent.run_sync(f"Entities to describe:\n{formatted_list}", model_settings=self.model_settings)
        return result.output.entities
