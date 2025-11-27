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
    def __init__(
        self,
        model_name: str = 'gemini-1.5-flash',
        api_key: Optional[str] = None
    ):
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set and no model provided."
            )
        os.environ["GEMINI_API_KEY"] = api_key
        self.model = GeminiModel(model_name)

    def extract_entities(self, text: str) -> List[str]:
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
