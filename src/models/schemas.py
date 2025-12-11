"""
Schemas for the Contextual Augmentation Agent.

This module defines the Pydantic models used for structured output
from the Gemini model.
"""

from pydantic import BaseModel, Field
from typing import List


class DetectedEntity(BaseModel):
    """An entity detected with reasoning and categorization."""

    reasoning: str = Field(
        description=(
            "A brief thought process explaining why this span is an entity "
            "and what category it belongs to."
        )
    )
    name: str = Field(
        description="The exact text span of the entity from the document."
    )
    category: str = Field(
        description="The category label (e.g., PERSON, NORP, GPE, ORG, EVENT)."
    )


class DetectedEntityList(BaseModel):
    """List of detected entities with metadata."""

    entities: List[DetectedEntity] = Field(
        description="List of detailed entity extractions"
    )


class NamedEntityList(BaseModel):
    """List of extracted named entities."""

    entities: List[str] = Field(
        description="List of named entities extracted from the text"
    )


class ExpandedEntity(BaseModel):
    """Represents an entity with its original and expanded forms."""

    original: str = Field(
        description="The original entity name extracted from text"
    )
    expanded: str = Field(
        description="The expanded entity name (e.g. full name)"
    )


class InternalExpandedEntity(BaseModel):
    """
    Internal model used only for LLM generation.
    Includes 'reasoning' to improve accuracy via Chain-of-Thought.
    """

    reasoning: str = Field(
        description=(
            "Brief analysis: 1. Identify domain. 2. Disambiguate based on "
            "context. 3. Select full canonical name."
        )
    )
    original: str = Field(
        description="The original entity name extracted from text"
    )
    expanded: str = Field(
        description="The full, canonical name (e.g., Wikipedia title)."
    )


class InternalExpandedEntityList(BaseModel):
    """List of internal expanded entities."""
    entities: List[InternalExpandedEntity] = Field(description="List of expanded entities")


class EntityWithDescription(BaseModel):
    """Represents an entity with a description."""

    name: str = Field(description="The expanded entity name")
    description: str = Field(description="A brief description of the entity")


class EntityWithDescriptionList(BaseModel):
    """List of entities with descriptions."""

    entities: List[EntityWithDescription] = Field(
        description="List of entities with descriptions"
    )
