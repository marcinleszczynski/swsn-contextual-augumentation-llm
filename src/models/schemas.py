"""
Schemas for the Contextual Augmentation Agent.

This module defines the Pydantic models used for structured output
from the Gemini model.
"""

from pydantic import BaseModel, Field
from typing import List


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


class ExpandedEntityList(BaseModel):
    """List of expanded entities."""

    entities: List[ExpandedEntity] = Field(
        description="List of expanded entities"
    )


class EntityWithDescription(BaseModel):
    """Represents an entity with a description."""

    name: str = Field(description="The expanded entity name")
    description: str = Field(description="A brief description of the entity")


class EntityWithDescriptionList(BaseModel):
    """List of entities with descriptions."""

    entities: List[EntityWithDescription] = Field(
        description="List of entities with descriptions"
    )
