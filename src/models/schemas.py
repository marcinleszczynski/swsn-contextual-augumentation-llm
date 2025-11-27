from pydantic import BaseModel, Field
from typing import List


class NamedEntityList(BaseModel):
    entities: List[str] = Field(
        description="List of named entities extracted from the text"
    )


class ExpandedEntity(BaseModel):
    original: str = Field(
        description="The original entity name extracted from text"
    )
    expanded: str = Field(
        description="The expanded entity name (e.g. full name)"
    )


class ExpandedEntityList(BaseModel):
    entities: List[ExpandedEntity] = Field(
        description="List of expanded entities"
    )


class EntityWithDescription(BaseModel):
    name: str = Field(description="The expanded entity name")
    description: str = Field(description="A brief description of the entity")


class EntityWithDescriptionList(BaseModel):
    entities: List[EntityWithDescription] = Field(
        description="List of entities with descriptions"
    )
