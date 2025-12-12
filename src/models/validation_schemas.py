"""
Validation Schemas Module.

Defines Pydantic models for validation results.
"""

from pydantic import BaseModel, Field
from typing import List, Literal


class ValidationDecision(BaseModel):
    """AI model's decision on entity validation."""
    
    reasoning: str = Field(
        description="Brief explanation of why the entity link is correct, incorrect, or uncertain"
    )
    decision: Literal["correct", "incorrect", "not_sure"] = Field(
        description="The validation decision: correct, incorrect, or not_sure"
    )


class EntityValidationResult(BaseModel):
    """Result of validating a single entity."""
    
    doc_id: str
    mention: str
    canonical_name: str
    our_description: str
    context: str
    dbpedia_link: str
    dbpedia_name: str
    dbpedia_type: str
    dbpedia_description: str
    decision: str  # "correct", "incorrect", "not_sure", "unlinked"
    reasoning: str
    

class ValidationSummary(BaseModel):
    """Summary statistics of validation results."""
    
    total_entities: int
    correct: int
    incorrect: int
    not_sure: int
    unlinked: int
    validation_rate: float  # percentage of entities with non-null links
    accuracy: float  # correct / (correct + incorrect + not_sure)
