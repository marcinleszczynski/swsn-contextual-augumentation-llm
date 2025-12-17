"""
Pipeline for creating NER and Entity Resolution datasets.

This module processes text data through the ContextualAugmentation agent
and builds structured datasets for Named Entity Recognition
and Entity Resolution with Entity Linking.
"""

import json
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import pandas as pd
from tqdm import tqdm

from src.agent.contextual_augmentation import ContextualAugmentation
from src.services.similarity import SemanticSelector
from src.services.dbpedia import DBpediaClient


@dataclass
class EntityMention:
    """Represents a single entity mention in text."""
    mention: str
    start: int
    end: int
    entity_id: str
    canonical_name: str
    description: str
    dbpedia_link: Optional[str]


@dataclass
class ProcessedDocument:
    """Represents a processed document with entities."""
    doc_id: str
    text: str
    entities: List[Dict[str, Any]]  # For NER dataset
    mentions: List[Dict[str, Any]]  # For ER dataset


class DatasetPipeline:
    """
    Pipeline for processing text and creating NER/ER datasets.
    """

    def __init__(
        self,
        augmentor: ContextualAugmentation,
        output_dir: str = "datasets"
    ):
        """
        Initialize the pipeline.

        Args:
            augmentor: ContextualAugmentation agent instance
            output_dir: Directory to save the datasets
        """
        self.augmentor = augmentor
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Entity knowledge base - maps canonical names to entity IDs
        self.entity_kb: Dict[str, str] = {}
        self.next_entity_id = 1

        # Store entity descriptions
        self.entity_descriptions: Dict[str, str] = {}

        # DBpedia Client
        self.dbpedia_client = DBpediaClient()
        self.semantic_selector = SemanticSelector()

        # Accumulated datasets
        self.ner_dataset: List[Dict[str, Any]] = []
        self.er_dataset: List[Dict[str, Any]] = []

    def _get_or_create_entity_id(
        self, canonical_name: str, description: str
    ) -> str:
        """
        Get existing entity ID or create a new one.

        Args:
            canonical_name: The canonical name of the entity
            description: Entity description

        Returns:
            Entity ID (e.g., "E001")
        """
        if canonical_name in self.entity_kb:
            return self.entity_kb[canonical_name]

        entity_id = f"E{self.next_entity_id:04d}"
        self.entity_kb[canonical_name] = entity_id
        self.entity_descriptions[entity_id] = description
        self.next_entity_id += 1
        return entity_id

    def _find_entity_positions(
        self, text: str, entity: str
    ) -> Optional[tuple[int, int]]:
        """
        Find the start and end positions of an entity in text.

        Args:
            text: The full text
            entity: The entity string to find

        Returns:
            Tuple of (start, end) positions, or None if not found
        """
        # Simple case-sensitive search
        start = text.find(entity)
        if start != -1:
            return (start, start + len(entity))

        # Case-insensitive fallback
        lower_text = text.lower()
        lower_entity = entity.lower()
        start = lower_text.find(lower_entity)
        if start != -1:
            return (start, start + len(entity))

        return None

    def process_document(
        self, doc_id: str, text: str
    ) -> Optional[ProcessedDocument]:
        """
        Process a single document through the pipeline.

        Args:
            doc_id: Document identifier
            text: Text to process

        Returns:
            ProcessedDocument with NER and ER information, or None if error
        """
        try:
            # Step 1: Extract entities
            entities = self.augmentor.extract_entities(text)
            if not entities:
                return None

            # Step 2: Expand entities
            expanded_entities = self.augmentor.expand_entities(text, entities)
            if not expanded_entities:
                return None

            # Step 3: Get descriptions
            descriptions = self.augmentor.describe_entities(expanded_entities)
            if not descriptions:
                return None

            # Build entity information
            ner_entities = []
            er_mentions = []

            for i, expanded_entity in enumerate(expanded_entities):
                original = expanded_entity.original
                canonical = expanded_entity.expanded

                # Get description (match by canonical name)
                description = ""
                for desc_obj in descriptions:
                    if desc_obj.name == canonical:
                        description = desc_obj.description
                        break

                if not description:
                    continue

                # Find position in text
                position = self._find_entity_positions(text, original)
                if not position:
                    continue

                start, end = position

                # Get or create entity ID
                entity_id = self._get_or_create_entity_id(canonical,
                                                          description)

                # Find best DBpedia link
                candidates = self.dbpedia_client.search_entities(canonical)
                best_match = self.semantic_selector.select_best_candidate(
                    description, candidates
                )

                dbpedia_link = best_match.get("uri") if best_match else None

                # Add to NER dataset format
                ner_entities.append({
                    "start": start,
                    "end": end,
                    "text": original
                })

                # Add to ER dataset format
                er_mentions.append({
                    "mention": original,
                    "start": start,
                    "end": end,
                    "entity_id": entity_id,
                    "canonical_name": canonical,
                    "description": description,
                    "dbpedia_link": dbpedia_link
                })

            return ProcessedDocument(
                doc_id=doc_id,
                text=text,
                entities=ner_entities,
                mentions=er_mentions
            )

        except Exception as e:
            import traceback
            print(f"Error processing document {doc_id}: {e}")
            print(traceback.format_exc())
            return None

    def process_dataframe(
        self, df: pd.DataFrame, dataset_name: str = "dataset"
    ) -> None:
        """
        Process a dataframe of texts through the pipeline.

        Args:
            df: DataFrame with a 'text' column
            dataset_name: Name prefix for this dataset batch
        """
        print(f"Processing {len(df)} documents from {dataset_name}...")

        for idx, row in tqdm(df.iterrows(), total=len(df)):
            text = row['text']
            if not text or len(text.strip()) == 0:
                continue

            doc_id = f"{dataset_name}_{idx:06d}"
            processed = self.process_document(doc_id, text)

            if processed and processed.entities:
                # Add to NER dataset
                self.ner_dataset.append({
                    "id": processed.doc_id,
                    "text": processed.text,
                    "entities": processed.entities
                })

                # Add to ER dataset
                self.er_dataset.append({
                    "id": processed.doc_id,
                    "text": processed.text,
                    "mentions": processed.mentions
                })

    def save_datasets(self) -> None:
        """Save the accumulated datasets to JSON files."""
        ner_path = os.path.join(self.output_dir, "ner_dataset.json")
        er_path = os.path.join(self.output_dir, "er_dataset.json")
        kb_path = os.path.join(self.output_dir, "entity_kb.json")

        print(f"Saving NER dataset to {ner_path}...")
        with open(ner_path, 'w', encoding='utf-8') as f:
            json.dump(self.ner_dataset, f, indent=2, ensure_ascii=False)

        print(f"Saving ER dataset to {er_path}...")
        with open(er_path, 'w', encoding='utf-8') as f:
            json.dump(self.er_dataset, f, indent=2, ensure_ascii=False)

        # Save entity knowledge base for reference
        kb_data = {
            "entities": [
                {
                    "entity_id": entity_id,
                    "canonical_name": canonical_name,
                    "description": self.entity_descriptions.get(entity_id, ""),
                    # Using cached value
                    "dbpedia_link": self.dbpedia_client.find_link(canonical_name)
                }
                for canonical_name, entity_id in self.entity_kb.items()
            ]
        }

        print(f"Saving entity KB to {kb_path}...")
        with open(kb_path, 'w', encoding='utf-8') as f:
            json.dump(kb_data, f, indent=2, ensure_ascii=False)

        print("\nDatasets saved:")
        print(f"  - NER dataset: {len(self.ner_dataset)} documents")
        print(f"  - ER dataset: {len(self.er_dataset)} documents")
        print(f"  - Entity KB: {len(self.entity_kb)} unique entities")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the generated datasets.

        Returns:
            Dictionary with dataset statistics
        """
        total_entities = sum(len(doc['entities']) for doc in self.ner_dataset)
        total_mentions = sum(len(doc['mentions']) for doc in self.er_dataset)

        return {
            "ner_documents": len(self.ner_dataset),
            "er_documents": len(self.er_dataset),
            "total_entities": total_entities,
            "total_mentions": total_mentions,
            "unique_entities": len(self.entity_kb)
        }
