"""
Entity Linking Validator.

This module validates entity linking results by comparing our extracted
entities with DBpedia data using an AI model.
"""

import json
import os
import re
import time
from typing import List, Dict, Any, Optional
import requests
from SPARQLWrapper import SPARQLWrapper, JSON
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.settings import ModelSettings
from pydantic import BaseModel, Field
from tqdm import tqdm

from src.models.validation_schemas import (
    ValidationDecision,
    EntityValidationResult,
    ValidationSummary
)


class BatchValidationDecision(BaseModel):
    """AI model's decision on multiple entity validations."""
    
    validations: List[ValidationDecision] = Field(
        description="List of validation decisions, one for each entity in the same order as provided"
    )


class EntityValidator:
    """Validates entity linking results against DBpedia."""
    
    def __init__(
        self,
        model_name: str = 'gemini-2.5-flash',
        api_key: Optional[str] = None,
        er_dataset_path: str = "datasets/er_dataset.json",
        output_path: str = "datasets/validation_results.json",
        batch_size: int = 50,
        rate_limit_delay: float = 12.0
    ):
        """
        Initialize the validator.
        
        Args:
            model_name: Gemini model name
            api_key: Google API key
            er_dataset_path: Path to the ER dataset JSON
            output_path: Path to save validation results
            batch_size: Number of entities to validate per API call (default: 5)
            rate_limit_delay: Seconds to wait between API calls (default: 12.0 for free tier)
        """
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set")
        
        os.environ["GEMINI_API_KEY"] = api_key
        self.model = GeminiModel(model_name)
        self.model_settings = ModelSettings(temperature=0.0)
        
        self.er_dataset_path = er_dataset_path
        self.output_path = output_path
        self.batch_size = batch_size
        self.rate_limit_delay = rate_limit_delay
        
        # SPARQL endpoint for DBpedia
        self.sparql = SPARQLWrapper("http://dbpedia.org/sparql")
        self.sparql.setReturnFormat(JSON)
        
    def get_context(self, text: str, start: int, end: int, window: int = 10) -> str:
        """
        Extract context around an entity mention.
        
        Args:
            text: Full text
            start: Entity start position
            end: Entity end position
            window: Number of words to include on each side
            
        Returns:
            Context string with entity highlighted
        """
        # Split into words
        words_before = text[:start].split()
        entity = text[start:end]
        words_after = text[end:].split()
        
        # Get window of words
        context_before = " ".join(words_before[-window:]) if words_before else ""
        context_after = " ".join(words_after[:window]) if words_after else ""
        
        # Build context with entity highlighted
        context_parts = []
        if context_before:
            context_parts.append(context_before)
        context_parts.append(f"**{entity}**")
        if context_after:
            context_parts.append(context_after)
            
        return " ".join(context_parts)
    
    def query_dbpedia(self, dbpedia_url: str) -> Dict[str, str]:
        """
        Query DBpedia for entity information using multiple strategies.
        
        Args:
            dbpedia_url: DBpedia resource URL
            
        Returns:
            Dictionary with name, type, and description
        """
        # Extract resource name from URL
        resource = dbpedia_url.replace("http://dbpedia.org/resource/", "")
        
        # Strategy 1: Try SPARQL with better query
        query = f"""
        PREFIX dbo: <http://dbpedia.org/ontology/>
        PREFIX dbr: <http://dbpedia.org/resource/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        
        SELECT ?label ?type ?abstract
        WHERE {{
            <{dbpedia_url}> rdfs:label ?label .
            FILTER (lang(?label) = 'en')
            
            OPTIONAL {{
                <{dbpedia_url}> rdf:type ?type .
                FILTER (
                    STRSTARTS(STR(?type), "http://dbpedia.org/ontology/") ||
                    STRSTARTS(STR(?type), "http://schema.org/")
                )
            }}
            
            OPTIONAL {{
                <{dbpedia_url}> dbo:abstract ?abstract .
                FILTER (lang(?abstract) = 'en')
            }}
        }}
        ORDER BY DESC(STRLEN(STR(?abstract)))
        LIMIT 1
        """
        
        try:
            self.sparql.setQuery(query)
            self.sparql.setTimeout(10)
            results = self.sparql.query().convert()
            
            if results["results"]["bindings"]:
                result = results["results"]["bindings"][0]
                
                # Extract label (name)
                label = result.get("label", {}).get("value", resource.replace("_", " "))
                
                # Extract type - get most specific type
                type_uri = result.get("type", {}).get("value", "")
                if type_uri:
                    # Extract last part of URI
                    if "/ontology/" in type_uri:
                        type_name = type_uri.split("/ontology/")[-1]
                    elif "/schema.org/" in type_uri:
                        type_name = type_uri.split("/schema.org/")[-1]
                    else:
                        type_name = type_uri.split("/")[-1].split("#")[-1]
                else:
                    type_name = "Entity"
                
                # Extract abstract (description)
                abstract = result.get("abstract", {}).get("value", "")
                
                if abstract:
                    # Limit abstract to first 500 chars
                    if len(abstract) > 500:
                        abstract = abstract[:497] + "..."
                else:
                    # Try REST API fallback for description
                    abstract = self._fetch_description_rest(dbpedia_url)
                
                return {
                    "name": label,
                    "type": type_name,
                    "description": abstract if abstract else "No description available"
                }
                
        except Exception as e:
            print(f"SPARQL query failed for {dbpedia_url}, trying REST API: {e}")
        
        # Strategy 2: Fallback to REST API
        try:
            import requests
            
            # Try to get JSON data from DBpedia
            json_url = dbpedia_url.replace("/resource/", "/data/") + ".json"
            response = requests.get(json_url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                resource_data = data.get(dbpedia_url, {})
                
                # Get label
                labels = resource_data.get("http://www.w3.org/2000/01/rdf-schema#label", [])
                label = None
                for l in labels:
                    if l.get("lang") == "en":
                        label = l.get("value")
                        break
                if not label:
                    label = resource.replace("_", " ")
                
                # Get type
                types = resource_data.get("http://www.w3.org/1999/02/22-rdf-syntax-ns#type", [])
                type_name = "Entity"
                for t in types:
                    type_uri = t.get("value", "")
                    if "/ontology/" in type_uri:
                        type_name = type_uri.split("/ontology/")[-1]
                        break
                
                # Get abstract
                abstracts = resource_data.get("http://dbpedia.org/ontology/abstract", [])
                abstract = None
                for a in abstracts:
                    if a.get("lang") == "en":
                        abstract = a.get("value", "")
                        if len(abstract) > 500:
                            abstract = abstract[:497] + "..."
                        break
                
                return {
                    "name": label,
                    "type": type_name,
                    "description": abstract if abstract else "No description available"
                }
        except Exception as e:
            print(f"REST API also failed for {dbpedia_url}: {e}")
        
        # Strategy 3: Ultimate fallback
        return {
            "name": resource.replace("_", " "),
            "type": "Entity",
            "description": f"DBpedia resource: {resource.replace('_', ' ')}"
        }
        
    def _fetch_description_rest(self, dbpedia_url: str) -> str:
        """Fetch description using DBpedia REST API (ontology:description first, fallback to abstract)."""
        try:
            data_url = dbpedia_url.replace("/resource/", "/data/") + ".json"
            response = requests.get(data_url, timeout=10)

            if response.status_code != 200:
                return ""

            data = response.json()

            resource_data = data.get(dbpedia_url, {})

            # 1. Try ontology:description (short, clean description)
            descriptions = resource_data.get("http://dbpedia.org/ontology/description", [])
            for d in descriptions:
                if d.get("lang") == "en":
                    return d.get("value", "")

            # 2. Fallback to dbo:abstract
            abstracts = resource_data.get("http://dbpedia.org/ontology/abstract", [])
            for a in abstracts:
                if a.get("lang") == "en":
                    abstract = a.get("value", "")
                    if len(abstract) > 500:
                        abstract = abstract[:497] + "..."
                    return abstract

        except Exception as ex:
            print("REST description fetch failed:", ex)

        return ""

    
    def validate_entity_batch(
        self,
        entities_data: List[Dict[str, Any]],
        max_retries: int = 3
    ) -> List[ValidationDecision]:
        """
        Use AI model to validate multiple entities at once with retry logic.
        
        Args:
            entities_data: List of dicts with canonical_name, our_description, 
                          context, and dbpedia_data for each entity
            max_retries: Maximum number of retry attempts
            
        Returns:
            List of ValidationDecision objects
        """
        agent = Agent(
            self.model,
            output_type=BatchValidationDecision,
            system_prompt=(
                "You are an expert Entity Linking Validator. Your task is to determine "
                "whether entities extracted from text correctly match their DBpedia resources.\n\n"
                "You will be given multiple entities to validate. For each one, you have:\n"
                "1. Our extracted canonical name and description\n"
                "2. The text context where the entity was mentioned\n"
                "3. Data from the DBpedia resource (name, type, description)\n\n"
                "Your job is to decide if each DBpedia link makes sense for the extracted entity.\n\n"
                "Guidelines:\n"
                "- 'correct': The DBpedia resource clearly matches the entity (same person, place, "
                "organization, etc.). Minor differences in description wording are OK.\n"
                "- 'incorrect': The DBpedia resource is clearly wrong (different entity entirely, "
                "wrong type, contradictory information).\n"
                "- 'not_sure': Ambiguous case where you cannot confidently determine correctness "
                "(e.g., common names, insufficient context, partial matches).\n\n"
                "Consider:\n"
                "- Do the names refer to the same entity?\n"
                "- Does the context support this being the DBpedia entity?\n"
                "- Are the descriptions semantically consistent?\n"
                "- Does the DBpedia type make sense?\n\n"
                "IMPORTANT: Return validations in the SAME ORDER as the entities provided."
            )
        )
        
        # Build prompt with all entities
        prompt_parts = ["Please validate the following entities:\n"]
        
        for i, entity in enumerate(entities_data, 1):
            prompt_parts.append(f"\n--- Entity {i} ---")
            prompt_parts.append(f"Context from text: {entity['context']}")
            prompt_parts.append(f"\nOur Data:")
            prompt_parts.append(f"- Canonical Name: {entity['canonical_name']}")
            prompt_parts.append(f"- Description: {entity['our_description']}")
            prompt_parts.append(f"\nDBpedia Data:")
            prompt_parts.append(f"- Name: {entity['dbpedia_data']['name']}")
            prompt_parts.append(f"- Type: {entity['dbpedia_data']['type']}")
            prompt_parts.append(f"- Description: {entity['dbpedia_data']['description']}\n")
        
        prompt = "\n".join(prompt_parts)
        
        # Retry logic with exponential backoff
        for attempt in range(max_retries):
            try:
                result = agent.run_sync(prompt, model_settings=self.model_settings)
                return result.output.validations
            except Exception as e:
                error_str = str(e)
                
                # Check if it's a rate limit error
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    if attempt < max_retries - 1:
                        # Extract retry delay from error message if available
                        import re
                        match = re.search(r'retry in (\d+(?:\.\d+)?)', error_str)
                        if match:
                            retry_delay = float(match.group(1)) + 1  # Add 1 second buffer
                        else:
                            retry_delay = (2 ** attempt) * 15  # Exponential backoff: 15s, 30s, 60s
                        
                        print(f"\nRate limit hit, waiting {retry_delay:.1f}s before retry {attempt + 1}/{max_retries}...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        print(f"\nMax retries reached due to rate limiting")
                        raise
                else:
                    # Non-rate-limit error, raise immediately
                    raise
        
        # Should not reach here
        raise Exception("Failed to validate batch after all retries")
    
    def validate_dataset(self) -> Dict[str, Any]:
        """
        Validate all entities in the ER dataset.
        
        Returns:
            Dictionary containing validation results and summary
        """
        print(f"Loading ER dataset from {self.er_dataset_path}...")
        
        with open(self.er_dataset_path, 'r', encoding='utf-8') as f:
            er_data = json.load(f)
        
        results = []
        stats = {
            "correct": 0,
            "incorrect": 0,
            "not_sure": 0,
            "unlinked": 0
        }
        
        # Collect all entities that need validation
        entities_to_validate = []
        
        print(f"Processing {len(er_data)} documents...\n")
        
        for doc in er_data:
            doc_id = doc["id"]
            text = doc["text"]
            mentions = doc["mentions"]
            
            for mention_data in mentions:
                mention = mention_data["mention"]
                canonical_name = mention_data["canonical_name"]
                our_description = mention_data["description"]
                dbpedia_link = mention_data["dbpedia_link"]
                start = mention_data["start"]
                end = mention_data["end"]
                
                # Get context
                context = self.get_context(text, start, end, window=10)
                
                # Handle unlinked entities immediately
                if dbpedia_link is None:
                    results.append(EntityValidationResult(
                        doc_id=doc_id,
                        mention=mention,
                        canonical_name=canonical_name,
                        our_description=our_description,
                        context=context,
                        dbpedia_link="null",
                        dbpedia_name="N/A",
                        dbpedia_type="N/A",
                        dbpedia_description="N/A",
                        decision="unlinked",
                        reasoning="No DBpedia link was found for this entity"
                    ).model_dump())
                    stats["unlinked"] += 1
                    continue
                
                # Query DBpedia
                dbpedia_data = self.query_dbpedia(dbpedia_link)
                
                # Add to validation queue
                entities_to_validate.append({
                    'doc_id': doc_id,
                    'mention': mention,
                    'canonical_name': canonical_name,
                    'our_description': our_description,
                    'context': context,
                    'dbpedia_link': dbpedia_link,
                    'dbpedia_data': dbpedia_data
                })
        
        # Process entities in batches
        print(f"Validating {len(entities_to_validate)} linked entities in batches of {self.batch_size}...\n")
        
        num_batches = (len(entities_to_validate) + self.batch_size - 1) // self.batch_size
        
        for i in tqdm(range(0, len(entities_to_validate), self.batch_size), 
                     total=num_batches, desc="Validating batches"):
            batch = entities_to_validate[i:i + self.batch_size]
            
            try:
                # Validate batch
                validations = self.validate_entity_batch(batch)
                
                # Match validations to entities
                for entity, validation in zip(batch, validations):
                    decision = validation.decision
                    reasoning = validation.reasoning
                    
                    # Store result
                    results.append(EntityValidationResult(
                        doc_id=entity['doc_id'],
                        mention=entity['mention'],
                        canonical_name=entity['canonical_name'],
                        our_description=entity['our_description'],
                        context=entity['context'],
                        dbpedia_link=entity['dbpedia_link'],
                        dbpedia_name=entity['dbpedia_data']['name'],
                        dbpedia_type=entity['dbpedia_data']['type'],
                        dbpedia_description=entity['dbpedia_data']['description'],
                        decision=decision,
                        reasoning=reasoning
                    ).model_dump())
                    
                    stats[decision] += 1
                
                # Rate limiting: wait between batches (except for last batch)
                if i + self.batch_size < len(entities_to_validate):
                    time.sleep(self.rate_limit_delay)
                    
            except Exception as e:
                print(f"\nError validating batch: {e}")
                # Mark all entities in failed batch as "not_sure"
                for entity in batch:
                    results.append(EntityValidationResult(
                        doc_id=entity['doc_id'],
                        mention=entity['mention'],
                        canonical_name=entity['canonical_name'],
                        our_description=entity['our_description'],
                        context=entity['context'],
                        dbpedia_link=entity['dbpedia_link'],
                        dbpedia_name=entity['dbpedia_data']['name'],
                        dbpedia_type=entity['dbpedia_data']['type'],
                        dbpedia_description=entity['dbpedia_data']['description'],
                        decision="not_sure",
                        reasoning=f"Validation failed: {str(e)}"
                    ).model_dump())
                    stats["not_sure"] += 1
        
        # Calculate summary statistics
        total_entities = len(results)
        linked_entities = total_entities - stats["unlinked"]
        validation_rate = (linked_entities / total_entities * 100) if total_entities > 0 else 0
        
        validated_entities = stats["correct"] + stats["incorrect"] + stats["not_sure"]
        accuracy = (stats["correct"] / validated_entities * 100) if validated_entities > 0 else 0
        
        summary = ValidationSummary(
            total_entities=total_entities,
            correct=stats["correct"],
            incorrect=stats["incorrect"],
            not_sure=stats["not_sure"],
            unlinked=stats["unlinked"],
            validation_rate=round(validation_rate, 2),
            accuracy=round(accuracy, 2)
        ).model_dump()
        
        return {
            "summary": summary,
            "results": results
        }
    
    def run_validation(self) -> None:
        """
        Run full validation and save results.
        """
        print("="*80)
        print("ENTITY LINKING VALIDATION")
        print("="*80 + "\n")
        
        # Run validation
        validation_data = self.validate_dataset()
        
        # Save to JSON
        print(f"\nSaving validation results to {self.output_path}...")
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(validation_data, f, indent=2, ensure_ascii=False)
        
        # Print summary
        print("\n" + "="*80)
        print("VALIDATION SUMMARY")
        print("="*80)
        
        summary = validation_data["summary"]
        print(f"Total Entities: {summary['total_entities']}")
        print(f"  ✓ Correct: {summary['correct']}")
        print(f"  ✗ Incorrect: {summary['incorrect']}")
        print(f"  ? Not Sure: {summary['not_sure']}")
        print(f"  - Unlinked: {summary['unlinked']}")
        print(f"\nValidation Rate: {summary['validation_rate']}% (entities with DBpedia links)")
        print(f"Accuracy: {summary['accuracy']}% (correct / validated entities)")
        print("\n" + "="*80)