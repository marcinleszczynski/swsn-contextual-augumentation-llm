"""
Semantic Similarity Service.

This module provides functionality to select the best matching entity
based on semantic similarity of descriptions.
"""

from typing import List, Dict
from sentence_transformers import SentenceTransformer, util


class SemanticSelector:
    """Selects best entities using semantic similarity."""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the semantic selector.

        Args:
            model_name: Name of the sentence-transformers model.
        """
        self.model = SentenceTransformer(model_name)

    def select_best_candidate(
        self,
        reference_text: str,
        candidates: List[Dict[str, str]]
    ) -> Dict[str, str]:
        """
        Select the best candidate from a list based on semantic similarity
        to the reference text.

        Args:
            reference_text: The source description/context.
            candidates: List of dictionaries with 'uri', 'description', etc.

        Returns:
            The candidate dictionary with the highest similarity score.
            Returns None if candidates list is empty.
        """
        if not candidates:
            return None

        descriptions = [c.get("description") or "" for c in candidates]

        # Encode
        ref_embedding = self.model.encode(reference_text, convert_to_tensor=True)
        cand_embeddings = self.model.encode(descriptions, convert_to_tensor=True)

        # Compute cosine similarities
        cosine_scores = util.cos_sim(ref_embedding, cand_embeddings)[0]

        # Find best
        best_idx = cosine_scores.argmax().item()
        best_candidate = candidates[best_idx].copy()
        best_candidate["semantic_score"] = cosine_scores[best_idx].item()

        return best_candidate
