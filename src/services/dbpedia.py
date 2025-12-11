"""
DBpedia Service Module.

This module handles interactions with the DBpedia Spotlight API
to find links for entities.
"""

import requests
from typing import Optional, Dict


class DBpediaClient:
    """Client for DBpedia Spotlight API."""

    def __init__(self):
        """Initialize the DBpedia client."""
        # Simple cache to avoid repeated requests for the same entity
        self._cache: Dict[str, Optional[str]] = {}

    def find_link(self, canonical_name: str) -> Optional[str]:
        """
        Find DBpedia link for an entity using DBpedia Spotlight API.

        Args:
            canonical_name: The canonical name of the entity

        Returns:
            DBpedia URL or None if not found
        """
        # Check cache first
        if canonical_name in self._cache:
            return self._cache[canonical_name]

        try:
            url = "https://api.dbpedia-spotlight.org/en/annotate"
            headers = {"Accept": "application/json"}
            params = {
                "text": canonical_name,
                "confidence": 0.4
            }

            response = requests.get(
                url,
                headers=headers,
                params=params,
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                resources = data.get("Resources", [])

                if resources:
                    # Find resource with highest similarity score
                    best_resource = max(
                        resources,
                        key=lambda r: float(r.get("@similarityScore", 0))
                    )
                    dbpedia_uri = best_resource.get("@URI")

                    if dbpedia_uri:
                        self._cache[canonical_name] = dbpedia_uri
                        return dbpedia_uri

            # No DBpedia link found
            self._cache[canonical_name] = None
            return None

        except Exception as e:
            print(f"Warning: Could not search for DBpedia link for '{canonical_name}': {e}")
            self._cache[canonical_name] = None
            return None
