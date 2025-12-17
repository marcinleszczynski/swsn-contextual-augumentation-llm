"""
DBpedia Service Module.

This module handles interactions with the DBpedia Lookup API
to find entities and links.
"""

import requests
import xml.etree.ElementTree as ET
from typing import Optional, Dict, List


class DBpediaClient:
    """Client for DBpedia Lookup API."""

    def __init__(self):
        """Initialize the DBpedia client."""
        # Simple cache to avoid repeated requests for the same entity
        self._cache: Dict[str, Optional[str]] = {}
        self.base_url = "https://lookup.dbpedia.org/api/search"

    def search_entities(
        self,
        query: str,
        max_results: int = 10,
        timeout: int = 15
    ) -> List[Dict[str, Optional[str]]]:
        """
        Searches DBpedia Lookup API for entities matching the query.

        Args:
            query (str): The search string (e.g., "Cat").
            max_results (int): The maximum number of results to return (default: 10).
            timeout (int): Request timeout in seconds (default: 15).

        Returns:
            list: A list of dictionaries, where each dictionary represents an entity
                  with keys: 'label', 'uri', and 'description'.
        """
        params = {
            'query': query,
            'maxResults': max_results
        }

        try:
            # Send GET request to the API
            response = requests.get(self.base_url, params=params, timeout=timeout)
            response.raise_for_status()

            # Parse the XML response
            root = ET.fromstring(response.content)

            entities = []

            # Iterate through each <Result> element in the XML
            for result in root.findall('Result'):
                description = result.find('Description').text if result.find(
                    'Description') is not None else None
                label = result.find('Label').text if result.find('Label') is not None else "Unknown"

                if not description:
                    print(f"Warning: No description found for entity '{label}' (Query: '{query}')")

                entity = {
                    'label': label,
                    'uri': result.find('URI').text if result.find('URI') is not None else None,
                    'description': description
                }
                entities.append(entity)

            return entities

        except requests.exceptions.RequestException as e:
            print(f"Error fetching data from DBpedia for '{query}': {e}")
            return []
        except ET.ParseError as e:
            print(f"Error parsing XML response for '{query}': {e}")
            return []
