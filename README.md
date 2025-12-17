# Contextual Augmentation for Entity Linking without external dataset

[![CI](https://github.com/marcinleszczynski/swsn-contextual-augumentation-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/marcinleszczynski/swsn-contextual-augumentation-llm/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)

This project implements a contextual augmentation pipeline for entity linking using Large Language Models (LLMs) via `pydantic-ai`.

## Features

1.  **Entity Extraction**: Extracts named entities from text.
2.  **Entity Expansion**: Expands extracted entities to their full names based on context.
3.  **Entity Description**: Provides brief descriptions for the expanded entities.

## Requirements

- Python 3.10+
- `pydantic-ai`
- `python-dotenv`
- `pandas`
- `datasets`
- A Google Gemini API Key
- aida_dev.json has to be in dataset folder

## Installation

1.  Clone the repository.
2.  Create a virtual environment:
    ```bash
    python -m venv .venv
    ```
3.  Activate the virtual environment:
    - Windows: `.\.venv\Scripts\activate`
    - Unix/MacOS: `source .venv/bin/activate`
4.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    # OR install in editable mode (recommended for development)
    pip install -e .
    ```

## Configuration

1.  Create `.env` and set your `GEMINI_API_KEY`.
2.  (Optional) Set `GEMINI_MODEL` in `.env` to use a specific model (default: `gemini-2.5-flash`).

## Testing

Run the tests using `pytest`:

```bash
pip install pytest
pytest
```

## Pipeline

The pipeline processes text through three stages:
1. **Named Entity Recognition (NER)**: Extract entity mentions from text
2. **Entity Resolution (ER)**: Expand entities to their canonical forms
3. **Entity Description**: Generate descriptions for each entity

The output consists of two datasets:
- **NER Dataset**: Span-based entity annotations
- **ER Dataset**: Entity mentions linked to canonical entities with descriptions

## Project Structure

```
.
├── src/
│   ├── agent/
│   │   ├── __init__.py
│   │   └── contextual_augmentation.py  # Gemini agent for NER/ER
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py                   # Pydantic schemas
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── pipeline.py                  # Main dataset generation pipeline
│   └── __init__.py
├── main.py                              # Entry point
└── requirements.txt
```

## Usage

### Demo Mode (Single Example)

Run a single example to see how the agent works:

```bash
python main.py --mode demo
```

Output:
```
================================================================================
DEMO: Single Example Processing
================================================================================
Initialized with model: gemini-2.5-flash

Original Text: Angelina met her partner Brad and her father Jon in AK.

--- Step 1: Extracting Entities ---
Extracted Entities: ['Angelina', 'Brad', 'Jon', 'AK']

--- Step 2: Expanding Entities ---
Original: Angelina -> Expanded: Angelina Jolie
Original: Brad -> Expanded: Brad Pitt
Original: Jon -> Expanded: Jon Voight
Original: AK -> Expanded: Alaska

--- Step 3: Describing Entities ---
Name: Angelina Jolie
Description: Angelina Jolie is an American actress, filmmaker, and humanitarian.

Name: Brad Pitt
Description: Brad Pitt is an American actor and film producer.

Name: Jon Voight
Description: Jon Voight is an American actor and the father of Angelina Jolie.

Name: Alaska
Description: Alaska is a U.S. state located in the northwest extremity of North America.

--- Step 4: Linking Entitites to DBpedia ---
Name: Angelina Jolie
Link: http://dbpedia.org/resource/Angelina_Jolie

Name: Brad Pitt
Link: http://dbpedia.org/resource/Brad_Pitt

Name: Jon Voight
Link: http://dbpedia.org/resource/Jon_Voight

Name: Alaska
Link: http://dbpedia.org/resource/Alaska


```

### Pipeline Mode (Full Dataset Generation)

Process datasets to generate NER and ER datasets:

```bash
# Process 100 samples from each dataset (default)
python main.py --mode pipeline

# Process custom number of samples
python main.py --mode pipeline --wikipedia-samples 500 --aida-samples 200

# Specify output directory
python main.py --mode pipeline --output-dir my_datasets
```

### Full Production Run

To process all available data (remove the sample limits):

```bash
python main.py --mode pipeline --wikipedia-samples 999999 --aida-samples 999999
```

### File Mode (Process a Single File)
```bash
# Process the text stored in example.txt
python main.py --mode file --filename example.txt
```

## Output Format

### NER Dataset (`ner_dataset.json`)

Span-based format with character offsets:

```json
[
  {
    "id": "wikipedia_000001",
    "text": "Angelina met her partner Brad and her father Jon in AK.",
    "entities": [
      {
        "start": 0,
        "end": 8,
        "text": "Angelina"
      },
      {
        "start": 25,
        "end": 29,
        "text": "Brad"
      },
      {
        "start": 45,
        "end": 48,
        "text": "Jon"
      },
      {
        "start": 52,
        "end": 54,
        "text": "AK"
      }
    ]
  }
]
```

### ER Dataset (`er_dataset.json`)

Entity mentions with canonical forms and descriptions:

```json
[
  {
    "id": "wikipedia_000001",
    "text": "Angelina met her partner Brad and her father Jon in AK.",
    "mentions": [
      {
        "mention": "Angelina",
        "start": 0,
        "end": 8,
        "entity_id": "E0001",
        "canonical_name": "Angelina Jolie",
        "description": "American actress, filmmaker, and humanitarian known for her roles in various films and her advocacy work.",
        "dbpedia_link": "http://dbpedia.org/resource/Angelina_Jolie"
      },
      {
        "mention": "Brad",
        "start": 25,
        "end": 29,
        "entity_id": "E0002",
        "canonical_name": "Brad Pitt",
        "description": "Highly acclaimed American actor and film producer, recognized for his diverse roles and numerous awards.",
        "dbpedia_link": "http://dbpedia.org/resource/Brad_Pitt"
      },
      {
        "mention": "Jon",
        "start": 45,
        "end": 48,
        "entity_id": "E0003",
        "canonical_name": "Jon Voight",
        "description": "American actor best known for his Emmy Award-winning role as Don Draper in the television series Mad Men.",
        "dbpedia_link": "http://dbpedia.org/resource/Jon_Voight"
      },
      {
        "mention": "AK",
        "start": 52,
        "end": 54,
        "entity_id": "E0004",
        "canonical_name": "Alaska",
        "description": "The largest U.S. state by area, located in the northwest extremity of North America.",
        "dbpedia_link": "http://dbpedia.org/resource/Alaska"
      }
    ]
  }
]
```

### Entity Knowledge Base (`entity_kb.json`)

Reference file mapping entity IDs to canonical names:

```json
{
  "entities": [
    {
      "entity_id": "E0001",
      "canonical_name": "Angelina Jolie",
      "description": "American actress, filmmaker, and humanitarian...",
        "dbpedia_link": "http://dbpedia.org/resource/Angelina_Jolie"
    },
    {
      "entity_id": "E0002",
      "canonical_name": "Brad Pitt",
      "description": "Highly acclaimed American actor and film producer...",
        "dbpedia_link": "http://dbpedia.org/resource/Brad_Pitt"
    }
  ]
}
```

## Features

### Entity ID Management
- Sequential entity IDs (E0001, E0002, ...)
- Automatic deduplication: same canonical entity gets same ID across documents
- Knowledge base grows incrementally as new entities are discovered

### Error Handling
- Continues processing even if individual documents fail
- Skips documents with no extractable entities
- Progress bar shows processing status