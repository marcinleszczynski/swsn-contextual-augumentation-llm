# Contextual Augmentation for Entity Linking without external dataset

[![Lint](https://github.com/marcinleszczynski/swsn-contextual-augumentation-llm/actions/workflows/lint.yml/badge.svg)](https://github.com/marcinleszczynski/swsn-contextual-augumentation-llm/actions/workflows/lint.yml)
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
- A Google Gemini API Key

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
    ```

## Configuration

1.  Create `.env` and set your `GEMINI_API_KEY`.
2.  (Optional) Set `GEMINI_MODEL` in `.env` to use a specific model (default: `gemini-2.5-flash`).

## Usage

Run the main script:

```bash
python main.py
```

## Structure

- `src/models/schemas.py`: Pydantic models for structured output.
- `src/agent/contextual_augmentation.py`: The main agent class encapsulating the LLM logic.
- `main.py`: Example usage script.
