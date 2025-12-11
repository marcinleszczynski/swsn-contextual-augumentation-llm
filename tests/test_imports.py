"""
Test imports validation.
"""


def test_imports():
    """Verify that main package modules can be imported."""

    from src.agent import ContextualAugmentation
    assert ContextualAugmentation is not None

    from src.pipeline import DatasetPipeline
    assert DatasetPipeline is not None

    from src.services.dbpedia import DBpediaClient
    assert DBpediaClient is not None

    from src.data.loaders import load_wikipedia_dataset
    assert load_wikipedia_dataset is not None
