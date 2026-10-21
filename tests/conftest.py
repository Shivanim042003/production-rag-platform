import pytest


def pytest_collection_modifyitems(items):
    for item in items:
        if item.path.name == "test_prompt_injection_model.py":
            item.add_marker(pytest.mark.model)
