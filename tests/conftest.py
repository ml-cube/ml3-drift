from importlib import import_module

import pytest

import numpy as np
from PIL import Image


def is_module_available(module_name):
    """
    Check if a module is available in the current environment.

    Args:
        module_name (str): The name of the module to check.

    Returns:
        bool: True if the module is available, False otherwise.
    """
    try:
        import_module(module_name)
        return True
    except ImportError:
        return False


# ------------------------------------
# Fixtures


@pytest.fixture
def text_data():
    """
    Fixture to provide text data for testing.
    """

    return "I am a drift detection warrior"


@pytest.fixture
def image_data():
    """
    Fixture to provide image data for testing.
    """

    return Image.fromarray(np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8))


# ------------------------------------
# Test factories
