import pytest

from strhub.data.utils import Tokenizer


@pytest.fixture
def tokenizer():
    return Tokenizer(("TÚL", "MI", "PAB", "AB", "NI"))
