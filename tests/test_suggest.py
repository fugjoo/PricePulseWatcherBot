import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # noqa: E402

import run  # noqa: E402


def test_suggest_coins_basic():
    run.COINS[:] = ["bitcoin", "ethereum", "dogecoin"]
    run.TOP_COINS[:] = []
    suggestions = run.suggest_coins("bitcin")
    assert suggestions[0] == "bitcoin"
