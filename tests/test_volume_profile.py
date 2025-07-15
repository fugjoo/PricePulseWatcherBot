import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from run import calculate_volume_profile  # noqa: E402


def test_calculate_volume_profile_basic():
    candles = [
        {"high": 2.0, "low": 1.0, "close": 1.5, "volume": 100.0},
        {"high": 1.7, "low": 1.5, "close": 1.6, "volume": 300.0},
    ]
    result = calculate_volume_profile(candles)
    assert result["val"] == 1.5
    assert abs(result["vah"] - 1.7) < 1e-6
    assert abs(result["poc"] - 1.595) < 1e-6
