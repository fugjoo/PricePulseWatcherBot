import pytest

import pricepulsebot.config as config


def test_parse_timeframe_days():
    assert config.parse_timeframe("3") == 3 * 86400


def test_parse_timeframe_hours():
    assert config.parse_timeframe("2h") == 2 * 3600


def test_parse_timeframe_minutes():
    assert config.parse_timeframe("30m") == 30 * 60


def test_parse_timeframe_invalid():
    with pytest.raises(ValueError):
        config.parse_timeframe("xh")
