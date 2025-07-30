import pricepulsebot.config as config


def test_parse_timeframe_days():
    assert config.parse_timeframe("3") == 3 * 86400


def test_parse_timeframe_hours():
    assert config.parse_timeframe("2h") == 2 * 3600


def test_parse_timeframe_invalid():
    try:
        config.parse_timeframe("xh")
    except ValueError:
        assert True
    else:
        assert False
