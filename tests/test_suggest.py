import pricepulsebot.api as api  # noqa: E402


def test_suggest_coins_basic():
    api.config.COINS[:] = ["bitcoin", "ethereum", "dogecoin"]
    api.config.TOP_COINS[:] = []
    suggestions = api.suggest_coins("bitcin")
    assert suggestions[0] == "bitcoin"
