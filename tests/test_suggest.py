import pricepulsebot.api as api  # noqa: E402


def test_suggest_coins_basic():
    api.config.COINS[:] = ["bitcoin", "ethereum", "dogecoin"]
    api.config.TOP_COINS[:] = []
    suggestions = api.suggest_coins("bitcin")
    assert suggestions[0] == "bitcoin"


def test_suggest_coins_symbol_lookup():
    api.config.COINS[:] = ["bitcoin", "ethereum", "dogecoin"]
    api.config.TOP_COINS[:] = []
    api.config.COIN_SYMBOLS.update({"bitcoin": "BTC", "ethereum": "ETH"})
    api.config.SYMBOL_TO_COIN.update({"btc": "bitcoin", "eth": "ethereum"})
    suggestions = api.suggest_coins("eth")
    assert suggestions == ["ethereum"]
