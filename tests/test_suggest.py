import pytest
from aresponses import Response, ResponsesMockServer

import pricepulsebot.api as api  # noqa: E402


@pytest.mark.asyncio
async def test_suggest_coins_basic():
    api.config.COINS[:] = ["bitcoin", "ethereum", "dogecoin"]
    api.config.TOP_COINS[:] = []
    suggestions = await api.suggest_coins("bitcin")
    assert suggestions[0] == "bitcoin"


@pytest.mark.asyncio
async def test_suggest_coins_symbol_lookup():
    api.config.COINS[:] = ["bitcoin", "ethereum", "dogecoin"]
    api.config.TOP_COINS[:] = []
    api.config.COIN_SYMBOLS.update({"bitcoin": "BTC", "ethereum": "ETH"})
    api.config.SYMBOL_TO_COIN.update({"btc": "bitcoin", "eth": "ethereum"})
    suggestions = await api.suggest_coins("eth")
    assert suggestions == ["ethereum"]


@pytest.mark.asyncio
async def test_suggest_coins_api_fallback():
    api.config.COINS[:] = ["bitcoin"]
    api.config.TOP_COINS[:] = []
    async with ResponsesMockServer() as ars:
        ars.add(
            "api.coingecko.com",
            "/api/v3/search",
            "GET",
            Response(
                text='{"coins": [{"id": "ripple", "symbol": "xrp", "name": "XRP"}]}',
                status=200,
                headers={"Content-Type": "application/json"},
            ),
        )
        suggestions = await api.suggest_coins("xrp")
        assert suggestions == ["ripple"]
