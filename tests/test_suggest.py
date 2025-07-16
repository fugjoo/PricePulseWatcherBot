import pytest
from aresponses import Response, ResponsesMockServer  # noqa: E402

import pricepulsebot.api as api  # noqa: E402


@pytest.mark.asyncio
async def test_suggest_coins_basic():
    api.config.COINS[:] = ["bitcoin", "ethereum", "dogecoin"]
    api.config.TOP_COINS[:] = []
    suggestions = await api.suggest_coins("bitcin")
    assert suggestions[0] == "bitcoin"


@pytest.mark.asyncio
async def test_suggest_coins_fallback():
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
        api.config.COINS[:] = []
        api.config.TOP_COINS[:] = []
        suggestions = await api.suggest_coins("xrp")
        assert suggestions == ["ripple"]
