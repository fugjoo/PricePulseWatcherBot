from importlib import reload  # noqa: E402

import pytest  # noqa: E402
from aresponses import Response, ResponsesMockServer  # noqa: E402

import pricepulsebot.api as api  # noqa: E402

PRICE_CACHE = api.PRICE_CACHE
get_price = api.get_price
get_prices = api.get_prices


@pytest.mark.asyncio
async def test_get_price():
    PRICE_CACHE.clear()
    async with ResponsesMockServer() as ars:
        ars.add(
            "api.coingecko.com",
            "/api/v3/simple/price",
            "GET",
            Response(
                text='{"bitcoin": {"usd": 5.0}}',
                status=200,
                headers={"Content-Type": "application/json"},
            ),
        )
        price = await get_price("bitcoin")
        assert price == 5.0


@pytest.mark.asyncio
async def test_get_prices_batch(monkeypatch):
    reload(api)
    PRICE_CACHE = api.PRICE_CACHE
    PRICE_CACHE.clear()
    get_prices = api.get_prices
    async with ResponsesMockServer() as ars:
        ars.add(
            "api.coingecko.com",
            "/api/v3/simple/price",
            "GET",
            Response(
                text='{"bitcoin": {"usd": 1.0}, "ethereum": {"usd": 2.0}}',
                status=200,
                headers={"Content-Type": "application/json"},
            ),
        )
        prices = await get_prices(["bitcoin", "ethereum"])
        assert prices == {"bitcoin": 1.0, "ethereum": 2.0}
