import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # noqa: E402

from importlib import reload  # noqa: E402

import pytest  # noqa: E402
from aresponses import Response, ResponsesMockServer  # noqa: E402

import run  # noqa: E402

PRICE_CACHE = run.PRICE_CACHE
get_price = run.get_price
get_prices = run.get_prices


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
async def test_get_price_coinmarketcap(monkeypatch):
    monkeypatch.setenv("PRICE_API_PROVIDER", "coinmarketcap")
    reload(run)
    PRICE_CACHE = run.PRICE_CACHE
    PRICE_CACHE.clear()
    get_price = run.get_price
    async with ResponsesMockServer() as ars:
        ars.add(
            "pro-api.coinmarketcap.com",
            "/v1/cryptocurrency/quotes/latest",
            "GET",
            Response(
                text='{"data": {"BTC": {"quote": {"USD": {"price": 6.0}}}}}',
                status=200,
                headers={"Content-Type": "application/json"},
            ),
        )
        price = await get_price("bitcoin")
        assert price == 6.0


@pytest.mark.asyncio
async def test_get_prices_batch(monkeypatch):
    monkeypatch.delenv("PRICE_API_PROVIDER", raising=False)
    reload(run)
    PRICE_CACHE = run.PRICE_CACHE
    PRICE_CACHE.clear()
    get_prices = run.get_prices
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
