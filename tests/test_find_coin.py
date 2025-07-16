import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # noqa: E402

import pytest  # noqa: E402
from aresponses import Response, ResponsesMockServer  # noqa: E402

import run  # noqa: E402

find_coin = run.find_coin


@pytest.mark.asyncio
async def test_find_coin():
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
        result = await find_coin("xrp")
        assert result == "ripple"
        assert run.SYMBOL_TO_COIN["xrp"] == "ripple"


@pytest.mark.asyncio
async def test_find_coin_encodes_query():
    async with ResponsesMockServer() as ars:
        ars.add(
            "api.coingecko.com",
            "/api/v3/search?query=bitcoin%20cash",
            "GET",
            Response(
                text="{}", status=200, headers={"Content-Type": "application/json"}
            ),
            match_querystring=True,
        )
        result = await find_coin("bitcoin cash")
        assert result is None
