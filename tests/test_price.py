import os
import sys

import pytest
from aresponses import Response, ResponsesMockServer

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from run import PRICE_CACHE, get_price  # noqa: E402


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
