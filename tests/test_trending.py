import pytest
from aresponses import Response, ResponsesMockServer

import pricepulsebot.api as api  # noqa: E402
import pricepulsebot.config as config  # noqa: E402
import pricepulsebot.db as db  # noqa: E402


@pytest.mark.asyncio
async def test_fetch_trending_coins_cached(tmp_path, monkeypatch):
    config.DB_FILE = str(tmp_path / "subs.db")
    await db.init_db()
    async with ResponsesMockServer() as ars:
        ars.add(
            "api.coingecko.com",
            "/api/v3/search/trending",
            "GET",
            Response(
                text='{"coins": [{"item": {"id": "solana", "symbol": "sol"}}]}',
                status=200,
                headers={"Content-Type": "application/json"},
            ),
        )
        ars.add(
            "api.coingecko.com",
            "/api/v3/coins/markets",
            "GET",
            Response(
                text=(
                    '[{"id": "solana", "symbol": "sol", "current_price": 1.0,'
                    ' "price_change_percentage_24h": 2.0}]'
                ),
                status=200,
                headers={"Content-Type": "application/json"},
            ),
        )
        await api.fetch_trending_coins()
    cached = await db.get_trending_coins()
    assert cached[0]["id"] == "solana"
    assert cached[0]["current_price"] == 1.0

    async def fail(*args, **kwargs):
        return None

    monkeypatch.setattr(api, "api_get", fail)
    config.COINS = []
    await api.fetch_trending_coins()
    assert config.COINS == ["solana"]
