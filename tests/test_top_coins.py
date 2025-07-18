import pytest
from aresponses import Response, ResponsesMockServer

import pricepulsebot.api as api
import pricepulsebot.config as config


@pytest.mark.asyncio
async def test_fetch_top_coins_updates_config(monkeypatch):
    async with ResponsesMockServer() as ars:
        ars.add(
            "api.coingecko.com",
            "/api/v3/coins/markets",
            "GET",
            Response(
                text=(
                    '[{"id": "bitcoin", "symbol": "btc", "current_price": 1.0, '
                    '"price_change_percentage_24h": 2.0}]'
                ),
                status=200,
                headers={"Content-Type": "application/json"},
            ),
        )
        result = await api.fetch_top_coins(per_page=100)
    assert result and result[0]["id"] == "bitcoin"
    assert config.TOP_COINS[0] == "bitcoin"
    assert config.COIN_SYMBOLS["bitcoin"] == "BTC"


@pytest.mark.asyncio
async def test_fetch_top_coins_excludes_usdt(monkeypatch):
    async with ResponsesMockServer() as ars:
        ars.add(
            "api.coingecko.com",
            "/api/v3/coins/markets",
            "GET",
            Response(
                text=(
                    '[{"id": "tether", "symbol": "usdt", "current_price": 1.0, '
                    '"price_change_percentage_24h": 0.0},'
                    '{"id": "bitcoin", "symbol": "btc", "current_price": 2.0, '
                    '"price_change_percentage_24h": 1.0}]'
                ),
                status=200,
                headers={"Content-Type": "application/json"},
            ),
        )
        result = await api.fetch_top_coins(per_page=100)
    assert all(item["id"] != "tether" for item in result)
    assert "tether" not in config.TOP_COINS
