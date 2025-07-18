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
                text=(
                    '{"coins": ['
                    '{"item": {"id": "solana", "symbol": "sol", "name": "Solana"}},'
                    '{"item": {"id": "bitcoin", "symbol": "btc", "name": "Bitcoin"}}'
                    "]}"
                ),
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
                    '[{"id": "solana", "current_price": 1.0, '
                    '"price_change_percentage_24h": 2.0},'
                    '{"id": "bitcoin", "current_price": 2.0, '
                    '"price_change_percentage_24h": 1.0}]'
                ),
                status=200,
                headers={"Content-Type": "application/json"},
            ),
        )
        trending = await api.fetch_trending_coins()
    cached = await db.get_trending_coins()
    assert cached == trending
    assert cached[0]["id"] == "solana"

    async def fail(*args, **kwargs):
        return None

    monkeypatch.setattr(api, "api_get", fail)
    config.COINS = []
    again = await api.fetch_trending_coins()
    assert again == cached
    assert config.COINS == ["solana", "bitcoin"]


@pytest.mark.asyncio
async def test_fetch_trending_coins_uses_cache(tmp_path, monkeypatch):
    config.DB_FILE = str(tmp_path / "subs.db")
    await db.init_db()
    cached = [{"id": "bitcoin", "symbol": "btc", "price": 1.0, "change_24h": 1.0}]
    await db.set_trending_coins(cached)

    async def fail(*args, **kwargs):
        raise AssertionError("network called")

    monkeypatch.setattr(api, "api_get", fail)
    result = await api.fetch_trending_coins()
    assert result == cached


@pytest.mark.asyncio
async def test_fetch_trending_coins_excludes_usdt(tmp_path):
    config.DB_FILE = str(tmp_path / "subs.db")
    await db.init_db()
    async with ResponsesMockServer() as ars:
        ars.add(
            "api.coingecko.com",
            "/api/v3/search/trending",
            "GET",
            Response(
                text=(
                    '{"coins": ['
                    '{"item": {"id": "tether", "symbol": "usdt", "name": "Tether"}},'
                    '{"item": {"id": "bitcoin", "symbol": "btc", "name": "Bitcoin"}}'
                    "]}"
                ),
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
                    '[{"id": "tether", "current_price": 1.0, '
                    '"price_change_percentage_24h": 0.0},'
                    '{"id": "bitcoin", "current_price": 2.0, '
                    '"price_change_percentage_24h": 1.0}]'
                ),
                status=200,
                headers={"Content-Type": "application/json"},
            ),
        )
        result = await api.fetch_trending_coins()
    assert all(item["id"] != "tether" for item in result)
    assert "tether" not in config.COINS
