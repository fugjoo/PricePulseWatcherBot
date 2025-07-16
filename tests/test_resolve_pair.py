import pytest

import pricepulsebot.api as api  # noqa: E402

resolve_pair = api.resolve_pair


@pytest.mark.asyncio
async def test_resolve_pair_existing_pair():
    result = await resolve_pair("BTCUSDT")
    assert result == "BTCUSDT"


@pytest.mark.asyncio
async def test_resolve_pair_symbol(monkeypatch):
    async def fake_resolve_coin(query, user=None):
        return "bitcoin" if query == "btc" else None

    monkeypatch.setattr(api, "resolve_coin", fake_resolve_coin)
    api.config.COIN_SYMBOLS["bitcoin"] = "BTC"
    api.config.SYMBOL_TO_COIN["btc"] = "bitcoin"
    result = await resolve_pair("btc")
    assert result == "BTCUSDT"


@pytest.mark.asyncio
async def test_resolve_pair_coin_name(monkeypatch):
    async def fake_resolve_coin(query, user=None):
        return "bitcoin" if query == "bitcoin" else None

    monkeypatch.setattr(api, "resolve_coin", fake_resolve_coin)
    api.config.COIN_SYMBOLS["bitcoin"] = "BTC"
    result = await resolve_pair("bitcoin")
    assert result == "BTCUSDT"
