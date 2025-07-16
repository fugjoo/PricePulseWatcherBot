import pytest  # noqa: E402

import pricepulsebot.api as api  # noqa: E402

resolve_coin = api.resolve_coin


@pytest.mark.asyncio
async def test_resolve_coin_direct(monkeypatch):
    async def fake_info(coin, user=None, session=None):
        if coin == "bitcoin":
            return {"current_price": 1.0}
        return None

    monkeypatch.setattr(api, "get_market_info", fake_info)
    result = await resolve_coin("bitcoin")
    assert result == "bitcoin"


@pytest.mark.asyncio
async def test_resolve_coin_fallback(monkeypatch):
    async def fake_info(coin, user=None, session=None):
        if coin == "ripple":
            return {"current_price": 1.0}
        return None

    async def fake_find(query):
        return "ripple"

    monkeypatch.setattr(api, "get_market_info", fake_info)
    monkeypatch.setattr(api, "find_coin", fake_find)

    result = await resolve_coin("xrp")
    assert result == "ripple"
