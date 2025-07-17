import pytest  # noqa: E402

import pricepulsebot.api as api  # noqa: E402
import pricepulsebot.config as config  # noqa: E402
import pricepulsebot.db as db  # noqa: E402

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


@pytest.mark.asyncio
async def test_resolve_coin_symbol_search(monkeypatch):
    async def fake_info(coin, user=None, session=None):
        if coin == "litecoin-cash":
            return {"current_price": 1.0}
        return None

    async def fake_find(query):
        if query == "litecoin":
            return None
        if query == "ltc":
            return "litecoin-cash"
        return None

    monkeypatch.setattr(api, "get_market_info", fake_info)
    monkeypatch.setattr(api, "find_coin", fake_find)

    result = await resolve_coin("ltc")
    assert result == "litecoin-cash"


@pytest.mark.asyncio
async def test_resolve_coin_uses_cache(tmp_path, monkeypatch):
    config.DB_FILE = str(tmp_path / "subs.db")
    await db.init_db()
    await db.set_coin_data(
        "bitcoin",
        {
            "price": 1.0,
            "market_info": {"current_price": 1.0},
            "info": {},
            "chart_7d": [],
        },
    )

    async def fail(*args, **kwargs):
        raise AssertionError("network called")

    monkeypatch.setattr(api, "get_market_info", fail)
    monkeypatch.setattr(api, "find_coin", fail)

    result = await resolve_coin("bitcoin")
    assert result == "bitcoin"
