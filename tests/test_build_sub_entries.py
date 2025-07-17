import pytest

import pricepulsebot.api as api
import pricepulsebot.config as config
import pricepulsebot.db as db
import pricepulsebot.handlers as handlers


@pytest.mark.asyncio
async def test_build_sub_entries_handles_non_dict_market_data(tmp_path, monkeypatch):
    config.DB_FILE = str(tmp_path / "subs.db")
    await db.init_db()
    await db.subscribe_coin(1, "bitcoin", 1.0, 60)
    await db.set_coin_data(
        "bitcoin",
        {
            "price": 10.0,
            "market_info": 0,
            "info": {"symbol": "btc", "name": "Bitcoin", "market_data": 0},
            "chart_7d": [],
        },
    )

    async def fail(*args, **kwargs):
        raise AssertionError("network called")

    monkeypatch.setattr(api, "get_coin_info", fail)
    monkeypatch.setattr(api, "get_price", fail)

    entries = await handlers.build_sub_entries(1)
    assert entries and entries[0][0] == "bitcoin"


@pytest.mark.asyncio
async def test_build_sub_entries_handles_non_dict_info(tmp_path, monkeypatch):
    config.DB_FILE = str(tmp_path / "subs.db")
    await db.init_db()
    await db.subscribe_coin(1, "bitcoin", 1.0, 60)
    await db.set_coin_data(
        "bitcoin",
        {
            "price": 10.0,
            "market_info": {"current_price": 10.0},
            "info": 0,
            "chart_7d": [],
        },
    )

    async def fail(*args, **kwargs):
        raise AssertionError("network called")

    monkeypatch.setattr(api, "get_coin_info", fail)
    monkeypatch.setattr(api, "get_price", fail)

    entries = await handlers.build_sub_entries(1)
    assert entries and entries[0][0] == "bitcoin"
