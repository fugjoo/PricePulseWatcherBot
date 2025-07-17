import time

import aiosqlite
import pytest

import pricepulsebot.api as api
import pricepulsebot.config as config
import pricepulsebot.db as db
import pricepulsebot.handlers as handlers
from pricepulsebot.handlers import MILESTONE_CACHE


class DummyBot:
    def __init__(self):
        self.sent = []

    async def send_message(self, chat_id, text):
        self.sent.append((chat_id, text))


class DummyApp:
    def __init__(self, bot):
        self.bot = bot


@pytest.mark.asyncio
async def test_refresh_coin_data_populates_table(tmp_path, monkeypatch):
    config.DB_FILE = str(tmp_path / "subs.db")
    await db.init_db()

    async def fake_price(coin, session=None, user=None):
        return 1.0

    async def fake_market_info(coin, session=None, user=None):
        return {"current_price": 1.0}

    async def fake_coin_info(coin, session=None, user=None):
        return {"id": coin}, None

    async def fake_chart(coin, days, session=None, user=None):
        return [(1, 2)], None

    monkeypatch.setattr(api, "get_price", fake_price)
    monkeypatch.setattr(api, "get_market_info", fake_market_info)
    monkeypatch.setattr(api, "get_coin_info", fake_coin_info)
    monkeypatch.setattr(api, "get_market_chart", fake_chart)

    await api.refresh_coin_data("bitcoin")
    data = await db.get_coin_data("bitcoin")
    assert data["price"] == 1.0
    assert data["market_info"]["current_price"] == 1.0
    assert data["info"]["id"] == "bitcoin"
    assert data["chart_7d"] == [[1, 2]]


@pytest.mark.asyncio
async def test_check_prices_uses_cached_data(tmp_path, monkeypatch):
    config.DB_FILE = str(tmp_path / "subs.db")
    await db.init_db()
    await db.subscribe_coin(1, "bitcoin", 0.1, 300)
    async with aiosqlite.connect(config.DB_FILE) as conn:
        await conn.execute(
            "UPDATE subscriptions SET last_price=?, last_alert_ts=? WHERE id=1",
            (100.0, time.time() - 600),
        )
        await conn.commit()

    await db.set_coin_data(
        "bitcoin",
        {
            "price": 110.0,
            "market_info": {"price_change_percentage_24h": 1.0},
            "info": {},
            "chart_7d": [],
        },
    )

    async def fail(*args, **kwargs):
        raise AssertionError("network called")

    monkeypatch.setattr(api, "get_price", fail)
    monkeypatch.setattr(api, "get_market_info", fail)

    bot = DummyBot()
    app = DummyApp(bot)
    MILESTONE_CACHE.clear()
    await handlers.check_prices(app)
    MILESTONE_CACHE.clear()
    assert bot.sent
