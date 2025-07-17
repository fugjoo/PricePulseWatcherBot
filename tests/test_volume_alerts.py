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
async def test_volume_spike_alert(tmp_path, monkeypatch):
    config.DB_FILE = str(tmp_path / "subs.db")
    await db.init_db()
    config.VOLUME_THRESHOLD = 10.0
    await db.subscribe_coin(1, "bitcoin", 0.1, 300)
    async with aiosqlite.connect(config.DB_FILE) as conn:
        await conn.execute(
            (
                "UPDATE subscriptions SET last_price=?, last_volume=?, "
                "last_alert_ts=? WHERE id=1"
            ),
            (100.0, 1000.0, time.time() - 600),
        )
        await conn.commit()

    async def fake_markets(coins, session=None, user=None):
        return {c: {"current_price": 101.0} for c in coins}

    async def fake_volume(coin, session=None, user=None):
        return 1500.0

    monkeypatch.setattr(api, "get_markets", fake_markets)
    monkeypatch.setattr(api, "get_volume", fake_volume)
    bot = DummyBot()
    app = DummyApp(bot)
    MILESTONE_CACHE.clear()
    await handlers.check_prices(app)
    MILESTONE_CACHE.clear()
    assert any("volume" in msg for _, msg in bot.sent)


@pytest.mark.asyncio
async def test_volume_drop_alert(tmp_path, monkeypatch):
    config.DB_FILE = str(tmp_path / "subs.db")
    await db.init_db()
    config.VOLUME_THRESHOLD = 10.0
    await db.subscribe_coin(1, "bitcoin", 0.1, 300)
    async with aiosqlite.connect(config.DB_FILE) as conn:
        await conn.execute(
            (
                "UPDATE subscriptions SET last_price=?, last_volume=?, "
                "last_alert_ts=? WHERE id=1"
            ),
            (100.0, 1000.0, time.time() - 600),
        )
        await conn.commit()

    async def fake_markets(coins, session=None, user=None):
        return {c: {"current_price": 99.0} for c in coins}

    async def fake_volume(coin, session=None, user=None):
        return 500.0

    monkeypatch.setattr(api, "get_markets", fake_markets)
    monkeypatch.setattr(api, "get_volume", fake_volume)
    bot = DummyBot()
    app = DummyApp(bot)
    MILESTONE_CACHE.clear()
    await handlers.check_prices(app)
    MILESTONE_CACHE.clear()
    assert any("volume" in msg for _, msg in bot.sent)


@pytest.mark.asyncio
async def test_volume_alerts_disabled(tmp_path, monkeypatch):
    config.DB_FILE = str(tmp_path / "subs.db")
    await db.init_db()
    config.VOLUME_THRESHOLD = 10.0
    await db.subscribe_coin(1, "bitcoin", 0.1, 300)
    async with aiosqlite.connect(config.DB_FILE) as conn:
        await conn.execute(
            (
                "UPDATE subscriptions SET last_price=?, last_volume=?, "
                "last_alert_ts=? WHERE id=1"
            ),
            (100.0, 1000.0, time.time() - 600),
        )
        await conn.commit()

    async def fake_markets(coins, session=None, user=None):
        return {c: {"current_price": 101.0} for c in coins}

    async def fake_volume(coin, session=None, user=None):
        return 2000.0

    monkeypatch.setattr(api, "get_markets", fake_markets)
    monkeypatch.setattr(api, "get_volume", fake_volume)
    bot = DummyBot()
    app = DummyApp(bot)
    MILESTONE_CACHE.clear()
    prev = config.ENABLE_VOLUME_ALERTS
    config.ENABLE_VOLUME_ALERTS = False
    await handlers.check_prices(app)
    MILESTONE_CACHE.clear()
    config.ENABLE_VOLUME_ALERTS = prev
    assert not any("volume" in msg for _, msg in bot.sent)
