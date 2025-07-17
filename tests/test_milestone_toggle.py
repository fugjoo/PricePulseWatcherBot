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
async def test_milestone_alerts_disabled(tmp_path, monkeypatch):
    db_file = tmp_path / "subs.db"
    config.DB_FILE = str(db_file)
    await db.init_db()
    await db.subscribe_coin(1, "bitcoin", 0.1, 300)
    async with aiosqlite.connect(config.DB_FILE) as conn:
        await conn.execute(
            "UPDATE subscriptions SET last_price=?, last_alert_ts=? WHERE id=1",
            (100.0, time.time() - 600),
        )
        await conn.commit()

    async def fake_markets(coins, session=None, user=None):
        return {c: {"current_price": 110.0} for c in coins}

    monkeypatch.setattr(api, "get_markets", fake_markets)
    bot = DummyBot()
    app = DummyApp(bot)
    MILESTONE_CACHE.clear()
    prev = config.ENABLE_MILESTONE_ALERTS
    config.ENABLE_MILESTONE_ALERTS = False
    await handlers.check_prices(app)
    MILESTONE_CACHE.clear()
    config.ENABLE_MILESTONE_ALERTS = prev
    assert len(bot.sent) == 1
