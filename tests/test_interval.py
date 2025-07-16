import time

import aiosqlite
import pytest

import pricepulsebot.api as api  # noqa: E402
import pricepulsebot.config as config  # noqa: E402
import pricepulsebot.db as db  # noqa: E402
import pricepulsebot.handlers as handlers  # noqa: E402


class DummyBot:
    def __init__(self):
        self.sent = []

    async def send_message(self, chat_id, text):
        self.sent.append((chat_id, text))


class DummyApp:
    def __init__(self, bot):
        self.bot = bot


def test_format_interval_basic():
    assert config.format_interval(300) == "5m"
    assert config.format_interval(3600) == "1h"
    assert config.format_interval(45) == "45s"
    assert config.format_interval(86400) == "1d"


@pytest.mark.asyncio
async def test_check_prices_interval_in_message(tmp_path, monkeypatch):
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

    async def fake_price(coin, user=None):
        return 105.0

    monkeypatch.setattr(api, "get_price", fake_price)
    bot = DummyBot()
    app = DummyApp(bot)
    await handlers.check_prices(app)
    assert config.format_interval(300) in bot.sent[0][1]
