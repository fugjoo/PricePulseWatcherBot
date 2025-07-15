import os
import sys
import time

import aiosqlite
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # noqa: E402

import run  # noqa: E402


class DummyBot:
    def __init__(self):
        self.sent = []

    async def send_message(self, chat_id, text):
        self.sent.append((chat_id, text))


class DummyApp:
    def __init__(self, bot):
        self.bot = bot


def test_format_interval_basic():
    assert run.format_interval(300) == "5m"
    assert run.format_interval(3600) == "1h"
    assert run.format_interval(45) == "45s"
    assert run.format_interval(86400) == "1d"


@pytest.mark.asyncio
async def test_check_prices_interval_in_message(tmp_path, monkeypatch):
    db_file = tmp_path / "subs.db"
    run.DB_FILE = str(db_file)
    await run.init_db()
    await run.subscribe_coin(1, "bitcoin", 0.1, 300)
    async with aiosqlite.connect(run.DB_FILE) as db:
        await db.execute(
            "UPDATE subscriptions SET last_price=?, last_alert_ts=? WHERE id=1",
            (100.0, time.time() - 600),
        )
        await db.commit()

    async def fake_price(coin, user=None):
        return 105.0

    monkeypatch.setattr(run, "get_price", fake_price)
    bot = DummyBot()
    app = DummyApp(bot)
    await run.check_prices(app)
    assert run.format_interval(300) in bot.sent[0][1]
