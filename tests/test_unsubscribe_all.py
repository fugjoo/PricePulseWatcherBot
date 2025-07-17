import pytest

import pricepulsebot.config as config
import pricepulsebot.db as db
import pricepulsebot.handlers as handlers


class DummyMessage:
    def __init__(self):
        self.texts = []

    async def reply_text(self, text, **kwargs):
        self.texts.append(text)


class DummyUpdate:
    def __init__(self):
        self.message = DummyMessage()
        self.effective_chat = type("Chat", (), {"id": 1})()


class DummyContext:
    def __init__(self):
        self.args = []


@pytest.mark.asyncio
async def test_unsubscribe_all(tmp_path):
    config.DB_FILE = str(tmp_path / "subs.db")
    await db.init_db()
    await db.subscribe_coin(1, "bitcoin", 0.1, 60)
    await db.subscribe_coin(1, "ethereum", 0.1, 60)
    await db.unsubscribe_all(1)
    subs = await db.list_subscriptions(1)
    assert subs == []


@pytest.mark.asyncio
async def test_clear_cmd(tmp_path):
    config.DB_FILE = str(tmp_path / "subs.db")
    await db.init_db()
    await db.subscribe_coin(1, "bitcoin", 0.1, 60)
    await db.subscribe_coin(1, "ethereum", 0.1, 60)
    update = DummyUpdate()
    ctx = DummyContext()
    await handlers.clear_cmd(update, ctx)
    subs = await db.list_subscriptions(1)
    assert subs == []
    assert update.message.texts
