import time

import pytest

import pricepulsebot.api as api
import pricepulsebot.handlers as handlers


class DummyBot:
    def __init__(self):
        self.photos = []

    async def send_photo(self, chat_id, photo):
        self.photos.append((chat_id, photo))


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
    def __init__(self, bot):
        self.bot = bot
        self.args = []


@pytest.mark.asyncio
async def test_status_cmd_basic():
    api.STATUS_HISTORY.clear()
    now = time.time()
    api.STATUS_HISTORY.extend([(now, 200), (now, 429), (now, 500)])
    bot = DummyBot()
    update = DummyUpdate()
    ctx = DummyContext(bot)
    await handlers.status_cmd(update, ctx)
    assert bot.photos
    assert update.message.texts
