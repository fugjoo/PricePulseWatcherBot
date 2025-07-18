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
    api.STATUS_HISTORY.extend(
        [
            (now - api.STATUS_WINDOW - 1, 200),
            (now - 1800, 429),
            (now - 10, 500),
        ]
    )
    bot = DummyBot()
    update = DummyUpdate()
    ctx = DummyContext(bot)
    await handlers.status_cmd(update, ctx)
    assert bot.photos
    assert update.message.texts
    text = update.message.texts[0]
    assert "API:" in text and "DB:" in text
    counts = api.status_counts()
    assert 200 not in counts
    assert counts[429] == 1
    assert counts[500] == 1
