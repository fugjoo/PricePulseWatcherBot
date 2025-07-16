import pytest

from pricepulsebot.handlers import global_messages  # noqa: E402
from pricepulsebot.handlers import send_rate_limited, user_messages


class DummyBot:
    def __init__(self):
        self.sent = []

    async def send_message(self, chat_id, text):
        self.sent.append((chat_id, text))


@pytest.mark.asyncio
async def test_send_rate_limited_format():
    user_messages.clear()
    global_messages.clear()
    bot = DummyBot()
    await send_rate_limited(bot, 123, "SOL moved -1% to $10", emoji="🔻")
    assert bot.sent[0][1] == "🔻 SOL moved -1% to $10"
