import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # noqa: E402

import pytest  # noqa: E402

from run import global_messages, send_rate_limited, user_messages  # noqa: E402


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
