import pytest

import pricepulsebot.config as config
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
    def __init__(self, args):
        self.args = args


@pytest.mark.asyncio
async def test_milestones_cmd_toggle():
    update = DummyUpdate()
    ctx = DummyContext(["on"])
    prev = config.ENABLE_MILESTONE_ALERTS
    config.ENABLE_MILESTONE_ALERTS = False
    await handlers.milestones_cmd(update, ctx)
    assert config.ENABLE_MILESTONE_ALERTS is True
    config.ENABLE_MILESTONE_ALERTS = prev
