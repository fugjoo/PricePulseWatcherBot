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
async def test_settings_update_threshold():
    update = DummyUpdate()
    ctx = DummyContext(["threshold", "2.0"])
    prev = config.DEFAULT_THRESHOLD
    await handlers.settings_cmd(update, ctx)
    assert config.DEFAULT_THRESHOLD == 2.0
    config.DEFAULT_THRESHOLD = prev


@pytest.mark.asyncio
async def test_settings_update_interval():
    update = DummyUpdate()
    ctx = DummyContext(["interval", "10m"])
    prev = config.DEFAULT_INTERVAL
    await handlers.settings_cmd(update, ctx)
    assert config.DEFAULT_INTERVAL == config.parse_duration("10m")
    config.DEFAULT_INTERVAL = prev


@pytest.mark.asyncio
async def test_settings_update_milestones():
    update = DummyUpdate()
    ctx = DummyContext(["milestones", "off"])
    prev = config.ENABLE_MILESTONE_ALERTS
    await handlers.settings_cmd(update, ctx)
    assert config.ENABLE_MILESTONE_ALERTS is False
    config.ENABLE_MILESTONE_ALERTS = prev


@pytest.mark.asyncio
async def test_settings_pricecheck_readonly():
    update = DummyUpdate()
    ctx = DummyContext(["pricecheck", "30s"])
    prev = config.PRICE_CHECK_INTERVAL
    await handlers.settings_cmd(update, ctx)
    assert config.PRICE_CHECK_INTERVAL == prev
    assert update.message.texts
