import pytest

import pricepulsebot.config as config
import pricepulsebot.handlers as handlers
from pricepulsebot.handlers import InlineKeyboardMarkup, ReplyKeyboardMarkup


class DummyMessage:
    def __init__(self):
        self.texts = []
        self.markups = []

    async def reply_text(self, text, **kwargs):
        self.texts.append(text)
        self.markups.append(kwargs.get("reply_markup"))


class DummyUpdate:
    def __init__(self):
        self.message = DummyMessage()
        self.effective_chat = type("Chat", (), {"id": 1})()


class DummyContext:
    def __init__(self, args, bot=None):
        self.args = args
        self.bot = bot


class DummyBot:
    def __init__(self):
        self.sent = []

    async def send_message(self, chat_id, text, **kwargs):
        self.sent.append((chat_id, text))


class DummyCallbackQuery:
    def __init__(self, data):
        self.data = data
        self.message = type("Msg", (), {"chat_id": 1})()
        self.reply_markup = None

    async def answer(self):
        pass

    async def edit_message_reply_markup(self, reply_markup=None):
        self.reply_markup = reply_markup


class DummyCallbackUpdate:
    def __init__(self, query):
        self.callback_query = query


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
async def test_settings_update_liquidations():
    update = DummyUpdate()
    ctx = DummyContext(["liquidations", "on"])
    prev = config.ENABLE_LIQUIDATION_ALERTS
    await handlers.settings_cmd(update, ctx)
    assert config.ENABLE_LIQUIDATION_ALERTS is True
    config.ENABLE_LIQUIDATION_ALERTS = prev


@pytest.mark.asyncio
async def test_settings_update_volume():
    update = DummyUpdate()
    ctx = DummyContext(["volume", "off"])
    prev = config.ENABLE_VOLUME_ALERTS
    await handlers.settings_cmd(update, ctx)
    assert config.ENABLE_VOLUME_ALERTS is False
    config.ENABLE_VOLUME_ALERTS = prev


@pytest.mark.asyncio
async def test_settings_pricecheck_readonly():
    update = DummyUpdate()
    ctx = DummyContext(["pricecheck", "30s"])
    prev = config.PRICE_CHECK_INTERVAL
    await handlers.settings_cmd(update, ctx)
    assert config.PRICE_CHECK_INTERVAL == prev
    assert update.message.texts


@pytest.mark.asyncio
async def test_settings_update_currency():
    update = DummyUpdate()
    ctx = DummyContext(["currency", "eur"])
    prev = config.VS_CURRENCY
    await handlers.settings_cmd(update, ctx)
    assert config.VS_CURRENCY == "eur"
    config.VS_CURRENCY = prev


@pytest.mark.asyncio
async def test_settings_keyboard():
    update = DummyUpdate()
    ctx = DummyContext([])
    await handlers.settings_cmd(update, ctx)
    assert update.message.markups and isinstance(
        update.message.markups[0], ReplyKeyboardMarkup
    )


@pytest.mark.asyncio
async def test_settings_button_toggle_milestones():
    bot = DummyBot()
    query = DummyCallbackQuery("settings:milestones")
    update = DummyCallbackUpdate(query)
    ctx = DummyContext([], bot)
    prev = config.ENABLE_MILESTONE_ALERTS
    await handlers.button(update, ctx)
    assert config.ENABLE_MILESTONE_ALERTS != prev
    assert isinstance(query.reply_markup, InlineKeyboardMarkup)
    assert bot.sent
    config.ENABLE_MILESTONE_ALERTS = prev


@pytest.mark.asyncio
async def test_settings_button_toggle_volume():
    bot = DummyBot()
    query = DummyCallbackQuery("settings:volume")
    update = DummyCallbackUpdate(query)
    ctx = DummyContext([], bot)
    prev = config.ENABLE_VOLUME_ALERTS
    await handlers.button(update, ctx)
    assert config.ENABLE_VOLUME_ALERTS != prev
    assert isinstance(query.reply_markup, InlineKeyboardMarkup)
    assert bot.sent
    config.ENABLE_VOLUME_ALERTS = prev
