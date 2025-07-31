import pytest

import pricepulsebot.config as config
import pricepulsebot.handlers as handlers
from pricepulsebot.handlers import BACK_EMOJI, SETTINGS_EMOJI, ReplyKeyboardMarkup


class DummyMessage:
    def __init__(self, text=""):
        self.text = text
        self.texts = []
        self.markups = []

    async def reply_text(self, text, **kwargs):
        self.texts.append(text)
        self.markups.append(kwargs.get("reply_markup"))


class DummyUpdate:
    def __init__(self, text=""):
        self.message = DummyMessage(text)
        self.effective_chat = type("Chat", (), {"id": 1})()


class DummyContext:
    def __init__(self):
        self.args = []


def test_keyboard_has_settings_button():
    kb = handlers.get_keyboard()
    assert kb.keyboard[1][1].text == SETTINGS_EMOJI


@pytest.mark.asyncio
async def test_settings_menu_toggle_and_back():
    prev = config.DEFAULT_THRESHOLD
    update = DummyUpdate(SETTINGS_EMOJI)
    ctx = DummyContext()
    await handlers.menu(update, ctx)
    assert update.message.markups and isinstance(
        update.message.markups[-1], ReplyKeyboardMarkup
    )

    update.message.text = f"threshold: ±{config.DEFAULT_THRESHOLD}%"
    await handlers.menu(update, ctx)
    assert config.DEFAULT_THRESHOLD != prev
    assert isinstance(update.message.markups[-1], ReplyKeyboardMarkup)

    update.message.text = f"{BACK_EMOJI} Back"
    await handlers.menu(update, ctx)
    kb = update.message.markups[-1]
    assert isinstance(kb, ReplyKeyboardMarkup)
    assert kb.keyboard[1][1].text == SETTINGS_EMOJI

    config.DEFAULT_THRESHOLD = prev


@pytest.mark.asyncio
async def test_settings_menu_volume_toggle():
    prev = config.ENABLE_VOLUME_ALERTS
    update = DummyUpdate(SETTINGS_EMOJI)
    ctx = DummyContext()
    await handlers.menu(update, ctx)
    assert any(
        "volume" in btn.text
        for row in update.message.markups[-1].keyboard
        for btn in row
    )

    update.message.text = (
        f"volume: {'on' if not config.ENABLE_VOLUME_ALERTS else 'off'}"
    )
    await handlers.menu(update, ctx)
    assert config.ENABLE_VOLUME_ALERTS != prev
    assert isinstance(update.message.markups[-1], ReplyKeyboardMarkup)
    config.ENABLE_VOLUME_ALERTS = prev


@pytest.mark.asyncio
async def test_settings_menu_deletechart_toggle():
    prev = config.DELETE_CHART_ON_RELOAD
    update = DummyUpdate(SETTINGS_EMOJI)
    ctx = DummyContext()
    await handlers.menu(update, ctx)
    assert any(
        "delete chart" in btn.text
        for row in update.message.markups[-1].keyboard
        for btn in row
    )

    update.message.text = (
        f"delete chart: {'on' if not config.DELETE_CHART_ON_RELOAD else 'off'}"
    )
    await handlers.menu(update, ctx)
    assert config.DELETE_CHART_ON_RELOAD != prev
    assert isinstance(update.message.markups[-1], ReplyKeyboardMarkup)
    config.DELETE_CHART_ON_RELOAD = prev


@pytest.mark.asyncio
async def test_settings_menu_overview_toggle():
    prev = config.DEFAULT_OVERVIEW
    update = DummyUpdate(SETTINGS_EMOJI)
    ctx = DummyContext()
    await handlers.menu(update, ctx)
    assert any(
        "overview" in btn.text
        for row in update.message.markups[-1].keyboard
        for btn in row
    )

    update.message.text = f"overview: {prev}"
    await handlers.menu(update, ctx)
    assert config.DEFAULT_OVERVIEW != prev
    assert isinstance(update.message.markups[-1], ReplyKeyboardMarkup)
    config.DEFAULT_OVERVIEW = prev
