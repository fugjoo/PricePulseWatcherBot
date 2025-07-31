import pytest

import pricepulsebot.config as config
import pricepulsebot.db as db
import pricepulsebot.handlers as handlers
from pricepulsebot.handlers import InlineKeyboardMarkup


class DummyBot:
    async def send_chat_action(self, chat_id, action):
        pass


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
    def __init__(self, bot):
        self.bot = bot
        self.args = []


class DummyCallbackQuery:
    def __init__(self, data):
        self.data = data
        self.message = type("Msg", (), {"chat_id": 1})()
        self.reply_markup = None

    async def answer(self):
        pass

    async def edit_message_text(self, text, **kwargs):
        self.reply_markup = kwargs.get("reply_markup")

    async def edit_message_reply_markup(self, reply_markup=None):
        self.reply_markup = reply_markup


class DummyCallbackUpdate:
    def __init__(self, query):
        self.callback_query = query


@pytest.mark.asyncio
async def test_list_cmd_shows_parameters(tmp_path):
    config.DB_FILE = str(tmp_path / "subs.db")
    await db.init_db()
    await db.subscribe_coin(1, "bitcoin", 1.0, 60)
    await db.set_coin_data(
        "bitcoin",
        {
            "price": 10.0,
            "market_info": {
                "current_price": 10.0,
                "market_cap": 1000,
                "price_change_percentage_24h": 1.0,
            },
            "info": {"symbol": "btc", "name": "Bitcoin"},
            "chart_7d": [],
        },
    )
    bot = DummyBot()
    update = DummyUpdate()
    ctx = DummyContext(bot)
    await handlers.list_cmd(update, ctx)
    assert update.message.markups
    kb = update.message.markups[0]
    assert isinstance(kb, InlineKeyboardMarkup)
    callbacks = [b.callback_data for b in kb.inline_keyboard[0]]
    assert any(cb.startswith("thr:") for cb in callbacks)
    assert any(cb.startswith("int:") for cb in callbacks)
    assert any(cb.startswith("del:") for cb in callbacks)


@pytest.mark.asyncio
async def test_threshold_button_updates(tmp_path):
    config.DB_FILE = str(tmp_path / "subs.db")
    await db.init_db()
    await db.subscribe_coin(1, "bitcoin", 1.0, 60)
    await db.set_coin_data(
        "bitcoin",
        {
            "price": 10.0,
            "market_info": {
                "current_price": 10.0,
                "market_cap": 1000,
                "price_change_percentage_24h": 1.0,
            },
            "info": {"symbol": "btc", "name": "Bitcoin"},
            "chart_7d": [],
        },
    )
    query = DummyCallbackQuery("thr:bitcoin")
    update = DummyCallbackUpdate(query)
    ctx = DummyContext(DummyBot())
    await handlers.button(update, ctx)
    subs = await db.list_subscriptions(1)
    assert subs[0][2] != 1.0
