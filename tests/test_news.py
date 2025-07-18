import pytest
from aresponses import Response, ResponsesMockServer

import pricepulsebot.api as api
import pricepulsebot.config as config
import pricepulsebot.db as db
import pricepulsebot.handlers as handlers


@pytest.mark.asyncio
async def test_get_news_basic():
    async with ResponsesMockServer() as ars:
        ars.add(
            "min-api.cryptocompare.com",
            "/data/v2/news/",
            "GET",
            Response(
                text='{"Data": [{"title": "Hello"}]}',
                status=200,
                headers={"Content-Type": "application/json"},
            ),
        )
        news = await api.get_news("bitcoin")
        assert news and news[0]["title"] == "Hello"


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
async def test_news_cmd_subscriptions(tmp_path, monkeypatch):
    config.DB_FILE = str(tmp_path / "subs.db")
    await db.init_db()
    await db.subscribe_coin(1, "bitcoin", 1.0, 60)

    async def fake_news(coin, session=None, user=None):
        return [{"title": "Hello"}]

    monkeypatch.setattr(api, "get_news", fake_news)

    update = DummyUpdate()
    ctx = DummyContext([])
    await handlers.news_cmd(update, ctx)
    assert update.message.texts


@pytest.mark.asyncio
async def test_news_cmd_dedup_and_links(tmp_path, monkeypatch):
    config.DB_FILE = str(tmp_path / "subs.db")
    await db.init_db()
    await db.subscribe_coin(1, "bitcoin", 1.0, 60)
    await db.subscribe_coin(1, "ethereum", 1.0, 60)

    async def fake_news(coin, session=None, user=None):
        return [{"title": "Hello", "url": "https://example.com"}]

    monkeypatch.setattr(api, "get_news", fake_news)

    update = DummyUpdate()
    ctx = DummyContext([])
    await handlers.news_cmd(update, ctx)
    assert len(update.message.texts) == 1
    assert "https://example.com" in update.message.texts[0]
    assert "Hello" in update.message.texts[0]
