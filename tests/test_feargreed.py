import pytest
from aresponses import Response, ResponsesMockServer

import pricepulsebot.api as api
import pricepulsebot.config as config
import pricepulsebot.db as db
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
    def __init__(self):
        self.args = []


@pytest.mark.asyncio
async def test_get_feargreed_basic(tmp_path):
    config.DB_FILE = str(tmp_path / "subs.db")
    await db.init_db()
    async with ResponsesMockServer() as ars:
        ars.add(
            "api.alternative.me",
            "/fng/",
            "GET",
            Response(
                text='{"data": [{"value": "55", "value_classification": "Greed"}]}',
                status=200,
                headers={"Content-Type": "application/json"},
            ),
        )
        data, err = await api.get_feargreed_index()
    cached = await db.get_feargreed()
    assert err is None
    assert data == cached
    assert data["value"] == "55"


@pytest.mark.asyncio
async def test_get_feargreed_uses_cache(tmp_path, monkeypatch):
    config.DB_FILE = str(tmp_path / "subs.db")
    await db.init_db()
    cached = {"value": "30", "value_classification": "Fear"}
    await db.set_feargreed(cached)

    async def fail(*args, **kwargs):
        raise AssertionError("network called")

    monkeypatch.setattr(api, "api_get", fail)
    data, err = await api.get_feargreed_index()
    assert err is None
    assert data == cached


@pytest.mark.asyncio
async def test_feargreed_cmd(monkeypatch):
    async def fake(*args, **kwargs):
        return {"value": "70", "value_classification": "Greed"}, None

    monkeypatch.setattr(api, "get_feargreed_index", fake)

    update = DummyUpdate()
    ctx = DummyContext()
    await handlers.feargreed_cmd(update, ctx)
    assert update.message.texts
    assert "70" in update.message.texts[0]
