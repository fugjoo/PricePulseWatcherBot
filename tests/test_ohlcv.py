import pytest

import pricepulsebot.api as api  # noqa: E402
import pricepulsebot.config as config  # noqa: E402


@pytest.mark.asyncio
async def test_fetch_ohlcv_ignores_default_headers(monkeypatch):
    captured = {}

    async def fake_api_get(url, session=None, headers=None, user=None):
        captured["headers"] = headers

        class Resp:
            status = 200

            async def json(self):
                return [[0, 0, 2, 1, 1.5, 100], [0, 0, 1.7, 1.5, 1.6, 300]]

        return Resp()

    monkeypatch.setattr(api, "api_get", fake_api_get)
    config.COINGECKO_HEADERS = {"x-cg-pro-api-key": "TOKEN"}
    candles, err = await api.fetch_ohlcv("BTCUSDT", "1h", 2, user=1)
    assert err is None
    assert len(candles) == 2
    assert captured["headers"] is None


@pytest.mark.asyncio
async def test_fetch_ohlcv_custom_headers(monkeypatch):
    captured = {}

    async def fake_api_get(url, session=None, headers=None, user=None):
        captured["headers"] = headers

        class Resp:
            status = 200

            async def json(self):
                return []

        return Resp()

    monkeypatch.setattr(api, "api_get", fake_api_get)
    headers = {"X-Test": "1"}
    candles, err = await api.fetch_ohlcv("BTCUSDT", "1h", 0, headers=headers)
    assert err is None
    assert captured["headers"] == headers
