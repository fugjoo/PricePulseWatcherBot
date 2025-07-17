"""Fetch and broadcast large futures liquidation events."""

from __future__ import annotations

import aiohttp

from . import config, db
from .handlers import send_rate_limited


async def check_liquidations(app) -> None:
    """Poll Binance for large liquidation orders and alert subscribers."""
    url = "https://fapi.binance.com/futures/data/forceOrders?limit=50"
    async with aiohttp.ClientSession() as session:
        resp = await session.get(url)
        if resp.status != 200:
            config.logger.warning("liquidation API status %s", resp.status)
            return
        data = await resp.json()

    events: list[str] = []
    for item in data:
        price = float(item.get("price", 0))
        qty = float(item.get("origQty") or item.get("qty") or 0)
        usd = price * qty
        if usd < 500_000:
            continue
        side = item.get("side", "")
        direction = "long" if side == "SELL" else "short"
        symbol = item.get("symbol", "")
        events.append(f"{symbol} {direction} liquidation ~${usd:,.0f}")

    if not events:
        return

    async with db.aiosqlite.connect(config.DB_FILE) as database:
        cursor = await database.execute("SELECT DISTINCT chat_id FROM subscriptions")
        chats = [row[0] for row in await cursor.fetchall()]
        await cursor.close()

    for chat_id in chats:
        for text in events:
            await send_rate_limited(app.bot, chat_id, text, emoji="\u26a0\ufe0f")
