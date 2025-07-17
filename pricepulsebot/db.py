"""Asynchronous SQLite helpers for caching coin data and subscriptions."""

import json
import time
from typing import List, Optional, Tuple

import aiosqlite

from . import config


async def init_db() -> None:
    """Create database tables if they do not already exist."""
    async with aiosqlite.connect(config.DB_FILE) as db:
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS subscriptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                coin_id TEXT NOT NULL,
                threshold REAL NOT NULL,
                interval INTEGER NOT NULL DEFAULT 60,
                target_price REAL,
                direction INTEGER,
                last_price REAL,
                last_alert_ts REAL
            )
            """
        )
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS global_info (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                data TEXT,
                fetched_at REAL
            )
            """
        )
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS coin_info (
                coin_id TEXT PRIMARY KEY,
                data TEXT,
                fetched_at REAL
            )
            """
        )
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS coin_charts (
                coin_id TEXT NOT NULL,
                days INTEGER NOT NULL,
                data TEXT,
                fetched_at REAL,
                PRIMARY KEY (coin_id, days)
            )
            """
        )
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS trending_coins (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                data TEXT,
                fetched_at REAL
            )
            """
        )
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS coin_data (
                coin_id TEXT PRIMARY KEY,
                price REAL,
                market_info TEXT,
                info TEXT,
                chart_7d TEXT,
                fetched_at REAL
            )
            """
        )
        cursor = await db.execute("PRAGMA table_info(subscriptions)")
        rows = await cursor.fetchall()
        await cursor.close()
        columns = {row[1] for row in rows}
        if "last_alert_ts" not in columns:
            await db.execute("ALTER TABLE subscriptions ADD COLUMN last_alert_ts REAL")
        if "interval" not in columns:
            await db.execute(
                "ALTER TABLE subscriptions "
                "ADD COLUMN interval INTEGER NOT NULL DEFAULT 60"
            )
        if "target_price" not in columns:
            await db.execute("ALTER TABLE subscriptions ADD COLUMN target_price REAL")
        if "direction" not in columns:
            await db.execute("ALTER TABLE subscriptions ADD COLUMN direction INTEGER")
        await db.commit()


async def subscribe_coin(
    chat_id: int,
    coin: str,
    threshold: float,
    interval: int,
    target_price: Optional[float] = None,
    direction: Optional[int] = None,
) -> None:
    """Add or update a subscription for ``chat_id`` and ``coin``."""
    async with aiosqlite.connect(config.DB_FILE) as db:
        cursor = await db.execute(
            (
                "SELECT id, threshold, interval, target_price, direction "
                "FROM subscriptions WHERE chat_id=? AND coin_id=?"
            ),
            (chat_id, coin),
        )
        row = await cursor.fetchone()
        await cursor.close()
        if row:
            sub_id, existing_th, existing_int, existing_price, existing_dir = row
            new_th = min(existing_th, threshold)
            new_int = min(existing_int, interval)
            new_price = target_price if target_price is not None else existing_price
            new_dir = direction if direction is not None else existing_dir
            await db.execute(
                "UPDATE subscriptions SET threshold=?, interval=?, target_price=?, direction=? WHERE id=?",
                (new_th, new_int, new_price, new_dir, sub_id),
            )
        else:
            await db.execute(
                (
                    "INSERT INTO subscriptions (chat_id, coin_id, threshold, interval, target_price, direction)"
                    " VALUES (?, ?, ?, ?, ?, ?)"
                ),
                (chat_id, coin, threshold, interval, target_price, direction),
            )
        await db.commit()
    config.logger.info(
        "chat %s subscribed to %s at ±%s%% every %ss target=%s dir=%s",
        chat_id,
        coin,
        threshold,
        interval,
        target_price,
        direction,
    )


async def unsubscribe_coin(chat_id: int, coin: str) -> None:
    """Remove a subscription for ``chat_id`` and ``coin``."""
    async with aiosqlite.connect(config.DB_FILE) as db:
        await db.execute(
            "DELETE FROM subscriptions WHERE chat_id=? AND coin_id=?",
            (chat_id, coin),
        )
        await db.commit()
    config.logger.info("chat %s unsubscribed from %s", chat_id, coin)


async def list_subscriptions(
    chat_id: int,
) -> List[Tuple[int, str, float, int, Optional[float], Optional[float]]]:
    """Return all subscriptions for ``chat_id``."""
    async with aiosqlite.connect(config.DB_FILE) as db:
        cursor = await db.execute(
            (
                "SELECT id, coin_id, threshold, interval, last_price, last_alert_ts "
                "FROM subscriptions WHERE chat_id=?"
            ),
            (chat_id,),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return [(row[0], row[1], row[2], row[3], row[4], row[5]) for row in rows]


async def set_last_price(sub_id: int, price: float) -> None:
    """Update the stored last price for a subscription."""
    async with aiosqlite.connect(config.DB_FILE) as db:
        await db.execute(
            "UPDATE subscriptions SET last_price=?, last_alert_ts=? WHERE id=?",
            (price, time.time(), sub_id),
        )
        await db.commit()


async def get_global_data() -> Optional[dict]:
    """Return cached global market data if present."""
    async with aiosqlite.connect(config.DB_FILE) as db:
        cursor = await db.execute("SELECT data, fetched_at FROM global_info WHERE id=1")
        row = await cursor.fetchone()
        await cursor.close()
    if row:
        return json.loads(row[0])
    return None


async def set_global_data(data: dict) -> None:
    """Persist global market data in the database."""
    async with aiosqlite.connect(config.DB_FILE) as db:
        await db.execute("DELETE FROM global_info WHERE id=1")
        await db.execute(
            "INSERT INTO global_info (id, data, fetched_at) VALUES (1, ?, ?)",
            (json.dumps(data), time.time()),
        )
        await db.commit()


async def get_coin_info(coin: str) -> Optional[dict]:
    """Return cached CoinGecko info for ``coin`` if available."""
    async with aiosqlite.connect(config.DB_FILE) as db:
        cursor = await db.execute(
            "SELECT data, fetched_at FROM coin_info WHERE coin_id=?",
            (coin,),
        )
        row = await cursor.fetchone()
        await cursor.close()
    if row:
        return json.loads(row[0])
    return None


async def set_coin_info(coin: str, data: dict) -> None:
    """Store detailed coin information in the database."""
    async with aiosqlite.connect(config.DB_FILE) as db:
        await db.execute(
            "REPLACE INTO coin_info (coin_id, data, fetched_at) VALUES (?, ?, ?)",
            (coin, json.dumps(data), time.time()),
        )
        await db.commit()


async def get_coin_chart(coin: str, days: int) -> Optional[list]:
    """Return cached chart data for ``coin`` if available."""
    async with aiosqlite.connect(config.DB_FILE) as db:
        cursor = await db.execute(
            "SELECT data, fetched_at FROM coin_charts WHERE coin_id=? AND days=?",
            (coin, days),
        )
        row = await cursor.fetchone()
        await cursor.close()
    if row:
        return json.loads(row[0])
    return None


async def set_coin_chart(coin: str, days: int, data: list) -> None:
    """Store chart data for ``coin`` for ``days`` days."""
    async with aiosqlite.connect(config.DB_FILE) as db:
        await db.execute(
            (
                "REPLACE INTO coin_charts (coin_id, days, data, fetched_at) "
                "VALUES (?, ?, ?, ?)"
            ),
            (coin, days, json.dumps(data), time.time()),
        )
        await db.commit()


async def get_trending_coins() -> Optional[list[dict]]:
    """Return cached trending coin information if available."""
    async with aiosqlite.connect(config.DB_FILE) as db:
        cursor = await db.execute(
            "SELECT data, fetched_at FROM trending_coins WHERE id=1"
        )
        row = await cursor.fetchone()
        await cursor.close()
    if row:
        return json.loads(row[0])
    return None


async def set_trending_coins(coins: list[dict]) -> None:
    """Cache the list of trending coins."""
    async with aiosqlite.connect(config.DB_FILE) as db:
        await db.execute("DELETE FROM trending_coins WHERE id=1")
        await db.execute(
            "INSERT INTO trending_coins (id, data, fetched_at) VALUES (1, ?, ?)",
            (json.dumps(coins), time.time()),
        )
        await db.commit()


async def get_coin_data(coin: str) -> Optional[dict]:
    """Return aggregated cached data for ``coin`` if present."""
    async with aiosqlite.connect(config.DB_FILE) as db:
        cursor = await db.execute(
            (
                "SELECT price, market_info, info, chart_7d, fetched_at "
                "FROM coin_data WHERE coin_id=?"
            ),
            (coin,),
        )
        row = await cursor.fetchone()
        await cursor.close()
    if row:
        price, market_json, info_json, chart_json, _ = row
        return {
            "price": price,
            "market_info": json.loads(market_json) if market_json else None,
            "info": json.loads(info_json) if info_json else None,
            "chart_7d": json.loads(chart_json) if chart_json else None,
        }
    return None


async def set_coin_data(coin: str, data: dict) -> None:
    """Store aggregated coin data in the database."""
    async with aiosqlite.connect(config.DB_FILE) as db:
        await db.execute(
            (
                "REPLACE INTO coin_data (coin_id, price, market_info, info, "
                "chart_7d, fetched_at) VALUES (?, ?, ?, ?, ?, ?)"
            ),
            (
                coin,
                data.get("price"),
                json.dumps(data.get("market_info")),
                json.dumps(data.get("info")),
                json.dumps(data.get("chart_7d")),
                time.time(),
            ),
        )
        await db.commit()
