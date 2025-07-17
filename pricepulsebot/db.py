import json
import time
from typing import List, Optional, Tuple

import aiosqlite

from . import config


async def init_db() -> None:
    async with aiosqlite.connect(config.DB_FILE) as db:
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS subscriptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                coin_id TEXT NOT NULL,
                threshold REAL NOT NULL,
                interval INTEGER NOT NULL DEFAULT 60,
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
        await db.commit()


async def subscribe_coin(
    chat_id: int, coin: str, threshold: float, interval: int
) -> None:
    async with aiosqlite.connect(config.DB_FILE) as db:
        cursor = await db.execute(
            (
                "SELECT id, threshold, interval "
                "FROM subscriptions WHERE chat_id=? AND coin_id=?"
            ),
            (chat_id, coin),
        )
        row = await cursor.fetchone()
        await cursor.close()
        if row:
            sub_id, existing_th, existing_int = row
            new_th = min(existing_th, threshold)
            new_int = min(existing_int, interval)
            await db.execute(
                "UPDATE subscriptions SET threshold=?, interval=? WHERE id=?",
                (new_th, new_int, sub_id),
            )
        else:
            await db.execute(
                (
                    "INSERT INTO subscriptions (chat_id, coin_id, threshold, interval)"
                    " VALUES (?, ?, ?, ?)"
                ),
                (chat_id, coin, threshold, interval),
            )
        await db.commit()
    config.logger.info(
        "chat %s subscribed to %s at ±%s%% every %ss",
        chat_id,
        coin,
        threshold,
        interval,
    )


async def unsubscribe_coin(chat_id: int, coin: str) -> None:
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
    async with aiosqlite.connect(config.DB_FILE) as db:
        await db.execute(
            "UPDATE subscriptions SET last_price=?, last_alert_ts=? WHERE id=?",
            (price, time.time(), sub_id),
        )
        await db.commit()


async def get_global_data(max_age: int = 300) -> Optional[dict]:
    async with aiosqlite.connect(config.DB_FILE) as db:
        cursor = await db.execute("SELECT data, fetched_at FROM global_info WHERE id=1")
        row = await cursor.fetchone()
        await cursor.close()
    if row and time.time() - row[1] < max_age:
        return json.loads(row[0])
    return None


async def set_global_data(data: dict) -> None:
    async with aiosqlite.connect(config.DB_FILE) as db:
        await db.execute("DELETE FROM global_info WHERE id=1")
        await db.execute(
            "INSERT INTO global_info (id, data, fetched_at) VALUES (1, ?, ?)",
            (json.dumps(data), time.time()),
        )
        await db.commit()


async def get_coin_info(coin: str, max_age: int = 300) -> Optional[dict]:
    async with aiosqlite.connect(config.DB_FILE) as db:
        cursor = await db.execute(
            "SELECT data, fetched_at FROM coin_info WHERE coin_id=?",
            (coin,),
        )
        row = await cursor.fetchone()
        await cursor.close()
    if row and time.time() - row[1] < max_age:
        return json.loads(row[0])
    return None


async def set_coin_info(coin: str, data: dict) -> None:
    async with aiosqlite.connect(config.DB_FILE) as db:
        await db.execute(
            "REPLACE INTO coin_info (coin_id, data, fetched_at) VALUES (?, ?, ?)",
            (coin, json.dumps(data), time.time()),
        )
        await db.commit()


async def get_coin_chart(coin: str, days: int, max_age: int = 300) -> Optional[list]:
    async with aiosqlite.connect(config.DB_FILE) as db:
        cursor = await db.execute(
            "SELECT data, fetched_at FROM coin_charts WHERE coin_id=? AND days=?",
            (coin, days),
        )
        row = await cursor.fetchone()
        await cursor.close()
    if row and time.time() - row[1] < max_age:
        return json.loads(row[0])
    return None


async def set_coin_chart(coin: str, days: int, data: list) -> None:
    async with aiosqlite.connect(config.DB_FILE) as db:
        await db.execute(
            (
                "REPLACE INTO coin_charts (coin_id, days, data, fetched_at) "
                "VALUES (?, ?, ?, ?)"
            ),
            (coin, days, json.dumps(data), time.time()),
        )
        await db.commit()


async def get_trending_coins(max_age: int = 300) -> Optional[list[dict]]:
    async with aiosqlite.connect(config.DB_FILE) as db:
        cursor = await db.execute(
            "SELECT data, fetched_at FROM trending_coins WHERE id=1"
        )
        row = await cursor.fetchone()
        await cursor.close()
    if row and time.time() - row[1] < max_age:
        return json.loads(row[0])
    return None


async def set_trending_coins(coins: list[dict]) -> None:
    async with aiosqlite.connect(config.DB_FILE) as db:
        await db.execute("DELETE FROM trending_coins WHERE id=1")
        await db.execute(
            "INSERT INTO trending_coins (id, data, fetched_at) VALUES (1, ?, ?)",
            (json.dumps(coins), time.time()),
        )
        await db.commit()


async def get_coin_data(coin: str, max_age: int = 300) -> Optional[dict]:
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
    if row and time.time() - row[4] < max_age:
        price, market_json, info_json, chart_json, _ = row
        return {
            "price": price,
            "market_info": json.loads(market_json) if market_json else None,
            "info": json.loads(info_json) if info_json else None,
            "chart_7d": json.loads(chart_json) if chart_json else None,
        }
    return None


async def set_coin_data(coin: str, data: dict) -> None:
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
