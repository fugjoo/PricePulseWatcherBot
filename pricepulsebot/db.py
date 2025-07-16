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
