import os
import aiosqlite

DB_PATH = os.getenv("DB_PATH", "./crypto.db")


async def init_db() -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """CREATE TABLE IF NOT EXISTS subscriptions (
            chat_id INTEGER,
            coin TEXT,
            price REAL
        )"""
        )
        await db.commit()


async def add_subscription(chat_id: int, coin: str, price: float) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO subscriptions (chat_id, coin, price) VALUES (?, ?, ?)",
            (chat_id, coin, price),
        )
        await db.commit()


async def remove_subscription(chat_id: int, coin: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "DELETE FROM subscriptions WHERE chat_id = ? AND coin = ?",
            (chat_id, coin),
        )
        await db.commit()


async def list_subscriptions(chat_id: int) -> list[tuple]:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT coin, price FROM subscriptions WHERE chat_id = ?",
            (chat_id,),
        ) as cursor:
            return await cursor.fetchall()


async def all_subscriptions() -> list[tuple]:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT chat_id, coin, price FROM subscriptions") as cursor:
            return await cursor.fetchall()
