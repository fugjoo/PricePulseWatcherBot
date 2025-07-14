import asyncio
import os
import signal
from itertools import cycle
from typing import Dict, Optional, Tuple

import aiohttp
import aiosqlite
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from dotenv import load_dotenv
from telegram import Bot, KeyboardButton, ReplyKeyboardMarkup, Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

DB_FILE = "subs.db"
DEFAULT_THRESHOLD = 3.0

COINS = ["bitcoin", "ethereum", "litecoin", "dogecoin"]
coin_cycle = cycle(COINS)


async def init_db() -> None:
    """Ensure the subscriptions table exists."""
    async with aiosqlite.connect(DB_FILE) as db:
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS subscriptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                coin_id TEXT NOT NULL,
                threshold REAL NOT NULL,
                last_price REAL
            )
            """
        )
        await db.commit()


async def get_price(coin: str) -> Optional[float]:
    """Return the current USD price for a coin from CoinGecko."""
    url = (
        "https://api.coingecko.com/api/v3/simple/price" f"?ids={coin}&vs_currencies=usd"
    )
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
            if coin not in data:
                return None
            return float(data[coin]["usd"])


async def subscribe_coin(chat_id: int, coin: str, threshold: float) -> None:
    async with aiosqlite.connect(DB_FILE) as db:
        await db.execute(
            """
            INSERT INTO subscriptions (chat_id, coin_id, threshold)
            VALUES (?, ?, ?)
            """,
            (chat_id, coin, threshold),
        )
        await db.commit()


async def unsubscribe_coin(chat_id: int, coin: str) -> None:
    async with aiosqlite.connect(DB_FILE) as db:
        await db.execute(
            "DELETE FROM subscriptions WHERE chat_id=? AND coin_id=?",
            (chat_id, coin),
        )
        await db.commit()


async def list_subscriptions(chat_id: int) -> list[Tuple[str, float]]:
    async with aiosqlite.connect(DB_FILE) as db:
        cursor = await db.execute(
            "SELECT coin_id, threshold FROM subscriptions WHERE chat_id=?",
            (chat_id,),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return [(row[0], row[1]) for row in rows]


async def set_last_price(sub_id: int, price: float) -> None:
    async with aiosqlite.connect(DB_FILE) as db:
        await db.execute(
            "UPDATE subscriptions SET last_price=? WHERE id=?",
            (price, sub_id),
        )
        await db.commit()


async def check_prices(bot: Bot) -> None:
    """Periodic job that checks prices and notifies users."""
    async with aiosqlite.connect(DB_FILE) as db:
        cursor = await db.execute(
            "SELECT id, chat_id, coin_id, threshold, last_price FROM subscriptions"
        )
        rows = await cursor.fetchall()
        await cursor.close()

    # Group subscriptions by coin to minimize API calls
    by_coin: Dict[str, list[Tuple[int, int, float, Optional[float]]]] = {}
    for sub_id, chat_id, coin, threshold, last_price in rows:
        by_coin.setdefault(coin, []).append((sub_id, chat_id, threshold, last_price))

    for coin, subscriptions in by_coin.items():
        price = await get_price(coin)
        if price is None:
            continue
        for sub_id, chat_id, threshold, last_price in subscriptions:
            if last_price is None:
                await set_last_price(sub_id, price)
                continue
            change = abs((price - last_price) / last_price * 100)
            if change >= threshold:
                await bot.send_message(
                    chat_id=chat_id,
                    text=f"{coin.upper()} price changed {change:.2f}% to ${price:.2f}",
                )
                await set_last_price(sub_id, price)


SUB_EMOJI = "\U0001fa99"
LIST_EMOJI = "\U0001f4cb"
HELP_EMOJI = "\u2753"


def get_keyboard() -> ReplyKeyboardMarkup:
    coin = next(coin_cycle)
    keyboard = [
        [KeyboardButton(f"{SUB_EMOJI} Subscribe {coin.upper()}")],
        [KeyboardButton(f"{LIST_EMOJI} List"), KeyboardButton(f"{HELP_EMOJI} Help")],
    ]
    return ReplyKeyboardMarkup(
        keyboard,
        resize_keyboard=True,
        is_persistent=True,
    )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Welcome! Choose an action:", reply_markup=get_keyboard()
    )


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "/subscribe <coin> [pct] - subscribe to price alerts\n"
        "/unsubscribe <coin> - remove subscription\n"
        "/list - list subscriptions",
        reply_markup=get_keyboard(),
    )


async def subscribe_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Usage: /subscribe <coin> [pct]", quote=True)
        return
    coin = context.args[0].lower()
    try:
        threshold = (
            float(context.args[1]) if len(context.args) > 1 else DEFAULT_THRESHOLD
        )
    except ValueError:
        await update.message.reply_text("Threshold must be a number")
        return

    await subscribe_coin(update.effective_chat.id, coin, threshold)
    await update.message.reply_text(
        f"Subscribed to {coin.upper()} price alerts at ±{threshold}%",
        reply_markup=get_keyboard(),
    )


async def unsubscribe_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Usage: /unsubscribe <coin>")
        return
    coin = context.args[0].lower()
    await unsubscribe_coin(update.effective_chat.id, coin)
    await update.message.reply_text(
        f"Unsubscribed from {coin.upper()} alerts",
        reply_markup=get_keyboard(),
    )


async def list_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    subs = await list_subscriptions(update.effective_chat.id)
    if not subs:
        text = "No active subscriptions"
    else:
        text = "\n".join(f"{c.upper()} ±{t}%" for c, t in subs)
    await update.message.reply_text(text, reply_markup=get_keyboard())


async def menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    text = update.message.text.strip()

    if text.startswith(SUB_EMOJI):
        parts = text.split()
        if len(parts) >= 3 and parts[1] == "Subscribe":
            coin = parts[2].lower()
            await subscribe_coin(update.effective_chat.id, coin, DEFAULT_THRESHOLD)
            await update.message.reply_text(
                f"Subscribed to {coin.upper()} alerts at ±{DEFAULT_THRESHOLD}%",
                reply_markup=get_keyboard(),
            )
    elif text == f"{LIST_EMOJI} List":
        subs = await list_subscriptions(update.effective_chat.id)
        if not subs:
            msg = "No active subscriptions"
        else:
            msg = "\n".join(f"{c.upper()} ±{t}%" for c, t in subs)
        await update.message.reply_text(msg, reply_markup=get_keyboard())
    elif text == f"{HELP_EMOJI} Help":
        await update.message.reply_text(
            "/subscribe <coin> [pct] - subscribe to price alerts\n"
            "/unsubscribe <coin> - remove subscription\n"
            "/list - list subscriptions",
            reply_markup=get_keyboard(),
        )


async def main() -> None:
    """Start the bot."""
    load_dotenv()
    await init_db()

    token = os.getenv("BOT_TOKEN")
    if not token:
        raise RuntimeError("BOT_TOKEN not set")

    app = ApplicationBuilder().token(token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("subscribe", subscribe_cmd))
    app.add_handler(CommandHandler("unsubscribe", unsubscribe_cmd))
    app.add_handler(CommandHandler("list", list_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, menu))

    scheduler = AsyncIOScheduler()
    scheduler.add_job(check_prices, "interval", seconds=10, args=(app.bot,))
    scheduler.start()

    await app.initialize()
    await app.start()
    await app.updater.start_polling()

    stop_event = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        asyncio.get_running_loop().add_signal_handler(sig, stop_event.set)

    await stop_event.wait()
    await app.updater.stop()
    await app.stop()
    await app.shutdown()
    scheduler.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
