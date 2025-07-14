import asyncio
import logging
import os
import signal
import time
from collections import defaultdict, deque
from itertools import cycle
from typing import Deque, Dict, Optional, Tuple

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

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger(__name__)

# price cache: coin -> (price, timestamp)
PRICE_CACHE: Dict[str, Tuple[float, float]] = {}
REQUEST_LOCK = asyncio.Lock()
LAST_REQUEST = 0.0

# telegram rate limits
user_messages: Dict[int, Deque[float]] = defaultdict(deque)
global_messages: Deque[float] = deque()


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
                last_price REAL,
                last_alert_ts REAL
            )
            """
        )
        columns = {
            row[1] for row in await db.execute("PRAGMA table_info(subscriptions)")
        }
        if "last_alert_ts" not in columns:
            await db.execute("ALTER TABLE subscriptions ADD COLUMN last_alert_ts REAL")
        await db.commit()


async def get_price(
    coin: str, session: Optional[aiohttp.ClientSession] = None
) -> Optional[float]:
    """Return the current USD price for a coin from CoinGecko."""
    now = time.time()
    cached = PRICE_CACHE.get(coin)
    if cached and now - cached[1] < 60:
        return cached[0]

    url = (
        "https://api.coingecko.com/api/v3/simple/price" f"?ids={coin}&vs_currencies=usd"
    )
    retries = 3
    owns_session = session is None
    if owns_session:
        session = aiohttp.ClientSession()
    try:
        for attempt in range(retries):
            async with REQUEST_LOCK:
                global LAST_REQUEST
                wait = max(0, LAST_REQUEST + 1.2 - time.time())
                if wait:
                    await asyncio.sleep(wait)
                resp = await session.get(url)
                LAST_REQUEST = time.time()
            if resp.status == 200:
                data = await resp.json()
                if coin in data:
                    price = float(data[coin]["usd"])
                    PRICE_CACHE[coin] = (price, time.time())
                    return price
            await asyncio.sleep(2**attempt)
    finally:
        if owns_session:
            await session.close()
    return None


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
    logger.info("chat %s subscribed to %s at ±%s%%", chat_id, coin, threshold)


async def unsubscribe_coin(chat_id: int, coin: str) -> None:
    async with aiosqlite.connect(DB_FILE) as db:
        await db.execute(
            "DELETE FROM subscriptions WHERE chat_id=? AND coin_id=?",
            (chat_id, coin),
        )
        await db.commit()
    logger.info("chat %s unsubscribed from %s", chat_id, coin)


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
            "UPDATE subscriptions SET last_price=?, last_alert_ts=? WHERE id=?",
            (price, time.time(), sub_id),
        )
        await db.commit()


async def send_rate_limited(bot: Bot, chat_id: int, text: str) -> None:
    now = time.time()
    # cleanup timestamps
    user_q = user_messages[chat_id]
    while user_q and now - user_q[0] > 60:
        user_q.popleft()
    while global_messages and now - global_messages[0] > 1:
        global_messages.popleft()

    # wait if limits exceeded
    if len(user_q) >= 20:
        wait = 60 - (now - user_q[0])
        await asyncio.sleep(wait)
    if len(global_messages) >= 30:
        wait = 1 - (now - global_messages[0])
        await asyncio.sleep(wait)

    await bot.send_message(chat_id=chat_id, text=text)
    user_q.append(time.time())
    global_messages.append(time.time())


async def check_prices(app) -> None:
    """Iterate subscriptions, alert on significant price changes."""
    async with aiosqlite.connect(DB_FILE) as db:
        cursor = await db.execute(
            "SELECT id, chat_id, coin_id, threshold, last_price FROM subscriptions"
        )
        rows = await cursor.fetchall()
        await cursor.close()

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
                text = f"{coin.upper()} moved {change:.2f}% to ${price:.2f}"
                await send_rate_limited(app.bot, chat_id, text)
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
    logger.debug("/start from %s", update.effective_chat.id)
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
    logger.info(
        "chat %s subscribes via command to %s at %.2f%%",
        update.effective_chat.id,
        coin,
        threshold,
    )
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

    logger.info(
        "chat %s unsubscribes via command from %s", update.effective_chat.id, coin
    )
    await update.message.reply_text(f"Unsubscribed from {coin.upper()} alerts")

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

    await update.message.reply_text(text)


async def button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    if query.data.startswith("sub:"):
        coin = query.data.split(":", 1)[1]
        await subscribe_coin(query.message.chat_id, coin, DEFAULT_THRESHOLD)
        logger.info("chat %s subscribed via button to %s", query.message.chat_id, coin)
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text=f"Subscribed to {coin.upper()} alerts at ±{DEFAULT_THRESHOLD}%",
        )
        await query.edit_message_reply_markup(reply_markup=get_keyboard())
    elif query.data == "list":
        subs = await list_subscriptions(query.message.chat_id)

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

    token = os.getenv("TELEGRAM_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_TOKEN not set")

    app = ApplicationBuilder().token(token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("subscribe", subscribe_cmd))
    app.add_handler(CommandHandler("unsubscribe", unsubscribe_cmd))
    app.add_handler(CommandHandler("list", list_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, menu))

    scheduler = AsyncIOScheduler()
    scheduler.add_job(check_prices, "interval", seconds=10, args=(app,))
    scheduler.start()

    await app.initialize()
    await app.start()
    await app.updater.start_polling()
    logger.info("Bot started")

    stop_event = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        asyncio.get_running_loop().add_signal_handler(sig, stop_event.set)

    await stop_event.wait()
    await app.updater.stop()
    await app.stop()
    await app.shutdown()
    scheduler.shutdown()
    logger.info("Bot stopped")


if __name__ == "__main__":
    asyncio.run(main())
