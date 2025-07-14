import asyncio
import json
import logging
import os
import random
import re
import signal
import time
from collections import defaultdict, deque
from decimal import Decimal
from io import BytesIO
from typing import Deque, Dict, Optional, Tuple

import aiohttp
import aiosqlite
import matplotlib
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from telegram import (
    Bot,
    BotCommand,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardMarkup,
    Update,
)
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

matplotlib.use("Agg")

DB_FILE = "subs.db"
DEFAULT_THRESHOLD = 0.1
DEFAULT_INTERVAL = 60
PRICE_CHECK_INTERVAL = 60


def parse_duration(value: str) -> int:
    """Return seconds for a duration string like '15m' or '1h'."""
    if value.isdigit():
        return int(value)
    match = re.fullmatch(r"(\d+)([dhms])", value.lower())
    if not match:
        raise ValueError("invalid interval format")
    num, unit = match.groups()
    factor = {"d": 86400, "h": 3600, "m": 60, "s": 1}[unit]
    return int(num) * factor


def load_config(path: str = "config.json") -> None:
    """Load defaults from a JSON config if present."""
    if not os.path.isfile(path):
        return
    try:
        with open(path) as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("failed to load config: %s", exc)
        return

    global DEFAULT_THRESHOLD, DEFAULT_INTERVAL, PRICE_CHECK_INTERVAL
    if "default_threshold" in data:
        try:
            DEFAULT_THRESHOLD = float(data["default_threshold"])
        except (TypeError, ValueError):
            logger.warning("invalid default_threshold in config")
    if "default_interval" in data:
        try:
            DEFAULT_INTERVAL = parse_duration(str(data["default_interval"]))
        except ValueError:
            logger.warning("invalid default_interval in config")
    if "price_check_interval" in data:
        try:
            PRICE_CHECK_INTERVAL = parse_duration(str(data["price_check_interval"]))
        except ValueError:
            logger.warning("invalid price_check_interval in config")


COINS = ["bitcoin", "ethereum", "litecoin", "dogecoin"]
COIN_SYMBOLS: Dict[str, str] = {
    "bitcoin": "BTC",
    "ethereum": "ETH",
    "litecoin": "LTC",
    "dogecoin": "DOGE",
}
SYMBOL_TO_COIN: Dict[str, str] = {v.lower(): k for k, v in COIN_SYMBOLS.items()}


def symbol_for(coin: str) -> str:
    """Return the symbol for a coin ID."""
    return COIN_SYMBOLS.get(coin, coin.upper())


def normalize_coin(value: str) -> str:
    """Return the coin ID for a given symbol or coin name."""
    return SYMBOL_TO_COIN.get(value.lower(), value.lower())


async def fetch_trending_coins() -> None:
    """Update COINS and symbol mappings using the trending list from CoinGecko."""
    url = "https://api.coingecko.com/api/v3/search/trending"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    logger.warning("trending request failed: %s", resp.status)
                    return
                data = await resp.json()
                coins: list[str] = []
                for c in data.get("coins", [])[:10]:
                    item = c.get("item", {})
                    coin_id = item.get("id")
                    symbol = item.get("symbol")
                    if coin_id:
                        coins.append(coin_id)
                        if symbol:
                            COIN_SYMBOLS[coin_id] = symbol.upper()
                            SYMBOL_TO_COIN[symbol.lower()] = coin_id
                if coins:
                    global COINS
                    COINS = coins
    except aiohttp.ClientError as exc:
        logger.error("error fetching trending coins: %s", exc)


logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger(__name__)

# price cache: coin -> (price, timestamp)
PRICE_CACHE: Dict[str, Tuple[float, float]] = {}
REQUEST_LOCK = asyncio.Lock()
LAST_REQUEST = 0.0

# telegram rate limits
user_messages: Dict[int, Deque[float]] = defaultdict(deque)
global_messages: Deque[float] = deque()

# milestone cache: (chat_id, coin) -> last checked price
MILESTONE_CACHE: Dict[Tuple[int, str], float] = {}


def milestone_step(price: float) -> float:
    """Return step size for round-number alerts based on price."""
    if price >= 1000:
        return 100.0
    if price >= 100:
        return 10.0
    if price >= 10:
        return 1.0
    if price >= 1:
        return 0.1
    if price >= 0.1:
        return 0.01
    return 0.001


def format_price(value: float) -> str:
    """Return price as decimal string without scientific notation."""
    return format(Decimal(str(value)), "f")


def milestones_crossed(last: float, current: float) -> list[float]:
    """Return a list of price levels crossed between two prices."""
    step = milestone_step(max(last, current))
    levels: list[float] = []
    if current > last:
        boundary = (int(last // step) * step) + step
        while boundary <= current:
            levels.append(boundary)
            boundary += step
    elif current < last:
        boundary = int(last // step) * step
        while boundary > current:
            levels.append(boundary)
            boundary -= step
    return levels


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
                (
                    "ALTER TABLE subscriptions "
                    "ADD COLUMN interval INTEGER NOT NULL DEFAULT 60"
                )
            )
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
                try:
                    resp = await session.get(url)
                except aiohttp.ClientError as exc:
                    logger.error("price request failed: %s", exc)
                    return None
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


async def get_coin_info(
    coin: str, session: Optional[aiohttp.ClientSession] = None
) -> Optional[dict]:
    """Return detailed coin info from CoinGecko."""
    url = f"https://api.coingecko.com/api/v3/coins/{coin}"
    owns_session = session is None
    if owns_session:
        session = aiohttp.ClientSession()
    try:
        async with REQUEST_LOCK:
            global LAST_REQUEST
            wait = max(0, LAST_REQUEST + 1.2 - time.time())
            if wait:
                await asyncio.sleep(wait)
            try:
                resp = await session.get(url)
            except aiohttp.ClientError as exc:
                logger.error("coin info request failed: %s", exc)
                return None
            LAST_REQUEST = time.time()
        if resp.status == 200:
            return await resp.json()
    finally:
        if owns_session and session:
            await session.close()
    return None


async def get_market_chart(
    coin: str, days: int, session: Optional[aiohttp.ClientSession] = None
) -> Optional[list[tuple[float, float]]]:
    """Return historical price chart data for a coin."""
    end_ts = int(time.time())
    start_ts = end_ts - days * 86400
    url = (
        f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart/range"
        f"?vs_currency=usd&from={start_ts}&to={end_ts}"
    )
    owns_session = session is None
    if owns_session:
        session = aiohttp.ClientSession()
    try:
        async with REQUEST_LOCK:
            global LAST_REQUEST
            wait = max(0, LAST_REQUEST + 1.2 - time.time())
            if wait:
                await asyncio.sleep(wait)
            try:
                resp = await session.get(url)
            except aiohttp.ClientError as exc:
                logger.error("chart request failed: %s", exc)
                return None
            LAST_REQUEST = time.time()
        if resp.status == 200:
            data = await resp.json()
            return [(p[0] / 1000, p[1]) for p in data.get("prices", [])]
    finally:
        if owns_session and session:
            await session.close()
    return None


async def get_global_overview(
    session: Optional[aiohttp.ClientSession] = None,
) -> Optional[dict]:
    """Return global market data from CoinGecko."""
    url = "https://api.coingecko.com/api/v3/global"
    owns_session = session is None
    if owns_session:
        session = aiohttp.ClientSession()
    try:
        async with REQUEST_LOCK:
            global LAST_REQUEST
            wait = max(0, LAST_REQUEST + 1.2 - time.time())
            if wait:
                await asyncio.sleep(wait)
            try:
                resp = await session.get(url)
            except aiohttp.ClientError as exc:
                logger.error("global overview request failed: %s", exc)
                return None
            LAST_REQUEST = time.time()
        if resp.status == 200:
            return await resp.json()
    finally:
        if owns_session and session:
            await session.close()
    return None


async def subscribe_coin(
    chat_id: int, coin: str, threshold: float, interval: int
) -> None:
    async with aiosqlite.connect(DB_FILE) as db:
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
                """
                INSERT INTO subscriptions (chat_id, coin_id, threshold, interval)
                VALUES (?, ?, ?, ?)
                """,
                (chat_id, coin, threshold, interval),
            )
        await db.commit()
    logger.info(
        "chat %s subscribed to %s at ±%s%% every %ss",
        chat_id,
        coin,
        threshold,
        interval,
    )


async def unsubscribe_coin(chat_id: int, coin: str) -> None:
    async with aiosqlite.connect(DB_FILE) as db:
        await db.execute(
            "DELETE FROM subscriptions WHERE chat_id=? AND coin_id=?",
            (chat_id, coin),
        )
        await db.commit()
    logger.info("chat %s unsubscribed from %s", chat_id, coin)


async def list_subscriptions(
    chat_id: int,
) -> list[Tuple[int, str, float, int, Optional[float], Optional[float]]]:
    async with aiosqlite.connect(DB_FILE) as db:
        cursor = await db.execute(
            "SELECT id, coin_id, threshold, interval, last_price, last_alert_ts "
            "FROM subscriptions WHERE chat_id=?",
            (chat_id,),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return [
            (
                row[0],
                row[1],
                row[2],
                row[3],
                row[4],
                row[5],
            )
            for row in rows
        ]


async def set_last_price(sub_id: int, price: float) -> None:
    async with aiosqlite.connect(DB_FILE) as db:
        await db.execute(
            "UPDATE subscriptions SET last_price=?, last_alert_ts=? WHERE id=?",
            (price, time.time(), sub_id),
        )
        await db.commit()


DEFAULT_ALERT_EMOJI = "\U0001f680"


async def send_rate_limited(
    bot: Bot, chat_id: int, text: str, emoji: str = DEFAULT_ALERT_EMOJI
) -> None:
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

    await bot.send_message(chat_id=chat_id, text=f"{emoji} {text}")
    user_q.append(time.time())
    global_messages.append(time.time())


async def check_prices(app) -> None:
    """Iterate subscriptions, alert on significant price changes."""
    async with aiosqlite.connect(DB_FILE) as db:
        cursor = await db.execute(
            "SELECT id, chat_id, coin_id, threshold, interval, last_price, "
            "last_alert_ts FROM subscriptions"
        )
        rows = await cursor.fetchall()
        await cursor.close()

    by_coin: Dict[
        str, list[Tuple[int, int, float, int, Optional[float], Optional[float]]]
    ] = {}
    for sub_id, chat_id, coin, threshold, interval, last_price, last_ts in rows:
        by_coin.setdefault(coin, []).append(
            (sub_id, chat_id, threshold, interval, last_price, last_ts)
        )

    for coin, subscriptions in by_coin.items():
        price = await get_price(coin)
        if price is None:
            continue
        for sub_id, chat_id, threshold, interval, last_price, last_ts in subscriptions:
            if last_price is None:
                await set_last_price(sub_id, price)
                MILESTONE_CACHE[(chat_id, coin)] = price
                continue

            prev = MILESTONE_CACHE.get((chat_id, coin), last_price)
            for level in milestones_crossed(prev, price):
                symbol = symbol_for(coin)
                if price > prev:
                    msg = (

                        f"{symbol} {COIN_EMOJI} blasts past ${level:.0f} "
                        f"(now ${price})"
                    )
                    await send_rate_limited(app.bot, chat_id, msg, emoji=UP_EMOJI)
                else:
                    msg = (
                        f"{symbol} {COIN_EMOJI} dives below ${level:.0f} "
                        f"(now ${price})"
                    )
                    await send_rate_limited(app.bot, chat_id, msg, emoji=DOWN_EMOJI)

            MILESTONE_CACHE[(chat_id, coin)] = price

            if last_ts is None or time.time() - last_ts >= interval:
                raw_change = (price - last_price) / last_price * 100
                change = abs(raw_change)
                if change >= threshold:

                    symbol = symbol_for(coin)
                    emoji = UP_EMOJI if raw_change >= 0 else DOWN_EMOJI
                    text = f"{symbol} {COIN_EMOJI} moved {raw_change:+.2f}% to ${price}"
                    await send_rate_limited(app.bot, chat_id, text, emoji=emoji)

                await set_last_price(sub_id, price)


SUB_EMOJI = "\U0001fa99"
RELOAD_EMOJI = "\U0001f504"
LIST_EMOJI = "\U0001f4cb"
HELP_EMOJI = "\u2753"
WELCOME_EMOJI = "\U0001f44b"
INFO_EMOJI = "\u2139\ufe0f"
SUCCESS_EMOJI = "\u2705"
ERROR_EMOJI = "\u26a0\ufe0f"
ALERT_EMOJI = "\U0001f680"  # rocket
UP_EMOJI = "\U0001f680"  # rocket for rising prices
DOWN_EMOJI = "\U0001f4a3"  # bomb for falling prices
COIN_EMOJI = "\u20bf"  # bitcoin sign


def get_keyboard() -> ReplyKeyboardMarkup:
    coin = random.choice(COINS[:10]) if COINS else "bitcoin"
    symbol = symbol_for(coin)
    keyboard = [
        [
            KeyboardButton(f"{SUB_EMOJI} Subscribe {symbol}"),
            KeyboardButton(RELOAD_EMOJI),
        ],
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
        f"{WELCOME_EMOJI} Welcome! Choose an action:",
        reply_markup=get_keyboard(),
    )


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        f"{INFO_EMOJI} /subscribe <coin> [pct] [interval] - subscribe to price alerts\n"
        "/unsubscribe <coin> - remove subscription\n"
        "/list - list subscriptions\n"
        "/info <coin> - coin information\n"
        "/chart <coin> [days] - price chart\n"
        "/global - global market stats\n"
        "Intervals can be like 1h, 15m or 30s",
        reply_markup=get_keyboard(),
    )


async def subscribe_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text(
            f"{ERROR_EMOJI} Usage: /subscribe <coin> [pct] [interval]",
            quote=True,
        )
        return
    coin = normalize_coin(context.args[0])
    try:
        threshold = (
            float(context.args[1]) if len(context.args) > 1 else DEFAULT_THRESHOLD
        )
    except ValueError:
        await update.message.reply_text(f"{ERROR_EMOJI} Threshold must be a number")
        return

    try:
        interval_str = (
            context.args[2] if len(context.args) > 2 else str(DEFAULT_INTERVAL)
        )
        interval = parse_duration(interval_str)
    except ValueError:
        await update.message.reply_text(
            f"{ERROR_EMOJI} Interval must be a number or like 1h, 15m, 30s"
        )
        return

    await subscribe_coin(update.effective_chat.id, coin, threshold, interval)
    logger.info(
        "chat %s subscribes via command to %s at %.2f%% every %ss",
        update.effective_chat.id,
        coin,
        threshold,
        interval,
    )
    await update.message.reply_text(
        (
            f"{SUCCESS_EMOJI} Subscribed to {symbol_for(coin)} at ±{threshold}% "
            f"every {interval}s"
        ),
        reply_markup=get_keyboard(),
    )


async def unsubscribe_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text(f"{ERROR_EMOJI} Usage: /unsubscribe <coin>")
        return
    coin = normalize_coin(context.args[0])
    await unsubscribe_coin(update.effective_chat.id, coin)

    logger.info(
        "chat %s unsubscribes via command from %s", update.effective_chat.id, coin
    )
    await update.message.reply_text(
        f"{SUCCESS_EMOJI} Unsubscribed from {symbol_for(coin)} alerts"
    )

    await update.message.reply_text(
        f"{SUCCESS_EMOJI} Unsubscribed from {symbol_for(coin)} alerts",
        reply_markup=get_keyboard(),
    )


async def list_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    subs = await list_subscriptions(update.effective_chat.id)
    if not subs:
        await update.message.reply_text(f"{INFO_EMOJI} No active subscriptions")
        return

    for _, coin, threshold, interval, last_price, last_ts in subs:
        price = await get_price(coin) or 0
        change = 0.0
        if last_price:
            change = (price - last_price) / last_price * 100
        text = (
            f"{symbol_for(coin)} ${format_price(price)} {change:+.2f}% "
            f"/ ±{threshold}% every {interval}s"
        )
        keyboard = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton("Unsubscribe", callback_data=f"del:{coin}"),
                    InlineKeyboardButton("Edit", callback_data=f"edit:{coin}"),
                ]
            ]
        )
        await update.message.reply_text(text, reply_markup=keyboard)


async def info_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text(f"{ERROR_EMOJI} Usage: /info <coin>")
        return
    coin = normalize_coin(context.args[0])
    data = await get_coin_info(coin)
    if not data:
        await update.message.reply_text(f"{ERROR_EMOJI} Coin not found")
        return
    market = data.get("market_data", {})
    price = market.get("current_price", {}).get("usd")
    cap = market.get("market_cap", {}).get("usd")
    change = market.get("price_change_percentage_24h")
    sym = data.get("symbol", "").upper()
    COIN_SYMBOLS[coin] = sym
    SYMBOL_TO_COIN[sym.lower()] = coin
    text = f"{INFO_EMOJI} {data.get('name')} ({sym})\n"
    if price is not None:
        text += f"Price: ${format_price(price)}\n"
    if cap is not None:
        text += f"Market Cap: ${cap:,.0f}\n"
    if change is not None:
        text += f"24h Change: {change:.2f}%"
    await update.message.reply_text(text)


async def chart_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text(f"{ERROR_EMOJI} Usage: /chart <coin> [days]")
        return
    coin = context.args[0].lower()
    days = 7
    if len(context.args) > 1:
        try:
            days = int(context.args[1])
        except ValueError:
            await update.message.reply_text(f"{ERROR_EMOJI} Days must be a number")
            return
    data = await get_market_chart(coin, days)
    if not data:
        await update.message.reply_text(f"{ERROR_EMOJI} No data available")
        return
    times, prices = zip(*data)
    plt.figure(figsize=(6, 3))
    plt.plot(times, prices)
    plt.title(f"{coin.upper()} last {days} days")
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    await context.bot.send_photo(update.effective_chat.id, buf)


async def global_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    data = await get_global_overview()
    if not data:
        await update.message.reply_text(f"{ERROR_EMOJI} Failed to fetch data")
        return
    info = data.get("data", {})
    cap = info.get("total_market_cap", {}).get("usd")
    volume = info.get("total_volume", {}).get("usd")
    btc_dom = info.get("market_cap_percentage", {}).get("btc")
    text = f"{INFO_EMOJI} "
    if cap is not None:
        text += f"Market Cap: ${cap:,.0f}\n"
    if volume is not None:
        text += f"24h Volume: ${volume:,.0f}\n"
    if btc_dom is not None:
        text += f"BTC Dominance: {btc_dom:.2f}%"
    await update.message.reply_text(text)


async def button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    if query.data.startswith("sub:"):
        coin = query.data.split(":", 1)[1]
        await subscribe_coin(
            query.message.chat_id,
            coin,
            DEFAULT_THRESHOLD,
            DEFAULT_INTERVAL,
        )
        logger.info("chat %s subscribed via button to %s", query.message.chat_id, coin)
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text=(
                f"{SUCCESS_EMOJI} Subscribed to {symbol_for(coin)} at "
                f"±{DEFAULT_THRESHOLD}% every {DEFAULT_INTERVAL}s"
            ),
        )
        await query.edit_message_reply_markup(reply_markup=get_keyboard())
    elif query.data.startswith("del:"):
        coin = query.data.split(":", 1)[1]
        await unsubscribe_coin(query.message.chat_id, coin)
        await query.edit_message_text(
            f"{SUCCESS_EMOJI} Unsubscribed from {symbol_for(coin)}"
        )
    elif query.data.startswith("edit:"):
        coin = query.data.split(":", 1)[1]
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text=f"{INFO_EMOJI} Use /subscribe {coin} [pct] [interval] to update",
        )
        await query.edit_message_reply_markup(reply_markup=None)
    elif query.data == "list":
        subs = await list_subscriptions(query.message.chat_id)
        if not subs:

            text = f"{INFO_EMOJI} No active subscriptions"

        else:
            for _, coin, threshold, interval, last_price, last_ts in subs:
                price = await get_price(coin) or 0
                change = 0.0
                if last_price:
                    change = (price - last_price) / last_price * 100
                text = (
                    f"{symbol_for(coin)} ${format_price(price)} {change:+.2f}% "
                    f"/ ±{threshold}% every {interval}s"
                )
                keyboard = InlineKeyboardMarkup(
                    [
                        [
                            InlineKeyboardButton(
                                "Unsubscribe", callback_data=f"del:{coin}"
                            ),
                            InlineKeyboardButton("Edit", callback_data=f"edit:{coin}"),
                        ]
                    ]
                )
                await context.bot.send_message(
                    chat_id=query.message.chat_id,
                    text=text,
                    reply_markup=keyboard,
                )
        await query.edit_message_reply_markup(reply_markup=get_keyboard())


async def menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    text = update.message.text.strip()

    if text.startswith(SUB_EMOJI):
        parts = text.split()
        if len(parts) >= 3 and parts[1] == "Subscribe":
            coin = normalize_coin(parts[2])
            await subscribe_coin(
                update.effective_chat.id,
                coin,
                DEFAULT_THRESHOLD,
                DEFAULT_INTERVAL,
            )
            await update.message.reply_text(
                (
                    f"{SUCCESS_EMOJI} Subscribed to {symbol_for(coin)} at "
                    f"±{DEFAULT_THRESHOLD}% every {DEFAULT_INTERVAL}s"
                ),
                reply_markup=get_keyboard(),
            )
    elif text == RELOAD_EMOJI:
        await update.message.reply_text("New suggestion:", reply_markup=get_keyboard())
    elif text == f"{LIST_EMOJI} List":
        subs = await list_subscriptions(update.effective_chat.id)

        if not subs:

            msg = f"{INFO_EMOJI} No active subscriptions"

        else:
            for _, coin, threshold, interval, last_price, last_ts in subs:
                price = await get_price(coin) or 0
                change = 0.0
                if last_price:
                    change = (price - last_price) / last_price * 100
                msg = (
                    f"{symbol_for(coin)} ${format_price(price)} {change:+.2f}% "
                    f"/ ±{threshold}% every {interval}s"
                )
                keyboard = InlineKeyboardMarkup(
                    [
                        [
                            InlineKeyboardButton(
                                "Unsubscribe", callback_data=f"del:{coin}"
                            ),
                            InlineKeyboardButton("Edit", callback_data=f"edit:{coin}"),
                        ]
                    ]
                )
                await update.message.reply_text(msg, reply_markup=keyboard)
    elif text == f"{HELP_EMOJI} Help":
        await update.message.reply_text(
            (
                f"{INFO_EMOJI} /subscribe <coin> [pct] [seconds] - subscribe to "
                "price alerts\n"
                "/unsubscribe <coin> - remove subscription\n"
                "/list - list subscriptions"
            ),
            reply_markup=get_keyboard(),
        )


async def main() -> None:
    """Start the bot."""
    load_dotenv()
    load_config()
    await init_db()
    await fetch_trending_coins()

    token = os.getenv("TELEGRAM_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_TOKEN not set")

    app = ApplicationBuilder().token(token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("subscribe", subscribe_cmd))
    app.add_handler(CommandHandler("unsubscribe", unsubscribe_cmd))
    app.add_handler(CommandHandler("list", list_cmd))
    app.add_handler(CommandHandler("info", info_cmd))
    app.add_handler(CommandHandler("chart", chart_cmd))
    app.add_handler(CommandHandler("global", global_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, menu))
    app.add_handler(CallbackQueryHandler(button))

    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        check_prices, "interval", seconds=PRICE_CHECK_INTERVAL, args=(app,)
    )
    scheduler.add_job(fetch_trending_coins, "interval", minutes=10)
    scheduler.start()

    await app.initialize()
    await app.bot.set_my_commands(
        [
            BotCommand("start", "Show menu"),
            BotCommand("help", "Show help"),
            BotCommand("subscribe", "Subscribe to price alerts"),
            BotCommand("unsubscribe", "Remove subscription"),
            BotCommand("list", "List subscriptions"),
            BotCommand("info", "Coin information"),
            BotCommand("chart", "Price chart"),
            BotCommand("global", "Global market"),
        ]
    )
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
