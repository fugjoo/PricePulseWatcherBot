"""Telegram command and callback handlers used by the bot."""

import asyncio
import random
import time
from collections import defaultdict, deque
from datetime import datetime
from decimal import Decimal
from io import BytesIO
from typing import Deque, Dict, List, Optional, Tuple

import aiohttp
import numpy as np
from matplotlib import dates as mdates
from matplotlib import pyplot as plt
from telegram import (
    Bot,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardMarkup,
    Update,
)
from telegram.constants import ChatAction
from telegram.ext import ContextTypes

from . import api, config, db

UP_ARROW = "\U0001f53a"
DOWN_ARROW = "\U0001f53b"
ROCKET = "\U0001f680"
BOMB = "\U0001f4a3"
POSITIVE_TREND_EMOJIS = [
    "\U0001f4c8",
    "\U0001f680",
    "\U0001f4aa",
    "\U0001f973",
    "\U0001f911",
    "\U0001f4b9",
    "\U0001f4ca",
]

NEGATIVE_TREND_EMOJIS = [
    "\U0001f4c9",
    "\U0001f61f",
    "\U0001f480",
    "\U0001f53b",
    "\U0001f630",
    "\U0001f525",
    "\U0001f9e8",
    "\u26d4",
]
DEFAULT_ALERT_EMOJI = UP_ARROW

user_messages: Dict[int, Deque[float]] = defaultdict(deque)
global_messages: Deque[float] = deque()
MILESTONE_CACHE: Dict[Tuple[int, str], float] = {}

SUB_EMOJI = "\U00002795"
LIST_EMOJI = "\U0001f4cb"
HELP_EMOJI = "\u2753"
WELCOME_EMOJI = "\U0001f44b"
INFO_EMOJI = "\u2139\ufe0f"
SUCCESS_EMOJI = "\u2705"
ERROR_EMOJI = "\u26a0\ufe0f"
SETTINGS_EMOJI = "\u2699\ufe0f"
BACK_EMOJI = "\u2b05\ufe0f"

# Commands organized by category for help output
COMMAND_CATEGORIES: dict[str, list[tuple[str, str]]] = {
    "Start": [
        ("start", "Show menu"),
    ],
    "Subscriptions": [
        ("add", "Subscribe to price alerts"),
        ("remove", "Remove subscription"),
        ("clear", "Remove all subscriptions"),
        ("list", "List subscriptions"),
    ],
    "Info & Tools": [
        ("info", "Coin information"),
        ("chart", "Price chart"),
        ("charts", "All coin charts"),
        ("news", "Latest news"),
        ("trends", "Trending coins"),
        ("top", "Top market cap"),
        ("global", "Global market"),
        ("feargreed", "Market sentiment"),
        ("valuearea", "Volume profile"),
    ],
    "General": [
        ("help", "Show help"),
    ],
    "Bot Settings": [
        ("settings", "Show or change defaults"),
    ],
    "Status": [
        ("status", "API status"),
    ],
}

# Flattened list for bot registration
COMMANDS: list[tuple[str, str]] = [
    cmd for cmds in COMMAND_CATEGORIES.values() for cmd in cmds
]


def format_coin_text(
    name: str,
    symbol: str,
    price: Optional[float],
    change_24h: Optional[float],
    cap: Optional[float],
    *,
    threshold: Optional[float] = None,
    interval: Optional[int] = None,
) -> str:
    """Return formatted text describing a coin."""
    text = f"{INFO_EMOJI} {name}"
    if symbol:
        text += f" ({symbol})"
    text += "\n"
    if threshold is not None and interval is not None:
        text += f"Alerts: ±{threshold}% every {config.format_interval(interval)}\n"
    if price is not None:
        text += f"Price: ${format_price(price)}"
        if change_24h is not None:
            text += f" ({change_24h:+.2f}% 24h)"
        text += "\n"
    if cap is not None:
        text += f"Cap: ${cap:,.0f}"
    return text


def milestone_step(price: float) -> float:
    """Return the milestone step size for ``price``."""
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
    if price >= 0.01:
        return 0.001
    if price >= 0.001:
        return 0.0001
    if price >= 0.0001:
        return 0.00001
    return 0.000001


def format_price(value: float) -> str:
    """Format ``value`` as a price string."""
    # limit precision to avoid floating point artifacts like
    # ``0.00013000000000000002``
    d = Decimal(value).quantize(Decimal("1e-8"))
    text = format(d.normalize(), "f")
    if "." in text:
        frac = text.split(".")[1]
        if len(frac) == 1:
            text += "0"
    return text


def milestones_crossed(last: float, current: float) -> List[float]:
    """Return milestone levels crossed between ``last`` and ``current``."""
    step = milestone_step(max(last, current))
    levels: List[float] = []
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


def trend_emojis(change: float) -> str:
    """Return arrow emoji based on the sign of ``change``."""
    return UP_ARROW if change >= 0 else DOWN_ARROW


def random_trend_suffix(change: float) -> str:
    """Return 2-3 random emojis representing the trend direction."""
    emojis = POSITIVE_TREND_EMOJIS if change >= 0 else NEGATIVE_TREND_EMOJIS
    count = random.randint(2, 3)
    chosen = random.sample(emojis, k=min(count, len(emojis)))
    return "".join(chosen)


def usd_value(value: Optional[object]) -> Optional[float]:
    """Return the configured currency float when given either a number or a dict."""
    if isinstance(value, dict):
        return value.get(config.VS_CURRENCY)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def calculate_volume_profile(candles: List[dict]) -> dict:
    """Calculate the volume profile statistics for given candles."""
    if not candles:
        raise ValueError("no candles provided")
    min_price = min(c["low"] for c in candles)
    max_price = max(c["high"] for c in candles)
    if min_price == max_price:
        raise ValueError("candle prices are constant")
    bins = 100
    edges = np.linspace(min_price, max_price, bins + 1)
    hist = np.zeros(bins)
    for candle in candles:
        low = candle["low"]
        high = candle["high"]
        vol = candle["volume"]
        if high <= low:
            idx = np.searchsorted(edges, high, side="right") - 1
            if 0 <= idx < bins:
                hist[idx] += vol
            continue
        start = np.searchsorted(edges, low, side="right") - 1
        end = np.searchsorted(edges, high, side="left")
        for idx in range(max(start, 0), min(end + 1, bins)):
            left = edges[idx]
            right = edges[idx + 1]
            overlap_left = max(left, low)
            overlap_right = min(right, high)
            if overlap_left >= overlap_right:
                continue
            proportion = (overlap_right - overlap_left) / (high - low)
            hist[idx] += vol * proportion
    total_volume = float(hist.sum())
    if total_volume == 0:
        raise ValueError("no volume data")
    poc_idx = int(hist.argmax())
    poc = float((edges[poc_idx] + edges[poc_idx + 1]) / 2)
    target = total_volume * 0.7
    sorted_idx = np.argsort(hist)[::-1]
    included = []
    volume_acc = 0.0
    for idx in sorted_idx:
        included.append(idx)
        volume_acc += hist[idx]
        if volume_acc >= target:
            break
    low_idx = min(included)
    high_idx = max(included)
    val = float(edges[low_idx])
    vah = float(edges[high_idx + 1])
    return {"val": val, "poc": poc, "vah": vah}


async def send_rate_limited(
    bot: Bot,
    chat_id: int,
    text: str,
    emoji: str = DEFAULT_ALERT_EMOJI,
    suffix: str = "",
) -> None:
    """Send a message while enforcing per-user and global rate limits."""
    now = time.time()
    user_q = user_messages[chat_id]
    while user_q and now - user_q[0] > 60:
        user_q.popleft()
    while global_messages and now - global_messages[0] > 1:
        global_messages.popleft()
    if len(user_q) >= 20:
        wait = max(0, 60 - (now - user_q[0]))
        await asyncio.sleep(wait)
    if len(global_messages) >= 30:
        wait = max(0, 1 - (now - global_messages[0]))
        await asyncio.sleep(wait)
    message = f"{emoji} {text}"
    if suffix:
        message += f" {suffix}"
    await bot.send_message(chat_id=chat_id, text=message)
    user_q.append(time.time())
    global_messages.append(time.time())


async def check_prices(app) -> None:
    """Check all subscriptions and send price alerts when needed."""
    async with aiohttp.ClientSession() as http_session:
        async with db.aiosqlite.connect(config.DB_FILE) as database:
            cursor = await database.execute(
                (
                    "SELECT id, chat_id, coin_id, threshold, interval, target_price, "
                    "direction, last_price, last_volume, last_alert_ts "
                    "FROM subscriptions"
                )
            )
            rows = await cursor.fetchall()
            await cursor.close()
        by_coin: Dict[
            str,
            List[
                Tuple[
                    int,
                    int,
                    float,
                    int,
                    Optional[float],
                    Optional[int],
                    Optional[float],
                    Optional[float],
                    Optional[float],
                ]
            ],
        ] = {}
        for (
            sub_id,
            chat_id,
            coin,
            threshold,
            interval,
            target_price,
            direction,
            last_price,
            last_volume,
            last_ts,
        ) in rows:
            by_coin.setdefault(coin, []).append(
                (
                    sub_id,
                    chat_id,
                    threshold,
                    interval,
                    target_price,
                    direction,
                    last_price,
                    last_volume,
                    last_ts,
                )
            )
        chat_ids = {row[1] for row in rows}
        settings_map = {cid: await db.get_user_settings(cid) for cid in chat_ids}
        coins = list(by_coin.keys())
        prices: Dict[str, float] = {}
        infos: Dict[str, dict] = {}
        missing: List[str] = []
        for coin in coins:
            cached = await db.get_coin_data(coin)
            if cached:
                if cached.get("price") is not None:
                    prices[coin] = float(cached["price"])
                info = cached.get("market_info")
                if info is not None:
                    infos[coin] = info
                else:
                    missing.append(coin)
            else:
                missing.append(coin)
        if missing:
            groups = [
                missing[i : i + 250] for i in range(0, len(missing), 250)  # noqa: E203
            ]
            for group in groups:
                markets = await api.get_markets(group, session=http_session, user=None)
                for c, info in markets.items():
                    price = info.get("current_price")
                    if price is not None:
                        prices[c] = float(price)
                    infos[c] = info
        volumes: Dict[str, float] = {}
        if any(s["volume"] for s in settings_map.values()):
            for coin in coins:
                vol = await api.get_volume(coin, session=http_session, user=None)
                if vol is not None:
                    volumes[coin] = vol
        for coin, subscriptions in by_coin.items():
            price = prices.get(coin)
            if price is None:
                continue
            volume = volumes.get(coin)
            for (
                sub_id,
                chat_id,
                threshold,
                interval,
                target_price,
                direction,
                last_price,
                last_volume,
                last_ts,
            ) in subscriptions:
                if last_price is None:
                    await db.set_last_price(sub_id, price, volume)
                    MILESTONE_CACHE[(chat_id, coin)] = price
                    continue
                prev = MILESTONE_CACHE.get((chat_id, coin), last_price)
                settings = settings_map.get(chat_id) or {}
                if settings.get("milestones", config.ENABLE_MILESTONE_ALERTS):
                    for level in milestones_crossed(prev, price):
                        symbol = api.symbol_for(coin)
                        if price > prev:
                            msg = (
                                f"{symbol} breaks through ${format_price(level)} "
                                f"(now ${format_price(price)})"
                            )
                            await send_rate_limited(
                                app.bot,
                                chat_id,
                                msg,
                                emoji=UP_ARROW,
                                suffix=random_trend_suffix(price - prev),
                            )
                        else:
                            msg = (
                                f"{symbol} falls below ${format_price(level)} "
                                f"(now ${format_price(price)})"
                            )
                            await send_rate_limited(
                                app.bot,
                                chat_id,
                                msg,
                                emoji=DOWN_ARROW,
                                suffix=random_trend_suffix(price - prev),
                            )
                MILESTONE_CACHE[(chat_id, coin)] = price
                if target_price is not None and direction is not None:
                    crossed_up = direction > 0 and prev < target_price <= price
                    crossed_down = direction < 0 and prev > target_price >= price
                    if crossed_up or crossed_down:
                        symbol = api.symbol_for(coin)
                        if crossed_up:
                            msg = (
                                f"{symbol} reached ${format_price(target_price)} "
                                f"(now ${format_price(price)})"
                            )
                            await send_rate_limited(
                                app.bot,
                                chat_id,
                                msg,
                                emoji=UP_ARROW,
                                suffix=random_trend_suffix(price - prev),
                            )
                        elif crossed_down:
                            msg = (
                                f"{symbol} fell below ${format_price(target_price)} "
                                f"(now ${format_price(price)})"
                            )
                            await send_rate_limited(
                                app.bot,
                                chat_id,
                                msg,
                                emoji=DOWN_ARROW,
                                suffix=random_trend_suffix(price - prev),
                            )
                if last_ts is None or time.time() - last_ts >= interval:
                    raw_change = (price - last_price) / last_price * 100
                    change = abs(raw_change)
                    if change >= threshold:
                        symbol = api.symbol_for(coin)
                        text = (
                            f"{symbol} moved {raw_change:+.2f}% in "
                            f"{config.format_interval(interval)} (now ${price}"
                        )
                        info = infos.get(coin)
                        if info is None:
                            info = await api.get_market_info(coin, user=chat_id)
                        change_24h = None
                        if info:
                            change_24h = info.get("price_change_percentage_24h")
                        if change_24h is not None:
                            text += f", {change_24h:+.2f}% 24h"
                        text += ")"
                        await send_rate_limited(
                            app.bot,
                            chat_id,
                            text,
                            emoji=trend_emojis(raw_change),
                            suffix=random_trend_suffix(raw_change),
                        )
                    if settings.get("volume", config.ENABLE_VOLUME_ALERTS):
                        if (
                            volume is not None
                            and last_volume is not None
                            and last_volume > 0
                        ):
                            raw_vol_change = (volume - last_volume) / last_volume * 100
                            if abs(raw_vol_change) >= config.VOLUME_THRESHOLD:
                                symbol = api.symbol_for(coin)
                                msg = (
                                    f"{symbol} volume {raw_vol_change:+.2f}% "
                                    f"(24h {volume:,.0f})"
                                )
                                await send_rate_limited(
                                    app.bot,
                                    chat_id,
                                    msg,
                                    emoji=trend_emojis(raw_vol_change),
                                    suffix=random_trend_suffix(raw_vol_change),
                                )
                    await db.set_last_price(sub_id, price, volume)


async def refresh_cache(app) -> None:
    """Refresh cached data for coins referenced in the database."""
    async with db.aiosqlite.connect(config.DB_FILE) as database:
        cursor = await database.execute("SELECT DISTINCT coin_id FROM subscriptions")
        coins = [row[0] for row in await cursor.fetchall()]
        await cursor.close()
    async with aiohttp.ClientSession() as session:
        await api.refresh_coins_data(coins, session=session)
    await api.get_global_overview(user=None)


def get_keyboard() -> ReplyKeyboardMarkup:
    """Return the default reply keyboard shown to users."""
    coins_source = config.COINS or config.TOP_COINS[:20] or ["bitcoin"]
    coins = random.sample(coins_source, k=min(3, len(coins_source)))
    subs = [KeyboardButton(f"{SUB_EMOJI} Add {api.symbol_for(c)}") for c in coins]
    keyboard = [
        subs,
        [
            KeyboardButton(f"{LIST_EMOJI} List"),
            KeyboardButton(SETTINGS_EMOJI),
            KeyboardButton(f"{HELP_EMOJI} Help"),
        ],
    ]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True, is_persistent=True)


async def get_settings_keyboard(chat_id: int) -> InlineKeyboardMarkup:
    """Return an inline keyboard showing current settings for ``chat_id``."""
    settings = await db.get_user_settings(chat_id)
    buttons = [
        [
            InlineKeyboardButton(
                f"threshold: ±{settings['threshold']}%",
                callback_data="settings:threshold",
            )
        ],
        [
            InlineKeyboardButton(
                f"interval: {config.format_interval(settings['interval'])}",
                callback_data="settings:interval",
            )
        ],
        [
            InlineKeyboardButton(
                f"milestones: {'on' if settings['milestones'] else 'off'}",
                callback_data="settings:milestones",
            )
        ],
        [
            InlineKeyboardButton(
                f"volume: {'on' if settings['volume'] else 'off'}",
                callback_data="settings:volume",
            )
        ],
        [
            InlineKeyboardButton(
                f"currency: {settings['currency']}", callback_data="settings:currency"
            )
        ],
    ]
    return InlineKeyboardMarkup(buttons)


async def get_settings_menu(chat_id: int) -> ReplyKeyboardMarkup:
    """Return a reply keyboard with current settings for ``chat_id``."""
    settings = await db.get_user_settings(chat_id)
    buttons = [
        [KeyboardButton(f"threshold: ±{settings['threshold']}%")],
        [KeyboardButton(f"interval: {config.format_interval(settings['interval'])}")],
        [KeyboardButton(f"milestones: {'on' if settings['milestones'] else 'off'}")],
        [KeyboardButton(f"volume: {'on' if settings['volume'] else 'off'}")],
        [KeyboardButton(f"currency: {settings['currency']}")],
        [KeyboardButton(f"{BACK_EMOJI} Back")],
    ]
    return ReplyKeyboardMarkup(buttons, resize_keyboard=True, is_persistent=True)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a welcome message and show the main keyboard."""
    await update.message.reply_text(
        f"{WELCOME_EMOJI} Welcome to {config.BOT_NAME}! Use /add or the "
        "buttons below to subscribe to price alerts.",
        reply_markup=get_keyboard(),
    )


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Display available commands and usage information."""
    lines: list[str] = []
    for category, commands in COMMAND_CATEGORIES.items():
        for name, desc in commands:
            lines.append(f"/{name} - {desc}")
    lines.append("Intervals can be like 1h, 15m or 30s")
    await update.message.reply_text(
        f"{INFO_EMOJI} Commands\n" + "\n".join(lines), reply_markup=get_keyboard()
    )


async def subscribe_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Subscribe the chat to price alerts for a coin.

    The second argument can be a percent change or an absolute price
    prefixed with ``>`` or ``<`` to alert when that value is crossed.
    """
    if not context.args:
        await update.message.reply_text(
            f"{ERROR_EMOJI} Usage: /add <coin> [pct] [interval]"
        )
        return
    coin_input = context.args[0]
    coin = await api.resolve_coin(coin_input, user=update.effective_chat.id)
    if not coin:
        suggestions = await api.suggest_coins(coin_input)
        msg = f"{ERROR_EMOJI} Unknown coin"
        if suggestions:
            syms = ", ".join(api.symbol_for(c) for c in suggestions)
            msg += f". Did you mean {syms}?"
        await update.message.reply_text(msg)
        return
    settings = await db.get_user_settings(update.effective_chat.id)
    target_price = None
    direction = None
    threshold = settings["threshold"]
    arg_idx = 1
    if len(context.args) > 1:
        arg = context.args[1]
        if arg and arg[0] in {">", "<"}:
            try:
                target_price = float(arg[1:])
            except ValueError:
                await update.message.reply_text(f"{ERROR_EMOJI} Invalid target price")
                return
            direction = 1 if arg[0] == ">" else -1
            arg_idx = 2
        else:
            try:
                threshold = float(arg)
            except ValueError:
                await update.message.reply_text(
                    f"{ERROR_EMOJI} Threshold must be a number"
                )
                return
            arg_idx = 2
    try:
        interval_str = (
            context.args[arg_idx]
            if len(context.args) > arg_idx
            else str(settings["interval"])
        )
        interval = config.parse_duration(interval_str)
    except ValueError:
        await update.message.reply_text(
            f"{ERROR_EMOJI} Interval must be a number or like 1h, 15m, 30s"
        )
        return
    await db.subscribe_coin(
        update.effective_chat.id,
        coin,
        threshold,
        interval,
        target_price,
        direction,
    )
    await update.message.reply_text(
        f"{SUB_EMOJI} Subscribed to {api.symbol_for(coin)}",
        reply_markup=get_keyboard(),
    )


async def unsubscribe_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Remove a price alert subscription."""
    if not context.args:
        await update.message.reply_text(f"{ERROR_EMOJI} Usage: /remove <coin>")
        return
    coin = api.normalize_coin(context.args[0])
    await db.unsubscribe_coin(update.effective_chat.id, coin)
    await update.message.reply_text(
        f"{SUCCESS_EMOJI} Unsubscribed from {api.symbol_for(coin)} alerts",
        reply_markup=get_keyboard(),
    )


async def clear_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Remove all subscriptions for the chat."""
    await db.unsubscribe_all(update.effective_chat.id)
    await update.message.reply_text(
        f"{SUCCESS_EMOJI} Removed all subscriptions",
        reply_markup=get_keyboard(),
    )


async def build_sub_entries(chat_id: int) -> List[Tuple[str, str]]:
    """Return formatted subscription entries for ``chat_id``."""
    subs = await db.list_subscriptions(chat_id)
    entries: List[Tuple[str, str]] = []
    for _, coin, threshold, interval, *_ in subs:
        cached = await db.get_coin_data(coin)
        info = cached.get("info") if cached else None
        if info is not None and not isinstance(info, dict):
            info = {}
        market = cached.get("market_info") if cached else None
        if market is not None and not isinstance(market, dict):
            market = None
        price = cached.get("price") if cached else None
        if info is None:
            info, _ = await api.get_coin_info(coin, user=chat_id)
        info = info or {}
        if market is None:
            market = info.get("market_data")
            if not isinstance(market, dict):
                market = {}
        if price is None:
            price = (
                usd_value(market.get("current_price"))
                or await api.get_price(coin, user=chat_id)
                or 0
            )
        cap = usd_value(market.get("market_cap"))
        change_24h = market.get("price_change_percentage_24h")
        sym = info.get("symbol")
        if sym:
            config.COIN_SYMBOLS[coin] = sym.upper()
            config.SYMBOL_TO_COIN[sym.lower()] = coin
        line = format_coin_text(
            info.get("name", coin.title()),
            sym.upper() if sym else "",
            price,
            change_24h,
            cap,
            threshold=threshold,
            interval=interval,
        )
        entries.append((coin, line))
    return entries


async def list_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """List all active subscriptions for the chat."""
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action=ChatAction.TYPING
    )
    entries = await build_sub_entries(update.effective_chat.id)
    if not entries:
        await update.message.reply_text(f"{INFO_EMOJI} No active subscriptions")
        return
    for coin, text in entries:
        keyboard = InlineKeyboardMarkup(
            [[InlineKeyboardButton("Remove", callback_data=f"del:{coin}")]]
        )
        await update.message.reply_text(text, reply_markup=keyboard)


async def info_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show detailed information about a coin."""
    if not context.args:
        await update.message.reply_text(f"{ERROR_EMOJI} Usage: /info <coin>")
        return
    coin_input = context.args[0]
    coin = await api.resolve_coin(coin_input, user=update.effective_chat.id)
    if not coin:
        suggestions = await api.suggest_coins(coin_input)
        msg = f"{ERROR_EMOJI} Unknown coin"
        if suggestions:
            syms = ", ".join(api.symbol_for(c) for c in suggestions)
            msg += f". Did you mean {syms}?"
        await update.message.reply_text(msg)
        return
    cached = await db.get_coin_data(coin)
    data = cached.get("info") if cached else None
    if data is not None and not isinstance(data, dict):
        data = None
    market = cached.get("market_info") if cached else None
    if market is not None and not isinstance(market, dict):
        market = None
    if data is None:
        data, err = await api.get_coin_info(coin, user=update.effective_chat.id)
        if err:
            await update.message.reply_text(f"{ERROR_EMOJI} {err}")
            return
    if data is None:
        await update.message.reply_text(f"{ERROR_EMOJI} No data available")
        return
    if market is None:
        market = data.get("market_data")
        if not isinstance(market, dict):
            market = {}
    price = usd_value(market.get("current_price"))
    cap = usd_value(market.get("market_cap"))
    change = market.get("price_change_percentage_24h")
    sym = data.get("symbol", "").upper()
    config.COIN_SYMBOLS[coin] = sym
    config.SYMBOL_TO_COIN[sym.lower()] = coin
    text = format_coin_text(
        data.get("name"),
        sym,
        price,
        change,
        cap,
    )
    await update.message.reply_text(text)


async def _send_chart(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    coin: str,
    seconds: int,
    *,
    force: bool = False,
) -> None:
    """Send a price chart for ``coin`` to ``chat_id``."""
    days = seconds / 86400
    cached = await db.get_coin_data(coin)
    if days == 7 and cached and cached.get("chart_7d") is not None:
        data = [(p[0], p[1]) for p in cached["chart_7d"]]
        err = None
    else:
        data, err = await api.get_market_chart(coin, days, user=chat_id, force=force)
    if err:
        await context.bot.send_message(chat_id, f"{ERROR_EMOJI} {err}")
        return
    if not data:
        await context.bot.send_message(
            chat_id,
            f"{ERROR_EMOJI} No data available for {api.symbol_for(coin)}",
        )
        return
    times, prices = zip(*data)
    times = [datetime.fromtimestamp(t) for t in times]
    plt.figure(figsize=(6, 3))
    plt.plot(times, prices)
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    if seconds <= 172800:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        plt.xlabel("Time")
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        plt.xlabel("Date")
    plt.title(f"{coin.upper()} last {config.format_interval(seconds)}")
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    keyboard = InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "Reload",
                    callback_data=f"chart:{coin}:{seconds}:reload",
                ),
                InlineKeyboardButton("1h", callback_data=f"chart:{coin}:3600"),
                InlineKeyboardButton("4h", callback_data=f"chart:{coin}:14400"),
                InlineKeyboardButton("1d", callback_data=f"chart:{coin}:86400"),
                InlineKeyboardButton("3d", callback_data=f"chart:{coin}:259200"),
            ]
        ]
    )
    await context.bot.send_photo(chat_id, buf, reply_markup=keyboard)


async def chart_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a price chart image for a coin."""
    if not context.args:
        await update.message.reply_text(f"{ERROR_EMOJI} Usage: /chart <coin> [period]")
        return
    coin_input = context.args[0]
    coin = await api.resolve_coin(coin_input, user=update.effective_chat.id)
    if not coin:
        suggestions = await api.suggest_coins(coin_input)
        msg = f"{ERROR_EMOJI} Unknown coin"
        if suggestions:
            syms = ", ".join(api.symbol_for(c) for c in suggestions)
            msg += f". Did you mean {syms}?"
        await update.message.reply_text(msg)
        return
    seconds = 86400
    if len(context.args) > 1:
        try:
            seconds = config.parse_timeframe(context.args[1])
        except ValueError:
            await update.message.reply_text(
                f"{ERROR_EMOJI} Period must be a number or like 1h, 30m"
            )
            return
    await _send_chart(context, update.effective_chat.id, coin, seconds)


async def charts_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send price charts for all subscribed coins."""
    subs = await db.list_subscriptions(update.effective_chat.id)
    coins = [coin for _, coin, *_ in subs]
    if not coins:
        await update.message.reply_text(f"{INFO_EMOJI} No subscriptions")
        return
    seconds = 86400
    if context.args:
        try:
            seconds = config.parse_timeframe(context.args[0])
        except ValueError:
            await update.message.reply_text(
                f"{ERROR_EMOJI} Period must be a number or like 1h, 30m"
            )
            return
    for coin in coins:
        await _send_chart(context, update.effective_chat.id, coin, seconds)


async def global_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Display global market statistics."""
    data, err = await api.get_global_overview(user=update.effective_chat.id)
    if err:
        await update.message.reply_text(f"{ERROR_EMOJI} {err}")
        return
    if data is None:
        await update.message.reply_text(f"{ERROR_EMOJI} Failed to fetch data")
        return
    info = data.get("data", {})
    cap = info.get("total_market_cap", {}).get(config.VS_CURRENCY)
    volume = info.get("total_volume", {}).get(config.VS_CURRENCY)
    btc_dom = info.get("market_cap_percentage", {}).get("btc")
    cap_change = info.get("market_cap_change_percentage_24h_usd")
    active = info.get("active_cryptocurrencies")
    markets = info.get("markets")
    text = f"{INFO_EMOJI} "
    if cap is not None:
        text += f"Market Cap: ${cap:,.0f}\n"
    if cap_change is not None:
        text += f"24h Cap Change: {cap_change:.2f}%\n"
    if volume is not None:
        text += f"24h Volume: ${volume:,.0f}\n"
    if btc_dom is not None:
        text += f"BTC Dominance: {btc_dom:.2f}%\n"
    if active is not None:
        text += f"Active Coins: {active}\n"
    if markets is not None:
        text += f"Markets: {markets}"
    await update.message.reply_text(text)


async def feargreed_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Display the daily Fear & Greed Index."""
    data, err = await api.get_feargreed_index(user=update.effective_chat.id)
    if err:
        await update.message.reply_text(f"{ERROR_EMOJI} {err}")
        return
    if not data:
        await update.message.reply_text(f"{ERROR_EMOJI} Failed to fetch data")
        return
    value_str = data.get("value")
    classification = data.get("value_classification")
    try:
        value = int(value_str)
    except (TypeError, ValueError):
        value = None
    if value is None:
        emoji = INFO_EMOJI
    elif value < 40:
        emoji = "\U0001f534"  # red
    elif value < 60:
        emoji = "\U0001f7e1"  # yellow
    else:
        emoji = "\U0001f7e2"  # green
    text = f"{emoji} Fear & Greed Index: {value_str}"
    if classification:
        text += f" ({classification})"
    await update.message.reply_text(text)


async def trends_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Display the current trending coins."""
    data = await api.fetch_trending_coins()
    if not data:
        await update.message.reply_text(f"{ERROR_EMOJI} Failed to fetch data")
        return
    data = sorted(
        data,
        key=lambda x: (x.get("change_24h") is None, -(x.get("change_24h") or 0)),
    )
    lines = []
    for item in data:
        coin_id = item.get("id")
        symbol = item.get("symbol") or api.symbol_for(coin_id)
        price = item.get("price")
        change_24h = item.get("change_24h")

        line = symbol.upper()
        if change_24h is not None:
            arrow = UP_ARROW if change_24h >= 0 else DOWN_ARROW
            line = f"{arrow} {line}"
        if price is not None:
            line += f" ${format_price(price)}"
        if change_24h is not None:
            line += f" ({change_24h:+.2f}% 24h)"
        lines.append(line)
    text = f"{INFO_EMOJI} Trending coins:\n" + "\n".join(lines)
    await update.message.reply_text(text)


async def top_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Display the top market cap coins."""
    data = await api.fetch_top_coins()
    if not data:
        await update.message.reply_text(f"{ERROR_EMOJI} Failed to fetch data")
        return
    lines = []
    for item in data[:10]:
        coin_id = item.get("id")
        symbol = item.get("symbol") or api.symbol_for(coin_id)
        price = item.get("price")
        change_24h = item.get("change_24h")

        line = symbol.upper()
        if change_24h is not None:
            arrow = UP_ARROW if change_24h >= 0 else DOWN_ARROW
            line = f"{arrow} {line}"
        if price is not None:
            line += f" ${format_price(price)}"
        if change_24h is not None:
            line += f" ({change_24h:+.2f}% 24h)"
        lines.append(line)
    text = f"{INFO_EMOJI} Top coins:\n" + "\n".join(lines)
    await update.message.reply_text(text)


async def news_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show recent news for a coin or subscribed coins."""
    if context.args:
        coin_input = context.args[0]
        coin = await api.resolve_coin(coin_input, user=update.effective_chat.id)
        if not coin:
            suggestions = await api.suggest_coins(coin_input)
            msg = f"{ERROR_EMOJI} Unknown coin"
            if suggestions:
                syms = ", ".join(api.symbol_for(c) for c in suggestions)
                msg += f". Did you mean {syms}?"
            await update.message.reply_text(msg)
            return
        coins = [coin]
    else:
        subs = await db.list_subscriptions(update.effective_chat.id)
        coins = [coin for _, coin, *_ in subs]
        if not coins:
            await update.message.reply_text(f"{INFO_EMOJI} No subscriptions")
            return

    async with aiohttp.ClientSession() as session:
        seen: set[str] = set()
        for coin in coins:
            items = await api.get_news(
                coin, session=session, user=update.effective_chat.id
            )
            if not items:
                await update.message.reply_text(
                    f"{ERROR_EMOJI} No news for {api.symbol_for(coin)}"
                )
                continue
            for item in items[:5]:
                url = item.get("url")
                title = item.get("title")
                if not title:
                    continue
                if not context.args and url in seen:
                    continue
                if url:
                    seen.add(url)
                    text = f'{INFO_EMOJI} <a href="{url}">{title}</a>'
                else:
                    text = f"{INFO_EMOJI} {title}"
                await update.message.reply_text(
                    text, parse_mode="HTML", disable_web_page_preview=True
                )


async def valuearea_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Calculate and display the volume value area for a symbol."""
    if len(context.args) < 3:
        await update.message.reply_text(
            f"{ERROR_EMOJI} Usage: /valuearea <symbol> <interval> <count>"
        )
        return
    symbol_input = context.args[0]
    symbol = await api.resolve_pair(symbol_input, user=update.effective_chat.id)
    interval = context.args[1]
    try:
        limit = int(context.args[2])
    except ValueError:
        await update.message.reply_text(f"{ERROR_EMOJI} Count must be a number")
        return
    candles, err = await api.fetch_ohlcv(
        symbol, interval, limit, user=update.effective_chat.id
    )
    if err:
        await update.message.reply_text(f"{ERROR_EMOJI} {err}")
        return
    if not candles:
        await update.message.reply_text(f"{ERROR_EMOJI} No data available")
        return
    try:
        profile = calculate_volume_profile(candles)
    except ValueError as exc:
        await update.message.reply_text(f"{ERROR_EMOJI} {exc}")
        return
    text = (
        f"\U0001f4ca Value Area {symbol} ({interval}, {limit} candles):\n"
        f"- VAL: ${format_price(profile['val'])}\n"
        f"- POC: ${format_price(profile['poc'])}\n"
        f"- VAH: ${format_price(profile['vah'])}"
    )
    await update.message.reply_text(text)


async def settings_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """View or modify default alert settings."""
    if not context.args:
        text = f"{INFO_EMOJI} Your settings"
        menu = await get_settings_menu(update.effective_chat.id)
        await update.message.reply_text(text, reply_markup=menu)
        return
    if len(context.args) < 2:
        usage = (
            f"{ERROR_EMOJI} Usage: "
            "/settings <threshold|interval|milestones|volume|currency> <value>"
        )
        await update.message.reply_text(usage)
        return
    key = context.args[0].lower()
    value = context.args[1]
    chat_id = update.effective_chat.id
    if key == "threshold":
        try:
            config.DEFAULT_THRESHOLD = float(value)
        except ValueError:
            await update.message.reply_text(f"{ERROR_EMOJI} Threshold must be a number")
            return
        await db.set_user_settings(chat_id, threshold=config.DEFAULT_THRESHOLD)
        await update.message.reply_text(
            f"{SUCCESS_EMOJI} Default threshold set to {config.DEFAULT_THRESHOLD}%"
        )
    elif key == "interval":
        try:
            config.DEFAULT_INTERVAL = config.parse_duration(value)
        except ValueError:
            await update.message.reply_text(
                f"{ERROR_EMOJI} Interval must be a number or like 1h, 15m, 30s"
            )
            return
        interval_text = config.format_interval(config.DEFAULT_INTERVAL)
        await db.set_user_settings(chat_id, interval=config.DEFAULT_INTERVAL)
        await update.message.reply_text(
            f"{SUCCESS_EMOJI} Default interval set to {interval_text}"
        )
    elif key == "milestones":
        val = value.lower()
        if val not in {"on", "off"}:
            await update.message.reply_text(
                f"{ERROR_EMOJI} Milestones must be on or off"
            )
            return
        config.ENABLE_MILESTONE_ALERTS = val == "on"
        await db.set_user_settings(
            chat_id, milestones=int(config.ENABLE_MILESTONE_ALERTS)
        )
        state = "enabled" if config.ENABLE_MILESTONE_ALERTS else "disabled"
        await update.message.reply_text(f"{SUCCESS_EMOJI} Milestone alerts {state}")
    elif key == "volume":
        val = value.lower()
        if val not in {"on", "off"}:
            await update.message.reply_text(f"{ERROR_EMOJI} Volume must be on or off")
            return
        config.ENABLE_VOLUME_ALERTS = val == "on"
        await db.set_user_settings(chat_id, volume=int(config.ENABLE_VOLUME_ALERTS))
        state = "enabled" if config.ENABLE_VOLUME_ALERTS else "disabled"
        await update.message.reply_text(f"{SUCCESS_EMOJI} Volume alerts {state}")
    elif key == "currency":
        config.VS_CURRENCY = value.lower()
        await db.set_user_settings(chat_id, currency=config.VS_CURRENCY)
        await update.message.reply_text(
            f"{SUCCESS_EMOJI} Default currency set to {config.VS_CURRENCY}"
        )
    elif key == "pricecheck":
        await update.message.reply_text(
            f"{ERROR_EMOJI} PRICE_CHECK_INTERVAL cannot be changed"
        )
    else:
        await update.message.reply_text(f"{ERROR_EMOJI} Unknown setting '{key}'")


async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show bot, API and database info plus a status timeline."""
    counts = api.status_counts()
    history = [(ts, status) for ts, status in api.STATUS_HISTORY]
    if not history:
        await update.message.reply_text(f"{INFO_EMOJI} No API requests recorded")
        return

    times = [datetime.fromtimestamp(ts) for ts, _ in history]
    statuses = [s for _, s in history]
    plt.figure(figsize=(6, 3))
    plt.plot(times, statuses, drawstyle="steps-post")
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.xlabel("Time")
    plt.ylabel("HTTP status")
    plt.title("API status last 3h")
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    await context.bot.send_photo(update.effective_chat.id, buf)

    db_count, db_size = await db.get_db_stats()
    lines = [f"{code}: {counts[code]}" for code in sorted(counts)]
    text = (
        f"{INFO_EMOJI} Bot: {config.BOT_NAME}\n"
        f"API: {config.COINGECKO_BASE_URL}\n"
        f"DB: {config.DB_FILE} ({db_count} subs, {db_size // 1024} kB)\n"
        "API responses:\n" + "\n".join(lines)
    )
    await update.message.reply_text(text)


async def button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle inline keyboard button callbacks."""
    query = update.callback_query
    await query.answer()
    if query.data.startswith("sub:"):
        coin = query.data.split(":", 1)[1]
        await db.subscribe_coin(
            query.message.chat_id,
            coin,
            config.DEFAULT_THRESHOLD,
            config.DEFAULT_INTERVAL,
        )
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text=f"{SUB_EMOJI} Subscribed to {api.symbol_for(coin)}",
        )
        await query.edit_message_reply_markup(reply_markup=get_keyboard())
    elif query.data.startswith("del:"):
        coin = query.data.split(":", 1)[1]
        await db.unsubscribe_coin(query.message.chat_id, coin)
        await query.edit_message_text(
            f"{SUCCESS_EMOJI} Unsubscribed from {api.symbol_for(coin)}"
        )
    elif query.data.startswith("edit:"):
        coin = query.data.split(":", 1)[1]
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text=f"{INFO_EMOJI} Use /add {coin} [pct] [interval] to update",
        )
        await query.edit_message_reply_markup(reply_markup=None)
    elif query.data.startswith("chart:"):
        parts = query.data.split(":")
        coin = parts[1]
        seconds = int(parts[2])
        force = len(parts) > 3 and parts[3] == "reload"
        await query.message.delete()
        await _send_chart(
            context,
            query.message.chat_id,
            coin,
            seconds,
            force=force,
        )
    elif query.data.startswith("settings:"):
        key = query.data.split(":", 1)[1]
        text = ""
        if key == "threshold":
            options = [0.1, 0.5, 1.0, 2.0, 5.0]
            try:
                idx = options.index(config.DEFAULT_THRESHOLD)
            except ValueError:
                idx = -1
            config.DEFAULT_THRESHOLD = options[(idx + 1) % len(options)]
            text = (
                f"{SUCCESS_EMOJI} Default threshold set to {config.DEFAULT_THRESHOLD}%"
            )
        elif key == "interval":
            options = [60, 300, 600, 1800, 3600]
            try:
                idx = options.index(config.DEFAULT_INTERVAL)
            except ValueError:
                idx = -1
            config.DEFAULT_INTERVAL = options[(idx + 1) % len(options)]
            text = (
                f"{SUCCESS_EMOJI} Default interval set to "
                f"{config.format_interval(config.DEFAULT_INTERVAL)}"
            )
        elif key == "milestones":
            config.ENABLE_MILESTONE_ALERTS = not config.ENABLE_MILESTONE_ALERTS
            state = "enabled" if config.ENABLE_MILESTONE_ALERTS else "disabled"
            text = f"{SUCCESS_EMOJI} Milestone alerts {state}"
        elif key == "volume":
            config.ENABLE_VOLUME_ALERTS = not config.ENABLE_VOLUME_ALERTS
            state = "enabled" if config.ENABLE_VOLUME_ALERTS else "disabled"
            text = f"{SUCCESS_EMOJI} Volume alerts {state}"
        elif key == "currency":
            options = ["usd", "eur", "btc"]
            try:
                idx = options.index(config.VS_CURRENCY)
            except ValueError:
                idx = -1
            config.VS_CURRENCY = options[(idx + 1) % len(options)]
            text = f"{SUCCESS_EMOJI} Default currency set to {config.VS_CURRENCY}"
        if text:
            await context.bot.send_message(query.message.chat_id, text)
        keyboard = await get_settings_keyboard(query.message.chat_id)
        await query.edit_message_reply_markup(reply_markup=keyboard)
    elif query.data == "list":
        entries = await build_sub_entries(query.message.chat_id)
        if not entries:
            await context.bot.send_message(
                chat_id=query.message.chat_id,
                text=f"{INFO_EMOJI} No active subscriptions",
            )
        else:
            for coin, text in entries:
                keyboard = InlineKeyboardMarkup(
                    [[InlineKeyboardButton("Remove", callback_data=f"del:{coin}")]]
                )
                await context.bot.send_message(
                    chat_id=query.message.chat_id, text=text, reply_markup=keyboard
                )
        await query.edit_message_reply_markup(reply_markup=get_keyboard())


async def menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle menu button presses from custom keyboards."""
    if not update.message:
        return
    text = update.message.text.strip()
    if text.startswith(SUB_EMOJI):
        parts = text.split()
        if len(parts) >= 3 and parts[1] == "Add":
            coin = api.normalize_coin(parts[2])
            await db.subscribe_coin(
                update.effective_chat.id,
                coin,
                config.DEFAULT_THRESHOLD,
                config.DEFAULT_INTERVAL,
            )
            await update.message.reply_text(
                f"{SUB_EMOJI} Subscribed to {api.symbol_for(coin)}",
                reply_markup=get_keyboard(),
            )
    elif text == f"{LIST_EMOJI} List":
        await list_cmd(update, context)
    elif text == f"{HELP_EMOJI} Help":
        await help_cmd(update, context)
    elif text == SETTINGS_EMOJI:
        menu = await get_settings_menu(update.effective_chat.id)
        await update.message.reply_text(
            f"{INFO_EMOJI} Your settings",
            reply_markup=menu,
        )
    elif text.startswith("threshold"):
        options = [0.1, 0.5, 1.0, 2.0, 5.0]
        try:
            idx = options.index(config.DEFAULT_THRESHOLD)
        except ValueError:
            idx = -1
        config.DEFAULT_THRESHOLD = options[(idx + 1) % len(options)]
        await db.set_user_settings(
            update.effective_chat.id, threshold=config.DEFAULT_THRESHOLD
        )
        menu = await get_settings_menu(update.effective_chat.id)
        await update.message.reply_text(
            f"{SUCCESS_EMOJI} Default threshold set to {config.DEFAULT_THRESHOLD}%",
            reply_markup=menu,
        )
    elif text.startswith("interval"):
        options = [60, 300, 600, 1800, 3600]
        try:
            idx = options.index(config.DEFAULT_INTERVAL)
        except ValueError:
            idx = -1
        config.DEFAULT_INTERVAL = options[(idx + 1) % len(options)]
        interval_text = config.format_interval(config.DEFAULT_INTERVAL)
        await db.set_user_settings(
            update.effective_chat.id, interval=config.DEFAULT_INTERVAL
        )
        menu = await get_settings_menu(update.effective_chat.id)
        await update.message.reply_text(
            f"{SUCCESS_EMOJI} Default interval set to {interval_text}",
            reply_markup=menu,
        )
    elif text.startswith("milestones"):
        config.ENABLE_MILESTONE_ALERTS = not config.ENABLE_MILESTONE_ALERTS
        await db.set_user_settings(
            update.effective_chat.id, milestones=int(config.ENABLE_MILESTONE_ALERTS)
        )
        state = "enabled" if config.ENABLE_MILESTONE_ALERTS else "disabled"
        menu = await get_settings_menu(update.effective_chat.id)
        await update.message.reply_text(
            f"{SUCCESS_EMOJI} Milestone alerts {state}",
            reply_markup=menu,
        )
    elif text.startswith("volume"):
        config.ENABLE_VOLUME_ALERTS = not config.ENABLE_VOLUME_ALERTS
        await db.set_user_settings(
            update.effective_chat.id, volume=int(config.ENABLE_VOLUME_ALERTS)
        )
        state = "enabled" if config.ENABLE_VOLUME_ALERTS else "disabled"
        menu = await get_settings_menu(update.effective_chat.id)
        await update.message.reply_text(
            f"{SUCCESS_EMOJI} Volume alerts {state}",
            reply_markup=menu,
        )
    elif text.startswith("currency"):
        options = ["usd", "eur", "btc"]
        try:
            idx = options.index(config.VS_CURRENCY)
        except ValueError:
            idx = -1
        config.VS_CURRENCY = options[(idx + 1) % len(options)]
        await db.set_user_settings(
            update.effective_chat.id, currency=config.VS_CURRENCY
        )
        menu = await get_settings_menu(update.effective_chat.id)
        await update.message.reply_text(
            f"{SUCCESS_EMOJI} Default currency set to {config.VS_CURRENCY}",
            reply_markup=menu,
        )
    elif text == f"{BACK_EMOJI} Back":
        await update.message.reply_text(
            f"{INFO_EMOJI} Settings saved", reply_markup=get_keyboard()
        )
