import asyncio
import random
import time
from collections import defaultdict, deque
from decimal import Decimal
from io import BytesIO
from typing import Deque, Dict, List, Optional, Tuple

import aiohttp
import numpy as np
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
DEFAULT_ALERT_EMOJI = ROCKET

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


def milestone_step(price: float) -> float:
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
    return format(Decimal(str(value)), "f")


def milestones_crossed(last: float, current: float) -> List[float]:
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
    if change >= 10:
        return f"{UP_ARROW} {ROCKET}"
    if change <= -10:
        return f"{DOWN_ARROW} {BOMB}"
    return UP_ARROW if change >= 0 else DOWN_ARROW


def calculate_volume_profile(candles: List[dict]) -> dict:
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
    bot: Bot, chat_id: int, text: str, emoji: str = DEFAULT_ALERT_EMOJI
) -> None:
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
    await bot.send_message(chat_id=chat_id, text=f"{emoji} {text}")
    user_q.append(time.time())
    global_messages.append(time.time())


async def check_prices(app) -> None:
    async with aiohttp.ClientSession() as http_session:
        async with db.aiosqlite.connect(config.DB_FILE) as database:
            cursor = await database.execute(
                "SELECT id, chat_id, coin_id, threshold, interval, last_price, "
                "last_alert_ts FROM subscriptions"
            )
            rows = await cursor.fetchall()
            await cursor.close()
        by_coin: Dict[
            str, List[Tuple[int, int, float, int, Optional[float], Optional[float]]]
        ] = {}
        for sub_id, chat_id, coin, threshold, interval, last_price, last_ts in rows:
            by_coin.setdefault(coin, []).append(
                (sub_id, chat_id, threshold, interval, last_price, last_ts)
            )
        coins = list(by_coin.keys())
        prices: Dict[str, float] = {}
        missing: List[str] = []
        for coin in coins:
            cached = await db.get_coin_data(coin, max_age=config.CACHE_TTL)
            if cached and cached.get("price") is not None:
                prices[coin] = float(cached["price"])
            else:
                missing.append(coin)
        if missing:
            if len(missing) > 1:
                groups = [
                    missing[i : i + 250]  # noqa: E203
                    for i in range(0, len(missing), 250)
                ]
                for group in groups:
                    prices.update(
                        await api.get_prices(group, session=http_session, user=None)
                    )
            else:
                coin = missing[0]
                price = await api.get_price(coin, user=None)
                if price is not None:
                    prices[coin] = price
        for coin, subscriptions in by_coin.items():
            price = prices.get(coin)
            if price is None:
                continue
            for (
                sub_id,
                chat_id,
                threshold,
                interval,
                last_price,
                last_ts,
            ) in subscriptions:
                if last_price is None:
                    await db.set_last_price(sub_id, price)
                    MILESTONE_CACHE[(chat_id, coin)] = price
                    continue
                prev = MILESTONE_CACHE.get((chat_id, coin), last_price)
                for level in milestones_crossed(prev, price):
                    symbol = api.symbol_for(coin)
                    if price > prev:
                        msg = f"{symbol} breaks through ${level:.0f} (now ${price})"
                        await send_rate_limited(
                            app.bot, chat_id, msg, emoji=f"{UP_ARROW} {ROCKET}"
                        )
                    else:
                        msg = f"{symbol} falls below ${level:.0f} (now ${price})"
                        await send_rate_limited(
                            app.bot, chat_id, msg, emoji=f"{DOWN_ARROW} {BOMB}"
                        )
                MILESTONE_CACHE[(chat_id, coin)] = price
                if last_ts is None or time.time() - last_ts >= interval:
                    raw_change = (price - last_price) / last_price * 100
                    change = abs(raw_change)
                    if change >= threshold:
                        symbol = api.symbol_for(coin)
                        text = (
                            f"{symbol} moved {raw_change:+.2f}% in "
                            f"{config.format_interval(interval)} (now ${price}"
                        )
                        cached = await db.get_coin_data(coin, max_age=config.CACHE_TTL)
                        info = cached.get("market_info") if cached else None
                        if info is None:
                            info = await api.get_market_info(coin, user=chat_id)
                        change_24h = None
                        if info:
                            change_24h = info.get("price_change_percentage_24h")
                        if change_24h is not None:
                            text += f", {change_24h:+.2f}% 24h"
                        text += ")"
                        await send_rate_limited(
                            app.bot, chat_id, text, emoji=trend_emojis(raw_change)
                        )
                    await db.set_last_price(sub_id, price)


async def refresh_cache(app) -> None:
    async with db.aiosqlite.connect(config.DB_FILE) as database:
        cursor = await database.execute("SELECT DISTINCT coin_id FROM subscriptions")
        coins = [row[0] for row in await cursor.fetchall()]
        await cursor.close()
    for coin in coins:
        await api.refresh_coin_data(coin)
    await api.get_global_overview(user=None)


def get_keyboard() -> ReplyKeyboardMarkup:
    coins_source = config.COINS or config.TOP_COINS[:20] or ["bitcoin"]
    coins = random.sample(coins_source, k=min(3, len(coins_source)))
    subs = [KeyboardButton(f"{SUB_EMOJI} Add {api.symbol_for(c)}") for c in coins]
    keyboard = [
        subs,
        [KeyboardButton(f"{LIST_EMOJI} List"), KeyboardButton(f"{HELP_EMOJI} Help")],
    ]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True, is_persistent=True)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        f"{WELCOME_EMOJI} Welcome to {config.BOT_NAME}! Use /add or the "
        "buttons below to subscribe to price alerts.",
        reply_markup=get_keyboard(),
    )


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        f"{INFO_EMOJI} /add <coin> [pct] [interval] - subscribe to price alerts\n"
        "/remove <coin> - remove subscription\n"
        "/list - list subscriptions\n"
        "/info <coin> - coin information\n"
        "/chart(s) <coin> [days] - price chart\n"
        "/trends - show trending coins\n"
        "/global - global market stats\n"
        "/valuearea <symbol> <interval> <count> - volume profile\n"
        "Intervals can be like 1h, 15m or 30s",
        reply_markup=get_keyboard(),
    )


async def subscribe_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text(
            f"{ERROR_EMOJI} Usage: /add <coin> [pct] [interval]", quote=True
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
    try:
        threshold = (
            float(context.args[1])
            if len(context.args) > 1
            else config.DEFAULT_THRESHOLD
        )
    except ValueError:
        await update.message.reply_text(f"{ERROR_EMOJI} Threshold must be a number")
        return
    try:
        interval_str = (
            context.args[2] if len(context.args) > 2 else str(config.DEFAULT_INTERVAL)
        )
        interval = config.parse_duration(interval_str)
    except ValueError:
        await update.message.reply_text(
            f"{ERROR_EMOJI} Interval must be a number or like 1h, 15m, 30s"
        )
        return
    await db.subscribe_coin(update.effective_chat.id, coin, threshold, interval)
    await update.message.reply_text(
        f"{SUB_EMOJI} Subscribed to {api.symbol_for(coin)}",
        reply_markup=get_keyboard(),
    )


async def unsubscribe_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text(f"{ERROR_EMOJI} Usage: /remove <coin>")
        return
    coin = api.normalize_coin(context.args[0])
    await db.unsubscribe_coin(update.effective_chat.id, coin)
    await update.message.reply_text(
        f"{SUCCESS_EMOJI} Unsubscribed from {api.symbol_for(coin)} alerts",
        reply_markup=get_keyboard(),
    )


async def build_sub_entries(chat_id: int) -> List[Tuple[str, str]]:
    subs = await db.list_subscriptions(chat_id)
    entries: List[Tuple[str, str]] = []
    for _, coin, threshold, interval, *_ in subs:
        cached = await db.get_coin_data(coin, max_age=config.CACHE_TTL)
        info = cached.get("info") if cached else None
        market = cached.get("market_info") if cached else None
        price = cached.get("price") if cached else None
        if info is None:
            info, _ = await api.get_coin_info(coin, user=chat_id)
        info = info or {}
        if market is None:
            market = info.get("market_data", {})
        if price is None:
            price = (
                market.get("current_price", {}).get("usd")
                or await api.get_price(coin, user=chat_id)
                or 0
            )
        cap = market.get("market_cap", {}).get("usd")
        change_24h = market.get("price_change_percentage_24h")
        sym = info.get("symbol")
        if sym:
            config.COIN_SYMBOLS[coin] = sym.upper()
            config.SYMBOL_TO_COIN[sym.lower()] = coin
        line = f"{INFO_EMOJI} {info.get('name', coin.title())}"
        if sym:
            line += f" ({sym.upper()})"
        line += "\n"
        line += f"Alerts: ±{threshold}% every {config.format_interval(interval)}"
        if price:
            line += f" - Price: ${format_price(price)}"
        if change_24h is not None:
            line += f" ({change_24h:+.2f}% 24h)"
        if cap is not None:
            line += f" - Cap: ${cap:,.0f}"
        entries.append((coin, line))
    return entries


async def list_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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
    cached = await db.get_coin_data(coin, max_age=config.CACHE_TTL)
    data = cached.get("info") if cached else None
    market = cached.get("market_info") if cached else None
    if data is None:
        data, err = await api.get_coin_info(coin, user=update.effective_chat.id)
        if err:
            await update.message.reply_text(f"{ERROR_EMOJI} {err}")
            return
    if data is None:
        await update.message.reply_text(f"{ERROR_EMOJI} No data available")
        return
    if market is None:
        market = data.get("market_data", {})
    price = market.get("current_price", {}).get("usd")
    cap = market.get("market_cap", {}).get("usd")
    change = market.get("price_change_percentage_24h")
    sym = data.get("symbol", "").upper()
    config.COIN_SYMBOLS[coin] = sym
    config.SYMBOL_TO_COIN[sym.lower()] = coin
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
    days = 7
    if len(context.args) > 1:
        try:
            days = int(context.args[1])
        except ValueError:
            await update.message.reply_text(f"{ERROR_EMOJI} Days must be a number")
            return
    cached = await db.get_coin_data(coin, max_age=config.CACHE_TTL)
    if days == 7 and cached and cached.get("chart_7d") is not None:
        data = [(p[0], p[1]) for p in cached["chart_7d"]]
        err = None
    else:
        data, err = await api.get_market_chart(
            coin, days, user=update.effective_chat.id
        )
    if err:
        await update.message.reply_text(f"{ERROR_EMOJI} {err}")
        return
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
    data, err = await api.get_global_overview(user=update.effective_chat.id)
    if err:
        await update.message.reply_text(f"{ERROR_EMOJI} {err}")
        return
    if data is None:
        await update.message.reply_text(f"{ERROR_EMOJI} Failed to fetch data")
        return
    info = data.get("data", {})
    cap = info.get("total_market_cap", {}).get("usd")
    volume = info.get("total_volume", {}).get("usd")
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


async def trends_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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


async def valuearea_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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


async def button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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
