#!/usr/bin/env python3
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
from difflib import get_close_matches
from io import BytesIO
from typing import Deque, Dict, Optional, Tuple

import aiohttp
import aiosqlite
import matplotlib
import numpy as np
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

load_dotenv()
DB_FILE = os.getenv("DB_PATH", "subs.db")
BOT_NAME = "PricePulseWatcherBot"
DEFAULT_THRESHOLD = 0.1
DEFAULT_INTERVAL = 300
PRICE_CHECK_INTERVAL = 60

# optional CoinGecko API key for higher rate limits
COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY")
COINGECKO_HEADERS = (
    {"x-cg-pro-api-key": COINGECKO_API_KEY} if COINGECKO_API_KEY else None
)

# API provider selection
API_PROVIDER = os.getenv("PRICE_API_PROVIDER", "coingecko").lower()
CMC_API_KEY = os.getenv("COINMARKETCAP_API_KEY")
CMC_HEADERS = {"X-CMC_PRO_API_KEY": CMC_API_KEY} if CMC_API_KEY else None

# emojis used for price movements
UP_ARROW = "\U0001f53a"  # up triangle
DOWN_ARROW = "\U0001f53b"  # down triangle
ROCKET = "\U0001f680"  # rocket for big gains
BOMB = "\U0001f4a3"  # bomb for big drops
DEFAULT_ALERT_EMOJI = ROCKET


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


def format_interval(seconds: int) -> str:
    """Return a short string representation for a duration in seconds."""
    if seconds % 86400 == 0:
        return f"{seconds // 86400}d"
    if seconds % 3600 == 0:
        return f"{seconds // 3600}h"
    if seconds % 60 == 0:
        return f"{seconds // 60}m"
    return f"{seconds}s"


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
TOP_COINS: list[str] = []


def symbol_for(coin: str) -> str:
    """Return the symbol for a coin ID."""
    return COIN_SYMBOLS.get(coin, coin.upper())


def normalize_coin(value: str) -> str:
    """Return the coin ID for a given symbol or coin name."""
    return SYMBOL_TO_COIN.get(value.lower(), value.lower())


def suggest_coins(name: str, limit: int = 3) -> list[str]:
    """Return close matches for a coin or symbol."""
    candidates = list(
        {
            *COINS,
            *TOP_COINS,
            *COIN_SYMBOLS.keys(),
        }
    )
    matches = get_close_matches(
        name.lower(), [c.lower() for c in candidates], n=limit, cutoff=0.6
    )
    # map back to coin ids and remove duplicates
    coins = [normalize_coin(m) for m in matches]
    seen: set[str] = set()
    result: list[str] = []
    for coin in coins:
        if coin not in seen:
            seen.add(coin)
            result.append(coin)
    return result


async def find_coin(query: str) -> Optional[str]:
    """Return coin id for a symbol or name via the CoinGecko search API."""
    url = f"https://api.coingecko.com/api/v3/search?query={query}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=COINGECKO_HEADERS) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                for item in data.get("coins", []):
                    symbol = item.get("symbol")
                    coin_id = item.get("id")
                    name = item.get("name")
                    if symbol and coin_id and symbol.lower() == query.lower():
                        COIN_SYMBOLS[coin_id] = symbol.upper()
                        SYMBOL_TO_COIN[symbol.lower()] = coin_id
                        return coin_id
                for item in data.get("coins", []):
                    symbol = item.get("symbol")
                    coin_id = item.get("id")
                    name = item.get("name", "")
                    if coin_id and name.lower() == query.lower():
                        if symbol:
                            COIN_SYMBOLS[coin_id] = symbol.upper()
                            SYMBOL_TO_COIN[symbol.lower()] = coin_id
                        return coin_id
    except aiohttp.ClientError as exc:
        logger.warning("search failed: %s", exc)
    return None


async def fetch_trending_coins() -> None:
    """Update COINS and symbol mappings using the trending list from CoinGecko."""
    url = "https://api.coingecko.com/api/v3/search/trending"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=COINGECKO_HEADERS) as resp:
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


async def fetch_top_coins() -> None:
    """Populate TOP_COINS with the coins that have the highest market cap."""
    url = (
        "https://api.coingecko.com/api/v3/coins/markets"
        "?vs_currency=usd&order=market_cap_desc&per_page=50&page=1"
    )
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=COINGECKO_HEADERS) as resp:
                if resp.status != 200:
                    logger.warning("top coins request failed: %s", resp.status)
                    return
                data = await resp.json()
                coins: list[str] = []
                for item in data[:20]:
                    coin_id = item.get("id")
                    symbol = item.get("symbol")
                    if coin_id:
                        coins.append(coin_id)
                        if symbol:
                            COIN_SYMBOLS[coin_id] = symbol.upper()
                            SYMBOL_TO_COIN[symbol.lower()] = coin_id
                global TOP_COINS
                TOP_COINS = coins
    except aiohttp.ClientError as exc:
        logger.error("error fetching top coins: %s", exc)


LOG_FILE = os.getenv("LOG_FILE")
_handlers = [logging.StreamHandler()]
if LOG_FILE:
    _handlers.append(logging.FileHandler(LOG_FILE))

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    handlers=_handlers,
    force=True,
)
logger = logging.getLogger(__name__)

# price cache: coin -> (price, timestamp)
PRICE_CACHE: Dict[str, Tuple[float, float]] = {}
REQUEST_LOCK = asyncio.Lock()
LAST_REQUEST = 0.0

# cache last observed price to avoid zero when API fails
LAST_KNOWN_PRICE: Dict[str, float] = {}

# telegram rate limits
user_messages: Dict[int, Deque[float]] = defaultdict(deque)
global_messages: Deque[float] = deque()

# milestone cache: (chat_id, coin) -> last checked price
MILESTONE_CACHE: Dict[Tuple[int, str], float] = {}


async def api_get(
    url: str,
    session: Optional[aiohttp.ClientSession] = None,
    headers: Optional[dict] = None,
    user: Optional[int] = None,
) -> Optional[aiohttp.ClientResponse]:
    """Perform an HTTP GET request with logging and rate limiting."""
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
                resp = await session.get(url, headers=headers)
            except aiohttp.ClientError as exc:
                logger.error("api request failed: %s", exc)
                return None
            LAST_REQUEST = time.time()
        logger.info("api_request user=%s url=%s status=%s", user, url, resp.status)
        return resp
    finally:
        if owns_session and session:
            await session.close()


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
    if price >= 0.01:
        return 0.001
    if price >= 0.001:
        return 0.0001
    if price >= 0.0001:
        return 0.00001
    return 0.000001


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


def trend_emojis(change: float) -> str:
    """Return arrow and rocket/bomb emojis for a price change."""
    if change >= 10:
        return f"{UP_ARROW} {ROCKET}"
    if change <= -10:
        return f"{DOWN_ARROW} {BOMB}"
    return UP_ARROW if change >= 0 else DOWN_ARROW


def calculate_volume_profile(candles: list[dict]) -> dict:
    """Return volume profile metrics for the given candles.

    Each candle should contain ``high``, ``low`` and ``volume`` keys. The
    volume of a candle is distributed evenly across its price range and added
    to one of 100 bins spanning the observed price range.
    """

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
    coin: str,
    session: Optional[aiohttp.ClientSession] = None,
    *,
    user: Optional[int] = None,
) -> Optional[float]:
    """Return the current USD price for a coin."""
    now = time.time()
    cached = PRICE_CACHE.get(coin)
    if cached and now - cached[1] < 60:
        return cached[0]

    if API_PROVIDER == "coinmarketcap":
        symbol = symbol_for(coin)
        url = (
            "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"
            f"?symbol={symbol}"
        )
        headers = CMC_HEADERS
        key = symbol
    else:
        url = (
            "https://api.coingecko.com/api/v3/simple/price"
            f"?ids={coin}&vs_currencies=usd"
        )
        headers = COINGECKO_HEADERS
        key = coin

    retries = 3
    owns_session = session is None
    if owns_session:
        session = aiohttp.ClientSession()
    try:
        for attempt in range(retries):
            resp = await api_get(url, session=session, headers=headers, user=user)
            if not resp:
                return None
            if resp.status == 200:
                data = await resp.json()
                if API_PROVIDER == "coinmarketcap":
                    info = data.get("data", {}).get(symbol, {})
                    quote = info.get("quote", {}).get("USD", {})
                    price = quote.get("price")
                else:
                    price = data.get(key, {}).get("usd")
                if price is not None:
                    price = float(price)
                    PRICE_CACHE[coin] = (price, time.time())
                    LAST_KNOWN_PRICE[coin] = price
                    return price
            await asyncio.sleep(2**attempt)
    finally:
        if owns_session:
            await session.close()
    return LAST_KNOWN_PRICE.get(coin)


async def get_coin_info(
    coin: str,
    session: Optional[aiohttp.ClientSession] = None,
    *,
    user: Optional[int] = None,
) -> tuple[Optional[dict], Optional[str]]:
    """Return detailed coin info."""
    if API_PROVIDER == "coinmarketcap":
        symbol = symbol_for(coin)
        url = (
            "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"
            f"?symbol={symbol}"
        )
        headers = CMC_HEADERS
    else:
        url = f"https://api.coingecko.com/api/v3/coins/{coin}"
        headers = COINGECKO_HEADERS

    owns_session = session is None
    if owns_session:
        session = aiohttp.ClientSession()
    try:
        for attempt in range(3):
            resp = await api_get(url, session=session, headers=headers, user=user)
            if not resp:
                return None, "request failed"
            if resp.status == 200:
                data = await resp.json()
                if API_PROVIDER == "coinmarketcap":
                    info = data.get("data", {}).get(symbol, {})
                    quote = info.get("quote", {}).get("USD", {})
                    return (
                        {
                            "name": info.get("name"),
                            "symbol": info.get("symbol"),
                            "market_data": {
                                "current_price": {"usd": quote.get("price")},
                                "market_cap": {"usd": quote.get("market_cap")},
                                "price_change_percentage_24h": quote.get(
                                    "percent_change_24h"
                                ),
                            },
                        },
                        None,
                    )
                return data, None
            if resp.status == 404:
                return None, "coin not found"
            await asyncio.sleep(2**attempt)
        return None, f"HTTP {resp.status}"
    finally:
        if owns_session and session:
            await session.close()


async def fetch_ohlcv(
    symbol: str,
    interval: str,
    limit: int,
    session: Optional[aiohttp.ClientSession] = None,
    *,
    user: Optional[int] = None,
) -> tuple[Optional[list[dict]], Optional[str]]:
    """Return OHLCV candles from Binance."""

    url = (
        "https://api.binance.com/api/v3/klines"
        f"?symbol={symbol.upper()}&interval={interval}&limit={limit}"
    )
    owns_session = session is None
    if owns_session:
        session = aiohttp.ClientSession()
    try:
        resp = await api_get(url, session=session, headers=COINGECKO_HEADERS, user=user)
        if not resp:
            return None, "request failed"
        if resp.status == 200:
            data = await resp.json()
            candles = [
                {
                    "high": float(item[2]),
                    "low": float(item[3]),
                    "close": float(item[4]),
                    "volume": float(item[5]),
                }
                for item in data
            ]
            return candles, None
        if resp.status == 429:
            return None, "rate limit exceeded"
        return None, f"HTTP {resp.status}"
    finally:
        if owns_session and session:
            await session.close()


async def get_market_info(
    coin: str,
    session: Optional[aiohttp.ClientSession] = None,
    *,
    user: Optional[int] = None,
) -> Optional[dict]:
    """Return basic market info for a coin."""
    if API_PROVIDER == "coinmarketcap":
        symbol = symbol_for(coin)
        url = (
            "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"
            f"?symbol={symbol}"
        )
        headers = CMC_HEADERS
    else:
        url = (
            "https://api.coingecko.com/api/v3/coins/markets"
            f"?vs_currency=usd&ids={coin}&price_change_percentage=24h"
        )
        headers = COINGECKO_HEADERS

    owns_session = session is None
    if owns_session:
        session = aiohttp.ClientSession()
    try:
        resp = await api_get(url, session=session, headers=headers, user=user)
        if not resp:
            return None
        if resp.status == 200:
            data = await resp.json()
            if API_PROVIDER == "coinmarketcap":
                info = data.get("data", {}).get(symbol, {})
                quote = info.get("quote", {}).get("USD", {})
                return {
                    "current_price": quote.get("price"),
                    "market_cap": quote.get("market_cap"),
                    "price_change_percentage_24h": quote.get("percent_change_24h"),
                }
            if data:
                return data[0]
    finally:
        if owns_session and session:
            await session.close()
    return None


async def get_market_chart(
    coin: str, days: int, session: Optional[aiohttp.ClientSession] = None
) -> tuple[Optional[list[tuple[float, float]]], Optional[str]]:
    """Return historical price chart data for a coin."""
    end_ts = int(time.time())
    start_ts = end_ts - days * 86400
    if API_PROVIDER == "coinmarketcap":
        symbol = symbol_for(coin)
        url = (
            "https://pro-api.coinmarketcap.com/v2/cryptocurrency/ohlcv/historical"
            f"?symbol={symbol}&time_start={start_ts}&time_end={end_ts}"
        )
        headers = CMC_HEADERS
    else:
        url = (
            f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart/range"
            f"?vs_currency=usd&from={start_ts}&to={end_ts}"
        )
        headers = COINGECKO_HEADERS
    owns_session = session is None
    if owns_session:
        session = aiohttp.ClientSession()
    try:
        resp = await api_get(url, session=session, headers=headers)
        if not resp:
            return None, "request failed"
        if resp.status == 200:
            data = await resp.json()
            if API_PROVIDER == "coinmarketcap":
                quotes = data.get("data", {}).get("quotes", [])
                return [
                    (
                        q.get("time_open", 0) / 1000,
                        q.get("quote", {}).get("USD", {}).get("close"),
                    )
                    for q in quotes
                ], None
            return [(p[0] / 1000, p[1]) for p in data.get("prices", [])], None
        if resp.status == 404:
            return None, "coin not found"
        return None, f"HTTP {resp.status}"
    finally:
        if owns_session and session:
            await session.close()


async def get_global_overview(
    session: Optional[aiohttp.ClientSession] = None,
    *,
    user: Optional[int] = None,
) -> tuple[Optional[dict], Optional[str]]:
    """Return global market data."""
    if API_PROVIDER == "coinmarketcap":
        url = "https://pro-api.coinmarketcap.com/v1/global-metrics/quotes/latest"
        headers = CMC_HEADERS
    else:
        url = "https://api.coingecko.com/api/v3/global"
        headers = COINGECKO_HEADERS
    owns_session = session is None
    if owns_session:
        session = aiohttp.ClientSession()
    try:
        for attempt in range(3):
            resp = await api_get(url, session=session, headers=headers, user=user)
            if not resp:
                return None, "request failed"
            if resp.status == 200:
                data = await resp.json()
                if API_PROVIDER == "coinmarketcap":
                    quote = data.get("data", {}).get("quote", {}).get("USD", {})
                    btc_dominance = data.get("data", {}).get("btc_dominance")
                    return (
                        {
                            "data": {
                                "total_market_cap": {
                                    "usd": quote.get("total_market_cap")
                                },
                                "total_volume": {"usd": quote.get("total_volume_24h")},
                                "market_cap_percentage": {"btc": btc_dominance},
                                "market_cap_change_percentage_24h_usd": quote.get(
                                    "percent_change_24h"
                                ),
                            }
                        },
                        None,
                    )
                return data, None
            await asyncio.sleep(2**attempt)
        return None, f"HTTP {resp.status}"
    finally:
        if owns_session and session:
            await session.close()


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


async def send_rate_limited(
    bot: Bot, chat_id: int, text: str, emoji: str = DEFAULT_ALERT_EMOJI
) -> None:
    """Send a message respecting basic rate limits."""
    now = time.time()
    # cleanup timestamps
    user_q = user_messages[chat_id]
    while user_q and now - user_q[0] > 60:
        user_q.popleft()
    while global_messages and now - global_messages[0] > 1:
        global_messages.popleft()

    # wait if limits exceeded
    if len(user_q) >= 20:
        wait = max(0, 60 - (now - user_q[0]))  # clamp negative sleeps
        await asyncio.sleep(wait)
    if len(global_messages) >= 30:
        wait = max(0, 1 - (now - global_messages[0]))  # clamp negative sleeps
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
        price = await get_price(coin, user=None)
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

                    msg = f"{symbol} breaks through ${level:.0f} " f"(now ${price})"
                    await send_rate_limited(
                        app.bot, chat_id, msg, emoji=f"{UP_ARROW} {ROCKET}"
                    )
                else:
                    msg = f"{symbol} falls below ${level:.0f} " f"(now ${price})"
                    await send_rate_limited(
                        app.bot, chat_id, msg, emoji=f"{DOWN_ARROW} {BOMB}"
                    )

            MILESTONE_CACHE[(chat_id, coin)] = price

            if last_ts is None or time.time() - last_ts >= interval:
                raw_change = (price - last_price) / last_price * 100
                change = abs(raw_change)
                if change >= threshold:

                    symbol = symbol_for(coin)
                    text = (
                        f"{symbol} moved {raw_change:+.2f}% in "
                        f"{format_interval(interval)} (now ${price})"
                    )
                    await send_rate_limited(
                        app.bot, chat_id, text, emoji=trend_emojis(raw_change)
                    )

                await set_last_price(sub_id, price)


SUB_EMOJI = "\U00002795"
RELOAD_EMOJI = "\U000027f3"
LIST_EMOJI = "\U0001f4cb"
HELP_EMOJI = "\u2753"
WELCOME_EMOJI = "\U0001f44b"
INFO_EMOJI = "\u2139\ufe0f"
SUCCESS_EMOJI = "\u2705"
ERROR_EMOJI = "\u26a0\ufe0f"


def get_keyboard() -> ReplyKeyboardMarkup:
    coins_source = TOP_COINS[:20] if TOP_COINS else (COINS or ["bitcoin"])
    coins = random.sample(coins_source, k=min(3, len(coins_source)))
    subs = [KeyboardButton(f"{SUB_EMOJI} Add {symbol_for(c)}") for c in coins]
    keyboard = [
        subs,
        [KeyboardButton(RELOAD_EMOJI)],
        [KeyboardButton(f"{LIST_EMOJI} List"), KeyboardButton(f"{HELP_EMOJI} Help")],
    ]
    return ReplyKeyboardMarkup(
        keyboard,
        resize_keyboard=True,
        is_persistent=True,
    )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command and show main menu."""
    logger.debug("/start from %s", update.effective_chat.id)
    await update.message.reply_text(
        f"{WELCOME_EMOJI} Welcome to {BOT_NAME}! Choose an action:",
        reply_markup=get_keyboard(),
    )


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Display usage information."""
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
    """Subscribe the chat to a coin at a given threshold and interval."""
    if not context.args:
        await update.message.reply_text(
            f"{ERROR_EMOJI} Usage: /add <coin> [pct] [interval]",
            quote=True,
        )
        return
    coin_input = context.args[0]
    coin = normalize_coin(coin_input)
    info = await get_market_info(coin, user=update.effective_chat.id)
    if not info or info.get("current_price") is None:
        alt = await find_coin(coin_input)
        if alt:
            coin = alt
            info = await get_market_info(coin, user=update.effective_chat.id)
    if not info or info.get("current_price") is None:
        suggestions = suggest_coins(coin_input)
        msg = f"{ERROR_EMOJI} Unknown coin"
        if suggestions:
            syms = ", ".join(symbol_for(c) for c in suggestions)
            msg += f". Meintest du: {syms}?"
        await update.message.reply_text(msg)
        return
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
            f"every {format_interval(interval)}"
        ),
        reply_markup=get_keyboard(),
    )


async def unsubscribe_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Remove an existing subscription."""
    if not context.args:
        await update.message.reply_text(f"{ERROR_EMOJI} Usage: /remove <coin>")
        return
    coin = normalize_coin(context.args[0])
    await unsubscribe_coin(update.effective_chat.id, coin)

    logger.info(
        "chat %s unsubscribes via command from %s", update.effective_chat.id, coin
    )

    await update.message.reply_text(
        f"{SUCCESS_EMOJI} Unsubscribed from {symbol_for(coin)} alerts",
        reply_markup=get_keyboard(),
    )


async def build_sub_entries(chat_id: int) -> list[tuple[str, str]]:
    """Return list of (coin, formatted text) for all subscriptions."""
    subs = await list_subscriptions(chat_id)
    entries: list[tuple[str, str]] = []
    for _, coin, threshold, interval, *_ in subs:
        info, _ = await get_coin_info(coin, user=chat_id)
        info = info or {}
        market = info.get("market_data", {})
        price = (
            market.get("current_price", {}).get("usd")
            or await get_price(coin, user=chat_id)
            or 0
        )
        cap = market.get("market_cap", {}).get("usd")
        change_24h = market.get("price_change_percentage_24h")
        sym = info.get("symbol")
        if sym:
            COIN_SYMBOLS[coin] = sym.upper()
            SYMBOL_TO_COIN[sym.lower()] = coin
        line = f"{INFO_EMOJI} {info.get('name', coin.title())}"
        if sym:
            line += f" ({sym.upper()})"
        line += "\n"
        line += f"Price: ${format_price(price)}\n"
        if cap is not None:
            line += f"Market Cap: ${cap:,.0f}\n"
        if change_24h is not None:
            line += f"24h Change: {change_24h:.2f}%\n"
        line += f"Alerts: ±{threshold}% every {format_interval(interval)}"
        entries.append((coin, line))
    return entries


async def build_list_keyboard(chat_id: int) -> Optional[InlineKeyboardMarkup]:
    """Return inline buttons to remove each subscription."""
    subs = await list_subscriptions(chat_id)
    if not subs:
        return None
    buttons = [
        [
            InlineKeyboardButton(
                f"Remove {symbol_for(coin)}",
                callback_data=f"del:{coin}",
            )
        ]
        for _, coin, *_ in subs
    ]
    return InlineKeyboardMarkup(buttons)


async def list_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """List all active subscriptions for the chat."""
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
    """Show basic information about a coin."""
    if not context.args:
        await update.message.reply_text(f"{ERROR_EMOJI} Usage: /info <coin>")
        return
    coin = normalize_coin(context.args[0])
    data, err = await get_coin_info(coin, user=update.effective_chat.id)
    if err:
        await update.message.reply_text(f"{ERROR_EMOJI} {err}")
        return
    if not data:
        await update.message.reply_text(f"{ERROR_EMOJI} No data available")
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
    """Send a price history chart as an image."""
    if not context.args:
        await update.message.reply_text(f"{ERROR_EMOJI} Usage: /chart <coin> [days]")
        return
    coin = normalize_coin(context.args[0])
    days = 7
    if len(context.args) > 1:
        try:
            days = int(context.args[1])
        except ValueError:
            await update.message.reply_text(f"{ERROR_EMOJI} Days must be a number")
            return
    data, err = await get_market_chart(coin, days, user=update.effective_chat.id)
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
    """Display global market statistics."""
    data, err = await get_global_overview(user=update.effective_chat.id)
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
    """Show currently trending coins and their prices."""
    await fetch_trending_coins()
    if not COINS:
        await update.message.reply_text(f"{ERROR_EMOJI} Failed to fetch data")
        return
    lines = []
    async with aiohttp.ClientSession() as session:
        for coin in COINS:
            info = (
                await get_market_info(
                    coin, session=session, user=update.effective_chat.id
                )
                or {}
            )
            price = info.get("current_price")
            change_24h = info.get("price_change_percentage_24h")
            line = f"{symbol_for(coin)}"
            if price is not None:
                line += f" ${format_price(price)}"
            if change_24h is not None:
                line += f" ({change_24h:+.2f}% 24h)"
            lines.append(line)
    text = f"{INFO_EMOJI} Trending coins:\n" + "\n".join(lines)
    await update.message.reply_text(text)


async def valuearea_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Display VAL, POC and VAH for a trading pair."""

    if len(context.args) < 3:
        await update.message.reply_text(
            f"{ERROR_EMOJI} Usage: /valuearea <symbol> <interval> <count>"
        )
        return

    symbol = context.args[0].upper()
    interval = context.args[1]
    try:
        limit = int(context.args[2])
    except ValueError:
        await update.message.reply_text(f"{ERROR_EMOJI} Count must be a number")
        return

    candles, err = await fetch_ohlcv(
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
    """Handle inline keyboard button callbacks."""
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
                f"±{DEFAULT_THRESHOLD}% every {format_interval(DEFAULT_INTERVAL)}"
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
                    chat_id=query.message.chat_id,
                    text=text,
                    reply_markup=keyboard,
                )
        await query.edit_message_reply_markup(reply_markup=get_keyboard())


async def menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle replies from the custom keyboard."""
    if not update.message:
        return
    text = update.message.text.strip()

    if text.startswith(SUB_EMOJI):
        parts = text.split()
        if len(parts) >= 3 and parts[1] == "Add":
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
                    f"±{DEFAULT_THRESHOLD}% every "
                    f"{format_interval(DEFAULT_INTERVAL)}"
                ),
                reply_markup=get_keyboard(),
            )
    elif text == RELOAD_EMOJI:
        await update.message.reply_text("New suggestion:", reply_markup=get_keyboard())
    elif text == f"{LIST_EMOJI} List":
        await list_cmd(update, context)
    elif text == f"{HELP_EMOJI} Help":
        await help_cmd(update, context)


async def main() -> None:
    """Start the bot."""
    load_dotenv()
    load_config()
    await init_db()
    await fetch_trending_coins()
    await fetch_top_coins()

    token = os.getenv("TELEGRAM_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_TOKEN not set")

    app = ApplicationBuilder().token(token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("add", subscribe_cmd))
    app.add_handler(CommandHandler("remove", unsubscribe_cmd))
    app.add_handler(CommandHandler("list", list_cmd))
    app.add_handler(CommandHandler("info", info_cmd))
    app.add_handler(CommandHandler("chart", chart_cmd))
    app.add_handler(CommandHandler("trends", trends_cmd))
    app.add_handler(CommandHandler("global", global_cmd))
    app.add_handler(CommandHandler("valuearea", valuearea_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, menu))
    app.add_handler(CallbackQueryHandler(button))

    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        check_prices, "interval", seconds=PRICE_CHECK_INTERVAL, args=(app,)
    )
    scheduler.add_job(fetch_trending_coins, "interval", minutes=10)
    scheduler.add_job(fetch_top_coins, "interval", minutes=10)
    scheduler.start()

    await app.initialize()
    await app.bot.set_my_commands(
        [
            BotCommand("start", "Show menu"),
            BotCommand("help", "Show help"),
            BotCommand("add", "Subscribe to price alerts"),
            BotCommand("remove", "Remove subscription"),
            BotCommand("list", "List subscriptions"),
            BotCommand("info", "Coin information"),
            BotCommand("chart", "Price chart"),
            BotCommand("trends", "Trending coins"),
            BotCommand("global", "Global market"),
            BotCommand("valuearea", "Volume profile"),
        ]
    )
    await app.start()
    await app.updater.start_polling()
    logger.info(f"{BOT_NAME} started")

    stop_event = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        asyncio.get_running_loop().add_signal_handler(sig, stop_event.set)

    await stop_event.wait()
    await app.updater.stop()
    await app.stop()
    await app.shutdown()
    scheduler.shutdown()
    logger.info(f"{BOT_NAME} stopped")


if __name__ == "__main__":
    asyncio.run(main())
