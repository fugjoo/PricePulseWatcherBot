"""Asynchronous helpers for interacting with cryptocurrency APIs.

Functions in this module wrap calls to external services such as CoinGecko and
Binance. Results are cached in memory and stored in the database when
appropriate.
"""

import asyncio
import time
from collections import deque
from difflib import get_close_matches
from typing import Deque, Dict, Optional, Tuple
from urllib.parse import quote

import aiohttp
from aiolimiter import AsyncLimiter

from . import config, db

PRICE_CACHE: Dict[str, Tuple[float, float]] = {}
COINGECKO_LIMITER = AsyncLimiter(30, 60)
LAST_KNOWN_PRICE: Dict[str, float] = {}
STATUS_HISTORY: Deque[Tuple[float, int]] = deque(maxlen=100)


def status_counts() -> Dict[int, int]:
    """Return a mapping of HTTP status codes to occurrence counts."""
    counts: Dict[int, int] = {}
    for _, status in STATUS_HISTORY:
        counts[status] = counts.get(status, 0) + 1
    return counts


def symbol_for(coin: str) -> str:
    """Return the trading symbol for a given coin ID."""

    return config.COIN_SYMBOLS.get(coin, coin.upper())


def normalize_coin(value: str) -> str:
    """Map a symbol or ID to a canonical coin ID."""

    return config.SYMBOL_TO_COIN.get(value.lower(), value.lower())


def encoded(coin: str) -> str:
    """URL-encode a coin ID for use in API requests."""

    return quote(coin, safe="-")


async def resolve_pair(
    value: str, quote: str = "USDT", *, user: Optional[int] = None
) -> str:
    """Return a Binance trading pair for a coin or symbol."""
    pair = value.replace("/", "").upper()
    quotes = ("USDT", "BUSD", "USDC", "USD", "BNB", "TRY", "EUR")
    if any(pair.endswith(q) for q in quotes):
        return pair
    coin = await resolve_coin(value, user=user)
    symbol = symbol_for(coin) if coin else pair
    return f"{symbol}{quote}"


async def suggest_coins(name: str, limit: int = 3) -> list[str]:
    """Return a list of coin IDs that closely match *name*.

    Parameters
    ----------
    name:
        Partial coin name or symbol provided by the user.
    limit:
        Maximum number of suggestions to return.

    Returns
    -------
    list[str]
        Possible coin IDs sorted by similarity.
    """
    candidates = list(
        {
            *config.COINS,
            *config.TOP_COINS,
            *config.COIN_SYMBOLS.keys(),
            *config.SYMBOL_TO_COIN.keys(),
        }
    )
    matches = get_close_matches(
        name.lower(), [c.lower() for c in candidates], n=limit, cutoff=0.6
    )
    coins = [normalize_coin(m) for m in matches]
    seen: set[str] = set()
    result: list[str] = []
    for coin in coins:
        if coin not in seen:
            seen.add(coin)
            result.append(coin)

    if not matches:
        found = await find_coin(name)
        if found:
            return [found]
    return result


async def api_get(
    url: str,
    session: Optional[aiohttp.ClientSession] = None,
    headers: Optional[dict] = None,
    user: Optional[int] = None,
) -> Optional[aiohttp.ClientResponse]:
    """Perform an HTTP GET request with optional rate limiting.

    Parameters
    ----------
    url:
        Endpoint to request.
    session:
        Existing ``ClientSession`` to use. If omitted a new one is created.
    headers:
        Optional headers to include in the request.
    user:
        User ID used for logging purposes.

    Returns
    -------
    Optional[aiohttp.ClientResponse]
        The response object or ``None`` when the request fails.
    """
    owns_session = session is None
    if owns_session:
        session = aiohttp.ClientSession()
    try:
        limiter = COINGECKO_LIMITER if "coingecko.com" in url else None
        for attempt in range(5):
            if limiter:
                async with limiter:
                    resp = await session.get(url, headers=headers)
            else:
                resp = await session.get(url, headers=headers)
            STATUS_HISTORY.append((time.time(), resp.status))
            config.logger.info(
                "api_request user=%s url=%s status=%s", user, url, resp.status
            )
            if resp.status != 429:
                return resp
            retry_after = resp.headers.get("Retry-After")
            wait = float(retry_after) if retry_after else 2**attempt
            await asyncio.sleep(wait)
        return resp
    except aiohttp.ClientError as exc:
        STATUS_HISTORY.append((time.time(), 0))
        config.logger.error("api request failed: %s", exc)
        return None
    finally:
        if owns_session and session:
            await session.close()


async def get_price(
    coin: str,
    session: Optional[aiohttp.ClientSession] = None,
    *,
    user: Optional[int] = None,
) -> Optional[float]:
    """Return the current USD price for ``coin``.

    Parameters
    ----------
    coin:
        Coin ID to query.
    session:
        Optional ``aiohttp`` session used for the request.
    user:
        User ID for logging.

    Returns
    -------
    Optional[float]
        The price in USD or ``None`` if it cannot be fetched.
    """
    now = time.time()
    cached = PRICE_CACHE.get(coin)
    if cached and now - cached[1] < 60:
        return cached[0]

    url = (
        f"{config.COINGECKO_BASE_URL}/simple/price"
        f"?ids={encoded(coin)}&vs_currencies=usd"
    )
    headers = config.COINGECKO_HEADERS
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


async def get_prices(
    coins: list[str],
    session: Optional[aiohttp.ClientSession] = None,
    *,
    user: Optional[int] = None,
) -> dict[str, float]:
    """Fetch USD prices for multiple coins at once.

    Parameters
    ----------
    coins:
        List of coin IDs to query.
    session:
        Optional session used for the HTTP request.
    user:
        User ID for logging.

    Returns
    -------
    dict[str, float]
        Mapping of coin ID to its current price.
    """
    ids = ",".join(encoded(c) for c in coins)
    url = f"{config.COINGECKO_BASE_URL}/simple/price?ids={ids}&vs_currencies=usd"
    retries = 3
    owns_session = session is None
    if owns_session:
        session = aiohttp.ClientSession()
    try:
        for attempt in range(retries):
            resp = await api_get(
                url, session=session, headers=config.COINGECKO_HEADERS, user=user
            )
            if not resp:
                return {}
            if resp.status == 200:
                data = await resp.json()
                now = time.time()
                result = {}
                for coin in coins:
                    price = data.get(coin, {}).get("usd")
                    if price is not None:
                        price = float(price)
                        PRICE_CACHE[coin] = (price, now)
                        LAST_KNOWN_PRICE[coin] = price
                        result[coin] = price
                return result
            await asyncio.sleep(2**attempt)
    finally:
        if owns_session and session:
            await session.close()
    return {}


async def get_markets(
    coins: list[str],
    session: Optional[aiohttp.ClientSession] = None,
    *,
    user: Optional[int] = None,
) -> dict[str, dict]:
    """Return market info for multiple ``coins``.

    Parameters
    ----------
    coins:
        List of coin IDs to query.
    session:
        Optional session used for the HTTP request.
    user:
        User ID for logging.

    Returns
    -------
    dict[str, dict]
        Mapping of coin ID to the market data dictionary.
    """
    ids = ",".join(encoded(c) for c in coins)
    url = (
        f"{config.COINGECKO_BASE_URL}/coins/markets"
        f"?vs_currency=usd&ids={ids}&price_change_percentage=24h"
    )
    retries = 3
    owns_session = session is None
    if owns_session:
        session = aiohttp.ClientSession()
    try:
        for attempt in range(retries):
            resp = await api_get(
                url, session=session, headers=config.COINGECKO_HEADERS, user=user
            )
            if not resp:
                return {}
            if resp.status == 200:
                data = await resp.json()
                return {item.get("id"): item for item in data if item.get("id")}
            await asyncio.sleep(2**attempt)
    finally:
        if owns_session and session:
            await session.close()
    return {}


async def get_coin_info(
    coin: str,
    session: Optional[aiohttp.ClientSession] = None,
    *,
    user: Optional[int] = None,
) -> tuple[Optional[dict], Optional[str]]:
    """Return detailed information about ``coin`` from CoinGecko.

    Parameters
    ----------
    coin:
        Coin ID to fetch information for.
    session:
        Optional HTTP session.
    user:
        User ID for logging.

    Returns
    -------
    tuple[Optional[dict], Optional[str]]
        Parsed JSON data on success and ``None`` otherwise with an error
        message.
    """
    cached = await db.get_coin_info(coin)
    if cached:
        return cached, None
    url = f"{config.COINGECKO_BASE_URL}/coins/{encoded(coin)}"
    headers = config.COINGECKO_HEADERS
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
                await db.set_coin_info(coin, data)
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
    headers: Optional[dict] = None,
    user: Optional[int] = None,
) -> tuple[Optional[list[dict]], Optional[str]]:
    """Fetch OHLCV candles from Binance.

    Parameters
    ----------
    symbol:
        Trading pair symbol, e.g. ``BTCUSDT``.
    interval:
        Candle interval such as ``1h`` or ``5m``.
    limit:
        Number of candles to return.
    session:
        Optional ``ClientSession`` to use.
    user:
        User ID for logging.

    Returns
    -------
    tuple[list[dict] | None, str | None]
        A list of candle dictionaries and ``None`` on success or ``None`` and an
        error message when something goes wrong.
    """
    url = (
        "https://api.binance.com/api/v3/klines"
        f"?symbol={symbol.upper()}&interval={interval}&limit={limit}"
    )
    owns_session = session is None
    if owns_session:
        session = aiohttp.ClientSession()
    try:
        resp = await api_get(url, session=session, headers=headers, user=user)
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
    """Return market data for ``coin`` such as price and 24h change."""
    url = (
        f"{config.COINGECKO_BASE_URL}/coins/markets"
        f"?vs_currency=usd&ids={encoded(coin)}&price_change_percentage=24h"
    )
    headers = config.COINGECKO_HEADERS
    owns_session = session is None
    if owns_session:
        session = aiohttp.ClientSession()
    try:
        resp = await api_get(url, session=session, headers=headers, user=user)
        if not resp:
            return None
        if resp.status == 200:
            data = await resp.json()
            if data:
                return data[0]
    finally:
        if owns_session and session:
            await session.close()
    return None


async def get_market_chart(
    coin: str,
    days: int,
    session: Optional[aiohttp.ClientSession] = None,
    *,
    user: Optional[int] = None,
) -> tuple[Optional[list[tuple[float, float]]], Optional[str]]:
    """Return historical price data for ``coin``.

    Parameters
    ----------
    coin:
        Coin ID to fetch prices for.
    days:
        Number of days in the past to include.
    session:
        Optional HTTP session.
    user:
        User ID for logging.

    Returns
    -------
    tuple[list[tuple[float, float]] | None, str | None]
        List of ``(timestamp, price)`` tuples or an error message.
    """
    cached = await db.get_coin_chart(coin, days)
    if cached is not None:
        return [(p[0], p[1]) for p in cached], None
    end_ts = int(time.time())
    start_ts = end_ts - days * 86400
    url = (
        f"{config.COINGECKO_BASE_URL}/coins/{encoded(coin)}/market_chart/range"
        f"?vs_currency=usd&from={start_ts}&to={end_ts}"
    )
    headers = config.COINGECKO_HEADERS
    owns_session = session is None
    if owns_session:
        session = aiohttp.ClientSession()
    try:
        resp = await api_get(url, session=session, headers=headers, user=user)
        if not resp:
            return None, "request failed"
        if resp.status == 200:
            data = await resp.json()
            prices = [(p[0] / 1000, p[1]) for p in data.get("prices", [])]
            await db.set_coin_chart(coin, days, prices)
            return prices, None
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
    """Return global cryptocurrency market statistics."""
    cached = await db.get_global_data()
    if cached:
        return cached, None
    url = f"{config.COINGECKO_BASE_URL}/global"
    headers = config.COINGECKO_HEADERS
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
                await db.set_global_data(data)
                return data, None
            await asyncio.sleep(2**attempt)
        return None, f"HTTP {resp.status}"
    finally:
        if owns_session and session:
            await session.close()


async def find_coin(query: str) -> Optional[str]:
    """Look up a coin ID on CoinGecko given a query string."""
    url = f"{config.COINGECKO_BASE_URL}/search?query={quote(query, safe='')}"
    try:
        async with aiohttp.ClientSession() as session:
            resp = await api_get(url, session=session, headers=config.COINGECKO_HEADERS)
            if not resp or resp.status != 200:
                return None
            data = await resp.json()
            for item in data.get("coins", []):
                symbol = item.get("symbol")
                coin_id = item.get("id")
                name = item.get("name", "")
                if coin_id and coin_id.lower() == query.lower():
                    if symbol:
                        config.COIN_SYMBOLS[coin_id] = symbol.upper()
                        config.SYMBOL_TO_COIN[symbol.lower()] = coin_id
                    return coin_id
                if symbol and coin_id and symbol.lower() == query.lower():
                    config.COIN_SYMBOLS[coin_id] = symbol.upper()
                    config.SYMBOL_TO_COIN[symbol.lower()] = coin_id
                    return coin_id
                if coin_id and name.lower() == query.lower():
                    if symbol:
                        config.COIN_SYMBOLS[coin_id] = symbol.upper()
                        config.SYMBOL_TO_COIN[symbol.lower()] = coin_id
                    return coin_id
    except aiohttp.ClientError as exc:
        config.logger.warning("search failed: %s", exc)
    return None


async def resolve_coin(query: str, user: Optional[int] = None) -> Optional[str]:
    """Resolve a user provided coin name or symbol to a coin ID."""
    coin = normalize_coin(query)

    cached = await db.get_coin_data(coin)
    if cached and isinstance(cached.get("market_info"), dict):
        if cached["market_info"].get("current_price") is not None:
            return coin

    info = await db.get_coin_info(coin)
    if info:
        market = info.get("market_data", {})
        if (
            isinstance(market, dict)
            and market.get("current_price", {}).get("usd") is not None
        ):
            return coin

    info = await get_market_info(coin, user=user)
    if info and info.get("current_price") is not None:
        return coin

    for candidate in (coin, query):
        alt = await find_coin(candidate)
        if not alt:
            continue
        cached = await db.get_coin_data(alt)
        if cached and isinstance(cached.get("market_info"), dict):
            if cached["market_info"].get("current_price") is not None:
                return alt
        info = await db.get_coin_info(alt)
        if info:
            market = info.get("market_data", {})
            if (
                isinstance(market, dict)
                and market.get("current_price", {}).get("usd") is not None
            ):
                return alt
        info = await get_market_info(alt, user=user)
        if info and info.get("current_price") is not None:
            return alt

    return None


async def fetch_trending_coins() -> Optional[list[dict]]:
    """Return currently trending coins from CoinGecko."""
    cached = await db.get_trending_coins()
    if cached:
        for item in cached:
            coin_id = item.get("id")
            symbol = item.get("symbol")
            if coin_id and symbol:
                config.COIN_SYMBOLS[coin_id] = symbol.upper()
                config.SYMBOL_TO_COIN[symbol.lower()] = coin_id
        config.COINS = [c.get("id") for c in cached if c.get("id")]
        cached.sort(key=lambda x: (x["change_24h"] is None, -(x["change_24h"] or 0)))
        return cached

    url = f"{config.COINGECKO_BASE_URL}/search/trending"
    try:
        async with aiohttp.ClientSession() as session:
            resp = await api_get(url, session=session, headers=config.COINGECKO_HEADERS)
            if not resp or resp.status != 200:
                raise RuntimeError(getattr(resp, "status", "n/a"))
            data = await resp.json()
            infos: list[tuple[str, Optional[str], Optional[str]]] = []
            ids: list[str] = []
            for c in data.get("coins", [])[:10]:
                item = c.get("item", {})
                coin_id = item.get("id")
                symbol = item.get("symbol")
                name = item.get("name")
                if not coin_id:
                    continue
                ids.append(coin_id)
                infos.append((coin_id, symbol, name))
                if symbol:
                    config.COIN_SYMBOLS[coin_id] = symbol.upper()
                    config.SYMBOL_TO_COIN[symbol.lower()] = coin_id

            markets: dict[str, dict] = {}
            if ids:
                markets_url = (
                    f"{config.COINGECKO_BASE_URL}/coins/markets"
                    f"?vs_currency=usd&ids={','.join(ids)}&price_change_percentage=24h"
                )
                market_resp = await api_get(
                    markets_url, session=session, headers=config.COINGECKO_HEADERS
                )
                if market_resp and market_resp.status == 200:
                    data = await market_resp.json()
                    markets = {m.get("id"): m for m in data}

            trending: list[dict] = []
            for coin_id, symbol, name in infos:
                market = markets.get(coin_id, {})
                trending.append(
                    {
                        "id": coin_id,
                        "symbol": symbol,
                        "name": name,
                        "price": market.get("current_price"),
                        "change_24h": market.get("price_change_percentage_24h"),
                    }
                )

            if trending:
                trending.sort(
                    key=lambda x: (x["change_24h"] is None, -(x["change_24h"] or 0))
                )
                config.COINS = ids
                await db.set_trending_coins(trending)
                return trending
    except aiohttp.ClientError as exc:
        config.logger.error("error fetching trending coins: %s", exc)
    except RuntimeError as err:
        config.logger.warning("trending request failed: %s", err)
    cached = await db.get_trending_coins()
    if cached:
        for item in cached:
            coin_id = item.get("id")
            symbol = item.get("symbol")
            if coin_id and symbol:
                config.COIN_SYMBOLS[coin_id] = symbol.upper()
                config.SYMBOL_TO_COIN[symbol.lower()] = coin_id
        config.COINS = [c.get("id") for c in cached if c.get("id")]
        cached.sort(key=lambda x: (x["change_24h"] is None, -(x["change_24h"] or 0)))
        return cached
    return None


async def fetch_top_coins() -> None:
    """Update :data:`config.TOP_COINS` with high market cap coins."""
    url = (
        f"{config.COINGECKO_BASE_URL}/coins/markets"
        "?vs_currency=usd&order=market_cap_desc&per_page=50&page=1"
    )
    try:
        async with aiohttp.ClientSession() as session:
            resp = await api_get(url, session=session, headers=config.COINGECKO_HEADERS)
            if not resp or resp.status != 200:
                config.logger.warning(
                    "top coins request failed: %s", getattr(resp, "status", "n/a")
                )
                return
            data = await resp.json()
            coins: list[str] = []
            for item in data[:20]:
                coin_id = item.get("id")
                symbol = item.get("symbol")
                if coin_id:
                    coins.append(coin_id)
                    if symbol:
                        config.COIN_SYMBOLS[coin_id] = symbol.upper()
                        config.SYMBOL_TO_COIN[symbol.lower()] = coin_id
            config.TOP_COINS = coins
    except aiohttp.ClientError as exc:
        config.logger.error("error fetching top coins: %s", exc)


async def refresh_coin_data(
    coin: str, session: Optional[aiohttp.ClientSession] = None
) -> None:
    """Refresh cached price, market info and chart data for ``coin``."""
    owns_session = session is None
    if owns_session:
        session = aiohttp.ClientSession()
    try:
        price = await get_price(coin, session=session, user=None)
        market_info = await get_market_info(coin, session=session, user=None)
        info, _ = await get_coin_info(coin, session=session, user=None)
        chart, _ = await get_market_chart(coin, 7, session=session, user=None)
    finally:
        if owns_session and session:
            await session.close()
    await db.set_coin_data(
        coin,
        {
            "price": price,
            "market_info": market_info,
            "info": info,
            "chart_7d": chart,
        },
    )
