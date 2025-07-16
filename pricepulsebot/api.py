import asyncio
import time
from difflib import get_close_matches
from typing import Dict, Optional, Tuple
from urllib.parse import quote

import aiohttp
from aiolimiter import AsyncLimiter

from . import config

PRICE_CACHE: Dict[str, Tuple[float, float]] = {}
COINGECKO_LIMITER = AsyncLimiter(30, 60)
LAST_KNOWN_PRICE: Dict[str, float] = {}


def symbol_for(coin: str) -> str:
    return config.COIN_SYMBOLS.get(coin, coin.upper())


def normalize_coin(value: str) -> str:
    return config.SYMBOL_TO_COIN.get(value.lower(), value.lower())


def encoded(coin: str) -> str:
    return quote(coin, safe="-")


def suggest_coins(name: str, limit: int = 3) -> list[str]:
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
    return result


async def api_get(
    url: str,
    session: Optional[aiohttp.ClientSession] = None,
    headers: Optional[dict] = None,
    user: Optional[int] = None,
) -> Optional[aiohttp.ClientResponse]:
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


async def get_coin_info(
    coin: str,
    session: Optional[aiohttp.ClientSession] = None,
    *,
    user: Optional[int] = None,
) -> tuple[Optional[dict], Optional[str]]:
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
    url = (
        "https://api.binance.com/api/v3/klines"
        f"?symbol={symbol.upper()}&interval={interval}&limit={limit}"
    )
    owns_session = session is None
    if owns_session:
        session = aiohttp.ClientSession()
    try:
        resp = await api_get(
            url, session=session, headers=config.COINGECKO_HEADERS, user=user
        )
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
                return data, None
            await asyncio.sleep(2**attempt)
        return None, f"HTTP {resp.status}"
    finally:
        if owns_session and session:
            await session.close()


async def find_coin(query: str) -> Optional[str]:
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
                if symbol and coin_id and symbol.lower() == query.lower():
                    config.COIN_SYMBOLS[coin_id] = symbol.upper()
                    config.SYMBOL_TO_COIN[symbol.lower()] = coin_id
                    return coin_id
            for item in data.get("coins", []):
                symbol = item.get("symbol")
                coin_id = item.get("id")
                name = item.get("name", "")
                if coin_id and name.lower() == query.lower():
                    if symbol:
                        config.COIN_SYMBOLS[coin_id] = symbol.upper()
                        config.SYMBOL_TO_COIN[symbol.lower()] = coin_id
                    return coin_id
    except aiohttp.ClientError as exc:
        config.logger.warning("search failed: %s", exc)
    return None


async def resolve_coin(query: str, user: Optional[int] = None) -> Optional[str]:
    coin = normalize_coin(query)
    info = await get_market_info(coin, user=user)
    if info and info.get("current_price") is not None:
        return coin
    alt = await find_coin(query)
    if alt:
        info = await get_market_info(alt, user=user)
        if info and info.get("current_price") is not None:
            return alt
    return None


async def fetch_trending_coins() -> None:
    url = f"{config.COINGECKO_BASE_URL}/search/trending"
    try:
        async with aiohttp.ClientSession() as session:
            resp = await api_get(url, session=session, headers=config.COINGECKO_HEADERS)
            if not resp or resp.status != 200:
                config.logger.warning(
                    "trending request failed: %s", getattr(resp, "status", "n/a")
                )
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
                        config.COIN_SYMBOLS[coin_id] = symbol.upper()
                        config.SYMBOL_TO_COIN[symbol.lower()] = coin_id
            if coins:
                config.COINS = coins
    except aiohttp.ClientError as exc:
        config.logger.error("error fetching trending coins: %s", exc)


async def fetch_top_coins() -> None:
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
