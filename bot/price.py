import asyncio
from typing import Any
import aiohttp


async def get_price(coin: str) -> float:
    url = (
        "https://api.coingecko.com/api/v3/simple/price?ids="
        f"{coin}&vs_currencies=usd"
    )
    delay = 1
    for _ in range(5):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        data: dict[str, Any] = await resp.json()
                        return float(data[coin]["usd"])
        except Exception:
            pass
        await asyncio.sleep(delay)
        delay *= 2
    raise RuntimeError("Failed to fetch price")
