import os
import re

from dotenv import load_dotenv


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


load_dotenv()

BOT_NAME = "PricePulseWatcherBot"
DB_FILE = os.getenv("DB_PATH", "subs.db")
DEFAULT_THRESHOLD = float(os.getenv("DEFAULT_THRESHOLD", "0.1"))
DEFAULT_INTERVAL = parse_duration(os.getenv("DEFAULT_INTERVAL", "5m"))
PRICE_CHECK_INTERVAL = parse_duration(os.getenv("PRICE_CHECK_INTERVAL", "60"))
COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY")
COINGECKO_BASE_URL = os.getenv("COINGECKO_BASE_URL", "https://api.coingecko.com/api/v3")
COINGECKO_HEADERS = (
    {"x-cg-pro-api-key": COINGECKO_API_KEY} if COINGECKO_API_KEY else None
)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "bot.log")
