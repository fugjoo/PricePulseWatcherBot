import asyncio
import os

from telegram.ext import ApplicationBuilder

from bot.config import load_env
from bot.handlers import register_handlers
from bot.scheduler import setup_scheduler
from bot.db import init_db


async def main() -> None:
    """Start the Telegram bot."""

    load_env()
    token = os.getenv("BOT_TOKEN")
    if not token:
        raise RuntimeError("BOT_TOKEN not set")

    await init_db()

    async with ApplicationBuilder().token(token).build() as app:
        register_handlers(app)
        setup_scheduler(app)

        await app.start()
        await app.updater.start_polling()
        await app.updater.idle()


if __name__ == "__main__":
    asyncio.run(main())
