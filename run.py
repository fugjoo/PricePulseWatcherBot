import asyncio
import os

from telegram.ext import ApplicationBuilder

from bot.config import load_env
from bot.handlers import register_handlers
from bot.scheduler import setup_scheduler
from bot.db import init_db


def main() -> None:
    """Start the Telegram bot."""

    load_env()
    token = os.getenv("BOT_TOKEN")
    if not token:
        raise RuntimeError("BOT_TOKEN not set")

    # initialize database
    asyncio.run(init_db())

    app = ApplicationBuilder().token(token).build()
    register_handlers(app)
    setup_scheduler(app)

    # create event loop for Application
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    app.run_polling()


if __name__ == "__main__":
    main()
