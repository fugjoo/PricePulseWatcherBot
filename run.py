import asyncio
import os
from bot.config import load_env
from bot.handlers import register_handlers
from bot.scheduler import setup_scheduler
from bot.db import init_db
from telegram.ext import ApplicationBuilder


async def main() -> None:
    load_env()
    token = os.getenv("BOT_TOKEN")
    if not token:
        raise RuntimeError("BOT_TOKEN not set")
    await init_db()
    app = ApplicationBuilder().token(token).build()
    register_handlers(app)
    setup_scheduler(app)
    await app.run_polling()


if __name__ == "__main__":
    asyncio.run(main())
