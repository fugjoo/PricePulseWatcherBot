"""Main entry point for starting the Telegram bot."""

import asyncio
import os
import signal

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from telegram import BotCommand
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    MessageHandler,
    filters,
)

from . import api, config, db, handlers


async def main() -> None:
    """Run the Telegram bot until the process receives a stop signal."""
    await db.init_db()
    await api.fetch_trending_coins()
    await api.fetch_top_coins()

    token = os.getenv("TELEGRAM_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_TOKEN not set")

    app = ApplicationBuilder().token(token).build()

    app.add_handler(CommandHandler("start", handlers.start))
    app.add_handler(CommandHandler("help", handlers.help_cmd))
    app.add_handler(CommandHandler("add", handlers.subscribe_cmd))
    app.add_handler(CommandHandler("remove", handlers.unsubscribe_cmd))
    app.add_handler(CommandHandler("clear", handlers.clear_cmd))
    app.add_handler(CommandHandler("list", handlers.list_cmd))
    app.add_handler(CommandHandler("info", handlers.info_cmd))
    app.add_handler(CommandHandler("chart", handlers.chart_cmd))
    app.add_handler(CommandHandler("charts", handlers.charts_cmd))
    app.add_handler(CommandHandler("news", handlers.news_cmd))
    app.add_handler(CommandHandler("trends", handlers.trends_cmd))
    app.add_handler(CommandHandler("top", handlers.top_cmd))
    app.add_handler(CommandHandler("global", handlers.global_cmd))
    app.add_handler(CommandHandler(["feargreed", "sentiment"], handlers.feargreed_cmd))
    app.add_handler(CommandHandler("status", handlers.status_cmd))
    app.add_handler(CommandHandler("valuearea", handlers.valuearea_cmd))
    app.add_handler(CommandHandler("settings", handlers.settings_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handlers.menu))
    app.add_handler(CallbackQueryHandler(handlers.button))

    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        handlers.check_prices,
        "interval",
        seconds=config.PRICE_CHECK_INTERVAL,
        args=(app,),
    )
    scheduler.add_job(
        handlers.refresh_cache,
        "interval",
        seconds=config.PRICE_CHECK_INTERVAL,
        args=(app,),
    )
    scheduler.add_job(api.fetch_trending_coins, "interval", minutes=10)
    scheduler.add_job(api.fetch_top_coins, "interval", minutes=10)
    scheduler.start()

    await app.initialize()
    await app.bot.set_my_commands(
        [BotCommand(name, desc) for name, desc in handlers.COMMANDS]
    )
    await app.start()
    await app.updater.start_polling()
    config.logger.info(f"{config.BOT_NAME} started")

    stop_event = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        asyncio.get_running_loop().add_signal_handler(sig, stop_event.set)

    await stop_event.wait()
    await app.updater.stop()
    await app.stop()
    await app.shutdown()
    scheduler.shutdown()
    config.logger.info(f"{config.BOT_NAME} stopped")
