from apscheduler.schedulers.asyncio import AsyncIOScheduler
from telegram.ext import Application
from bot import db, price

scheduler = AsyncIOScheduler()


async def check_prices(app: Application) -> None:
    subs = await db.all_subscriptions()
    for chat_id, coin, target in subs:
        current = await price.get_price(coin)
        if current >= target:
            # TODO: refine message formatting
            await app.bot.send_message(chat_id, text=f"{coin} reached {current}")
            await db.remove_subscription(chat_id, coin)


def setup_scheduler(app: Application) -> None:
    scheduler.add_job(check_prices, "interval", minutes=1, args=(app,))
    scheduler.start()
