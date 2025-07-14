import asyncio
import os

from dotenv import load_dotenv
import aiohttp
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
)


async def get_bitcoin_price() -> float:
    """Fetch the current Bitcoin price in USD."""
    url = (
        "https://api.coingecko.com/api/v3/simple/price"
        "?ids=bitcoin&vs_currencies=usd"
    )
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            data = await resp.json()
            return float(data["bitcoin"]["usd"])


async def send_price(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send the current Bitcoin price to the chat."""
    price = await get_bitcoin_price()
    await context.bot.send_message(
        chat_id=context.job.chat_id,
        text=f"Bitcoin price: ${price}",
    )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Start sending price updates every minute."""
    await update.message.reply_text(
        "Bitcoin price updates will be sent every minute."
    )

    # Remove any existing job for this chat
    if job := context.chat_data.get("price_job"):
        job.schedule_removal()

    job = context.job_queue.run_repeating(
        send_price,
        interval=60,
        first=0,
        chat_id=update.effective_chat.id,
    )
    context.chat_data["price_job"] = job


async def main() -> None:
    load_dotenv()
    token = os.getenv("BOT_TOKEN")
    if not token:
        raise RuntimeError("BOT_TOKEN not set")

    async with ApplicationBuilder().token(token).build() as app:
        app.add_handler(CommandHandler("start", start))
        await app.run_polling()


if __name__ == "__main__":
    asyncio.run(main())
