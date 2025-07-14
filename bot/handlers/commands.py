from telegram import Update
from telegram.ext import CommandHandler, ContextTypes
from bot import db


async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Welcome to Crypto Price Alert Bot!")


async def subscribe_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if len(context.args) != 2:
        await update.message.reply_text("Usage: /subscribe <coin> <price>")
        return
    coin, price = context.args[0], float(context.args[1])
    await db.add_subscription(update.effective_chat.id, coin, price)
    await update.message.reply_text(f"Subscribed to {coin} at {price}")


async def unsubscribe_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Usage: /unsubscribe <coin>")
        return
    coin = context.args[0]
    await db.remove_subscription(update.effective_chat.id, coin)
    await update.message.reply_text(f"Unsubscribed from {coin}")


async def list_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    subs = await db.list_subscriptions(update.effective_chat.id)
    if not subs:
        await update.message.reply_text("No subscriptions")
        return
    msg = "\n".join(f"{c} at {p}" for c, p in subs)
    await update.message.reply_text(msg)


start = CommandHandler("start", start_cmd)
subscribe = CommandHandler("subscribe", subscribe_cmd)
unsubscribe = CommandHandler("unsubscribe", unsubscribe_cmd)
list_subs = CommandHandler("list", list_cmd)
