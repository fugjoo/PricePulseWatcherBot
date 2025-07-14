from .commands import start, subscribe, unsubscribe, list_subs
from telegram.ext import Application


def register_handlers(app: Application) -> None:
    app.add_handler(start)
    app.add_handler(subscribe)
    app.add_handler(unsubscribe)
    app.add_handler(list_subs)
