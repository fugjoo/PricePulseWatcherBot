import os
from dotenv import load_dotenv


def load_env(path: str | None = None) -> None:
    load_dotenv(dotenv_path=path)
    os.environ.setdefault("DB_PATH", "./crypto.db")
