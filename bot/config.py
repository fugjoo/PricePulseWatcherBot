import os
from dotenv import load_dotenv
from typing import Optional


def load_env(path: Optional[str] = None) -> None:
    load_dotenv(dotenv_path=path)
    os.environ.setdefault("DB_PATH", "./crypto.db")
