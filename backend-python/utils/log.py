import json
import logging
from typing import Any
from fastapi import Request
from pydantic import BaseModel
from enum import Enum


logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s\n%(message)s")
fh = logging.handlers.RotatingFileHandler(
    "api.log", mode="a", maxBytes=3 * 1024 * 1024, backupCount=3, encoding="utf-8"
)
fh.setFormatter(formatter)
logger.addHandler(fh)


class ClsEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, BaseModel):
            return obj.dict()
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)


def quick_log(request: Request, body: Any, response: str):
    try:
        logger.info(
            f"Client: {request.client if request else ''}\nUrl: {request.url if request else ''}\n"
            + (
                f"Body: {json.dumps(body.__dict__, ensure_ascii=False, cls=ClsEncoder)}\n"
                if body
                else ""
            )
            + (f"Data:\n{response}\n" if response else "")
        )
    except Exception as e:
        logger.info(f"Error quick_log request:\n{e}")


async def log_middleware(request: Request):
    try:
        logger.info(
            f"Client: {request.client}\nUrl: {request.url}\nBody: {await request.body()}\n"
        )
    except Exception as e:
        logger.info(f"Error log_middleware request:\n{e}")
