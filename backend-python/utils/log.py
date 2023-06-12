import json
import logging
from typing import Any
from fastapi import Request


logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s\n%(message)s")
fh = logging.handlers.RotatingFileHandler(
    "api.log", mode="a", maxBytes=3 * 1024 * 1024, backupCount=3
)
fh.setFormatter(formatter)
logger.addHandler(fh)


def quick_log(request: Request, body: Any, response: str):
    logger.info(
        f"Client: {request.client if request else ''}\nUrl: {request.url if request else ''}\n"
        + (
            f"Body: {json.dumps(body.__dict__, default=vars, ensure_ascii=False)}\n"
            if body
            else ""
        )
        + (f"Data:\n{response}\n" if response else "")
    )


async def log_middleware(request: Request):
    logger.info(
        f"Client: {request.client}\nUrl: {request.url}\nBody: {await request.body()}\n"
    )
