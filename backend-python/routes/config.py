import pathlib
import sys

from fastapi import APIRouter, HTTPException, Response, status
from pydantic import BaseModel
from langchain.llms import RWKV
from utils.rwkv import *
from utils.torch import *
import global_var

router = APIRouter()


class UpdateConfigBody(BaseModel):
    model: str = None
    strategy: str = None
    max_response_token: int = None
    temperature: float = None
    top_p: float = None
    presence_penalty: float = None
    count_penalty: float = None


@router.post("/update-config")
def update_config(body: UpdateConfigBody, response: Response):
    if (global_var.get(global_var.Model_Status) is global_var.ModelStatus.Loading):
        response.status_code = status.HTTP_304_NOT_MODIFIED
        return

    global_var.set(global_var.Model_Status, global_var.ModelStatus.Offline)
    global_var.set(global_var.Model, None)
    torch_gc()

    global_var.set(global_var.Model_Status, global_var.ModelStatus.Loading)
    try:
        global_var.set(global_var.Model, RWKV(
            model=sys.argv[2],
            strategy=sys.argv[1],
            tokens_path=f"{pathlib.Path(__file__).parent.parent.resolve()}/20B_tokenizer.json"
        ))
    except Exception:
        global_var.set(global_var.Model_Status, global_var.ModelStatus.Offline)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "failed to load")

    global_var.set(global_var.Model_Status, global_var.ModelStatus.Working)

    return "success"
