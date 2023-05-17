import pathlib

from fastapi import APIRouter, HTTPException, Response, status
from pydantic import BaseModel
from langchain.llms import RWKV
from utils.rwkv import *
from utils.torch import *
import global_var

router = APIRouter()


class SwitchModelBody(BaseModel):
    model: str
    strategy: str


@router.post("/switch-model")
def switch_model(body: SwitchModelBody, response: Response):
    if global_var.get(global_var.Model_Status) is global_var.ModelStatus.Loading:
        response.status_code = status.HTTP_304_NOT_MODIFIED
        return

    global_var.set(global_var.Model_Status, global_var.ModelStatus.Offline)
    global_var.set(global_var.Model, None)
    torch_gc()

    global_var.set(global_var.Model_Status, global_var.ModelStatus.Loading)
    try:
        global_var.set(
            global_var.Model,
            RWKV(
                model=body.model,
                strategy=body.strategy,
                tokens_path=f"{pathlib.Path(__file__).parent.parent.resolve()}/20B_tokenizer.json",
            ),
        )
    except Exception:
        global_var.set(global_var.Model_Status, global_var.ModelStatus.Offline)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "failed to load")

    if global_var.get(global_var.Model_Config) is None:
        global_var.set(
            global_var.Model_Config, get_rwkv_config(global_var.get(global_var.Model))
        )
    global_var.set(global_var.Model_Status, global_var.ModelStatus.Working)

    return "success"


@router.post("/update-config")
def update_config(body: ModelConfigBody):
    """
    Will not update the model config immediately, but set it when completion called to avoid modifications during generation
    """

    print(body)
    global_var.set(global_var.Model_Config, body)

    return "success"
