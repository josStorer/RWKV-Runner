import pathlib
from utils.log import quick_log

from fastapi import APIRouter, HTTPException, Request, Response, status as Status
from pydantic import BaseModel
from utils.rwkv import *
from utils.torch import *
import global_var

router = APIRouter()


def get_tokens_path(model_path: str):
    model_path = model_path.lower()
    tokenizer_dir = f"{pathlib.Path(__file__).parent.parent.resolve()}/rwkv_pip/"

    default_tokens_path = tokenizer_dir + "20B_tokenizer.json"

    if "raven" in model_path:
        return default_tokens_path
    elif "world" in model_path:
        return "rwkv_vocab_v20230424"
    elif "midi" in model_path:
        return tokenizer_dir + "tokenizer-midi.json"
    else:
        return default_tokens_path


class SwitchModelBody(BaseModel):
    model: str
    strategy: str
    tokenizer: Union[str, None] = None
    customCuda: bool = False

    class Config:
        schema_extra = {
            "example": {
                "model": "models/RWKV-4-World-3B-v1-20230619-ctx4096.pth",
                "strategy": "cuda fp16",
                "tokenizer": None,
                "customCuda": False,
            }
        }


@router.post("/switch-model", tags=["Configs"])
def switch_model(body: SwitchModelBody, response: Response, request: Request):
    if global_var.get(global_var.Model_Status) is global_var.ModelStatus.Loading:
        response.status_code = Status.HTTP_304_NOT_MODIFIED
        return

    global_var.set(global_var.Model_Status, global_var.ModelStatus.Offline)
    global_var.set(global_var.Model, None)
    torch_gc()

    if body.model == "":
        return "success"

    if "->" in body.strategy:
        state_cache.disable_state_cache()
    else:
        try:
            state_cache.enable_state_cache()
        except HTTPException:
            pass

    os.environ["RWKV_CUDA_ON"] = "1" if body.customCuda else "0"

    global_var.set(global_var.Model_Status, global_var.ModelStatus.Loading)
    tokenizer = (
        get_tokens_path(body.model)
        if body.tokenizer is None or body.tokenizer == ""
        else body.tokenizer
    )
    try:
        global_var.set(
            global_var.Model,
            TextRWKV(
                model=body.model,
                strategy=body.strategy,
                tokens_path=tokenizer,
            )
            if "midi" not in body.model.lower()
            else MusicRWKV(
                model=body.model,
                strategy=body.strategy,
                tokens_path=tokenizer,
            ),
        )
    except Exception as e:
        print(e)
        quick_log(request, body, f"Exception: {e}")
        global_var.set(global_var.Model_Status, global_var.ModelStatus.Offline)
        raise HTTPException(
            Status.HTTP_500_INTERNAL_SERVER_ERROR, f"failed to load: {e}"
        )

    if global_var.get(global_var.Model_Config) is None:
        global_var.set(
            global_var.Model_Config, get_rwkv_config(global_var.get(global_var.Model))
        )
    global_var.set(global_var.Model_Status, global_var.ModelStatus.Working)

    return "success"


@router.post("/update-config", tags=["Configs"])
def update_config(body: ModelConfigBody):
    """
    Will not update the model config immediately, but set it when completion called to avoid modifications during generation
    """

    print(body)
    global_var.set(global_var.Model_Config, body)

    return "success"


@router.get("/status", tags=["Configs"])
def status():
    import GPUtil

    gpus = GPUtil.getGPUs()
    if len(gpus) == 0:
        device_name = "CPU"
    else:
        device_name = gpus[0].name
    return {
        "status": global_var.get(global_var.Model_Status),
        "pid": os.getpid(),
        "device_name": device_name,
    }
