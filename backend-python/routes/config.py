import pathlib
from utils.log import quick_log

from fastapi import APIRouter, HTTPException, Request, Response, status as Status
from pydantic import BaseModel
from utils.rwkv import *
from utils.torch import *
import global_var

router = APIRouter()


class SwitchModelBody(BaseModel):
    model: str
    strategy: str
    tokenizer: Union[str, None] = None
    customCuda: bool = False
    deploy: bool = Field(
        False,
        description="Deploy mode. If success, will disable /switch-model, /exit and other dangerous APIs (state cache APIs, part of midi APIs)",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "model": "models/RWKV-4-World-3B-v1-20230619-ctx4096.pth",
                "strategy": "cuda fp16",
                "tokenizer": "",
                "customCuda": False,
                "deploy": False,
            }
        }
    }


@router.post("/switch-model", tags=["Configs"])
def switch_model(body: SwitchModelBody, response: Response, request: Request):
    if global_var.get(global_var.Deploy_Mode) is True:
        raise HTTPException(Status.HTTP_403_FORBIDDEN)

    if global_var.get(global_var.Model_Status) is global_var.ModelStatus.Loading:
        response.status_code = Status.HTTP_304_NOT_MODIFIED
        return

    global_var.set(global_var.Model_Status, global_var.ModelStatus.Offline)
    global_var.set(global_var.Model, None)
    torch_gc()

    if body.model == "":
        return "success"

    STRATEGY_REGEX = r"^(?:(?:^|->) *(?:cuda(?::[\d]+)?|cpu|mps|dml) (?:fp(?:16|32)|bf16)(?:i8|i4|i3)?(?: \*[\d]+\+?)? *)+$"
    if not re.match(STRATEGY_REGEX, body.strategy):
        raise HTTPException(
            Status.HTTP_400_BAD_REQUEST,
            "Invalid strategy. Please read https://pypi.org/project/rwkv/",
        )
    devices = set(
        [
            x.strip().split(" ")[0].replace("cuda:0", "cuda")
            for x in body.strategy.split("->")
        ]
    )
    print(f"Devices: {devices}")
    # if len(devices) > 1:
    #     state_cache.disable_state_cache()
    # else:
    try:
        state_cache.enable_state_cache()
    except HTTPException:
        pass

    os.environ["RWKV_CUDA_ON"] = "1" if body.customCuda else "0"

    global_var.set(global_var.Model_Status, global_var.ModelStatus.Loading)
    try:
        global_var.set(
            global_var.Model,
            RWKV(model=body.model, strategy=body.strategy, tokenizer=body.tokenizer),
        )
    except Exception as e:
        print(e)
        quick_log(request, body, f"Exception: {e}")
        global_var.set(global_var.Model_Status, global_var.ModelStatus.Offline)
        raise HTTPException(
            Status.HTTP_500_INTERNAL_SERVER_ERROR, f"failed to load: {e}"
        )

    if body.deploy:
        global_var.set(global_var.Deploy_Mode, True)
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
