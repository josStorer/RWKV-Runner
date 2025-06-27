import pathlib
from utils.log import quick_log

from fastapi import APIRouter, HTTPException, Request, Response, status as Status
from pydantic import BaseModel
from utils.rwkv import *
from utils.llama import *
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


@router.post(
    "/switch-model",
    tags=["Configs"],
    description="pass a .gguf file to use llama.cpp, otherwise use rwkv cli args and strategy",
)
def switch_model(body: SwitchModelBody, response: Response, request: Request):
    if global_var.get(global_var.Deploy_Mode) is True:
        raise HTTPException(Status.HTTP_403_FORBIDDEN)

    if global_var.get(global_var.Model_Status) is global_var.ModelStatus.Loading:
        response.status_code = Status.HTTP_304_NOT_MODIFIED
        return

    global_var.set(global_var.Model_Status, global_var.ModelStatus.Offline)
    global_var.set(global_var.Model, None)
    if not body.model.endswith(".gguf"):
        torch_gc()

    if body.model == "":
        return "success"

    devices = set(
        [
            x.strip().split(" ")[0].replace("cuda:0", "cuda")
            for x in body.strategy.split("->")
        ]
    )
    print(f"Strategy Devices: {devices}")
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
        if not body.model.endswith(".gguf"):
            global_var.set(
                global_var.Model,
                RWKV(
                    model=body.model, strategy=body.strategy, tokenizer=body.tokenizer
                ),
            )
        else:
            global_var.set(
                global_var.Model,
                Llama(model_path=body.model, strategy=body.strategy),
            )
    except Exception as e:
        print(e)
        import traceback

        print(traceback.format_exc())

        quick_log(request, body, f"Exception: {e}")
        global_var.set(global_var.Model_Status, global_var.ModelStatus.Offline)
        raise HTTPException(
            Status.HTTP_500_INTERNAL_SERVER_ERROR, f"failed to load: {e}"
        )

    if body.deploy:
        global_var.set(global_var.Deploy_Mode, True)

    saved_model_config = global_var.get(global_var.Model_Config)
    model = global_var.get(global_var.Model)
    if isinstance(model, AbstractRWKV):
        init_model_config = get_rwkv_config(model)
    else:
        init_model_config = get_llama_config(model)
    if saved_model_config is not None:
        merge_model(init_model_config, saved_model_config)
    global_var.set(global_var.Model_Config, init_model_config)
    global_var.set(global_var.Model_Status, global_var.ModelStatus.Working)

    return "success"


def merge_model(to_model: BaseModel, from_model: BaseModel):
    from_model_fields = [x for x in from_model.dict().keys()]
    to_model_fields = [x for x in to_model.dict().keys()]

    for field_name in from_model_fields:
        if field_name in to_model_fields:
            from_value = getattr(from_model, field_name)

            if from_value is not None:
                setattr(to_model, field_name, from_value)


@router.post("/update-config", tags=["Configs"])
def update_config(body: ModelConfigBody):
    """
    Will not update the model config immediately, but set it when completion called to avoid modifications during generation
    """

    model_config = global_var.get(global_var.Model_Config)
    if model_config is None:
        model_config = ModelConfigBody()
        global_var.set(global_var.Model_Config, model_config)
    merge_model(model_config, body)
    model = global_var.get(global_var.Model)
    if isinstance(model, AbstractRWKV):
        exception = load_rwkv_state(model, model_config.state, True)
    else:
        exception = None
    if exception is not None:
        raise exception
    print("Updated Model Config:", model_config)

    return "success"


@router.get("/status", tags=["Configs"])
def status():
    try:
        import GPUtil

        gpus = GPUtil.getGPUs()
    except:
        gpus = []
    if len(gpus) == 0:
        device_name = "CPU"
    else:
        device_name = gpus[0].name
    return {
        "status": global_var.get(global_var.Model_Status),
        "pid": os.getpid(),
        "device_name": device_name,
    }
