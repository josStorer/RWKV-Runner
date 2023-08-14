from enum import Enum, auto

Args = "args"
Model = "model"
Model_Status = "model_status"
Model_Config = "model_config"


class ModelStatus(Enum):
    Offline = 0
    Loading = 2
    Working = 3


def init():
    global GLOBALS
    GLOBALS = {}
    set(Model_Status, ModelStatus.Offline)


def set(key, value):
    GLOBALS[key] = value


def get(key):
    if key in GLOBALS:
        return GLOBALS[key]
    else:
        return None
