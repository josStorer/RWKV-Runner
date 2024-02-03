from enum import Enum, auto

Args = "args"
Model = "model"
Model_Status = "model_status"
Model_Config = "model_config"
Deploy_Mode = "deploy_mode"
Midi_Vocab_Config_Type = "midi_vocab_config_type"


class ModelStatus(Enum):
    Offline = 0
    Loading = 2
    Working = 3


class MidiVocabConfig(Enum):
    Default = auto()
    Piano = auto()


def init():
    global GLOBALS
    GLOBALS = {}
    set(Model_Status, ModelStatus.Offline)
    set(Deploy_Mode, False)
    set(Midi_Vocab_Config_Type, MidiVocabConfig.Default)


def set(key, value):
    GLOBALS[key] = value


def get(key):
    if key in GLOBALS:
        return GLOBALS[key]
    else:
        return None
