from typing import Any, Dict, List
from utils.log import quick_log
from fastapi import APIRouter, HTTPException, Request, Response, status
from pydantic import BaseModel
import gc
import copy
import torch

router = APIRouter()

trie = None
dtrie: Dict = {}


def init():
    global trie
    try:
        import cyac

        # import mmap
        # import os
        #
        # if os.path.exists("state_cache.trie"):
        #     with open("state_cache.trie", "r") as bf:
        #         buff_object = mmap.mmap(bf.fileno(), 0, access=mmap.ACCESS_READ)
        #     trie = cyac.Trie.from_buff(buff_object, copy=False)
        # else:
        trie = cyac.Trie()
    except ModuleNotFoundError:
        print("cyac not found")


class AddStateBody(BaseModel):
    prompt: str
    tokens: List[str]
    state: Any
    logits: Any


@router.post("/add-state")
def add_state(body: AddStateBody):
    global trie, dtrie
    if trie is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "trie not loaded")

    id = trie.insert(body.prompt)
    device = body.state[0].device
    dtrie[id] = {
        "tokens": copy.deepcopy(body.tokens),
        "state": [tensor.cpu() for tensor in body.state]
        if device != torch.device("cpu")
        else copy.deepcopy(body.state),
        "logits": copy.deepcopy(body.logits),
        "device": device,
    }

    return "success"


@router.post("/reset-state")
def reset_state():
    global trie, dtrie
    if trie is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "trie not loaded")

    trie = cyac.Trie()
    dtrie = {}
    gc.collect()

    return "success"


class LongestPrefixStateBody(BaseModel):
    prompt: str


@router.post("/longest-prefix-state")
def longest_prefix_state(body: LongestPrefixStateBody, request: Request):
    global trie
    if trie is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "trie not loaded")

    id = -1
    for id, len in trie.prefix(body.prompt):
        pass
    if id != -1:
        v = dtrie[id]
        device = v["device"]
        prompt = trie[id]
        quick_log(request, body, "Hit:\n" + prompt)
        return {
            "prompt": prompt,
            "tokens": v["tokens"],
            "state": [tensor.to(device) for tensor in v["state"]]
            if device != torch.device("cpu")
            else v["state"],
            "logits": v["logits"],
            "device": device,
        }
    else:
        return {
            "prompt": "",
            "tokens": [],
            "state": None,
            "logits": None,
            "device": None,
        }


@router.post("/save-state")
def save_state():
    global trie
    if trie is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "trie not loaded")

    # trie.save("state_cache.trie")

    return "not implemented"
