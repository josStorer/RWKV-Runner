from typing import Any, Dict
from fastapi import APIRouter, HTTPException, Response, status
from pydantic import BaseModel
import gc
import copy

router = APIRouter()

trie = None
dtrie: Dict = {}


def init():
    global trie
    try:
        import cyac
        import mmap
        import os

        if os.path.exists("state_cache.trie"):
            with open("state_cache.trie", "r") as bf:
                buff_object = mmap.mmap(bf.fileno(), 0, access=mmap.ACCESS_READ)
            trie = cyac.Trie.from_buff(buff_object, copy=False)
        else:
            trie = cyac.Trie()
    except ModuleNotFoundError:
        print("cyac not found")


class AddStateBody(BaseModel):
    prompt: str
    tokens: list[str]
    state: Any
    logits: Any


@router.post("/add-state")
def add_state(body: AddStateBody):
    global trie, dtrie
    if trie is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "trie not loaded")

    id = trie.insert(body.prompt)
    dtrie[id] = {
        "tokens": copy.deepcopy(body.tokens),
        "state": copy.deepcopy(body.state),
        "logits": copy.deepcopy(body.logits),
    }

    return "success"


@router.post("/reset-state")
def reset_state():
    global trie
    if trie is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "trie not loaded")

    trie = cyac.Trie()
    gc.collect()

    return "success"


class LongestPrefixStateBody(BaseModel):
    prompt: str


@router.post("/longest-prefix-state")
def longest_prefix_state(body: LongestPrefixStateBody):
    global trie
    if trie is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "trie not loaded")

    id = -1
    for id, len in trie.prefix(body.prompt):
        pass
    if id != -1:
        v = dtrie[id]
        return {
            "prompt": trie[id],
            "tokens": v["tokens"],
            "state": v["state"],
            "logits": v["logits"],
        }
    else:
        return {"prompt": "", "tokens": [], "state": None, "logits": None}


@router.post("/save-state")
def save_state():
    global trie
    if trie is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "trie not loaded")

    trie.save("state_cache.trie")

    return "success"
