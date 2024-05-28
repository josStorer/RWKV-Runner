import setuptools

if setuptools.__version__ >= "70.0.0":
    raise ImportError("setuptools>=70.0.0 is not supported")

import multipart
import fitz
import safetensors
import midi2audio
import mido
import lm_dataformat
import ftfy
import tqdm
import tiktoken

import torch
import rwkv
import langchain
import numpy
import tokenizers
import fastapi
import uvicorn
import sse_starlette
import pydantic
import psutil
