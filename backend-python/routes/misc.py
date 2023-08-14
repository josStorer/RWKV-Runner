from fastapi import APIRouter, HTTPException, status
from utils.rwkv import AbstractRWKV
import global_var

router = APIRouter()


@router.get("/dashboard/billing/credit_grants", tags=["MISC"])
def credit_grants():
    return {
        "object": "credit_summary",
        "total_granted": 10000,
        "total_used": 0,
        "total_available": 10000,
        "grants": {
            "object": "list",
            "data": [
                {
                    "object": "credit_grant",
                    "grant_amount": 10000,
                    "used_amount": 0,
                    "effective_at": 1672531200,
                    "expires_at": 33229440000,
                }
            ],
        },
    }


fake_models = [
    {
        "id": "gpt-3.5-turbo",
        "object": "model",
        "created": 1677610602,
        "owned_by": "openai",
        "permission": [
            {
                "id": "modelperm-zy5TOjnE2zVaicIcKO9bQDgX",
                "object": "model_permission",
                "created": 1690864883,
                "allow_create_engine": False,
                "allow_sampling": True,
                "allow_logprobs": True,
                "allow_search_indices": False,
                "allow_view": True,
                "allow_fine_tuning": False,
                "organization": "*",
                "group": None,
                "is_blocking": False,
            }
        ],
        "root": "gpt-3.5-turbo",
        "parent": None,
    },
    {
        "id": "text-davinci-003",
        "object": "model",
        "created": 1669599635,
        "owned_by": "openai-internal",
        "permission": [
            {
                "id": "modelperm-a6niqBmW2JaGmo0fDO7FEt1n",
                "object": "model_permission",
                "created": 1690930172,
                "allow_create_engine": False,
                "allow_sampling": True,
                "allow_logprobs": True,
                "allow_search_indices": False,
                "allow_view": True,
                "allow_fine_tuning": False,
                "organization": "*",
                "group": None,
                "is_blocking": False,
            }
        ],
        "root": "text-davinci-003",
        "parent": None,
    },
]


@router.get("/v1/models", tags=["MISC"])
@router.get("/models", tags=["MISC"])
def models():
    model: AbstractRWKV = global_var.get(global_var.Model)
    model_name = model.name if model else "rwkv"

    return {
        "object": "list",
        "data": [
            {
                "id": model_name,
                "object": "model",
                "owned_by": "rwkv",
                "root": model_name,
                "parent": None,
            },
            *fake_models,
        ],
    }


@router.get("/v1/models/{model_id}", tags=["MISC"])
@router.get("/models/{model_id}", tags=["MISC"])
def model(model_id: str):
    for fake_model in fake_models:
        if fake_model["id"] == model_id:
            return fake_model

    if "rwkv" in model_id.lower():
        model: AbstractRWKV = global_var.get(global_var.Model)
        model_name = model.name if model else "rwkv"
        return {
            "id": model_name,
            "object": "model",
            "owned_by": "rwkv",
            "root": model_name,
            "parent": None,
        }

    raise HTTPException(
        status.HTTP_404_NOT_FOUND,
        {
            "error": {
                "message": f"The model '{model_id}' does not exist",
                "type": "invalid_request_error",
                "param": "model",
                "code": "model_not_found",
            }
        },
    )
