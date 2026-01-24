"""Internal module use at your own risk

This module provides a minimal interface for working with ggml tensors from llama-cpp-python
"""
import enum
import os
import pathlib
import ctypes

import llama_cpp._ctypes_extensions as ctypes_ext

from typing import (
    Callable,
    Union,
    NewType,
    Optional,
    TYPE_CHECKING,
)

libggml_base_path = pathlib.Path(os.path.abspath(os.path.dirname(__file__))) / "lib"
libggml = ctypes_ext.load_shared_library("ggml", libggml_base_path)

# // ====== ggml.h ======

# enum ggml_status {
#     GGML_STATUS_ALLOC_FAILED = -2,
#     GGML_STATUS_FAILED = -1,
#     GGML_STATUS_SUCCESS = 0,
#     GGML_STATUS_ABORTED = 1,
# };
class GGMLStatus(enum.IntEnum):
    GGML_STATUS_ALLOC_FAILED = -2
    GGML_STATUS_FAILED       = -1
    GGML_STATUS_SUCCESS      = 0
    GGML_STATUS_ABORTED      = 1


# // NOTE: always add types at the end of the enum to keep backward compatibility
# enum ggml_type {
#     GGML_TYPE_F32     = 0,
#     GGML_TYPE_F16     = 1,
#     GGML_TYPE_Q4_0    = 2,
#     GGML_TYPE_Q4_1    = 3,
#     // GGML_TYPE_Q4_2 = 4, support has been removed
#     // GGML_TYPE_Q4_3 = 5, support has been removed
#     GGML_TYPE_Q5_0    = 6,
#     GGML_TYPE_Q5_1    = 7,
#     GGML_TYPE_Q8_0    = 8,
#     GGML_TYPE_Q8_1    = 9,
#     GGML_TYPE_Q2_K    = 10,
#     GGML_TYPE_Q3_K    = 11,
#     GGML_TYPE_Q4_K    = 12,
#     GGML_TYPE_Q5_K    = 13,
#     GGML_TYPE_Q6_K    = 14,
#     GGML_TYPE_Q8_K    = 15,
#     GGML_TYPE_IQ2_XXS = 16,
#     GGML_TYPE_IQ2_XS  = 17,
#     GGML_TYPE_IQ3_XXS = 18,
#     GGML_TYPE_IQ1_S   = 19,
#     GGML_TYPE_IQ4_NL  = 20,
#     GGML_TYPE_IQ3_S   = 21,
#     GGML_TYPE_IQ2_S   = 22,
#     GGML_TYPE_IQ4_XS  = 23,
#     GGML_TYPE_I8      = 24,
#     GGML_TYPE_I16     = 25,
#     GGML_TYPE_I32     = 26,
#     GGML_TYPE_I64     = 27,
#     GGML_TYPE_F64     = 28,
#     GGML_TYPE_IQ1_M   = 29,
#     GGML_TYPE_BF16    = 30,
#     // GGML_TYPE_Q4_0_4_4 = 31, support has been removed from gguf files
#     // GGML_TYPE_Q4_0_4_8 = 32,
#     // GGML_TYPE_Q4_0_8_8 = 33,
#     GGML_TYPE_TQ1_0   = 34,
#     GGML_TYPE_TQ2_0   = 35,
#     // GGML_TYPE_IQ4_NL_4_4 = 36,
#     // GGML_TYPE_IQ4_NL_4_8 = 37,
#     // GGML_TYPE_IQ4_NL_8_8 = 38,
#     GGML_TYPE_MXFP4   = 39, // MXFP4 (1 block)
#     GGML_TYPE_COUNT   = 40,
# };
class GGMLType(enum.IntEnum):
    GGML_TYPE_F32  = 0
    GGML_TYPE_F16  = 1
    GGML_TYPE_Q4_0 = 2
    GGML_TYPE_Q4_1 = 3
    GGML_TYPE_Q5_0 = 6
    GGML_TYPE_Q5_1 = 7
    GGML_TYPE_Q8_0 = 8
    GGML_TYPE_Q8_1 = 9
    GGML_TYPE_Q2_K = 10
    GGML_TYPE_Q3_K = 11
    GGML_TYPE_Q4_K = 12
    GGML_TYPE_Q5_K = 13
    GGML_TYPE_Q6_K = 14
    GGML_TYPE_Q8_K = 15
    GGML_TYPE_IQ2_XXS = 16
    GGML_TYPE_IQ2_XS  = 17
    GGML_TYPE_IQ3_XXS = 18
    GGML_TYPE_IQ1_S  = 19
    GGML_TYPE_IQ4_NL = 20
    GGML_TYPE_IQ3_S  = 21
    GGML_TYPE_IQ2_S  = 22
    GGML_TYPE_IQ4_XS = 23
    GGML_TYPE_I8  = 24
    GGML_TYPE_I16 = 25
    GGML_TYPE_I32 = 26
    GGML_TYPE_I64 = 27
    GGML_TYPE_F64 = 28
    GGML_TYPE_IQ1_M = 29
    GGML_TYPE_BF16  = 30
    GGML_TYPE_TQ1_0 = 34
    GGML_TYPE_TQ2_0 = 35
    GGML_TYPE_MXFP4 = 39
    GGML_TYPE_COUNT = 40


# // precision
# enum ggml_prec {
#     GGML_PREC_DEFAULT =  0, // stored as ggml_tensor.op_params, 0 by default
#     GGML_PREC_F32     = 10,
# };
class GGMLPrec(enum.IntEnum):
    GGML_PREC_DEFAULT =  0
    GGML_PREC_F32     = 10


# // model file types
# enum ggml_ftype {
#     GGML_FTYPE_UNKNOWN        = -1,
#     GGML_FTYPE_ALL_F32        = 0,
#     GGML_FTYPE_MOSTLY_F16     = 1,  // except 1d tensors
#     GGML_FTYPE_MOSTLY_Q4_0    = 2,  // except 1d tensors
#     GGML_FTYPE_MOSTLY_Q4_1    = 3,  // except 1d tensors
#     GGML_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4, // tok_embeddings.weight and output.weight are F16
#     GGML_FTYPE_MOSTLY_Q8_0    = 7,  // except 1d tensors
#     GGML_FTYPE_MOSTLY_Q5_0    = 8,  // except 1d tensors
#     GGML_FTYPE_MOSTLY_Q5_1    = 9,  // except 1d tensors
#     GGML_FTYPE_MOSTLY_Q2_K    = 10, // except 1d tensors
#     GGML_FTYPE_MOSTLY_Q3_K    = 11, // except 1d tensors
#     GGML_FTYPE_MOSTLY_Q4_K    = 12, // except 1d tensors
#     GGML_FTYPE_MOSTLY_Q5_K    = 13, // except 1d tensors
#     GGML_FTYPE_MOSTLY_Q6_K    = 14, // except 1d tensors
#     GGML_FTYPE_MOSTLY_IQ2_XXS = 15, // except 1d tensors
#     GGML_FTYPE_MOSTLY_IQ2_XS  = 16, // except 1d tensors
#     GGML_FTYPE_MOSTLY_IQ3_XXS = 17, // except 1d tensors
#     GGML_FTYPE_MOSTLY_IQ1_S   = 18, // except 1d tensors
#     GGML_FTYPE_MOSTLY_IQ4_NL  = 19, // except 1d tensors
#     GGML_FTYPE_MOSTLY_IQ3_S   = 20, // except 1d tensors
#     GGML_FTYPE_MOSTLY_IQ2_S   = 21, // except 1d tensors
#     GGML_FTYPE_MOSTLY_IQ4_XS  = 22, // except 1d tensors
#     GGML_FTYPE_MOSTLY_IQ1_M   = 23, // except 1d tensors
#     GGML_FTYPE_MOSTLY_BF16    = 24, // except 1d tensors
#     GGML_FTYPE_MOSTLY_MXFP4   = 25, // except 1d tensors
# };
class GGMLFType(enum.IntEnum):
    GGML_FTYPE_UNKNOWN        = -1
    GGML_FTYPE_ALL_F32        = 0
    GGML_FTYPE_MOSTLY_F16     = 1
    GGML_FTYPE_MOSTLY_Q4_0    = 2
    GGML_FTYPE_MOSTLY_Q4_1    = 3
    GGML_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4
    GGML_FTYPE_MOSTLY_Q8_0    = 7
    GGML_FTYPE_MOSTLY_Q5_0    = 8
    GGML_FTYPE_MOSTLY_Q5_1    = 9
    GGML_FTYPE_MOSTLY_Q2_K    = 10
    GGML_FTYPE_MOSTLY_Q3_K    = 11
    GGML_FTYPE_MOSTLY_Q4_K    = 12
    GGML_FTYPE_MOSTLY_Q5_K    = 13
    GGML_FTYPE_MOSTLY_Q6_K    = 14
    GGML_FTYPE_MOSTLY_IQ2_XXS = 15
    GGML_FTYPE_MOSTLY_IQ2_XS  = 16
    GGML_FTYPE_MOSTLY_IQ3_XXS = 17
    GGML_FTYPE_MOSTLY_IQ1_S   = 18
    GGML_FTYPE_MOSTLY_IQ4_NL  = 19
    GGML_FTYPE_MOSTLY_IQ3_S   = 20
    GGML_FTYPE_MOSTLY_IQ2_S   = 21
    GGML_FTYPE_MOSTLY_IQ4_XS  = 22
    GGML_FTYPE_MOSTLY_IQ1_M   = 23
    GGML_FTYPE_MOSTLY_BF16    = 24
    GGML_FTYPE_MOSTLY_MXFP4   = 25


# enum ggml_object_type {
#     GGML_OBJECT_TYPE_TENSOR,
#     GGML_OBJECT_TYPE_GRAPH,
#     GGML_OBJECT_TYPE_WORK_BUFFER
# };
class GGMLObjectType(enum.IntEnum):
    GGML_OBJECT_TYPE_TENSOR      = 0
    GGML_OBJECT_TYPE_GRAPH       = 1
    GGML_OBJECT_TYPE_WORK_BUFFER = 2


# enum ggml_log_level {
#     GGML_LOG_LEVEL_NONE  = 0,
#     GGML_LOG_LEVEL_DEBUG = 1,
#     GGML_LOG_LEVEL_INFO  = 2,
#     GGML_LOG_LEVEL_WARN  = 3,
#     GGML_LOG_LEVEL_ERROR = 4,
#     GGML_LOG_LEVEL_CONT  = 5, // continue previous log
# };

class GGMLLogLevel(enum.IntEnum):
    GGML_LOG_LEVEL_NONE  = 0
    GGML_LOG_LEVEL_DEBUG = 1
    GGML_LOG_LEVEL_INFO  = 2
    GGML_LOG_LEVEL_WARN  = 3
    GGML_LOG_LEVEL_ERROR = 4
    GGML_LOG_LEVEL_CONT  = 5 # continue previous log


# // this tensor...
# enum ggml_tensor_flag {
#     GGML_TENSOR_FLAG_INPUT  =  1, // ...is an input for the GGML compute graph
#     GGML_TENSOR_FLAG_OUTPUT =  2, // ...is an output for the GGML compute graph
#     GGML_TENSOR_FLAG_PARAM  =  4, // ...contains trainable parameters
#     GGML_TENSOR_FLAG_LOSS   =  8, // ...defines loss for numerical optimization (multiple loss tensors add up)
# };
class GGMLTensorFlag(enum.IntEnum):
    GGML_TENSOR_FLAG_INPUT  = 1  # ...is an input for the GGML compute graph
    GGML_TENSOR_FLAG_OUTPUT = 2  # ...is an output for the GGML compute graph
    GGML_TENSOR_FLAG_PARAM  = 4  # ...contains trainable parameters
    GGML_TENSOR_FLAG_LOSS   = 8  # ...defines loss for numerical optimization (multiple loss tensors add up)


# enum ggml_tri_type {
#     GGML_TRI_TYPE_UPPER_DIAG = 0,
#     GGML_TRI_TYPE_UPPER      = 1,
#     GGML_TRI_TYPE_LOWER_DIAG = 2,
#     GGML_TRI_TYPE_LOWER      = 3
# };
class GGMLTriType(enum.IntEnum):
    GGML_TRI_TYPE_UPPER_DIAG = 0
    GGML_TRI_TYPE_UPPER      = 1
    GGML_TRI_TYPE_LOWER_DIAG = 2
    GGML_TRI_TYPE_LOWER      = 3


# struct ggml_init_params {
#     // memory pool
#     size_t mem_size;   // bytes
#     void * mem_buffer; // if NULL, memory will be allocated internally
#     bool   no_alloc;   // don't allocate memory for the tensor data
# };
class ggml_init_params(ctypes.Structure):
    _fields_ = [
        ('mem_size', ctypes.c_size_t),
        ('mem_buffer', ctypes.c_void_p),
        ('no_alloc', ctypes.c_bool),
    ]


# // Abort callback
# // If not NULL, called before ggml computation
# // If it returns true, the computation is aborted
# typedef bool (*ggml_abort_callback)(void * data);
ggml_abort_callback = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_void_p)


# // TODO these functions were sandwiched in the old optimization interface, is there a better place for them?
# typedef void (*ggml_log_callback)(enum ggml_log_level level, const char * text, void * user_data);
ggml_log_callback = ctypes.CFUNCTYPE(
    None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p
)


# // ====== ggml-opt.h ======

# // built-in loss types, i.e. the built-in quantities minimized by the optimizer
# // custom loss types can be defined via mean or sum which simply reduce the outputs for all datapoints to a single value
# enum ggml_opt_loss_type {
#     GGML_OPT_LOSS_TYPE_MEAN,
#     GGML_OPT_LOSS_TYPE_SUM,
#     GGML_OPT_LOSS_TYPE_CROSS_ENTROPY,
#     GGML_OPT_LOSS_TYPE_MEAN_SQUARED_ERROR,
# };
class GGMLOptLossType(enum.IntEnum):
    GGML_OPT_LOSS_TYPE_MEAN               = 0
    GGML_OPT_LOSS_TYPE_SUM                = 1
    GGML_OPT_LOSS_TYPE_CROSS_ENTROPY      = 2
    GGML_OPT_LOSS_TYPE_MEAN_SQUARED_ERROR = 3


# enum ggml_opt_build_type {
#     GGML_OPT_BUILD_TYPE_FORWARD = 10,
#     GGML_OPT_BUILD_TYPE_GRAD    = 20,
#     GGML_OPT_BUILD_TYPE_OPT     = 30,
# };
class GGMLOptBuildType(enum.IntEnum):
    GGML_OPT_BUILD_TYPE_FORWARD = 10
    GGML_OPT_BUILD_TYPE_GRAD    = 20
    GGML_OPT_BUILD_TYPE_OPT     = 30


# enum ggml_opt_optimizer_type {
#     GGML_OPT_OPTIMIZER_TYPE_ADAMW,
#     GGML_OPT_OPTIMIZER_TYPE_SGD,

#     GGML_OPT_OPTIMIZER_TYPE_COUNT
# };
class GGMLOptBuildType(enum.IntEnum):
    GGML_OPT_OPTIMIZER_TYPE_ADAMW = 0
    GGML_OPT_OPTIMIZER_TYPE_SGD   = 1
    GGML_OPT_OPTIMIZER_TYPE_COUNT = 2


# // parameters that control which optimizer is used and how said optimizer tries to find the minimal loss
# struct ggml_opt_optimizer_params {
#     struct {
#         float alpha; // learning rate
#         float beta1; // first AdamW momentum
#         float beta2; // second AdamW momentum
#         float eps;   // epsilon for numerical stability
#         float wd;    // weight decay - 0.0f to disable
#     } adamw;
#     struct {
#         float alpha; // learning rate
#         float wd;    // weight decay
#     } sgd;
# };
class ggml_opt_adamw_params(ctypes.Structure):
    _fields_ = [
        ('alpha', ctypes.c_float), # learning rate
        ('beta1', ctypes.c_float), # first AdamW momentum
        ('beta2', ctypes.c_float), # second AdamW momentum
        ('eps',   ctypes.c_float), # epsilon for numerical stability
        ('wd',    ctypes.c_float), # weight decay - 0.0f to disable
    ]

class ggml_opt_sgd_params(ctypes.Structure):
    _fields_ = [
        ('alpha', ctypes.c_float), # learning rate
        ('wd',    ctypes.c_float), # weight decay
    ]

class ggml_opt_optimizer_params(ctypes.Structure):
    _fields_ = [
        ('adamw', ggml_opt_adamw_params), # Nested AdamW parameters
        ('sgd', ggml_opt_sgd_params), # Nested SGD parameters
    ]


# // callback to calculate optimizer parameters prior to a backward pass
# // userdata can be used to pass arbitrary data
# typedef struct ggml_opt_optimizer_params (*ggml_opt_get_optimizer_params)(void * userdata);
ggml_opt_get_optimizer_params = ctypes.CFUNCTYPE(
    ctypes.POINTER(ggml_opt_optimizer_params), ctypes.c_void_p
)


# from ggml-backend.h
# // Evaluation callback for each node in the graph (set with ggml_backend_sched_set_eval_callback)
# // when ask == true, the scheduler wants to know if the user wants to observe this node
# // this allows the scheduler to batch nodes together in order to evaluate them in a single call
# //
# // when ask == false, the scheduler is passing the node tensor to the user for observation
# // if the user returns false, the scheduler will cancel the graph compute
# //
# typedef bool (*ggml_backend_sched_eval_callback)(struct ggml_tensor * t, bool ask, void * user_data);
ggml_backend_sched_eval_callback = ctypes.CFUNCTYPE(
    ctypes.c_bool, ctypes.c_void_p, ctypes.c_bool, ctypes.c_void_p
)
