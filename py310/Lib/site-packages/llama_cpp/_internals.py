from __future__ import annotations

import os
import ctypes

from typing import (
    Dict,
    List,
    Tuple,
    Optional,
    Sequence,
)
from dataclasses import dataclass, field
from contextlib import ExitStack

import numpy as np
import numpy.typing as npt

from .llama_types import *
from .llama_grammar import LlamaGrammar
from ._utils import suppress_stdout_stderr

import llama_cpp.llama_cpp as llama_cpp


# Python wrappers over llama.h structs


class LlamaModel:
    """Intermediate Python wrapper for a llama.cpp llama_model.
    NOTE: For stability it's recommended you use the Llama class instead."""

    def __init__(
        self,
        *,
        path_model: str,
        params: llama_cpp.llama_model_params,
        verbose: bool = True,
    ):
        self.path_model = path_model
        self.params = params
        self.verbose = verbose
        self._exit_stack = ExitStack()

        model = None

        if not os.path.exists(path_model):
            raise ValueError(f"Model path does not exist: {path_model}")

        with suppress_stdout_stderr(disable=verbose):
            model = llama_cpp.llama_model_load_from_file(
                self.path_model.encode("utf-8"), self.params
            )

        if model is None:
            raise ValueError(f"Failed to load model from file: {path_model}")

        vocab = llama_cpp.llama_model_get_vocab(model)

        if vocab is None:
            raise ValueError(f"Failed to get vocab from model: {path_model}")

        self.model = model
        self.vocab = vocab

        def free_model():
            if self.model is None:
                return
            llama_cpp.llama_model_free(self.model)
            self.model = None

        self._exit_stack.callback(free_model)

    def close(self):
        self._exit_stack.close()

    def __del__(self):
        self.close()

    def vocab_type(self) -> int:
        return llama_cpp.llama_vocab_type(self.model)

    def n_vocab(self) -> int:
        return llama_cpp.llama_n_vocab(self.vocab)

    def n_ctx_train(self) -> int:
        return llama_cpp.llama_model_n_ctx_train(self.model)

    def n_cls_out(self) -> int:
        return llama_cpp.llama_model_n_cls_out(self.model)

    def n_embd(self) -> int:
        return llama_cpp.llama_model_n_embd(self.model)

    def n_embd_inp(self) -> int:
        return llama_cpp.llama_model_n_embd_inp(self.model)

    def n_embd_out(self) -> int:
        return llama_cpp.llama_model_n_embd_out(self.model)

    def n_layer(self) -> int:
        return llama_cpp.llama_model_n_layer(self.model)

    def n_head(self) -> int:
        return llama_cpp.llama_model_n_head(self.model)

    def n_head_kv(self) -> int:
        return llama_cpp.llama_model_n_head_kv(self.model)

    def n_swa(self) -> int:
        return llama_cpp.llama_model_n_swa(self.model)

    def n_params(self) -> int:
        return llama_cpp.llama_model_n_params(self.model)

    def has_encoder(self) -> bool:
        return llama_cpp.llama_model_has_encoder(self.model)

    def has_decoder(self) -> bool:
        return llama_cpp.llama_model_has_decoder(self.model)

    def decoder_start_token(self) -> int:
        return llama_cpp.llama_model_decoder_start_token(self.model)

    def is_recurrent(self) -> bool:
        return llama_cpp.llama_model_is_recurrent(self.model)

    def is_hybrid(self) -> bool:
        return llama_cpp.llama_model_is_hybrid(self.model)

    def is_diffusion(self) -> bool:
        return llama_cpp.llama_model_is_diffusion(self.model)

    def rope_freq_scale_train(self) -> float:
        return llama_cpp.llama_model_rope_freq_scale_train(self.model)

    def desc(self) -> str:
        buf = ctypes.create_string_buffer(1024)
        llama_cpp.llama_model_desc(self.model, buf, 1024)
        return buf.value.decode("utf-8")

    def size(self) -> int:
        return llama_cpp.llama_model_size(self.model)

    def get_tensor(self, name: str) -> ctypes.c_void_p:
        raise NotImplementedError("get_tensor is not implemented in llama.cpp")

    # Vocab

    def token_get_text(self, token: int) -> str:
        return llama_cpp.llama_vocab_get_text(self.vocab, token).decode("utf-8")

    def token_get_score(self, token: int) -> float:
        return llama_cpp.llama_vocab_get_score(self.vocab, token)

    def token_get_attr(self, token: int) -> int:
        return llama_cpp.llama_vocab_get_attr(self.vocab, token)

    def token_is_eog(self, token: int) -> bool:
        return llama_cpp.llama_vocab_is_eog(self.vocab, token)

    def token_is_control(self, token: int) -> bool:
        return llama_cpp.llama_vocab_is_control(self.vocab, token)

    # Special tokens

    def token_bos(self) -> int:
        return llama_cpp.llama_vocab_bos(self.vocab)

    def token_eos(self) -> int:
        return llama_cpp.llama_vocab_eos(self.vocab)

    def token_eot(self) -> int:
        return llama_cpp.llama_vocab_eot(self.vocab)

    def token_sep(self) -> int:
        return llama_cpp.llama_vocab_sep(self.vocab)

    def token_nl(self) -> int:
        return llama_cpp.llama_vocab_nl(self.vocab)

    def token_pad(self) -> int:
        return llama_cpp.llama_vocab_pad(self.vocab)

    def token_mask(self) -> int:
        return llama_cpp.llama_vocab_mask(self.vocab)

    def token_cls(self) -> int:
        return llama_cpp.llama_vocab_cls(self.vocab)

    def token_fim_pre(self) -> int:
        return llama_cpp.llama_vocab_fim_pre(self.vocab)

    def token_fim_suf(self) -> int:
        return llama_cpp.llama_vocab_fim_suf(self.vocab)

    def token_fim_mid(self) -> int:
        return llama_cpp.llama_vocab_fim_mid(self.vocab)

    def token_fim_pad(self) -> int:
        return llama_cpp.llama_vocab_fim_pad(self.vocab)

    def token_fim_rep(self) -> int:
        return llama_cpp.llama_vocab_fim_rep(self.vocab)

    def token_fim_sep(self) -> int:
        return llama_cpp.llama_vocab_fim_sep(self.vocab)

    def get_add_bos(self) -> bool:
        return llama_cpp.llama_vocab_get_add_bos(self.vocab)

    def get_add_eos(self) -> bool:
        return llama_cpp.llama_vocab_get_add_eos(self.vocab)

    def get_add_sep(self) -> bool:
        return llama_cpp.llama_vocab_get_add_sep(self.vocab)

    # Tokenization

    def tokenize(self, text: bytes, add_bos: bool, special: bool):
        """
        Tokenize a string.
        Optimized to use dynamic buffer allocation.
        """
        n_tokens_alloc = len(text) + 2
        tokens = (llama_cpp.llama_token * n_tokens_alloc)()

        n_tokens = llama_cpp.llama_tokenize(
            self.vocab, text, len(text), tokens, n_tokens_alloc, add_bos, special
        )

        # If the buffer is insufficient (returns a negative number), reallocate the buffer.
        if n_tokens < 0:
            n_tokens_alloc = -n_tokens
            tokens = (llama_cpp.llama_token * n_tokens_alloc)()
            n_tokens = llama_cpp.llama_tokenize(
                self.vocab, text, len(text), tokens, n_tokens_alloc, add_bos, special
            )
            if n_tokens < 0:
                raise RuntimeError(
                    f'Failed to tokenize: text="{text}" n_tokens={n_tokens}'
                )

        # return a buffer of n_tokens size.
        return list(tokens[:n_tokens])

    def token_to_piece(self, token: int, special: bool = False) -> bytes:
        """
        Convert a single token to bytes.
        Optimized to handle dynamic resizing for ultra-long tokens.
        """
        size = 32
        buf = (ctypes.c_char * size)()
        n = llama_cpp.llama_token_to_piece(self.vocab, token, buf, size, 0, special)

        # If the token is very long (returns a negative number), redistribute it according to the returned size.
        if n < 0:
            size = -n
            buf = (ctypes.c_char * size)()
            n = llama_cpp.llama_token_to_piece(self.vocab, token, buf, size, 0, special)
            if n < 0:
                raise RuntimeError(f"Failed to get piece for token {token}")

        # return a buffer of n size.
        return bytes(buf[:n])

    def detokenize(self, tokens: List[int], special: bool = False) -> bytes:
        """
        Convert a list of tokens to bytes.
        Optimized to handle dynamic resizing for ultra-long tokens.
        """
        if not tokens:
            return b""

        n_tokens = len(tokens)
        # Convert a Python list to a C int array
        tokens_array = (llama_cpp.llama_token * n_tokens)(*tokens)

        # Initial buffer size estimation
        buffer_size = max(n_tokens, 64)
        buffer = (ctypes.c_char * buffer_size)()

        n_chars = llama_cpp.llama_detokenize(
            self.vocab, tokens_array, n_tokens, buffer, buffer_size, False, special
        )

        # If the buffer is insufficient, expand it and retry.
        if n_chars < 0:
            buffer_size = -n_chars
            buffer = (ctypes.c_char * buffer_size)()
            n_chars = llama_cpp.llama_detokenize(
                self.vocab, tokens_array, n_tokens, buffer, buffer_size, False, special
            )
            if n_chars < 0:
                raise RuntimeError("Failed to detokenize")

        return bytes(buffer[:n_chars])


    # Extra
    def metadata(self) -> Dict[str, str]:
        metadata: Dict[str, str] = {}
        # Pre-allocate a 16KB buffer. This is large enough to handle almost all
        # metadata values (including gpt-oss large chat templates ~15KB) in a single pass,
        # eliminating the need for resize-and-retry in most cases.
        buffer_size = 16384
        buffer = ctypes.create_string_buffer(buffer_size)

        # Caching function references reduces the overhead of property lookups within loops.
        get_key_by_index = llama_cpp.llama_model_meta_key_by_index
        get_val_by_index = llama_cpp.llama_model_meta_val_str_by_index
        metadata_count = llama_cpp.llama_model_meta_count(self.model)
        # iterate over model keys
        for i in range(metadata_count):
            # 1. Get Key
            nbytes = get_key_by_index(self.model, i, buffer, buffer_size)
            # Handle buffer resize if the key exceeds current size
            if nbytes > buffer_size:
                buffer_size = nbytes + 1024
                buffer = ctypes.create_string_buffer(buffer_size)
                # Retry with the larger buffer
                nbytes = get_key_by_index(self.model, i, buffer, buffer_size)
            key = buffer.value.decode("utf-8")

            # 2. Get Value
            nbytes = get_val_by_index(self.model, i, buffer, buffer_size)
            # Handle buffer resize if the value exceeds current size
            if nbytes > buffer_size:
                buffer_size = nbytes + 1024
                buffer = ctypes.create_string_buffer(buffer_size)
                # Retry with the larger buffer
                nbytes = get_val_by_index(self.model, i, buffer, buffer_size)
            value = buffer.value.decode("utf-8")

            metadata[key] = value
        return metadata

    @staticmethod
    def default_params():
        """Get the default llama_model_params."""
        return llama_cpp.llama_model_default_params()


class LlamaContext:
    """Intermediate Python wrapper for a llama.cpp llama_context.
    NOTE: For stability it's recommended you use the Llama class instead."""

    def __init__(
        self,
        *,
        model: LlamaModel,
        params: llama_cpp.llama_context_params,
        verbose: bool = True,
    ):
        self.model = model
        self.params = params
        self.verbose = verbose
        self._exit_stack = ExitStack()

        ctx = llama_cpp.llama_init_from_model(self.model.model, self.params)

        if ctx is None:
            llama_cpp.llama_model_free(self.model.model)
            raise ValueError("Failed to create context with model")

        self.ctx = ctx

        def free_ctx():
            if self.ctx is None:
                return
            llama_cpp.llama_free(self.ctx)
            self.ctx = None

        self._exit_stack.callback(free_ctx)

    def close(self):
        self._exit_stack.close()

    def __del__(self):
        self.close()

    def n_ctx(self) -> int:
        return llama_cpp.llama_n_ctx(self.ctx)

    def n_ctx_seq(self) -> int:
        return llama_cpp.llama_n_ctx_seq(self.ctx)

    def n_batch(self) -> int:
        return llama_cpp.llama_n_batch(self.ctx)

    def n_ubatch(self) -> int:
        return llama_cpp.llama_n_ubatch(self.ctx)

    def n_seq_max(self) -> int:
        return llama_cpp.llama_n_seq_max(self.ctx)

    def pooling_type(self) -> int:
        return llama_cpp.llama_pooling_type(self.ctx)

    # // Memory API

    def get_memory(self):
        return llama_cpp.llama_get_memory(self.ctx)

    def memory_clear(self, data: bool):
        llama_cpp.llama_memory_clear(self.get_memory(), data)

    def memory_seq_rm(self, seq_id: int, p0: int, p1: int) -> bool:
        if self.ctx is not None:
            return llama_cpp.llama_memory_seq_rm(self.get_memory(), seq_id, p0, p1)
        else:
            return False

    def memory_seq_cp(self, seq_id_src: int, seq_id_dst: int, p0: int, p1: int):
        llama_cpp.llama_memory_seq_cp(self.get_memory(), seq_id_src, seq_id_dst, p0, p1)

    def memory_seq_keep(self, seq_id: int):
        llama_cpp.llama_memory_seq_keep(self.get_memory(), seq_id)

    def memory_seq_add(self, seq_id: int, p0: int, p1: int, delta: int):
        llama_cpp.llama_memory_seq_add(self.get_memory(), seq_id, p0, p1, delta)

    def memory_seq_div(self, seq_id: int, p0: int, p1: int, d: int):
        llama_cpp.llama_memory_seq_div(self.get_memory(), seq_id, p0, p1, d)

    def memory_seq_pos_max(self, seq_id: int) -> int:
        return llama_cpp.llama_memory_seq_pos_max(self.get_memory(), seq_id)

    def memory_seq_pos_min(self, seq_id: int) -> int:
        return llama_cpp.llama_memory_seq_pos_min(self.get_memory(), seq_id)

    # // State / sessions API

    def get_state_size(self) -> int:
        return llama_cpp.llama_state_get_size(self.ctx)

    def get_state_data(self, dst:ctypes.Array[ctypes.c_uint8], size: int) -> int:
        return llama_cpp.llama_state_get_data(self.ctx, dst, size)

    def set_state_data(self, src:ctypes.Array[ctypes.c_uint8], size: int) -> int:
        return llama_cpp.llama_state_set_data(self.ctx, src, size)

    def load_state_file(
        self,
        path_session: bytes,
        tokens_out: ctypes.Array[llama_cpp.llama_token],
        n_token_capacity: ctypes.c_size_t,
        n_token_count_out: ctypes.pointer(ctypes.c_size_t)
    ) -> bool:
        return llama_cpp.llama_state_load_file(self.ctx, path_session, tokens_out, n_token_capacity, n_token_count_out)

    def save_state_file(
        self,
        path_session: bytes,
        tokens: ctypes.Array[llama_cpp.llama_token],
        n_token_count: ctypes.c_size_t
    ) -> bool:
        return llama_cpp.llama_state_save_file(self.ctx, path_session, tokens, n_token_count)

    def get_state_seq_size(self, seq_id: int) -> int:
        return llama_cpp.llama_state_seq_get_size(self.ctx, seq_id)

    def get_state_seq_data(self, dst: ctypes.Array[ctypes.c_uint8], size: int, seq_id: int) -> int:
        return llama_cpp.llama_state_seq_get_data(self.ctx, dst, size, seq_id)

    def set_state_seq_data(self, src: ctypes.Array[ctypes.c_uint8], size: int, dest_seq_id: int) -> int:
        return llama_cpp.llama_state_seq_set_data(self.ctx, src, size, dest_seq_id)

    def load_state_seq_file(
        self,
        filepath: bytes,
        dest_seq_id: int,
        tokens_out: ctypes.Array[llama_cpp.llama_token],
        n_token_capacity: ctypes.c_size_t,
        n_token_count_out: ctypes.pointer(ctypes.c_size_t)
    ) -> int:
        return llama_cpp.llama_state_seq_load_file(self.ctx, filepath, dest_seq_id, tokens_out, n_token_capacity, n_token_count_out)

    def save_state_seq_file(
        self,
        filepath: bytes,
        seq_id: int,
        tokens: ctypes.Array[llama_cpp.llama_token],
        n_token_count: ctypes.c_size_t
    ) -> int:
        return llama_cpp.llama_state_seq_save_file(self.ctx, filepath, seq_id, tokens, n_token_count)

    def get_state_seq_size_ext(self, seq_id: int, flags: llama_cpp.llama_state_seq_flags) -> int:
        return llama_cpp.llama_state_seq_get_size_ext(self.ctx, seq_id, flags)

    def get_state_seq_data_ext(
        self,
        dst:ctypes.Array[ctypes.c_uint8],
        size: int,
        seq_id: int,
        flags: llama_cpp.llama_state_seq_flags
    ) -> int:
        return llama_cpp.llama_state_seq_get_data_ext(self.ctx, dst, size, seq_id, flags)

    def set_state_seq_data_ext(
        self,
        src:ctypes.Array[ctypes.c_uint8],
        size: int,
        dest_seq_id: int,
        flags: llama_cpp.llama_state_seq_flags
    ) -> int:
        return llama_cpp.llama_state_seq_set_data_ext(self.ctx, src, size, dest_seq_id, flags)

    # // Decoding API

    def encode(self, batch: LlamaBatch):
        return_code = llama_cpp.llama_encode(
            self.ctx,
            batch.batch,
        )
        if return_code != 0:
            raise RuntimeError(f"llama_encode returned {return_code}")

    def decode(self, batch: LlamaBatch):
        return_code = llama_cpp.llama_decode(self.ctx, batch.batch)

        if return_code == 0:
            return

        error_map = {
             1: "No KV slot available: try reducing batch size or increasing context window",
             2: "Decoding aborted",
            -1: "Invalid input batch",
        }

        msg = error_map.get(return_code, "Fatal internal error")
        raise RuntimeError(f"llama_decode failed (code {return_code}): {msg}")

    def set_n_threads(self, n_threads: int, n_threads_batch: int):
        llama_cpp.llama_set_n_threads(self.ctx, n_threads, n_threads_batch)

    def n_threads(self) -> int:
        return llama_cpp.llama_n_threads(self.ctx)

    def n_threads_batch(self) -> int:
        return llama_cpp.llama_n_threads_batch(self.ctx)

    def set_causal_attn(self, causal_attn: bool):
        llama_cpp.llama_set_causal_attn(self.ctx, causal_attn)

    def set_warmup(self, warmup: bool):
        llama_cpp.llama_set_warmup(self.ctx, warmup)

    def synchronize(self):
        llama_cpp.llama_synchronize(self.ctx)

    def get_logits(self):
        return llama_cpp.llama_get_logits(self.ctx)

    def get_logits_ith(self, i: int):
        return llama_cpp.llama_get_logits_ith(self.ctx, i)

    def set_embeddings(self, embeddings: bool):
        llama_cpp.llama_set_embeddings(self.ctx, embeddings)

    def get_embeddings(self):
        return llama_cpp.llama_get_embeddings(self.ctx)

    def get_embeddings_ith(self, i: int):
        return llama_cpp.llama_get_embeddings_ith(self.ctx, i)

    def get_embeddings_seq(self, seq_id: int):
        return llama_cpp.llama_get_embeddings_seq(self.ctx, seq_id)

    # Sampling functions

    def set_rng_seed(self, seed: int):
        # TODO: Fix
        # llama_cpp.llama_set_rng_seed(self.ctx, seed)
        raise NotImplementedError("set_rng_seed is not implemented in llama.cpp")

    def sample_repetition_penalties(
        self,
        candidates: "_LlamaTokenDataArray",
        last_tokens_data: "llama_cpp.Array[llama_cpp.llama_token]",
        penalty_last_n: int,
        penalty_repeat: float,
        penalty_freq: float,
        penalty_present: float,
    ):
        # llama_cpp.llama_sample_repetition_penalties(
        #     self.ctx,
        #     llama_cpp.byref(candidates.candidates),
        #     last_tokens_data,
        #     penalty_last_n,
        #     penalty_repeat,
        #     penalty_freq,
        #     penalty_present,
        # )
        raise NotImplementedError("sample_repetition_penalties is not implemented in llama.cpp")

    def sample_top_k(self, candidates: "_LlamaTokenDataArray", k: int, min_keep: int):
        # llama_cpp.llama_sample_top_k(
        #     self.ctx, llama_cpp.byref(candidates.candidates), k, min_keep
        # )
        raise NotImplementedError("sample_top_k is not implemented in llama.cpp")

    def sample_top_p(self, candidates: "_LlamaTokenDataArray", p: float, min_keep: int):
        # llama_cpp.llama_sample_top_p(
        #     self.ctx, llama_cpp.byref(candidates.candidates), p, min_keep
        # )
        raise NotImplementedError("sample_top_p is not implemented in llama.cpp")

    def sample_min_p(self, candidates: "_LlamaTokenDataArray", p: float, min_keep: int):
        # llama_cpp.llama_sample_min_p(
        #     self.ctx, llama_cpp.byref(candidates.candidates), p, min_keep
        # )
        raise NotImplementedError("sample_min_p is not implemented in llama.cpp")

    def sample_typical(
        self, candidates: "_LlamaTokenDataArray", p: float, min_keep: int
    ):
        # llama_cpp.llama_sample_typical(
        #     self.ctx, llama_cpp.byref(candidates.candidates), p, min_keep
        # )
        raise NotImplementedError("sample_typical is not implemented in llama.cpp")

    def sample_temp(self, candidates: "_LlamaTokenDataArray", temp: float):
        # llama_cpp.llama_sample_temp(
        #     self.ctx, llama_cpp.byref(candidates.candidates), temp
        # )
        raise NotImplementedError("sample_temp is not implemented in llama.cpp")

    def sample_grammar(self, candidates: "_LlamaTokenDataArray", grammar: LlamaGrammar):
        # llama_cpp.llama_sample_grammar(
        #     self.ctx,
        #     llama_cpp.byref(candidates.candidates),
        #     grammar.grammar,
        # )
        raise NotImplementedError("sample_grammar is not implemented in llama.cpp")

    def sample_token_mirostat(
        self,
        candidates: "_LlamaTokenDataArray",
        tau: float,
        eta: float,
        m: int,
        mu: llama_cpp.CtypesPointerOrRef[ctypes.c_float],
    ) -> int:
        raise NotImplementedError("sample_token_mirostat is not implemented in llama.cpp")
        # return llama_cpp.llama_sample_token_mirostat(
        #     self.ctx,
        #     llama_cpp.byref(candidates.candidates),
        #     tau,
        #     eta,
        #     m,
        #     mu,
        # )

    def sample_token_mirostat_v2(
        self,
        candidates: "_LlamaTokenDataArray",
        tau: float,
        eta: float,
        mu: llama_cpp.CtypesPointerOrRef[ctypes.c_float],
    ) -> int:
        raise NotImplementedError("sample_token_mirostat_v2 is not implemented in llama.cpp")
        # return llama_cpp.llama_sample_token_mirostat_v2(
        #     self.ctx,
        #     llama_cpp.byref(candidates.candidates),
        #     tau,
        #     eta,
        #     mu,
        # )

    def sample_token_greedy(self, candidates: "_LlamaTokenDataArray") -> int:
        raise NotImplementedError("sample_token_greedy is not implemented in llama.cpp")
        # return llama_cpp.llama_sample_token_greedy(
        #     self.ctx,
        #     llama_cpp.byref(candidates.candidates),
        # )

    def sample_token(self, candidates: "_LlamaTokenDataArray") -> int:
        raise NotImplementedError("sample_token is not implemented in llama.cpp")
        # return llama_cpp.llama_sample_token(
        #     self.ctx,
        #     llama_cpp.byref(candidates.candidates),
        # )

    # Grammar
    def grammar_accept_token(self, grammar: LlamaGrammar, token: int):
        raise NotImplementedError("grammar_accept_token is not implemented in llama.cpp")
        # llama_cpp.llama_grammar_accept_token(grammar.grammar, self.ctx, token)

    def reset_timings(self):
        llama_cpp.llama_perf_context_reset(self.ctx)

    def print_timings(self):
        llama_cpp.llama_perf_context_print(self.ctx)

    # Utility functions
    @staticmethod
    def default_params():
        """Get the default llama_context_params."""
        return llama_cpp.llama_context_default_params()


class LlamaBatch:
    def __init__(
        self,
        *,
        n_tokens: int,
        embd: int,
        n_seq_max: int,
        verbose: bool = True
    ):
        # logical validity of parameters
        if n_tokens <= 0:
            raise ValueError(f"n_tokens must be positive, got {n_tokens}")
        if n_seq_max <= 0:
            raise ValueError(f"n_seq_max must be positive, got {n_seq_max}")

        self.n_tokens_capacity = n_tokens
        self.embd = embd
        self.n_seq_max = n_seq_max
        self.verbose = verbose
        self._exit_stack = ExitStack()

        batch = llama_cpp.llama_batch_init(self.n_tokens_capacity, self.embd, self.n_seq_max)

        if batch is None:
            raise MemoryError(
                f"Failed to allocate memory for llama_batch via llama_batch_init({n_tokens},{embd},{n_seq_max})"
            )

        self.batch = batch

        def free_batch():
            if self.batch is None:
                return
            llama_cpp.llama_batch_free(self.batch)
            self.batch = None

        self._exit_stack.callback(free_batch)

    def close(self):
        """Manually free resources."""
        self._exit_stack.close()

    def __del__(self):
        self.close()

    def n_tokens(self) -> int:
        """
        Current number of tokens stored in the batch.
        """
        if self.batch is None: return 0
        return self.batch.n_tokens

    def capacity(self) -> int:
        """
        Total capacity of the batch.
        """
        return self.n_tokens_capacity

    def space_left(self) -> int:
        """
        Returns the number of empty slots remaining in the batch.
        Throws a RuntimeError if internal state implies an overflow.
        """
        if self.batch is None: return 0
        elif self.n_tokens_capacity >= self.batch.n_tokens:
            return self.n_tokens_capacity - self.batch.n_tokens
        else:
            raise RuntimeError(
                f"LlamaBatch Critical Error: n_tokens ({self.batch.n_tokens}) exceeds capacity ({self.n_tokens_capacity}). "
                "This implies a buffer overflow or corrupted internal state."
            )

    def reset(self):
        """
        Resets the batch counter to 0. Does not free memory, just resets the index.
        Call this before starting a new decoding step.
        """
        if self.batch is not None:
            self.batch.n_tokens = 0

    def set_batch(self, batch: Sequence[int], n_past: llama_cpp.llama_pos, logits_all: bool):
        if len(batch) > self.n_tokens_capacity:
             raise IndexError(f"Input batch size {len(batch)} exceeds capacity {self.n_tokens_capacity}")

        n_tokens = len(batch)
        self.batch.n_tokens = n_tokens
        for i in range(n_tokens):
            self.batch.token[i] = batch[i]
            self.batch.pos[i] = n_past + i
            self.batch.seq_id[i][0] = 0
            self.batch.n_seq_id[i] = 1
            self.batch.logits[i] = logits_all
        self.batch.logits[n_tokens - 1] = True

    def add_sequence(self, batch: Sequence[int], seq_id: int, logits_all: bool):
        n_tokens = len(batch)
        current_count = self.batch.n_tokens
        if current_count + n_tokens > self.n_tokens_capacity:
            raise IndexError(
                f"LlamaBatch overflow: Cannot add {n_tokens} tokens. "
                f"Space left: {self.n_tokens_capacity - current_count}"
            )
        self.batch.n_tokens += n_tokens
        for i in range(n_tokens):
            j = current_count + i
            self.batch.token[j] = batch[i]
            self.batch.pos[j] = i
            self.batch.seq_id[j][0] = seq_id
            self.batch.n_seq_id[j] = 1
            self.batch.logits[j] = logits_all
        self.batch.logits[current_count + n_tokens - 1] = True


class LlamaTokenDataArray:
    def __init__(self, *, n_vocab: int):
        self.n_vocab = n_vocab
        self.candidates_data = np.recarray(
            (self.n_vocab,),
            dtype=np.dtype(
                [("id", np.intc), ("logit", np.single), ("p", np.single)], align=True
            ),
        )
        self.candidates = llama_cpp.llama_token_data_array(
            data=self.candidates_data.ctypes.data_as(llama_cpp.llama_token_data_p),
            size=self.n_vocab,
            selected=-1,
            sorted=False,
        )
        self.default_candidates_data_id = np.arange(self.n_vocab, dtype=np.intc)  # type: ignore
        self.default_candidates_data_p = np.zeros(self.n_vocab, dtype=np.single)

    def copy_logits(self, logits: npt.NDArray[np.single]):
        self.candidates_data.id[:] = self.default_candidates_data_id
        self.candidates_data.logit[:] = logits
        self.candidates_data.p[:] = self.default_candidates_data_p
        self.candidates.sorted = False
        self.candidates.size = self.n_vocab


# Embedding functions


def normalize_embedding(embedding):
    norm = float(np.linalg.norm(embedding))
    if norm == 0.0:
        return embedding
    return [v / norm for v in embedding]


# Python wrappers over common/sampling structs


@dataclass
class LlamaSamplingParams:
    n_prev: int = 64
    n_probs: int = 0
    top_k: int = 40
    top_n_sigma: float = -1.00
    top_p: float = 0.95
    min_p: float = 0.05
    typical_p: float = 1.00
    temp: float = 0.80
    penalty_last_n: int = 64
    penalty_repeat: float = 1.0
    penalty_freq: float = 0.00
    penalty_present: float = 0.00
    mirostat: int = 0
    mirostat_tau: float = 5.00
    mirostat_eta: float = 0.10
    penalize_nl: bool = True

    xtc_threshold: float = 0.1
    xtc_probability: float = 0.0

    grammar: str = ""

    cfg_negative_prompt: str = ""
    cfg_scale: float = 1.00

    logit_bias: dict[int, float] = field(default_factory=dict)


@dataclass
class LlamaSamplingContext:
    params: LlamaSamplingParams = field(default_factory=LlamaSamplingParams)
    mirostat_mu: ctypes.c_float = field(default_factory=ctypes.c_float)
    grammar: Optional[LlamaGrammar] = None
    # NOTE: Missing parsed_grammar
    prev: list[int] = field(default_factory=list)
    cur: list[llama_cpp.llama_token_data] = field(default_factory=list)

    def reset(self):
        self.prev = []
        self.cur = []
        if self.grammar is not None:
            self.grammar.reset()

    def cp(self):
        return LlamaSamplingContext(
            params=self.params,
            mirostat_mu=self.mirostat_mu,
            grammar=self.grammar,
            prev=self.prev.copy(),
            cur=self.cur.copy(),
        )

    def last(self) -> Optional[int]:
        if len(self.prev) > 0:
            return self.prev[-1]
        else:
            return None

    def prev_str(self, ctx_main: LlamaContext, n: int) -> str:
        return ctx_main.model.detokenize(self.prev[-n:]).decode("utf-8")

    def sample(
        self,
        ctx_main: LlamaContext,
        idx: int = 0,
        logits_array: Optional[npt.NDArray[np.single]] = None,
    ):
        n_vocab = ctx_main.model.n_vocab()
        id: int = 0

        if logits_array is None:
            logits = ctx_main.get_logits_ith(idx)
            logits_array = np.array(
                ctypes.cast(logits, ctypes.POINTER(ctypes.c_float * n_vocab)).contents,
                dtype=np.single,
            )

        # apply logit_bias
        for token, logit_bias in self.params.logit_bias.items():
            logits_array[token] += logit_bias

        token_data_array = LlamaTokenDataArray(
            n_vocab=n_vocab
        )  # TODO: Only create this once
        token_data_array.copy_logits(logits_array)

        # apply penalties
        if len(self.prev) > 0:
            nl_token = ctx_main.model.token_nl()
            nl_logit = logits_array[nl_token]
            last_tokens = self.prev[-self.params.penalty_last_n :]
            last_tokens_size = min(len(last_tokens), self.params.penalty_last_n)
            if last_tokens_size > 0:
                last_tokens_p = (llama_cpp.llama_token * len(last_tokens))(*last_tokens)
                ctx_main.sample_repetition_penalties(
                    token_data_array,
                    last_tokens_p,
                    last_tokens_size,
                    self.params.penalty_repeat,
                    self.params.penalty_freq,
                    self.params.penalty_present,
                )
            if not self.params.penalize_nl:
                token_data_array.candidates_data.logit[nl_token] = nl_logit

        if self.grammar is not None:
            ctx_main.sample_grammar(token_data_array, self.grammar)

        if self.params.temp < 0:
            id = token_data_array.candidates_data.id[0]
        elif self.params.temp == 0:
            id = ctx_main.sample_token_greedy(token_data_array)
        else:
            if self.params.mirostat == 1:
                mirostat_m = 100
                ctx_main.sample_temp(token_data_array, self.params.temp)
                id = ctx_main.sample_token_mirostat(
                    token_data_array,
                    self.params.mirostat_tau,
                    self.params.mirostat_eta,
                    mirostat_m,
                    ctypes.pointer(self.mirostat_mu),
                )
            elif self.params.mirostat == 2:
                ctx_main.sample_temp(token_data_array, self.params.temp)
                id = ctx_main.sample_token_mirostat_v2(
                    token_data_array,
                    self.params.mirostat_tau,
                    self.params.mirostat_eta,
                    ctypes.pointer(self.mirostat_mu),
                )
            else:
                min_keep = max(1, self.params.n_probs)
                ctx_main.sample_top_k(
                    token_data_array, self.params.top_k, min_keep=min_keep
                )
                ctx_main.sample_typical(
                    token_data_array, self.params.typical_p, min_keep=min_keep
                )
                ctx_main.sample_top_p(
                    token_data_array, self.params.top_p, min_keep=min_keep
                )
                ctx_main.sample_min_p(
                    token_data_array, self.params.min_p, min_keep=min_keep
                )
                ctx_main.sample_temp(token_data_array, self.params.temp)
                id = ctx_main.sample_token(token_data_array)
        return id

    def accept(self, ctx_main: LlamaContext, id: int, apply_grammar: bool):
        if apply_grammar and self.grammar is not None:
            ctx_main.grammar_accept_token(self.grammar, id)
        self.prev.append(id)


from typing import List, Callable, Optional, Union
import ctypes
import llama_cpp


class CustomSampler:
    def __init__(
        self, apply_func: Callable[[llama_cpp.llama_token_data_array], None]
    ):
        self.apply_func = apply_func

        def apply_wrapper(
            sampler: llama_cpp.llama_sampler_p,
            cur_p: llama_cpp.llama_token_data_array_p,
        ):
            self.apply_func(cur_p)

        def free_wrapper(sampler: llama_cpp.llama_sampler_p):
            pass

        sampler_i = llama_cpp.llama_sampler_i()
        sampler_i.apply = llama_cpp.llama_sampler_i_apply(apply_wrapper)
        self._apply_wrapper_ref = apply_wrapper

        sampler_i.name = llama_cpp.llama_sampler_i_name(0)
        sampler_i.accept = llama_cpp.llama_sampler_i_accept(0)
        sampler_i.reset = llama_cpp.llama_sampler_i_reset(0)
        sampler_i.clone = llama_cpp.llama_sampler_i_clone(0)
        sampler_i.free = llama_cpp.llama_sampler_i_free(0)

        self.sampler = llama_cpp.llama_sampler_init(ctypes.pointer(sampler_i), None)

    def get_sampler(self) -> llama_cpp.llama_sampler_p:
        return self.sampler


class LlamaSampler:
    def __init__(self):
        params = llama_cpp.llama_sampler_chain_params()
        self.sampler = llama_cpp.llama_sampler_chain_init(params)
        self.samplers: List[llama_cpp.llama_sampler_p] = []
        self.custom_samplers: List[Tuple[int, CustomSampler]] = []

    def add_greedy(self):
        sampler = llama_cpp.llama_sampler_init_greedy()
        self._add_sampler(sampler)

    def add_dist(self, seed: int):
        sampler = llama_cpp.llama_sampler_init_dist(seed)
        self._add_sampler(sampler)

    def add_top_k(self, k: int):
        sampler = llama_cpp.llama_sampler_init_top_k(k)
        self._add_sampler(sampler)

    def add_top_p(self, p: float, min_keep: int):
        sampler = llama_cpp.llama_sampler_init_top_p(p, min_keep)
        self._add_sampler(sampler)

    def add_min_p(self, p: float, min_keep: int):
        sampler = llama_cpp.llama_sampler_init_min_p(p, min_keep)
        self._add_sampler(sampler)

    def add_typical(self, p: float, min_keep: int):
        sampler = llama_cpp.llama_sampler_init_typical(p, min_keep)
        self._add_sampler(sampler)

    def add_xtc(self, p: float, t: float, min_keep: int, seed: int):
        sampler = llama_cpp.llama_sampler_init_xtc(p, t, min_keep, seed)
        self._add_sampler(sampler)

    def add_temp(self, temp: float):
        sampler = llama_cpp.llama_sampler_init_temp(temp)
        self._add_sampler(sampler)

    def add_temp_ext(self, t: float, delta: float, exponent: float):
        sampler = llama_cpp.llama_sampler_init_temp_ext(t, delta, exponent)
        self._add_sampler(sampler)

    def add_top_n_sigma(self, n: float):
        sampler = llama_cpp.llama_sampler_init_top_n_sigma(n)
        self._add_sampler(sampler)

    def add_mirostat(self, n_vocab: int, seed: int, tau: float, eta: float, m: int):
        sampler = llama_cpp.llama_sampler_init_mirostat(n_vocab, seed, tau, eta, m)
        self._add_sampler(sampler)

    def add_mirostat_v2(self, seed: int, tau: float, eta: float):
        sampler = llama_cpp.llama_sampler_init_mirostat_v2(seed, tau, eta)
        self._add_sampler(sampler)

    def add_grammar(self, model: LlamaModel, grammar: LlamaGrammar):
        sampler = llama_cpp.llama_sampler_init_grammar(
            model.vocab, grammar._grammar.encode("utf-8"), grammar._root.encode("utf-8")
        )
        self._add_sampler(sampler)

    def convert_list_str_to_char_array_ptr(self, str_list: List[str]):
        """
        Converts a list of strings to a char** array for C interop, and returns two values:
        the char** array and the number of bytes in the list.

        Args:
            str_list: List of string objects.

        Returns:
            - A ctypes pointer to a char** array.
            - The number of strings in the input list.
        """
        # Encode strings to bytes
        byte_list = [s.encode('utf-8') for s in str_list]
        # Calculate the number of breakers
        num_byte_list= len(byte_list)
        # Define the type of a char pointer
        char_ptr_type = ctypes.POINTER(ctypes.c_char)
        # Define the type of an array of char pointers
        char_ptr_array_type = char_ptr_type * num_byte_list

        # Allocate memory for the array of char pointers
        char_ptr_array = char_ptr_array_type()

        # Populate the array with pointers to the byte strings
        for i, byte_string in enumerate(byte_list):
            # Create a null-terminated C-style string buffer
            c_char_array = ctypes.create_string_buffer(byte_string)
            # Cast the buffer to a char pointer and assign it to the array
            char_ptr_array[i] = ctypes.cast(c_char_array, char_ptr_type)

        char_array_ptr = ctypes.cast(char_ptr_array, ctypes.POINTER(char_ptr_type))

        # Return the char** pointer and the number of strings
        return char_array_ptr, num_byte_list

    def add_grammar_lazy_patterns(
            self,
            model: LlamaModel,
            grammar: LlamaGrammar,
            num_trigger_patterns: int,
            trigger_tokens:list[llama_cpp.llama_token],
            num_trigger_tokens: int,
            trigger_patterns: list[str]=[]
        ):
        trigger_patterns_char_array_ptr, num_trigger_patterns = self.convert_list_str_to_char_array_ptr(trigger_patterns)
        sampler = llama_cpp.llama_sampler_init_grammar_lazy_patterns(
            model.vocab,
            grammar._grammar.encode("utf-8"),
            grammar._root.encode("utf-8"),
            trigger_patterns_char_array_ptr,
            num_trigger_patterns,
            trigger_tokens,
            num_trigger_tokens
        )
        self._add_sampler(sampler)


    def add_penalties(
        self,
        n_vocab: int,
        special_eos_id: int,
        linefeed_id: int,
        penalty_last_n: int,
        penalty_repeat: float,
        penalty_freq: float,
        penalty_present: float,
        penalize_nl: bool,
        ignore_eos: bool,
    ):
        sampler = llama_cpp.llama_sampler_init_penalties(
            penalty_last_n,
            penalty_repeat,
            penalty_freq,
            penalty_present,
        )
        self._add_sampler(sampler)

    def add_dry(
        self,
        model: LlamaModel,
        dry_multiplier: float,
        dry_base: float,
        dry_allowed_length: int,
        dry_penalty_last_n: int,
        dry_seq_breakers: list[str] = ["\n", ":", "\"", "*"]
    ):

        dry_seq_breakers_char_array_ptr, num_seq_breakers = self.convert_list_str_to_char_array_ptr(dry_seq_breakers)

        sampler = llama_cpp.llama_sampler_init_dry(
            model.vocab,
            model.n_ctx_train(),
            dry_multiplier,
            dry_base,
            dry_allowed_length,
            dry_penalty_last_n,
            dry_seq_breakers_char_array_ptr,
            num_seq_breakers
        )
        self._add_sampler(sampler)

    def add_adaptive_p(
        self,
        target: float,
        decay: float,
        seed: int,
    ):
        sampler = llama_cpp.llama_sampler_init_adaptive_p(
            target,
            decay,
            seed
        )
        self._add_sampler(sampler)

    def add_logit_bias(
        self, n_vocab: int, logit_bias: Dict[int, float]
    ):
        # Construct a C array to store the contents of the logit_bias dictionary
        logit_bias_array = (llama_cpp.llama_logit_bias * len(logit_bias))()

        for i, (token, bias) in enumerate(logit_bias.items()):
            logit_bias_array[i].token = token
            logit_bias_array[i].bias = bias

        sampler = llama_cpp.llama_sampler_init_logit_bias(n_vocab, len(logit_bias), logit_bias_array)
        self._add_sampler(sampler)

    def add_infill(self, model: LlamaModel):
        sampler = llama_cpp.llama_sampler_init_infill(model.vocab)
        self._add_sampler(sampler)

    def add_custom(
        self, apply_func: Callable[[llama_cpp.llama_token_data_array], None]
    ):
        custom_sampler = CustomSampler(apply_func)
        sampler = custom_sampler.get_sampler()
        self._add_sampler(sampler)
        # NOTE: Must remove custom samplers before free or llama.cpp will try to free them
        self.custom_samplers.append(
            [llama_cpp.llama_sampler_chain_n(self.sampler) - 1, custom_sampler]
        )

    def _add_sampler(self, sampler: llama_cpp.llama_sampler_p):
        assert self.sampler is not None
        llama_cpp.llama_sampler_chain_add(self.sampler, sampler)
        self.samplers.append(sampler)

    def get_seed(self) -> int:
        assert self.sampler is not None
        return llama_cpp.llama_sampler_get_seed(self.sampler)

    def sample(self, ctx: LlamaContext, idx: ctypes.c_int32) -> ctypes.c_int32:
        assert self.sampler is not None
        assert ctx.ctx is not None
        return llama_cpp.llama_sampler_sample(self.sampler, ctx.ctx, idx)

    def close(self):
        if self.sampler:
            # NOTE: Must remove custom samplers before free or llama.cpp will try to free them
            for i, _ in reversed(self.custom_samplers):
                llama_cpp.llama_sampler_chain_remove(self.sampler, i)
            llama_cpp.llama_sampler_free(self.sampler)
            self.sampler = None
        self.samplers.clear()
        self.custom_samplers.clear()

    def __del__(self):
        self.close()
