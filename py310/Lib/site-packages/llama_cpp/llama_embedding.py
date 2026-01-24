import numpy as np
from typing import Union, List, Optional, Dict, Any, Tuple
import llama_cpp.llama_cpp as llama_cpp
from .llama_types import Embedding
from .llama import Llama
# Pooling types from .llama_cpp
from .llama_cpp import (
    LLAMA_POOLING_TYPE_UNSPECIFIED,
    LLAMA_POOLING_TYPE_NONE,
    LLAMA_POOLING_TYPE_MEAN,
    LLAMA_POOLING_TYPE_CLS,
    LLAMA_POOLING_TYPE_LAST,
    LLAMA_POOLING_TYPE_RANK, # Specifically for Reranking models
)

# Normalization modes for embedding vectors
# See: https://github.com/ggml-org/llama.cpp/tree/master/examples/embedding#--embd-normalize-integer
NORM_MODE_NONE = -1
NORM_MODE_MAX_INT16 = 0
NORM_MODE_TAXICAB = 1
NORM_MODE_EUCLIDEAN = 2
NORM_MODE_PNORM = 6

# TODO(JamePeng): Needs more extensive testing with various embedding and reranking models.
class LlamaEmbedding(Llama):
    """
    A specialized class for high-performance Text Embedding and Reranking.
    Inherits from the base Llama class but is optimized for vector operations.

    Key Features:
    1. Auto-configuration: Automatically sets embeddings=True.
    2. Streaming Batch: Handles massive datasets without OOM (Out Of Memory).
    3. Native Reranking Support: Specifically handles `LLAMA_POOLING_TYPE_RANK` models (like BGE-Reranker). /
       It correctly identifies classification heads to output scalar relevance scores instead of high-dimensional vectors.
    4. Advanced Normalization: Implements MaxInt16, Taxicab (L1), and Euclidean (L2) normalization strategies /
       using NumPy for optimal performance and compatibility with various vector databases.
    """

    def __init__(
            self,
            model_path: str,
            n_ctx: int = 0,
            n_batch: int = 512,
            n_ubatch: int = 512,
            pooling_type: int = LLAMA_POOLING_TYPE_UNSPECIFIED,
            n_gpu_layers: int = 0,
            verbose: bool = True,
            **kwargs):
        """
        Initialize the embedding model with enforced configuration.

        Args:
            model_path: Path to the GGUF model file.
            n_ctx: Text context, 0 = from model
            n_batch: Prompt processing maximum batch size
            n_ubatch: Physical batch size
            pooling_type: The pooling strategy used by the model.
                          - Use `LLAMA_POOLING_TYPE_RANK` (4) for Reranker models.
                          - Use `LLAMA_POOLING_TYPE_UNSPECIFIED` (-1) to let the model metadata decide (for standard embeddings).
            n_gpu_layers: Number of model layers to offload to GPU.
                          - Set to 0 for CPU only.
                          - Set to -1 for all layers (recommended for best performance).
            **kwargs: Additional arguments passed to the Llama base class (e.g., n_batch, n_ctx, verbose).
        """
        kwargs["embeddings"] = True
        kwargs["n_gpu_layers"] = n_gpu_layers
        kwargs["n_ctx"] = n_ctx
        kwargs["n_batch"] = n_batch
        kwargs["n_ubatch"] = n_ubatch
        kwargs["verbose"] = verbose

        # Enable Unified KV Cache (Crucial for Batching)
        # This allows us to assign arbitrary seq_ids in a batch, enabling the parallel /
        #     encoding of multiple unrelated documents without "invalid seq_id" errors.
        kwargs["kv_unified"] = True

        # Set pooling type
        kwargs["pooling_type"] = pooling_type

        super().__init__(model_path=model_path, **kwargs)

        if self.verbose:
            print(f"LlamaEmbedding initialized with pooling_type: {self.pooling_type()}")

    def _normalize_vector(self, vector: List[float], mode: int) -> List[float]:
        """
        Apply mathematical normalization to a vector.
        Uses numpy for performance.
        """
        if mode == NORM_MODE_NONE: return vector
        arr = np.array(vector, dtype=np.float32)

        # Mode 0: Max Absolute Int16 -> 32760 * x_i / max|x_i|
        if mode == NORM_MODE_MAX_INT16:
            max_abs = np.max(np.abs(arr))
            if max_abs == 0: return vector
            return ((arr / max_abs) * 32760.0).tolist()

        # Mode 1: Taxicab (L1 Norm) -> x_i / sum|x_i|
        elif mode == NORM_MODE_TAXICAB:
            norm = np.sum(np.abs(arr))
            if norm == 0: return vector
            return (arr / norm).tolist()

        # Mode 2: Euclidean (L2 Norm) -> x_i / sqrt(sum x_i^2)
        elif mode == NORM_MODE_EUCLIDEAN:
            norm = np.linalg.norm(arr)
            if norm == 0: return vector
            return (arr / norm).tolist()

        # Mode > 2: p-norm
        elif mode > 2:
            norm = np.sum(np.abs(arr) ** mode) ** (1.0 / mode)
            if norm == 0: return vector
            return (arr / norm).tolist()

        return vector

    def embed(
        self,
        input: Union[str, List[str], List[List[int]]],
        normalize: int = NORM_MODE_EUCLIDEAN,
        truncate: bool = True,
        separator: Optional[str] = None,
        return_count: bool = False,
    ) -> Union[List[float], List[List[float]], Tuple[Any, int]]:

        ctx = self._ctx.ctx
        n_batch = self.n_batch
        n_ctx = self._n_ctx
        n_ubatch = self.context_params.n_ubatch

        # Determine if it is in Rerank mode
        try:
            pooling_type = self.pooling_type()
        except AttributeError:
            pooling_type = LLAMA_POOLING_TYPE_UNSPECIFIED
        is_rank = (pooling_type == LLAMA_POOLING_TYPE_RANK)
        is_none = (pooling_type == LLAMA_POOLING_TYPE_NONE) # Token-level embedding
        logits_all = True if is_none else False

        # Determine the output dimension
        if is_rank:
            out_dim = llama_cpp.llama_model_n_cls_out(self._model.model)
        else:
            out_dim = self.n_embd()

        if self.verbose:
            type_str = "TOKEN (None)" if is_none else ("RANK (Score)" if is_rank else "SEQ (Vector)")
            print(f"LlamaEmbedding Debug: Mode={type_str} | Pooling={pooling_type} | Dim={out_dim}")

        # Preprocess Input
        inputs: List[Union[str, List[int]]] = []
        is_single = False

        if isinstance(input, str):
            if separator:
                inputs = input.split(separator)
                is_single = False
            else:
                inputs = [input]
                is_single = True
        else:
            inputs = input
            is_single = False

        # Reset Context and Batch
        if self.verbose:
            llama_cpp.llama_perf_context_reset(ctx)
        self._batch.reset()
        llama_cpp.llama_memory_clear(llama_cpp.llama_get_memory(ctx), True)

        # Initialize State Variables
        results: List[Any] = []
        batch_seq_lens: List[int] = []
        total_tokens_processed = 0

        # --- Decode Current Batch ---
        def _decode_batch():
            nonlocal batch_seq_lens
            if not batch_seq_lens: return

            self._ctx.decode(self._batch)

            # Extract Embeddings
            # Branch A: LLAMA_POOLING_TYPE_NONE (Token Level)
            if is_none:
                curr_token_idx = 0
                for seq_len in batch_seq_lens:
                    doc_tokens_embd = []
                    for _ in range(seq_len):
                        # Get the vector of the i-th token
                        ptr = llama_cpp.llama_get_embeddings_ith(ctx, curr_token_idx)
                        if ptr is None:
                            # Fallback: append zero vector or skip (here we zero-pad to keep shape)
                            doc_tokens_embd.append([0.0] * out_dim)
                        else:
                            data = ptr[:out_dim]
                            # Normalization
                            data = self._normalize_vector(data, normalize)
                            doc_tokens_embd.append(data)

                        curr_token_idx += 1
                    results.append(doc_tokens_embd)

            # Branch B: Sequence Level (Mean, Cls, Rank, Unspecified)
            else:
                for i in range(len(batch_seq_lens)):
                    # Obtain the vector of the i-th sequence.
                    ptr = llama_cpp.llama_get_embeddings_seq(ctx, i)
                    data = ptr[:out_dim]

                    if not is_rank:
                        data = self._normalize_vector(data, normalize)

                    if is_rank and len(data) == 1:
                        results.append(data[0])
                    else:
                        results.append(data)

            self._batch.reset()
            llama_cpp.llama_memory_clear(llama_cpp.llama_get_memory(ctx), True)
            batch_seq_lens = []

        # Main Streaming Loop
        idx_in_batch = 0

        for item in inputs:
            # Tokenize
            tokens: List[int] = []
            if isinstance(item, list) and (not item or isinstance(item[0], int)):
                tokens = item
            elif isinstance(item, str):
                tokens = self.tokenize(item.encode("utf-8"))
            else:
                raise ValueError("Input item must be str or List[int]")

            # Truncate
            if truncate and len(tokens) > n_ctx:
                tokens = tokens[:n_ctx]

            n_tokens = len(tokens)
            total_tokens_processed += n_tokens

            if n_tokens == 0:
                results.append(0.0 if is_rank else [])
                continue

            # Check Batch Capacity
            if (self._batch.n_tokens() + n_tokens > n_batch) or (idx_in_batch >= n_ubatch):
                _decode_batch()
                idx_in_batch = 0

            # Add to Batch
            self._batch.add_sequence(tokens, idx_in_batch, logits_all=logits_all)
            batch_seq_lens.append(n_tokens)
            idx_in_batch += 1

        # Process Remaining Items
        _decode_batch()

        if self.verbose:
            llama_cpp.llama_perf_context_print(ctx)

        final_result = results[0] if is_single else results

        if return_count:
            return final_result, total_tokens_processed

        return final_result

    def rank(self, query: str, documents: List[str]) -> List[float]:
        """
        Calculate relevance scores for a list of documents against a query using a Reranking model.

        This method constructs a specific prompt structure ([BOS] Query [SEP] Doc [EOS])
        typically used by Cross-Encoders to estimate similarity.

        Args:
            query: The search query string.
            documents: A list of candidate document strings to be scored.

        Returns:
            A list of float scores, where higher values indicate greater relevance.
        """
        if self.pooling_type() != LLAMA_POOLING_TYPE_RANK:
            raise ValueError(f"Model pooling_type is {self.pooling_type()}, but LLAMA_POOLING_TYPE_RANK is required.")

        # Prepare Special Tokens
        sep_id = self.token_sep()
        if sep_id == -1: sep_id = self.token_eos()
        eos_id = self.token_eos()

        # Pre-process Query
        q_tokens = self.tokenize(query.encode("utf-8"), add_bos=True, special=True)
        # Remove the automatically added EOS token from the query
        # because we need to append the separator and document tokens after it.
        if q_tokens and q_tokens[-1] == eos_id:
            q_tokens.pop()

        # Construct Batch Inputs
        batch_inputs: List[List[int]] = []
        for doc in documents:
            d_tokens = self.tokenize(doc.encode("utf-8"), add_bos=False, special=True)
            full_seq = q_tokens + [sep_id] + d_tokens
            # Ensure the sequence ends with an EOS token to mark the end of inference.
            if not full_seq or full_seq[-1] != eos_id:
                full_seq.append(eos_id)
            batch_inputs.append(full_seq)

        # We use NORM_MODE_NONE because rerankers output raw logits/scores, not vectors that need normalization.
        return self.embed(batch_inputs, normalize=NORM_MODE_NONE)

    def create_embedding(
        self,
        input: Union[str, List[str]],
        model: Optional[str] = None,
        normalize: int = NORM_MODE_EUCLIDEAN,
        output_format: str = "json"
    ) -> Union[Dict[str, Any], List[float], List[List[float]]]:
        """
        High-level API compatible with OpenAI format.

        Args:
            output_format:
                - 'json': OpenAI style dict (Default)
                - 'json+': OpenAI style dict + cosineSimilarity matrix
                - 'array': Raw python list (List[float] or List[List[float]])
        """
        model_name = model if model is not None else self.model_path

        # Normalize input to list
        inputs_list = [input] if isinstance(input, str) else input

        # Generate Embeddings(and get token count)
        embeddings, token_count = self.embed(
            inputs_list,
            normalize=normalize,
            return_count=True
        )

        if output_format == "array":
            return embeddings

        # Structure the OpenAI-style response ('json' or 'json+')
        # Ensure embeddings is a list for iteration
        # (If input was single string, embeddings is List[float], wrap it for the loop)
        iter_embeddings = [embeddings] if isinstance(embeddings[0], float) else embeddings

        data: List[Embedding] = [
            {
                "object": "embedding",
                "embedding": emb,
                "index": idx,
            }
            for idx, emb in enumerate(iter_embeddings)
        ]

        response = {
            "object": "list",
            "data": data,
            "model": model_name,
            "usage": {
                "prompt_tokens": token_count,  # Input consumption
                "completion_tokens": 0,        # The Embedding task does not generate text, so the value is 0.
                "total_tokens": token_count,   # Total consumption = Input consumption + Output
            }
        }

        # Calculate Cosine Similarity Matrix (Optimized via Numpy)
        # Only if output_format is 'json+' and we have vectors
        if output_format == "json+" and len(embeddings) > 1 and isinstance(embeddings[0], list):
            try:
                # Assuming embeddings are already L2 normalized if normalize=2
                mat = np.array(embeddings)

                # Safety check: Force normalize if not already done, to ensure Cosine (not Dot Product)
                if normalize != NORM_MODE_EUCLIDEAN:
                    norm = np.linalg.norm(mat, axis=1, keepdims=True)
                    # Avoid division by zero
                    norm[norm == 0] = 1e-10
                    mat = mat / norm

                # Matrix multiplication: A @ A.T
                sim_matrix = np.dot(mat, mat.T)
                response["cosineSimilarity"] = sim_matrix.tolist()
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Failed to calculate similarity matrix: {e}")

        return response
