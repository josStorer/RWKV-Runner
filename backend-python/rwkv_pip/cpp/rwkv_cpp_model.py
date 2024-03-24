import os
import multiprocessing

# Pre-import PyTorch, if available.
# This fixes "OSError: [WinError 127] The specified procedure could not be found".
try:
    import torch
except ModuleNotFoundError:
    pass

# I'm sure this is not strictly correct, but let's keep this crutch for now.
try:
    import rwkv_cpp_shared_library
except ModuleNotFoundError:
    from . import rwkv_cpp_shared_library

from typing import TypeVar, Optional, Tuple, List

# A value of this type is either a numpy's ndarray or a PyTorch's Tensor.
NumpyArrayOrPyTorchTensor: TypeVar = TypeVar('NumpyArrayOrPyTorchTensor')

class RWKVModel:
    """
    An RWKV model managed by rwkv.cpp library.
    """

    def __init__(
            self,
            shared_library: rwkv_cpp_shared_library.RWKVSharedLibrary,
            model_path: str,
            thread_count: int = max(1, multiprocessing.cpu_count() // 2),
            gpu_layer_count: int = 0,
            **kwargs
    ) -> None:
        """
        Loads the model and prepares it for inference.
        In case of any error, this method will throw an exception.

        Parameters
        ----------
        shared_library : RWKVSharedLibrary
            rwkv.cpp shared library.
        model_path : str
            Path to RWKV model file in ggml format.
        thread_count : int
            Thread count to use. If not set, defaults to CPU count / 2.
        gpu_layer_count : int
            Count of layers to offload onto the GPU, must be >= 0.
            See documentation of `gpu_offload_layers` for details about layer offloading.
        """

        if 'gpu_layers_count' in kwargs:
            gpu_layer_count = kwargs['gpu_layers_count']

        if not os.path.isfile(model_path):
            raise ValueError(f'{model_path} is not a file')
        
        if not (thread_count > 0):
            raise ValueError('Thread count must be > 0')  

        if not (gpu_layer_count >= 0):
            raise ValueError('GPU layer count must be >= 0')

        self._library: rwkv_cpp_shared_library.RWKVSharedLibrary = shared_library

        self._ctx: rwkv_cpp_shared_library.RWKVContext = self._library.rwkv_init_from_file(model_path, thread_count)

        if gpu_layer_count > 0:
            self.gpu_offload_layers(gpu_layer_count)

        self._state_buffer_element_count: int = self._library.rwkv_get_state_buffer_element_count(self._ctx)
        self._logits_buffer_element_count: int = self._library.rwkv_get_logits_buffer_element_count(self._ctx)

        self._valid: bool = True

    def gpu_offload_layers(self, layer_count: int) -> bool:
        """
        Offloads specified count of model layers onto the GPU. Offloaded layers are evaluated using cuBLAS or CLBlast.
        For the purposes of this function, model head (unembedding matrix) is treated as an additional layer:
        - pass `model.n_layer` to offload all layers except model head
        - pass `model.n_layer + 1` to offload all layers, including model head

        Returns true if at least one layer was offloaded.
        If rwkv.cpp was compiled without cuBLAS and CLBlast support, this function is a no-op and always returns false.

        Parameters
        ----------
        layer_count : int
            Count of layers to offload onto the GPU, must be >= 0.
        """

        if not (layer_count >= 0):
            raise ValueError('Layer count must be >= 0')

        return self._library.rwkv_gpu_offload_layers(self._ctx, layer_count)

    @property
    def arch_version_major(self) -> int:
        return self._library.rwkv_get_arch_version_major(self._ctx)

    @property
    def arch_version_minor(self) -> int:
        return self._library.rwkv_get_arch_version_minor(self._ctx)

    @property
    def n_vocab(self) -> int:
        return self._library.rwkv_get_n_vocab(self._ctx)

    @property
    def n_embed(self) -> int:
        return self._library.rwkv_get_n_embed(self._ctx)

    @property
    def n_layer(self) -> int:
        return self._library.rwkv_get_n_layer(self._ctx)

    def eval(
            self,
            token: int,
            state_in: Optional[NumpyArrayOrPyTorchTensor],
            state_out: Optional[NumpyArrayOrPyTorchTensor] = None,
            logits_out: Optional[NumpyArrayOrPyTorchTensor] = None,
            use_numpy: bool = False
    ) -> Tuple[NumpyArrayOrPyTorchTensor, NumpyArrayOrPyTorchTensor]:
        """
        Evaluates the model for a single token.
        In case of any error, this method will throw an exception.

        Parameters
        ----------
        token : int
            Index of next token to be seen by the model. Must be in range 0 <= token < n_vocab.
        state_in : Optional[NumpyArrayOrTorchTensor]
            State from previous call of this method. If this is a first pass, set it to None.
        state_out : Optional[NumpyArrayOrTorchTensor]
            Optional output tensor for state. If provided, must be of type float32, contiguous and of shape (state_buffer_element_count).
        logits_out : Optional[NumpyArrayOrTorchTensor]
            Optional output tensor for logits. If provided, must be of type float32, contiguous and of shape (logits_buffer_element_count).
        use_numpy : bool
            If set to True, numpy's ndarrays will be created instead of PyTorch's Tensors.
            This parameter is ignored if any tensor parameter is not None; in such case,
            type of returned tensors will match the type of received tensors.

        Returns
        -------
        logits, state
            Logits vector of shape (n_vocab); state for the next step.
        """

        if not self._valid:
            raise ValueError('Model was freed')

        use_numpy = self._detect_numpy_usage([state_in, state_out, logits_out], use_numpy)

        if state_in is not None:
            self._validate_tensor(state_in, 'state_in', self._state_buffer_element_count)

            state_in_ptr = self._get_data_ptr(state_in)
        else:
            state_in_ptr = 0

        if state_out is not None:
            self._validate_tensor(state_out, 'state_out', self._state_buffer_element_count)
        else:
            state_out = self._zeros_float32(self._state_buffer_element_count, use_numpy)

        if logits_out is not None:
            self._validate_tensor(logits_out, 'logits_out', self._logits_buffer_element_count)
        else:
            logits_out = self._zeros_float32(self._logits_buffer_element_count, use_numpy)

        self._library.rwkv_eval(
            self._ctx,
            token,
            state_in_ptr,
            self._get_data_ptr(state_out),
            self._get_data_ptr(logits_out)
        )

        return logits_out, state_out
    
    def eval_sequence(
            self,
            tokens: List[int],
            state_in: Optional[NumpyArrayOrPyTorchTensor],
            state_out: Optional[NumpyArrayOrPyTorchTensor] = None,
            logits_out: Optional[NumpyArrayOrPyTorchTensor] = None,
            use_numpy: bool = False
    ) -> Tuple[NumpyArrayOrPyTorchTensor, NumpyArrayOrPyTorchTensor]:
        """
        Evaluates the model for a sequence of tokens.

        NOTE ON GGML NODE LIMIT

        ggml has a hard-coded limit on max amount of nodes in a computation graph. The sequence graph is built in a way that quickly exceedes
        this limit when using large models and/or large sequence lengths.
        Fortunately, rwkv.cpp's fork of ggml has increased limit which was tested to work for sequence lengths up to 64 for 14B models.

        If you get `GGML_ASSERT: ...\\ggml.c:16941: cgraph->n_nodes < GGML_MAX_NODES`, this means you've exceeded the limit.
        To get rid of the assertion failure, reduce the model size and/or sequence length.

        In case of any error, this method will throw an exception.

        Parameters
        ----------
        tokens : List[int]
            Indices of the next tokens to be seen by the model. Must be in range 0 <= token < n_vocab.
        state_in : Optional[NumpyArrayOrTorchTensor]
            State from previous call of this method. If this is a first pass, set it to None.
        state_out : Optional[NumpyArrayOrTorchTensor]
            Optional output tensor for state. If provided, must be of type float32, contiguous and of shape (state_buffer_element_count).
        logits_out : Optional[NumpyArrayOrTorchTensor]
            Optional output tensor for logits. If provided, must be of type float32, contiguous and of shape (logits_buffer_element_count).
        use_numpy : bool
            If set to True, numpy's ndarrays will be created instead of PyTorch's Tensors.
            This parameter is ignored if any tensor parameter is not None; in such case,
            type of returned tensors will match the type of received tensors.

        Returns
        -------
        logits, state
            Logits vector of shape (n_vocab); state for the next step.
        """

        if not self._valid:
            raise ValueError('Model was freed')

        use_numpy = self._detect_numpy_usage([state_in, state_out, logits_out], use_numpy)

        if state_in is not None:
            self._validate_tensor(state_in, 'state_in', self._state_buffer_element_count)

            state_in_ptr = self._get_data_ptr(state_in)
        else:
            state_in_ptr = 0

        if state_out is not None:
            self._validate_tensor(state_out, 'state_out', self._state_buffer_element_count)
        else:
            state_out = self._zeros_float32(self._state_buffer_element_count, use_numpy)

        if logits_out is not None:
            self._validate_tensor(logits_out, 'logits_out', self._logits_buffer_element_count)
        else:
            logits_out = self._zeros_float32(self._logits_buffer_element_count, use_numpy)

        self._library.rwkv_eval_sequence(
            self._ctx,
            tokens,
            state_in_ptr,
            self._get_data_ptr(state_out),
            self._get_data_ptr(logits_out)
        )

        return logits_out, state_out

    def eval_sequence_in_chunks(
            self,
            tokens: List[int],
            state_in: Optional[NumpyArrayOrPyTorchTensor],
            state_out: Optional[NumpyArrayOrPyTorchTensor] = None,
            logits_out: Optional[NumpyArrayOrPyTorchTensor] = None,
            chunk_size: int = 16,
            use_numpy: bool = False
    ) -> Tuple[NumpyArrayOrPyTorchTensor, NumpyArrayOrPyTorchTensor]:
        """
        Evaluates the model for a sequence of tokens using `eval_sequence`, splitting a potentially long sequence into fixed-length chunks.
        This function is useful for processing complete prompts and user input in chat & role-playing use-cases.
        It is recommended to use this function instead of `eval_sequence` to avoid mistakes and get maximum performance.

        Chunking allows processing sequences of thousands of tokens, while not reaching the ggml's node limit and not consuming too much memory.
        A reasonable and recommended value of chunk size is 16. If you want maximum performance, try different chunk sizes in range [2..64]
        and choose one that works the best in your use case.

        In case of any error, this method will throw an exception.

        Parameters
        ----------
        tokens : List[int]
            Indices of the next tokens to be seen by the model. Must be in range 0 <= token < n_vocab.
        chunk_size : int
            Size of each chunk in tokens, must be positive.
        state_in : Optional[NumpyArrayOrTorchTensor]
            State from previous call of this method. If this is a first pass, set it to None.
        state_out : Optional[NumpyArrayOrTorchTensor]
            Optional output tensor for state. If provided, must be of type float32, contiguous and of shape (state_buffer_element_count).
        logits_out : Optional[NumpyArrayOrTorchTensor]
            Optional output tensor for logits. If provided, must be of type float32, contiguous and of shape (logits_buffer_element_count).
        use_numpy : bool
            If set to True, numpy's ndarrays will be created instead of PyTorch's Tensors.
            This parameter is ignored if any tensor parameter is not None; in such case,
            type of returned tensors will match the type of received tensors.

        Returns
        -------
        logits, state
            Logits vector of shape (n_vocab); state for the next step.
        """

        if not self._valid:
            raise ValueError('Model was freed')

        use_numpy = self._detect_numpy_usage([state_in, state_out, logits_out], use_numpy)

        if state_in is not None:
            self._validate_tensor(state_in, 'state_in', self._state_buffer_element_count)

            state_in_ptr = self._get_data_ptr(state_in)
        else:
            state_in_ptr = 0

        if state_out is not None:
            self._validate_tensor(state_out, 'state_out', self._state_buffer_element_count)
        else:
            state_out = self._zeros_float32(self._state_buffer_element_count, use_numpy)

        if logits_out is not None:
            self._validate_tensor(logits_out, 'logits_out', self._logits_buffer_element_count)
        else:
            logits_out = self._zeros_float32(self._logits_buffer_element_count, use_numpy)

        self._library.rwkv_eval_sequence_in_chunks(
            self._ctx,
            tokens,
            chunk_size,
            state_in_ptr,
            self._get_data_ptr(state_out),
            self._get_data_ptr(logits_out)
        )

        return logits_out, state_out

    def free(self) -> None:
        """
        Frees all allocated resources.
        In case of any error, this method will throw an exception.
        The object must not be used anymore after calling this method.
        """

        if not self._valid:
            raise ValueError('Already freed')

        self._valid = False

        self._library.rwkv_free(self._ctx)

    def __del__(self) -> None:
        # Free the context on GC in case user forgot to call free() explicitly.
        if hasattr(self, '_valid') and self._valid:
            self.free()

    def _is_pytorch_tensor(self, tensor: NumpyArrayOrPyTorchTensor) -> bool:
        return hasattr(tensor, '__module__') and tensor.__module__ == 'torch'

    def _detect_numpy_usage(self, tensors: List[Optional[NumpyArrayOrPyTorchTensor]], use_numpy_by_default: bool) -> bool:
        for tensor in tensors:
            if tensor is not None:
                return False if self._is_pytorch_tensor(tensor) else True

        return use_numpy_by_default

    def _validate_tensor(self, tensor: NumpyArrayOrPyTorchTensor, name: str, size: int) -> None:
        if self._is_pytorch_tensor(tensor):
            tensor: torch.Tensor = tensor
            
            if tensor.device != torch.device('cpu'):
                raise ValueError(f'{name} is not on CPU')
            if tensor.dtype != torch.float32:
                raise ValueError(f'{name} is not of type float32')
            if tensor.shape != (size,):
                raise ValueError(f'{name} has invalid shape {tensor.shape}, expected ({size})')
            if not tensor.is_contiguous():
                raise ValueError(f'{name} is not contiguous')
        else:
            import numpy as np
            tensor: np.ndarray = tensor

            if tensor.dtype != np.float32:
                raise ValueError(f'{name} is not of type float32')
            if tensor.shape != (size,):
                raise ValueError(f'{name} has invalid shape {tensor.shape}, expected ({size})')
            if not tensor.data.contiguous:
                raise ValueError(f'{name} is not contiguous')

    def _get_data_ptr(self, tensor: NumpyArrayOrPyTorchTensor):
        if self._is_pytorch_tensor(tensor):
            return tensor.data_ptr()
        else:
            return tensor.ctypes.data

    def _zeros_float32(self, element_count: int, use_numpy: bool) -> NumpyArrayOrPyTorchTensor:
        if use_numpy:
            import numpy as np
            return np.zeros(element_count, dtype=np.float32)
        else:
            return torch.zeros(element_count, dtype=torch.float32, device='cpu')
