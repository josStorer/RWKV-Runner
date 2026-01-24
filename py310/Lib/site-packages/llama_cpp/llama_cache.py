import sys
from abc import ABC, abstractmethod
from typing import (
    Optional,
    Sequence,
    Tuple,
)
from collections import OrderedDict

import diskcache

import llama_cpp.llama

from .llama_types import *


class BaseLlamaCache(ABC):
    """Base cache class for a llama.cpp model."""

    def __init__(self, capacity_bytes: int = (2 << 30)):
        self.capacity_bytes = capacity_bytes

    @property
    @abstractmethod
    def cache_size(self) -> int:
        raise NotImplementedError

    def _find_longest_prefix_key(
        self,
        key: Tuple[int, ...],
    ) -> Optional[Tuple[int, ...]]:
        pass

    @abstractmethod
    def __getitem__(self, key: Sequence[int]) -> "llama_cpp.llama.LlamaState":
        raise NotImplementedError

    @abstractmethod
    def __contains__(self, key: Sequence[int]) -> bool:
        raise NotImplementedError

    @abstractmethod
    def __setitem__(
        self, key: Sequence[int], value: "llama_cpp.llama.LlamaState"
    ) -> None:
        raise NotImplementedError


class LlamaRAMCache(BaseLlamaCache):
    """Cache for a llama.cpp model using RAM."""

    def __init__(self, capacity_bytes: int = (2 << 30)):
        super().__init__(capacity_bytes)
        self.capacity_bytes = capacity_bytes
        self.cache_state: OrderedDict[
            Tuple[int, ...], "llama_cpp.llama.LlamaState"
        ] = OrderedDict()

    @property
    def cache_size(self):
        return sum([state.llama_state_size for state in self.cache_state.values()])

    def _find_longest_prefix_key(
        self,
        key: Tuple[int, ...],
    ) -> Optional[Tuple[int, ...]]:
        min_len = 0
        min_key = None
        keys = (
            (k, llama_cpp.llama.Llama.longest_token_prefix(k, key))
            for k in self.cache_state.keys()
        )
        for k, prefix_len in keys:
            if prefix_len > min_len:
                min_len = prefix_len
                min_key = k
        return min_key

    def __getitem__(self, key: Sequence[int]) -> "llama_cpp.llama.LlamaState":
        key = tuple(key)
        _key = self._find_longest_prefix_key(key)
        if _key is None:
            raise KeyError("Key not found")
        value = self.cache_state[_key]
        self.cache_state.move_to_end(_key)
        return value

    def __contains__(self, key: Sequence[int]) -> bool:
        return self._find_longest_prefix_key(tuple(key)) is not None

    def __setitem__(self, key: Sequence[int], value: "llama_cpp.llama.LlamaState"):
        key = tuple(key)
        if key in self.cache_state:
            del self.cache_state[key]
        self.cache_state[key] = value
        while self.cache_size > self.capacity_bytes and len(self.cache_state) > 0:
            self.cache_state.popitem(last=False)


class TrieNode:
    """A node in the prefix tree (Trie)."""
    def __init__(self):
        # Child nodes: {token_id: TrieNode}
        self.children: Dict[int, "TrieNode"] = {}
        # Stores the LlamaState if this node marks the end of a cached sequence.
        self.state: Optional["llama_cpp.llama.LlamaState"] = None


class LlamaTrieCache(BaseLlamaCache):
    """
    A Llama cache implementation using a Trie for O(K) prefix lookup
    and an OrderedDict for O(1) LRU eviction.

    - K = length of the query key (number of tokens)
    - N = total number of items in the cache

    This solves the O(N*K) lookup bottleneck of the linear scan cache.
    """

    def __init__(self, capacity_bytes: int = (2 << 30)):
        super().__init__(capacity_bytes)
        self.root = TrieNode() # The root node of the Trie
        self._current_size = 0  # O(1) tracking of cache size in bytes

        # LRU Tracker:
        # Key: Cached token sequence (Tuple[int, ...])
        # Value: The *terminal* TrieNode for that key
        self.lru_tracker: OrderedDict[
            Tuple[int, ...], TrieNode
        ] = OrderedDict()

    @property
    def cache_size(self) -> int:
        """Returns the current total size of the cache in bytes (O(1))."""
        return self._current_size

    def _find_longest_prefix_node(
        self, key: Tuple[int, ...]
    ) -> Tuple[Optional[TrieNode], Optional[Tuple[int, ...]]]:
        """
        Finds the longest cached prefix for a given key in O(K) time.

        Returns: (The matching TrieNode, The matching key)
        """
        node = self.root
        longest_prefix_node: Optional[TrieNode] = None
        longest_prefix_key: Optional[Tuple[int, ...]] = None
        current_prefix: List[int] = []

        # Check if the empty prefix (root) is cached
        if node.state is not None:
            longest_prefix_node = node
            longest_prefix_key = tuple(current_prefix)

        for token in key:
            if token not in node.children:
                # Path ends, no further prefix matches
                break

            node = node.children[token]
            current_prefix.append(token)

            if node.state is not None:
                # Found a valid, longer prefix; update our best match
                longest_prefix_node = node
                longest_prefix_key = tuple(current_prefix)

        return longest_prefix_node, longest_prefix_key

    def __getitem__(self, key: Sequence[int]) -> "llama_cpp.llama.LlamaState":
        """
        Retrieves the state for the longest matching prefix in O(K) time.
        Updates the LRU status.
        """
        key_tuple = tuple(key)
        node, prefix_key = self._find_longest_prefix_node(key_tuple)

        if node is None or node.state is None or prefix_key is None:
            raise KeyError(f"Key prefix not found in cache for: {key_tuple}")

        # Move the accessed key to the end (most recently used) in O(1)
        self.lru_tracker.move_to_end(prefix_key)

        return node.state

    def __contains__(self, key: Sequence[int]) -> bool:
        """Checks if any prefix of the key is cached in O(K) time."""
        node, _ = self._find_longest_prefix_node(tuple(key))
        return node is not None

    def _prune(self, key: Tuple[int, ...]):
        """
        (Helper) Removes a key and its state from the Trie.
        Also removes empty parent nodes (branch pruning).
        """
        path: List[Tuple[TrieNode, int]] = [] # Stores (parent_node, token)
        node = self.root

        # 1. Find the node and record the path
        for token in key:
            if token not in node.children:
                return # Key not found
            path.append((node, token))
            node = node.children[token]

        # 2. Remove the state
        if node.state is None:
            return # Node has no state

        self._current_size -= node.state.llama_state_size
        node.state = None

        # 3. Prune empty parent nodes backward
        for parent, token in reversed(path):
            child = parent.children[token]

            # If the child node is now empty (no children, no state), delete it
            if not child.children and child.state is None:
                del parent.children[token]
            else:
                # Node is still in use, stop pruning
                break

    def __setitem__(self, key: Sequence[int], value: "llama_cpp.llama.LlamaState"):
        """
        Adds a (key, state) pair to the cache in O(K) time.
        Handles LRU updates and eviction.
        """
        key_tuple = tuple(key)

        # 1. Find or create nodes for the key (O(K))
        node = self.root
        for token in key_tuple:
            node = node.children.setdefault(token, TrieNode())

        # 2. Check if updating an existing item
        if node.state is not None:
            self._current_size -= node.state.llama_state_size

        # 3. Set new state and update O(1) size
        node.state = value
        self._current_size += value.llama_state_size

        # 4. Update LRU tracker (O(1))
        if key_tuple in self.lru_tracker:
            self.lru_tracker.move_to_end(key_tuple)
        else:
            self.lru_tracker[key_tuple] = node

        # 5. Eviction logic
        while self._current_size > self.capacity_bytes and self.lru_tracker:
            # Get the least recently used item in O(1)
            evicted_key, _ = self.lru_tracker.popitem(last=False)

            # Remove the evicted item from the Trie
            self._prune(evicted_key)

# Alias for backwards compatibility
LlamaCache = LlamaRAMCache


class LlamaDiskCache(BaseLlamaCache):
    """Cache for a llama.cpp model using disk."""

    def __init__(
        self, cache_dir: str = ".cache/llama_cache", capacity_bytes: int = (2 << 30)
    ):
        super().__init__(capacity_bytes)
        self.cache = diskcache.Cache(cache_dir)

    @property
    def cache_size(self):
        return int(self.cache.volume())  # type: ignore

    def _find_longest_prefix_key(
        self,
        key: Tuple[int, ...],
    ) -> Optional[Tuple[int, ...]]:
        min_len = 0
        min_key: Optional[Tuple[int, ...]] = None
        for k in self.cache.iterkeys():  # type: ignore
            prefix_len = llama_cpp.llama.Llama.longest_token_prefix(k, key)
            if prefix_len > min_len:
                min_len = prefix_len
                min_key = k  # type: ignore
        return min_key

    def __getitem__(self, key: Sequence[int]) -> "llama_cpp.llama.LlamaState":
        key = tuple(key)
        _key = self._find_longest_prefix_key(key)
        if _key is None:
            raise KeyError("Key not found")
        value: "llama_cpp.llama.LlamaState" = self.cache.pop(_key)  # type: ignore
        # NOTE: This puts an integer as key in cache, which breaks,
        # Llama.longest_token_prefix(k, key) above since k is not a tuple of ints/tokens
        # self.cache.push(_key, side="front")  # type: ignore
        return value

    def __contains__(self, key: Sequence[int]) -> bool:
        return self._find_longest_prefix_key(tuple(key)) is not None

    def __setitem__(self, key: Sequence[int], value: "llama_cpp.llama.LlamaState"):
        print("LlamaDiskCache.__setitem__: called", file=sys.stderr)
        key = tuple(key)
        if key in self.cache:
            print("LlamaDiskCache.__setitem__: delete", file=sys.stderr)
            del self.cache[key]
        self.cache[key] = value
        print("LlamaDiskCache.__setitem__: set", file=sys.stderr)
        while self.cache_size > self.capacity_bytes and len(self.cache) > 0:
            key_to_remove = next(iter(self.cache))
            del self.cache[key_to_remove]
        print("LlamaDiskCache.__setitem__: trim", file=sys.stderr)
