"""
Enterprise Caching Layer.

Provides a thread-safe, hashed in-memory cache to memoize expensive operations
such as external web search requests, RAG retrieval context, compliance audits, 
and costing formulas to minimize external API costs and eliminate redundant runs.
"""

import hashlib
import threading
from typing import Any, Dict, Optional


class EnterpriseCache:
    """
    Thread-safe in-memory cache that uses argument hashing for key generation.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._cache: Dict[str, Any] = {}

    def _make_key(self, key_type: str, *args, **kwargs) -> str:
        """
        Generate a unique MD5 hash for the combination of key_type and arguments.
        """
        # Convert arguments to string representations, sorting kwargs for determinism
        repr_args = tuple(repr(arg) for arg in args)
        repr_kwargs = tuple((k, repr(v)) for k, v in sorted(kwargs.items()))
        
        serialized = f"{key_type}:{repr_args}:{repr_kwargs}"
        return hashlib.md5(serialized.encode("utf-8")).hexdigest()

    def get(self, key_type: str, *args, **kwargs) -> Optional[Any]:
        """
        Retrieve a value from the cache. Returns None if cache miss.
        """
        key = self._make_key(key_type, *args, **kwargs)
        with self._lock:
            return self._cache.get(key)

    def set(self, key_type: str, value: Any, *args, **kwargs):
        """
        Set a value in the cache.
        """
        key = self._make_key(key_type, *args, **kwargs)
        with self._lock:
            self._cache[key] = value

    def clear(self):
        """
        Clear all cached elements.
        """
        with self._lock:
            self._cache.clear()


# Single-user global cache instance
GLOBAL_CACHE = EnterpriseCache()
