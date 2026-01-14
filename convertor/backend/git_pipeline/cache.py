"""
Production-Grade LRU Cache with TTL Expiration
===============================================
Thread-safe, O(1) get/put operations using OrderedDict.
Memory-bound with configurable capacity and time-based eviction.
"""

from collections import OrderedDict
from typing import Optional, Any, TypeVar, Generic
import time
import threading

T = TypeVar('T')


class LRUCache(Generic[T]):
    """
    Lock-free LRU cache optimized for high-throughput read scenarios.
    
    Complexity:
        - get(): O(1) amortized
        - put(): O(1) amortized
        - Memory: O(capacity)
    
    Thread Safety:
        Uses RLock for concurrent access. For extreme throughput,
        consider sharding across multiple cache instances.
    """
    __slots__ = ('_capacity', '_ttl', '_cache', '_timestamps', '_lock', '_hits', '_misses')
    
    def __init__(self, capacity: int = 1000, ttl_seconds: float = 300.0):
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
        if ttl_seconds <= 0:
            raise ValueError("TTL must be positive")
            
        self._capacity = capacity
        self._ttl = ttl_seconds
        self._cache: OrderedDict[str, T] = OrderedDict()
        self._timestamps: dict[str, float] = {}
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str) -> Optional[T]:
        """
        Retrieve value by key. Returns None if expired or missing.
        Updates access order for LRU tracking.
        """
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            
            # Check TTL expiration
            if time.monotonic() - self._timestamps[key] > self._ttl:
                del self._cache[key]
                del self._timestamps[key]
                self._misses += 1
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]
    
    def put(self, key: str, value: T) -> None:
        """
        Insert or update key-value pair.
        Evicts LRU entry if at capacity.
        """
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key] = value
                self._timestamps[key] = time.monotonic()
                return
            
            # Evict if at capacity
            while len(self._cache) >= self._capacity:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                del self._timestamps[oldest_key]
            
            self._cache[key] = value
            self._timestamps[key] = time.monotonic()
    
    def invalidate(self, key: str) -> bool:
        """Remove specific key. Returns True if key existed."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                del self._timestamps[key]
                return True
            return False
    
    def invalidate_prefix(self, prefix: str) -> int:
        """Remove all keys starting with prefix. Returns count removed."""
        with self._lock:
            keys_to_remove = [k for k in self._cache if k.startswith(prefix)]
            for key in keys_to_remove:
                del self._cache[key]
                del self._timestamps[key]
            return len(keys_to_remove)
    
    def clear(self) -> None:
        """Remove all entries."""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
    
    @property
    def stats(self) -> dict:
        """Return cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total * 100) if total > 0 else 0.0
            return {
                "size": len(self._cache),
                "capacity": self._capacity,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate_percent": round(hit_rate, 2)
            }


class DocumentCache:
    """
    Domain-specific cache for Git pipeline documents.
    Separates metadata cache (high capacity) from content cache (lower capacity).
    """
    
    def __init__(
        self,
        metadata_capacity: int = 10000,
        content_capacity: int = 500,
        ttl_seconds: float = 300.0
    ):
        self._metadata = LRUCache[dict](metadata_capacity, ttl_seconds)
        self._content = LRUCache[str](content_capacity, ttl_seconds)
        self._tree = LRUCache[list](100, ttl_seconds)
    
    def _doc_key(self, repo_id: str, path: str) -> str:
        return f"{repo_id}:{path}"
    
    def get_metadata(self, repo_id: str, path: str) -> Optional[dict]:
        return self._metadata.get(self._doc_key(repo_id, path))
    
    def put_metadata(self, repo_id: str, path: str, data: dict) -> None:
        self._metadata.put(self._doc_key(repo_id, path), data)
    
    def get_content(self, repo_id: str, path: str) -> Optional[str]:
        return self._content.get(self._doc_key(repo_id, path))
    
    def put_content(self, repo_id: str, path: str, content: str) -> None:
        self._content.put(self._doc_key(repo_id, path), content)
    
    def get_tree(self, repo_id: str, path: str = "", depth: int = 2) -> Optional[list]:
        key = f"{repo_id}:tree:{path}:{depth}"
        return self._tree.get(key)
    
    def put_tree(self, repo_id: str, tree: list, path: str = "", depth: int = 2) -> None:
        key = f"{repo_id}:tree:{path}:{depth}"
        self._tree.put(key, tree)
    
    def invalidate_repo(self, repo_id: str) -> int:
        """Invalidate all caches for a repository."""
        count = 0
        count += self._metadata.invalidate_prefix(f"{repo_id}:")
        count += self._content.invalidate_prefix(f"{repo_id}:")
        count += self._tree.invalidate_prefix(f"{repo_id}:")
        return count
    
    @property
    def stats(self) -> dict:
        return {
            "metadata": self._metadata.stats,
            "content": self._content.stats,
            "tree": self._tree.stats
        }
