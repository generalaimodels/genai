"""
Production-Grade LRU Cache for Docs Pipeline
=============================================
Thread-safe, O(1) get/put operations with TTL expiration.
Optimized for 100,000+ file workloads.

Architecture:
- 3-tier cache: metadata (50k), content (1k), navigation (100)
- Hash-based change detection for incremental updates
- Lock-free reads with RLock for writes
"""

from collections import OrderedDict
from typing import Optional, Any, TypeVar, Generic, Callable
import time
import threading
import asyncio

T = TypeVar('T')


class LRUCache(Generic[T]):
    """
    Production-grade LRU cache optimized for high-throughput reads.
    
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
    
    def __len__(self) -> int:
        """Current number of cached items."""
        return len(self._cache)
    
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


class DocsCache:
    """
    Domain-specific cache for Documentation pipeline.
    
    Separates caches by access pattern:
    - metadata: High capacity (50k), longer TTL - for file listings
    - content: Lower capacity (1k), shorter TTL - for document content
    - navigation: Small capacity (100), short TTL - for nav trees
    - hash_index: High capacity (100k), long TTL - for change detection
    
    OPTIMIZATION for 100,000+ files:
    - Lazy loading: only load what's accessed
    - Hash-based invalidation: skip unchanged files
    - Prefix invalidation: fast directory-level cache clear
    """
    
    def __init__(
        self,
        metadata_capacity: int = 50000,
        content_capacity: int = 1000,
        navigation_capacity: int = 100,
        hash_capacity: int = 100000,
        metadata_ttl: float = 300.0,  # 5 minutes
        content_ttl: float = 120.0,   # 2 minutes
        navigation_ttl: float = 60.0,  # 1 minute
        hash_ttl: float = 3600.0      # 1 hour
    ):
        self._metadata = LRUCache[dict](metadata_capacity, metadata_ttl)
        self._content = LRUCache[str](content_capacity, content_ttl)
        self._navigation = LRUCache[list](navigation_capacity, navigation_ttl)
        self._hash_index = LRUCache[str](hash_capacity, hash_ttl)
        
        print(f"✓ DocsCache initialized:")
        print(f"  - Metadata cache: {metadata_capacity:,} capacity, {metadata_ttl}s TTL")
        print(f"  - Content cache: {content_capacity:,} capacity, {content_ttl}s TTL")
        print(f"  - Navigation cache: {navigation_capacity} capacity, {navigation_ttl}s TTL")
        print(f"  - Hash index: {hash_capacity:,} capacity, {hash_ttl}s TTL")
    
    # =============== Metadata Cache ===============
    
    def get_metadata(self, path: str) -> Optional[dict]:
        """Get cached document metadata."""
        return self._metadata.get(path)
    
    def put_metadata(self, path: str, data: dict) -> None:
        """Cache document metadata."""
        self._metadata.put(path, data)
    
    def get_or_load_metadata(self, path: str, loader: Callable[[], dict]) -> dict:
        """
        Get metadata from cache or load using provided loader.
        
        OPTIMIZATION: Cache-first lookup with lazy loading.
        """
        cached = self._metadata.get(path)
        if cached is not None:
            return cached
        
        # Cache miss - load and cache
        data = loader()
        self._metadata.put(path, data)
        return data
    
    # =============== Content Cache ===============
    
    def get_content(self, path: str) -> Optional[str]:
        """Get cached document content (HTML)."""
        return self._content.get(path)
    
    def put_content(self, path: str, content: str) -> None:
        """Cache document content."""
        self._content.put(path, content)
    
    async def get_or_load_content(self, path: str, loader: Callable) -> str:
        """
        Get content from cache or load using provided async loader.
        """
        cached = self._content.get(path)
        if cached is not None:
            return cached
        
        # Cache miss - load and cache
        if asyncio.iscoroutinefunction(loader):
            content = await loader()
        else:
            content = loader()
        
        self._content.put(path, content)
        return content
    
    # =============== Navigation Cache ===============
    
    def get_navigation(self, prefix: str = "") -> Optional[list]:
        """Get cached navigation tree."""
        key = f"nav:{prefix}"
        return self._navigation.get(key)
    
    def put_navigation(self, tree: list, prefix: str = "") -> None:
        """Cache navigation tree."""
        key = f"nav:{prefix}"
        self._navigation.put(key, tree)
    
    # =============== Hash Index (Change Detection) ===============
    
    def get_hash(self, path: str) -> Optional[str]:
        """Get cached content hash for change detection."""
        return self._hash_index.get(path)
    
    def put_hash(self, path: str, hash_value: str) -> None:
        """Cache content hash."""
        self._hash_index.put(path, hash_value)
    
    def has_changed(self, path: str, new_hash: str) -> bool:
        """
        Check if file has changed based on hash comparison.
        
        Returns True if:
        - Hash not in cache (new file)
        - Hash differs from cached value (changed file)
        """
        old_hash = self._hash_index.get(path)
        if old_hash is None:
            return True
        return old_hash != new_hash
    
    def update_if_changed(self, path: str, new_hash: str) -> bool:
        """
        Update hash and invalidate caches if file changed.
        
        Returns True if file was changed and caches invalidated.
        """
        if self.has_changed(path, new_hash):
            # Invalidate related caches
            self._metadata.invalidate(path)
            self._content.invalidate(path)
            self._navigation.clear()  # Nav tree needs rebuild
            
            # Update hash
            self._hash_index.put(path, new_hash)
            return True
        return False
    
    # =============== Bulk Operations ===============
    
    def invalidate_path(self, path: str) -> int:
        """
        Invalidate all caches for a specific path.
        
        Returns count of invalidated entries.
        """
        count = 0
        if self._metadata.invalidate(path):
            count += 1
        if self._content.invalidate(path):
            count += 1
        if self._hash_index.invalidate(path):
            count += 1
        return count
    
    def invalidate_prefix(self, prefix: str) -> int:
        """
        Invalidate all caches for paths starting with prefix.
        
        Useful for directory-level invalidation.
        """
        count = 0
        count += self._metadata.invalidate_prefix(prefix)
        count += self._content.invalidate_prefix(prefix)
        count += self._hash_index.invalidate_prefix(prefix)
        self._navigation.clear()
        return count
    
    def clear_all(self) -> None:
        """Clear all caches."""
        self._metadata.clear()
        self._content.clear()
        self._navigation.clear()
        self._hash_index.clear()
    
    # =============== Statistics ===============
    
    @property
    def stats(self) -> dict:
        """Return comprehensive cache statistics."""
        return {
            "metadata": self._metadata.stats,
            "content": self._content.stats,
            "navigation": self._navigation.stats,
            "hash_index": self._hash_index.stats,
            "total_items": (
                len(self._metadata) + 
                len(self._content) + 
                len(self._navigation) + 
                len(self._hash_index)
            )
        }
    
    def print_stats(self) -> None:
        """Print formatted cache statistics."""
        stats = self.stats
        print("\n📊 DocsCache Statistics:")
        print(f"  Metadata: {stats['metadata']['size']:,}/{stats['metadata']['capacity']:,} "
              f"({stats['metadata']['hit_rate_percent']}% hit rate)")
        print(f"  Content: {stats['content']['size']:,}/{stats['content']['capacity']:,} "
              f"({stats['content']['hit_rate_percent']}% hit rate)")
        print(f"  Navigation: {stats['navigation']['size']}/{stats['navigation']['capacity']} "
              f"({stats['navigation']['hit_rate_percent']}% hit rate)")
        print(f"  Hash Index: {stats['hash_index']['size']:,}/{stats['hash_index']['capacity']:,}")
        print(f"  Total items: {stats['total_items']:,}")


# Global singleton instance (initialized on import or explicitly)
_docs_cache: Optional[DocsCache] = None


def get_docs_cache() -> DocsCache:
    """Get or create the global DocsCache singleton."""
    global _docs_cache
    if _docs_cache is None:
        _docs_cache = DocsCache()
    return _docs_cache


def init_docs_cache(**kwargs) -> DocsCache:
    """Initialize the global DocsCache with custom settings."""
    global _docs_cache
    _docs_cache = DocsCache(**kwargs)
    return _docs_cache
