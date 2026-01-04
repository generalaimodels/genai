/**
 * SOTA LRU Cache Implementation
 * 
 * Features:
 * - LRU (Least Recently Used) eviction policy
 * - Bounded memory with configurable max entries
 * - Per-entry TTL with lazy expiration
 * - O(1) get, set, delete operations
 * - Per-document invalidation (not full cache clear)
 * - Hit/miss rate tracking
 * - Memory usage monitoring
 * 
 * Algorithm:
 * - HashMap + Doubly Linked List for O(1) operations
 * - Most recently used at head, least recently used at tail
 * - On cache miss: evict tail if at capacity
 * - On cache hit: move entry to head
 */

interface CacheEntry<T> {
    key: string;
    value: T;
    timestamp: number;
    size: number; // Estimated size in bytes
    prev: CacheEntry<T> | null;
    next: CacheEntry<T> | null;
}

interface CacheStats {
    hits: number;
    misses: number;
    evictions: number;
    hitRate: number;
    size: number;
    maxSize: number;
    memoryUsage: number; // Estimated bytes
}

export class LRUCache<T = unknown> {
    private capacity: number;
    private ttl: number; // milliseconds
    private cache: Map<string, CacheEntry<T>>;
    private head: CacheEntry<T> | null = null;
    private tail: CacheEntry<T> | null = null;
    private hits = 0;
    private misses = 0;
    private evictions = 0;
    private totalMemory = 0; // Estimated bytes

    constructor(capacity = 100, ttlSeconds = 60) {
        this.capacity = capacity;
        this.ttl = ttlSeconds * 1000;
        this.cache = new Map();
    }

    /**
     * Get value from cache
     * O(1) complexity
     */
    get(key: string): T | null {
        const entry = this.cache.get(key);

        if (!entry) {
            this.misses++;
            return null;
        }

        // Check TTL expiration
        if (Date.now() - entry.timestamp > this.ttl) {
            this.delete(key);
            this.misses++;
            return null;
        }

        // Move to head (most recently used)
        this.moveToHead(entry);
        this.hits++;
        return entry.value;
    }

    /**
     * Set value in cache
     * O(1) complexity
     */
    set(key: string, value: T): void {
        // Update existing entry
        const existing = this.cache.get(key);
        if (existing) {
            this.totalMemory -= existing.size;
            existing.value = value;
            existing.timestamp = Date.now();
            existing.size = this.estimateSize(value);
            this.totalMemory += existing.size;
            this.moveToHead(existing);
            return;
        }

        // Create new entry
        const size = this.estimateSize(value);
        const entry: CacheEntry<T> = {
            key,
            value,
            timestamp: Date.now(),
            size,
            prev: null,
            next: null,
        };

        // Add to HashMap
        this.cache.set(key, entry);
        this.totalMemory += size;

        // Add to linked list head
        this.addToHead(entry);

        // Evict LRU if at capacity
        if (this.cache.size > this.capacity) {
            this.evictTail();
        }
    }

    /**
     * Delete specific entry
     * O(1) complexity
     */
    delete(key: string): boolean {
        const entry = this.cache.get(key);
        if (!entry) return false;

        this.removeEntry(entry);
        this.cache.delete(key);
        this.totalMemory -= entry.size;
        return true;
    }

    /**
     * Clear entire cache
     */
    clear(): void {
        this.cache.clear();
        this.head = null;
        this.tail = null;
        this.totalMemory = 0;
    }

    /**
     * Invalidate entries matching a pattern
     * Useful for invalidating all documents or specific paths
     */
    invalidatePattern(pattern: string | RegExp): number {
        const keysToDelete: string[] = [];
        const regex = typeof pattern === 'string' ? new RegExp(pattern) : pattern;

        for (const key of this.cache.keys()) {
            if (regex.test(key)) {
                keysToDelete.push(key);
            }
        }

        keysToDelete.forEach(key => this.delete(key));
        return keysToDelete.length;
    }

    /**
     * Get cache statistics
     */
    getStats(): CacheStats {
        const total = this.hits + this.misses;
        return {
            hits: this.hits,
            misses: this.misses,
            evictions: this.evictions,
            hitRate: total > 0 ? this.hits / total : 0,
            size: this.cache.size,
            maxSize: this.capacity,
            memoryUsage: this.totalMemory,
        };
    }

    /**
     * Check if key exists (without updating LRU order)
     */
    has(key: string): boolean {
        const entry = this.cache.get(key);
        if (!entry) return false;

        // Check TTL
        if (Date.now() - entry.timestamp > this.ttl) {
            this.delete(key);
            return false;
        }

        return true;
    }

    // ========================================
    // Private Helper Methods
    // ========================================

    private moveToHead(entry: CacheEntry<T>): void {
        if (entry === this.head) return;

        // Remove from current position
        this.removeEntry(entry);

        // Add to head
        this.addToHead(entry);
    }

    private addToHead(entry: CacheEntry<T>): void {
        entry.prev = null;
        entry.next = this.head;

        if (this.head) {
            this.head.prev = entry;
        }

        this.head = entry;

        if (!this.tail) {
            this.tail = entry;
        }
    }

    private removeEntry(entry: CacheEntry<T>): void {
        if (entry.prev) {
            entry.prev.next = entry.next;
        } else {
            this.head = entry.next;
        }

        if (entry.next) {
            entry.next.prev = entry.prev;
        } else {
            this.tail = entry.prev;
        }
    }

    private evictTail(): void {
        if (!this.tail) return;

        const key = this.tail.key;
        this.removeEntry(this.tail);
        this.cache.delete(key);
        this.totalMemory -= this.tail.size;
        this.evictions++;
    }

    /**
     * Estimate object size in bytes (rough approximation)
     */
    private estimateSize(value: T): number {
        try {
            const json = JSON.stringify(value);
            // UTF-8 encoding: ~2 bytes per character (conservative estimate)
            return json.length * 2;
        } catch {
            // Fallback for non-serializable objects
            return 1024; // 1KB default
        }
    }
}

export default LRUCache;
