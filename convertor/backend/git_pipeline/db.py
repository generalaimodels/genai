"""
Production-Grade Git Repository Database Layer
===============================================
Optimized for high-throughput document indexing with:
- Connection pooling for concurrent access
- Batch inserts with single transaction commit
- Covering indexes for O(1) lookups
- Lazy content loading (metadata-first pattern)
"""

import aiosqlite
import asyncio
import json
import time
from pathlib import Path
from typing import List, Optional, Dict, Any, AsyncIterator
from contextlib import asynccontextmanager

from .schemas import RepoMetadata, DocumentNode


class ConnectionPool:
    """
    Bounded async connection pool for aiosqlite.
    Prevents lock contention by limiting concurrent connections.
    
    Thread Safety: asyncio.Queue is coroutine-safe.
    Complexity: acquire/release O(1)
    """
    __slots__ = ('_db_path', '_pool_size', '_pool', '_initialized')
    
    def __init__(self, db_path: Path, pool_size: int = 4):
        if pool_size <= 0:
            raise ValueError("Pool size must be positive")
        self._db_path = db_path
        self._pool_size = pool_size
        self._pool: asyncio.Queue[aiosqlite.Connection] = asyncio.Queue(maxsize=pool_size)
        self._initialized = False
    
    async def initialize(self) -> None:
        """Pre-warm the pool with connections."""
        for _ in range(self._pool_size):
            conn = await self._create_connection()
            await self._pool.put(conn)
        self._initialized = True
    
    async def _create_connection(self) -> aiosqlite.Connection:
        conn = await aiosqlite.connect(self._db_path)
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA synchronous=NORMAL")
        await conn.execute("PRAGMA foreign_keys=ON")
        await conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
        await conn.execute("PRAGMA temp_store=MEMORY")
        return conn
    
    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[aiosqlite.Connection]:
        """Context manager for pool connection."""
        conn = await self._pool.get()
        try:
            yield conn
        finally:
            await self._pool.put(conn)
    
    async def close(self) -> None:
        """Close all pooled connections."""
        while not self._pool.empty():
            conn = await self._pool.get()
            await conn.close()


class GitRepoDatabase:
    """
    High-performance document storage with O(1) lookups.
    
    Architecture:
    - Single writer, multiple readers via WAL mode
    - Batch inserts eliminate per-document commit overhead
    - Covering indexes for hot query paths
    """
    
    def __init__(self, db_path: Path, pool_size: int = 4):
        self.db_path = db_path
        self._pool: Optional[ConnectionPool] = None
        self._pool_size = pool_size
        # Single connection for simple use cases (backwards compat)
        self._db: Optional[aiosqlite.Connection] = None

    async def connect(self) -> None:
        """Initialize connection pool and schema."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize pool
        self._pool = ConnectionPool(self.db_path, self._pool_size)
        await self._pool.initialize()
        
        # Also keep single connection for simple queries
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA synchronous=NORMAL")
        await self._db.execute("PRAGMA foreign_keys=ON")
        
        await self._init_schema()

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
        if self._db:
            await self._db.close()

    async def _init_schema(self) -> None:
        """Create tables and optimized indexes."""
        
        # Repositories table
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS repositories (
                id TEXT PRIMARY KEY,
                url TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                branch TEXT,
                status TEXT,
                total_files INTEGER DEFAULT 0,
                processed_files INTEGER DEFAULT 0,
                size_bytes INTEGER DEFAULT 0,
                last_synced REAL,
                created_at REAL
            )
        """)
        
        # Documents table with content separation for lazy loading
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                repo_id TEXT NOT NULL,
                path TEXT NOT NULL,
                content TEXT,
                content_html TEXT,
                metadata JSON,
                file_type TEXT,
                size_bytes INTEGER,
                content_hash TEXT,
                updated_at REAL,
                FOREIGN KEY (repo_id) REFERENCES repositories(id) ON DELETE CASCADE,
                UNIQUE(repo_id, path)
            )
        """)
        
        # Processing checkpoints for resumable jobs
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS processing_checkpoints (
                repo_id TEXT PRIMARY KEY,
                last_path TEXT,
                processed_count INTEGER DEFAULT 0,
                updated_at REAL,
                FOREIGN KEY (repo_id) REFERENCES repositories(id) ON DELETE CASCADE
            )
        """)

        # ============================================================
        # COVERING INDEXES - Optimized for production query patterns
        # ============================================================
        
        # Primary lookup: O(1) document by path
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_docs_repo_path ON documents(repo_id, path)"
        )
        
        # Tree construction: covering index avoids table lookup
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_tree_covering ON documents(repo_id, path, file_type, size_bytes)"
        )
        
        # Filename fuzzy search
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_docs_filename ON documents(repo_id, file_type)"
        )
        
        # Content deduplication
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_docs_content_hash ON documents(content_hash)"
        )
        
        # Status filtering for job queue
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_repo_status ON repositories(status)"
        )
        
        await self._db.commit()

    # ================================================================
    # REPOSITORY OPERATIONS
    # ================================================================

    async def upsert_repo(self, repo_data: Dict[str, Any]) -> None:
        fields = ", ".join(repo_data.keys())
        placeholders = ", ".join(["?" for _ in repo_data])
        updates = ", ".join([f"{k}=Excluded.{k}" for k in repo_data.keys()])
        
        sql = f"""
            INSERT INTO repositories ({fields})
            VALUES ({placeholders})
            ON CONFLICT(id) DO UPDATE SET {updates}
        """
        await self._db.execute(sql, list(repo_data.values()))
        await self._db.commit()

    async def get_repo(self, repo_id: str) -> Optional[Dict[str, Any]]:
        async with self._db.execute(
            "SELECT * FROM repositories WHERE id = ?", (repo_id,)
        ) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None
    
    async def get_repo_by_url(self, url: str) -> Optional[Dict[str, Any]]:
        async with self._db.execute(
            "SELECT * FROM repositories WHERE url = ?", (url,)
        ) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None

    async def update_status(self, repo_id: str, status: str, progress: int = 0) -> None:
        await self._db.execute(
            "UPDATE repositories SET status = ?, processed_files = ? WHERE id = ?",
            (status, progress, repo_id)
        )
        await self._db.commit()

    # ================================================================
    # DOCUMENT OPERATIONS - Optimized for batch processing
    # ================================================================

    async def batch_insert_documents(self, docs: List[Dict[str, Any]]) -> int:
        """
        Insert N documents in single transaction.
        Critical optimization: O(1) disk sync vs O(N) per-document.
        
        Args:
            docs: List of document dictionaries
            
        Returns:
            Number of documents inserted
        """
        if not docs:
            return 0
        
        # Use pooled connection for isolation
        async with self._pool.acquire() as conn:
            await conn.execute("BEGIN IMMEDIATE")
            
            try:
                for doc in docs:
                    fields = ", ".join(doc.keys())
                    placeholders = ", ".join(["?" for _ in doc])
                    
                    sql = f"""
                        INSERT INTO documents ({fields})
                        VALUES ({placeholders})
                        ON CONFLICT(repo_id, path) DO UPDATE SET
                            content=Excluded.content,
                            content_html=Excluded.content_html,
                            metadata=Excluded.metadata,
                            size_bytes=Excluded.size_bytes,
                            content_hash=Excluded.content_hash,
                            updated_at=Excluded.updated_at
                    """
                    await conn.execute(sql, list(doc.values()))
                
                await conn.commit()
                return len(docs)
                
            except Exception:
                await conn.rollback()
                raise

    async def insert_document(self, doc_data: Dict[str, Any]) -> None:
        """Single document insert (backwards compatibility)."""
        await self.batch_insert_documents([doc_data])

    async def get_document(self, repo_id: str, path: str) -> Optional[Dict[str, Any]]:
        """O(1) lookup via covering index."""
        async with self._db.execute(
            "SELECT * FROM documents WHERE repo_id = ? AND path = ?", 
            (repo_id, path)
        ) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None

    async def get_document_by_filename(self, repo_id: str, filename: str) -> Optional[Dict[str, Any]]:
        """Fallback fuzzy lookup by filename suffix."""
        async with self._db.execute(
            "SELECT * FROM documents WHERE repo_id = ? AND (path = ? OR path LIKE ?) LIMIT 1",
            (repo_id, filename, f"%/{filename}")
        ) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None

    async def get_document_metadata(self, repo_id: str, path: str) -> Optional[Dict[str, Any]]:
        """
        Lazy loading: fetch metadata only, no content.
        Reduces memory footprint for tree/listing operations.
        """
        async with self._db.execute(
            "SELECT id, repo_id, path, file_type, size_bytes, metadata, content_hash, updated_at "
            "FROM documents WHERE repo_id = ? AND path = ?",
            (repo_id, path)
        ) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None

    async def get_document_content(self, doc_id: int) -> Optional[Dict[str, str]]:
        """Fetch content separately for streaming/lazy load."""
        async with self._db.execute(
            "SELECT content, content_html FROM documents WHERE id = ?",
            (doc_id,)
        ) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None

    # ================================================================
    # TREE OPERATIONS - Optimized for large repos
    # ================================================================

    async def get_repo_files(self, repo_id: str) -> List[Dict[str, Any]]:
        """Return all file metadata for tree construction."""
        async with self._db.execute(
            "SELECT path, file_type, size_bytes FROM documents WHERE repo_id = ?",
            (repo_id,)
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def get_subtree_files(
        self, 
        repo_id: str, 
        path_prefix: str = "", 
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Paginated subtree query for on-demand loading.
        Uses covering index for zero table lookups.
        
        Args:
            repo_id: Repository ID
            path_prefix: Filter paths starting with this prefix
            limit: Maximum results to return
        """
        if path_prefix:
            sql = """
                SELECT path, file_type, size_bytes 
                FROM documents 
                WHERE repo_id = ? AND path LIKE ?
                ORDER BY path
                LIMIT ?
            """
            params = (repo_id, f"{path_prefix}%", limit)
        else:
            sql = """
                SELECT path, file_type, size_bytes 
                FROM documents 
                WHERE repo_id = ?
                ORDER BY path
                LIMIT ?
            """
            params = (repo_id, limit)
        
        async with self._db.execute(sql, params) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    # ================================================================
    # CHECKPOINT OPERATIONS - Resumable processing
    # ================================================================

    async def save_checkpoint(self, repo_id: str, last_path: str, count: int) -> None:
        await self._db.execute("""
            INSERT INTO processing_checkpoints (repo_id, last_path, processed_count, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(repo_id) DO UPDATE SET
                last_path = Excluded.last_path,
                processed_count = Excluded.processed_count,
                updated_at = Excluded.updated_at
        """, (repo_id, last_path, count, time.time()))
        await self._db.commit()

    async def get_checkpoint(self, repo_id: str) -> Optional[Dict[str, Any]]:
        async with self._db.execute(
            "SELECT * FROM processing_checkpoints WHERE repo_id = ?",
            (repo_id,)
        ) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None

    async def clear_checkpoint(self, repo_id: str) -> None:
        await self._db.execute(
            "DELETE FROM processing_checkpoints WHERE repo_id = ?",
            (repo_id,)
        )
        await self._db.commit()

    async def commit(self) -> None:
        """Explicit commit for external transaction control."""
        await self._db.commit()
