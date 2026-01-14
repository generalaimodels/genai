"""
Production-Grade Git Pipeline API Router
=========================================
Optimized endpoints with:
- HTTP caching (ETag, Cache-Control)
- Paginated tree endpoint for large repos
- Lazy content loading
- Cache statistics endpoint
"""

from fastapi import APIRouter, HTTPException, Depends, Request, Response
from fastapi.responses import JSONResponse
from typing import List, Optional
from pathlib import Path
import json
import hashlib

from .schemas import RepoRequest, ProcessingJob, ApiResponse, RepoMetadata
from .service import GitService
from .db import GitRepoDatabase


router = APIRouter(prefix="/api/git", tags=["git-pipeline"])


def get_git_service(request: Request) -> GitService:
    """Dependency injection for GitService."""
    service = getattr(request.app.state, "git_service", None)
    if not service:
        raise HTTPException(status_code=500, detail="Git Service not initialized")
    return service


# ================================================================
# REPOSITORY MANAGEMENT
# ================================================================

@router.post("/process", response_model=ApiResponse)
async def process_repo(
    request: RepoRequest,
    service: GitService = Depends(get_git_service)
):
    """
    Submit GitHub repository for processing.
    
    Idempotent: Returns existing job ID if already processing/processed.
    Retries automatically for failed or zombie (0-file) repositories.
    """
    try:
        job_id = await service.submit_repo(request)
        return ApiResponse(success=True, data={"job_id": job_id})
    except Exception as e:
        return ApiResponse(success=False, error=str(e))


@router.get("/{repo_id}/status", response_model=ApiResponse)
async def get_repo_status(
    repo_id: str,
    service: GitService = Depends(get_git_service)
):
    """Get repository processing status."""
    repo = await service.db.get_repo(repo_id)
    if not repo:
        raise HTTPException(status_code=404, detail="Repository not found")
    return ApiResponse(success=True, data=repo)


# ================================================================
# TREE NAVIGATION - Optimized for 100K+ files
# ================================================================

@router.get("/{repo_id}/tree", response_model=ApiResponse)
async def get_repo_tree(
    repo_id: str,
    path: str = "",
    depth: int = 2,
    limit: int = 100,
    service: GitService = Depends(get_git_service)
):
    """
    Paginated file tree for repository navigation.
    
    Args:
        repo_id: Repository UUID
        path: Subtree root path (default: root)
        depth: Maximum tree depth to return (default: 2)
        limit: Maximum children per level (default: 100)
    
    Performance: O(1) cache hit, O(depth * limit) cache miss
    """
    try:
        if path:
            tree = await service.get_subtree(repo_id, path, depth, limit)
        else:
            tree = await service.get_tree(repo_id)
        return ApiResponse(success=True, data=tree)
    except Exception as e:
        return ApiResponse(success=False, error=str(e))


# ================================================================
# DOCUMENT RETRIEVAL - With HTTP Caching
# ================================================================

@router.get("/{repo_id}/doc")
async def get_document(
    repo_id: str,
    path: str,
    request: Request,
    service: GitService = Depends(get_git_service)
):
    """
    Fetch document content with HTTP caching.
    
    Supports:
    - ETag for conditional requests (304 Not Modified)
    - Cache-Control for browser caching
    - Fuzzy filename matching as fallback
    
    Performance:
    - O(1) with cache hit
    - O(1) with index lookup (cache miss)
    - O(N) fuzzy fallback
    """
    # Fetch document (uses internal LRU cache)
    doc = await service.get_cached_document(repo_id, path)
    
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Parse metadata JSON
    if doc.get('metadata') and isinstance(doc['metadata'], str):
        try:
            doc['metadata'] = json.loads(doc['metadata'])
        except json.JSONDecodeError:
            doc['metadata'] = {}
    
    # Generate ETag from content hash
    content_hash = doc.get('content_hash', '')
    etag = hashlib.md5(content_hash.encode()).hexdigest()
    
    # Check If-None-Match header
    if_none_match = request.headers.get("If-None-Match", "").strip('"')
    if if_none_match == etag:
        return Response(status_code=304)
    
    # Return with cache headers
    response_data = ApiResponse(success=True, data=doc)
    
    return JSONResponse(
        content=response_data.model_dump(),
        headers={
            "ETag": f'"{etag}"',
            "Cache-Control": "public, max-age=3600",  # 1 hour
            "X-Content-Hash": content_hash[:16]
        }
    )


@router.get("/{repo_id}/doc/meta")
async def get_document_metadata(
    repo_id: str,
    path: str,
    service: GitService = Depends(get_git_service)
):
    """
    Fetch document metadata only (no content).
    
    Optimized for file listings and previews where
    full content is not needed.
    """
    meta = await service.db.get_document_metadata(repo_id, path)
    
    if not meta:
        raise HTTPException(status_code=404, detail="Document not found")
    
    if meta.get('metadata') and isinstance(meta['metadata'], str):
        try:
            meta['metadata'] = json.loads(meta['metadata'])
        except json.JSONDecodeError:
            meta['metadata'] = {}
    
    return ApiResponse(success=True, data=meta)


# ================================================================
# CACHE MANAGEMENT
# ================================================================

@router.get("/cache/stats")
async def get_cache_stats(
    service: GitService = Depends(get_git_service)
):
    """
    Get cache statistics for monitoring.
    
    Returns hit rates, sizes, and capacity for each cache tier.
    """
    return ApiResponse(success=True, data=service.cache.stats)


@router.post("/{repo_id}/cache/invalidate")
async def invalidate_repo_cache(
    repo_id: str,
    service: GitService = Depends(get_git_service)
):
    """
    Invalidate all cached data for a repository.
    
    Use after manual refresh or re-processing.
    """
    count = service.cache.invalidate_repo(repo_id)
    return ApiResponse(success=True, data={"invalidated_entries": count})


# ================================================================
# HEALTH & DIAGNOSTICS
# ================================================================

@router.get("/health")
async def health_check(
    service: GitService = Depends(get_git_service)
):
    """
    Health check endpoint for load balancers.
    
    Returns database connection status and cache stats.
    """
    try:
        # Quick DB check
        await service.db.get_repo("test")
        db_healthy = True
    except Exception:
        db_healthy = False
    
    return {
        "status": "healthy" if db_healthy else "degraded",
        "database": "connected" if db_healthy else "disconnected",
        "cache": service.cache.stats
    }
