"""
FastAPI routes for document API.

Endpoints:
- GET /api/documents - List all documents
- GET /api/documents/{path} - Get single document
- GET /api/search - Full-text search
- GET /api/navigation - Navigation tree
- GET /api/health - Health check
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Any


router = APIRouter(prefix="/api", tags=["documents"])


# Pydantic models for API responses
class HeadingModel(BaseModel):
    """Document heading for TOC."""
    level: int
    text: str
    id: str


class DocumentMetadataModel(BaseModel):
    """Document metadata for listings."""
    path: str
    title: str
    description: str | None = None
    modified_at: datetime
    size_bytes: int
    heading_count: int


class DocumentContentModel(BaseModel):
    """Full document with parsed content."""
    metadata: DocumentMetadataModel
    content_html: str
    headings: list[HeadingModel]
    front_matter: dict[str, Any] = Field(default_factory=dict)


class NavigationNodeModel(BaseModel):
    """Navigation tree node."""
    name: str
    path: str | None = None
    is_directory: bool
    children: list["NavigationNodeModel"] = Field(default_factory=list)


class SearchMatchModel(BaseModel):
    """Individual search match."""
    text: str
    heading: str | None = None
    line: int


class SearchResultModel(BaseModel):
    """Search result item."""
    path: str
    title: str
    score: float
    matches: list[SearchMatchModel] = Field(default_factory=list)


class DocumentListResponse(BaseModel):
    """Response for document listing."""
    documents: list[DocumentMetadataModel]
    total: int


class SearchResponse(BaseModel):
    """Response for search query."""
    results: list[SearchResultModel]
    query: str
    total: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    document_count: int
    index_stats: dict[str, int]


# Enable recursive model for navigation
NavigationNodeModel.model_rebuild()


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(request: Request) -> DocumentListResponse:
    """
    List all available documents.
    
    Returns document metadata sorted by path.
    """
    scanner = request.app.state.scanner
    documents = await scanner.scan_all()
    
    doc_models = [
        DocumentMetadataModel(
            path=doc.path,
            title=doc.title,
            description=doc.description,
            modified_at=doc.modified_at,
            size_bytes=doc.size_bytes,
            heading_count=doc.heading_count
        )
        for doc in documents
    ]
    
    return DocumentListResponse(
        documents=doc_models,
        total=len(doc_models)
    )


@router.get("/documents/{path:path}", response_model=DocumentContentModel)
async def get_document(request: Request, path: str) -> DocumentContentModel:
    """
    Get a single document by path.
    
    Returns fully parsed document with HTML content.
    """
    scanner = request.app.state.scanner
    full_doc = await scanner.get_document(path)
    
    if full_doc is None:
        raise HTTPException(status_code=404, detail=f"Document not found: {path}")
    
    return DocumentContentModel(
        metadata=DocumentMetadataModel(
            path=full_doc.info.path,
            title=full_doc.info.title,
            description=full_doc.info.description,
            modified_at=full_doc.info.modified_at,
            size_bytes=full_doc.info.size_bytes,
            heading_count=full_doc.info.heading_count
        ),
        content_html=full_doc.parsed.content_html,
        headings=[
            HeadingModel(level=h.level, text=h.text, id=h.id)
            for h in full_doc.parsed.headings
        ],
        front_matter=full_doc.parsed.front_matter
    )


@router.get("/search", response_model=SearchResponse)
async def search_documents(
    request: Request,
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results")
) -> SearchResponse:
    """
    Full-text search across all documents.
    
    Supports multiple search terms.
    """
    search_engine = request.app.state.search_engine
    results = search_engine.search(q, limit=limit)
    
    result_models = [
        SearchResultModel(
            path=r.path,
            title=r.title,
            score=r.score,
            matches=[
                SearchMatchModel(text=m.text, heading=m.heading, line=m.line)
                for m in r.matches
            ]
        )
        for r in results
    ]
    
    return SearchResponse(
        results=result_models,
        query=q,
        total=len(result_models)
    )


@router.get("/navigation", response_model=NavigationNodeModel)
async def get_navigation(request: Request) -> NavigationNodeModel:
    """
    Get navigation tree for sidebar.
    
    Returns hierarchical structure matching directory layout.
    """
    scanner = request.app.state.scanner
    nav_tree = await scanner.build_navigation()
    
    def convert_node(node) -> NavigationNodeModel:
        return NavigationNodeModel(
            name=node.name,
            path=node.path,
            is_directory=node.is_directory,
            children=[convert_node(child) for child in node.children]
        )
    
    return convert_node(nav_tree)


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    """
    Health check endpoint.
    
    Returns server status and document count.
    """
    scanner = request.app.state.scanner
    search_engine = request.app.state.search_engine
    
    documents = await scanner.scan_all()
    
    return HealthResponse(
        status="healthy",
        document_count=len(documents),
        index_stats=search_engine.get_stats()
    )
