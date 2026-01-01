"""
FastAPI Application Entry Point.

SOTA Markdown Documentation Converter Backend.

Features:
- Async document scanning and parsing
- Full-text search with inverted index
- RESTful API for frontend integration
- CORS middleware for cross-origin requests
- Static file serving for development
"""

from __future__ import annotations

import os
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from core.parser import MarkdownParser
from core.scanner import DocumentScanner
from core.search import SearchEngine
from api.routes import router


# Configuration from environment
DATA_DIR = os.environ.get("DATA_DIR", "../data")
FRONTEND_DIR = os.environ.get("FRONTEND_DIR", "../frontend/dist")
DEBUG = os.environ.get("DEBUG", "false").lower() == "true"
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))


async def build_search_index(scanner: DocumentScanner, search_engine: SearchEngine) -> None:
    """Build search index from all documents."""
    documents = await scanner.scan_all()
    
    for doc_info in documents:
        full_doc = await scanner.get_document(doc_info.path)
        if full_doc:
            # Index with headings for section context
            headings = [
                (h.text, h.line) for h in full_doc.parsed.headings
            ]
            # Get raw content for indexing (from cache or re-read)
            # We index the title and parsed content text
            search_engine.index_document(
                path=doc_info.path,
                title=doc_info.title,
                content=full_doc.parsed.content_html,  # Index HTML for now, could strip tags
                headings=headings
            )
    
    print(f"Indexed {len(documents)} documents for search")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events for startup and shutdown."""
    # Startup: Initialize components
    print("Starting Markdown Documentation Server...")
    
    # Resolve data directory
    data_path = Path(DATA_DIR).resolve()
    if not data_path.exists():
        print(f"Warning: Data directory not found: {data_path}")
        # Create empty data directory
        data_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    parser = MarkdownParser()
    scanner = DocumentScanner(data_path, parser)
    search_engine = SearchEngine()
    
    # Store in app state
    app.state.parser = parser
    app.state.scanner = scanner
    app.state.search_engine = search_engine
    
    # Build search index
    await build_search_index(scanner, search_engine)
    
    print(f"Server ready. Data directory: {data_path}")
    
    yield
    
    # Shutdown: Cleanup
    print("Shutting down...")


# Create FastAPI application
app = FastAPI(
    title="Markdown Documentation Converter",
    description="SOTA Markdown to Documentation Website Converter API",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if DEBUG else None,
    redoc_url="/redoc" if DEBUG else None,
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(router)


# Serve frontend static files if directory exists
frontend_path = Path(FRONTEND_DIR).resolve()
if frontend_path.exists():
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc) if DEBUG else "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=DEBUG,
        log_level="info" if DEBUG else "warning"
    )
