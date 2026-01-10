# API module exports
from .routes import router as main_router
from .editor_api import router as editor_router, init_preview_cache

__all__ = ["router", "init_preview_cache"]

# Combine routers
from fastapi import APIRouter
router = APIRouter()
router.include_router(main_router)
router.include_router(editor_router)
