"""
Advanced Terminal Service - Main Application.

Production-grade FastAPI application for terminal service with:
- REST API for session management
- WebSocket endpoint for real-time terminal I/O
- Redis-backed session persistence
- 6-layer connection stability architecture
- Health monitoring and metrics

Author: Backend Lead Developer
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
import redis.asyncio as redis

from .config import TerminalConfig
from .core.pty_manager import PTYManager
from .core.session_manager import SessionManager
from .core.output_buffer_manager import OutputBufferManager
from .core.websocket_handler import WebSocketHandler
from .api import router, init_api

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Global state
app_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Startup:
    - Load configuration
    - Connect to Redis
    - Initialize managers
    - Start background tasks
    
    Shutdown:
    - Cleanup sessions
    - Close Redis connections
    """
    logger.info("Starting Terminal Service...")
    
    # Load configuration
    config = TerminalConfig()
    app_state['config'] = config
    
    # Connect to Redis
    redis_client = redis.from_url(
        config.redis_cluster_urls[0],  # Use first URL for now
        password=config.redis_password,
        decode_responses=False  # We handle encoding
    )
    app_state['redis'] = redis_client
    
    # Initialize managers
    pty_manager = PTYManager()
    session_manager = SessionManager(redis_client, default_ttl=config.session_ttl_seconds)
    output_buffer = OutputBufferManager(redis_client)
    
    app_state['pty_manager'] = pty_manager
    app_state['session_manager'] = session_manager
    app_state['output_buffer'] = output_buffer
    
    # Initialize API with dependencies
    init_api(pty_manager, session_manager)
    
    # Start background cleanup task
    async def cleanup_loop():
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                cleaned = await session_manager.cleanup_expired()
                if cleaned > 0:
                    logger.info(f"Cleaned up {cleaned} expired sessions")
            except Exception as e:
                logger.error(f"Cleanup error: {e}", exc_info=True)
    
    cleanup_task = asyncio.create_task(cleanup_loop())
    app_state['cleanup_task'] = cleanup_task
    
    logger.info("Terminal Service started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Terminal Service...")
    
    # Cancel cleanup task
    cleanup_task.cancel()
    
    # Close Redis
    await redis_client.close()
    
    logger.info("Terminal Service shut down")


# Create FastAPI app
app = FastAPI(
    title="Advanced Terminal Service",
    description="Production-grade terminal-as-a-service with 6-layer connection stability",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include REST API router
app.include_router(router)


@app.websocket("/ws/terminal/{session_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    session_id: str,
    recover_from: int = Query(default=None, description="Sequence to recover from")
):
    """
    WebSocket endpoint for terminal I/O.
    
    Path Parameters:
        session_id: Terminal session ID
    
    Query Parameters:
        recover_from: Sequence number to replay from (for reconnection)
    
    Message Protocol:
        Client → Server:
        - {"type": "input", "data": "base64_encoded_input"}
        - {"type": "resize", "rows": 30, "cols": 120}
        - {"type": "ping", "timestamp": 1234567890.0}
        - {"type": "ack", "bytes": 1024}
        
        Server → Client:
        - {"type": "output", "data": "base64_encoded_output", "seq": 123}
        - {"type": "pong", "timestamp": 1234567890.0}
        - {"type": "heartbeat", "timestamp": 1234567890.0}
        - {"type": "flow_pause"}
        - {"type": "session_ended", "reason": "..."}
    """
    handler = WebSocketHandler(
        pty_manager=app_state['pty_manager'],
        session_manager=app_state['session_manager'],
        output_buffer=app_state['output_buffer']
    )
    
    await handler.handle_connection(
        websocket,
        session_id,
        recover_from_seq=recover_from
    )


@app.get("/")
async def root():
    """Root endpoint with service information."""
    stats = await app_state['session_manager'].get_stats()
    
    return {
        "service": "Advanced Terminal Service",
        "version": "1.0.0",
        "status": "running",
        "active_sessions": stats.get("active_sessions", 0),
        "endpoints": {
            "rest_api": "/api/v1",
            "websocket": "/ws/terminal/{session_id}",
            "health": "/api/v1/health",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "terminal_service.main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )
