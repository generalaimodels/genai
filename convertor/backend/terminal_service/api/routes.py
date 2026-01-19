"""
REST API - Session Management Endpoints.

FastAPI router for terminal session CRUD operations.

Endpoints:
- POST /api/v1/sessions - Create new terminal session
- GET /api/v1/sessions - List active sessions
- GET /api/v1/sessions/{id} - Get session details
- DELETE /api/v1/sessions/{id} - Terminate session
- PUT /api/v1/sessions/{id}/resize - Resize terminal

Author: Backend Lead Developer
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import uuid
import time
import logging

from ..core.session_manager import SessionManager, TerminalSession, SessionNotFoundError
from ..core.pty_manager import PTYManager, ShellType

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["sessions"])


# Request/Response Models

class CreateSessionRequest(BaseModel):
    """Request to create new terminal session."""
    shell: str = Field(default="bash", description="Shell type (bash, powershell, cmd, etc.)")
    rows: int = Field(default=24, ge=1, le=500, description="Terminal rows")
    cols: int = Field(default=80, ge=1, le=500, description="Terminal columns")
    env: Optional[Dict[str, str]] = Field(default=None, description="Environment variables")
    user_id: Optional[str] = Field(default=None, description="User ID for multi-tenancy")


class CreateSessionResponse(BaseModel):
    """Response from session creation."""
    session_id: str
    websocket_url: str
    shell: str
    rows: int
    cols: int
    created_at: float


class SessionInfo(BaseModel):
    """Session metadata response."""
    session_id: str
    shell: str
    pid: int
    rows: int
    cols: int
    created_at: float
    last_activity: float
    uptime: float
    user_id: Optional[str] = None


class ResizeRequest(BaseModel):
    """Terminal resize request."""
    rows: int = Field(ge=1, le=500)
    cols: int = Field(ge=1, le=500)


# Global dependencies (injected at startup)
_pty_manager: Optional[PTYManager] = None
_session_manager: Optional[SessionManager] = None


def init_api(pty_manager: PTYManager, session_manager: SessionManager):
    """Initialize API with dependencies."""
    global _pty_manager, _session_manager
    _pty_manager = pty_manager
    _session_manager = session_manager


@router.post("/sessions", response_model=CreateSessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(request: CreateSessionRequest):
    """
    Create new terminal session.
    
    Process:
    1. Generate session ID
    2. Map shell name to ShellType enum
    3. Create PTY with specified shell
    4. Create session in Redis
    5. Return session details + WebSocket URL
    """
    try:
        # Generate session ID
        session_id = str(uuid.uuid4())
        
        # Map shell string to ShellType
        shell_map = {
            "bash": ShellType.BASH,
            "powershell": ShellType.POWERSHELL,
            "pwsh": ShellType.POWERSHELL,  # Use POWERSHELL for pwsh
            "cmd": ShellType.CMD,
            "wsl": ShellType.WSL,
            "zsh": ShellType.ZSH,
            "fish": ShellType.FISH,
            "sh": ShellType.SH,
        }
        
        shell_type = shell_map.get(request.shell.lower(), ShellType.BASH)
        
        # Create PTY session
        pty_session = await _pty_manager.create_session(
            session_id=session_id,
            shell_type=shell_type,
            env=request.env,
            rows=request.rows,
            cols=request.cols
        )
        
        # Create session metadata
        terminal_session = TerminalSession(
            session_id=session_id,
            shell_type=shell_type.value,
            pid=pty_session.pid,
            master_fd=pty_session.master_fd,
            rows=request.rows,
            cols=request.cols,
            created_at=pty_session.created_at,
            last_activity=time.time(),
            user_id=request.user_id,
            env=request.env,
            output_sequence=0
        )
        
        # Store in Redis
        await _session_manager.create_session(terminal_session)
        
        return CreateSessionResponse(
            session_id=session_id,
            websocket_url=f"/ws/terminal/{session_id}",
            shell=request.shell,
            rows=request.rows,
            cols=request.cols,
            created_at=terminal_session.created_at
        )
        
    except Exception as e:
        logger.error(f"Failed to create session: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create session: {str(e)}"
        )


@router.get("/sessions", response_model=List[SessionInfo])
async def list_sessions(user_id: Optional[str] = None):
    """
    List active terminal sessions.
    
    Query Parameters:
        user_id: Filter sessions by user (optional)
    """
    try:
        sessions = await _session_manager.list_sessions(user_id=user_id)
        
        return [
            SessionInfo(
                session_id=s.session_id,
                shell=s.shell_type,
                pid=s.pid,
                rows=s.rows,
                cols=s.cols,
                created_at=s.created_at,
                last_activity=s.last_activity,
                uptime=time.time() - s.created_at,
                user_id=s.user_id
            )
            for s in sessions
        ]
        
    except Exception as e:
        logger.error(f"Failed to list sessions: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list sessions: {str(e)}"
        )


@router.get("/sessions/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str):
    """Get terminal session details."""
    try:
        session = await _session_manager.get_session(session_id)
        
        return SessionInfo(
            session_id=session.session_id,
            shell=session.shell_type,
            pid=session.pid,
            rows=session.rows,
            cols=session.cols,
            created_at=session.created_at,
            last_activity=session.last_activity,
            uptime=time.time() - session.created_at,
            user_id=session.user_id
        )
        
    except SessionNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found"
        )
    except Exception as e:
        logger.error(f"Failed to get session {session_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get session: {str(e)}"
        )


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(session_id: str):
    """
    Terminate terminal session.
    
    Process:
    1. Terminate PTY process
    2. Delete session from Redis
    3. Clean up output buffer
    """
    try:
        # Terminate PTY
        terminated = await _pty_manager.terminate_session(session_id)
        
        # Delete from Redis
        deleted = await _session_manager.delete_session(session_id)
        
        if not (terminated or deleted):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session {session_id} not found"
            )
        
        logger.info(f"Deleted session {session_id}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete session {session_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete session: {str(e)}"
        )


@router.put("/sessions/{session_id}/resize", status_code=status.HTTP_200_OK)
async def resize_session(session_id: str, request: ResizeRequest):
    """
    Resize terminal session.
    
    Updates PTY window size via TIOCSWINSZ ioctl.
    """
    try:
        # Resize PTY
        await _pty_manager.resize_terminal(session_id, request.rows, request.cols)
        
        logger.info(f"Resized session {session_id} to {request.rows}x{request.cols}")
        
        return {"status": "resized", "rows": request.rows, "cols": request.cols}
        
    except Exception as e:
        logger.error(f"Failed to resize session {session_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to resize session: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """Health check endpoint for load balancers."""
    try:
        stats = await _session_manager.get_stats()
        return {
            "status": "healthy",
            "active_sessions": stats.get("active_sessions", 0),
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }
