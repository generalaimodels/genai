"""
Advanced Terminal User Interface (TUI) Backend Microservice.

Production-grade terminal service with:
- Multi-shell support (PowerShell, Bash, CMD, WSL, Zsh, Fish)
- 6-layer connection stability architecture
- Redis-backed session persistence
- OAuth2/JWT authentication
- OpenTelemetry distributed tracing
- Prometheus metrics
"""

__version__ = "1.0.0"
__author__ = "Backend Lead Developer"

from .core import (
    PTYManager,
    SessionManager,
    WebSocketHandler,
    ReconnectionManager,
    OutputBufferManager,
    ConnectionMonitor,
    FlowController,
    optimize_socket,
)

__all__ = [
    "PTYManager",
    "SessionManager",
    "WebSocketHandler",
    "ReconnectionManager",
    "OutputBufferManager",
    "ConnectionMonitor",
    "FlowController",
    "optimize_socket",
]
