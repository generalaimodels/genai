"""
Terminal Service Configuration.

Environment-driven configuration using Pydantic Settings.
All sensitive values loaded from environment variables with validation.

Security Note:
- Never commit .env files to version control
- Use secrets management (Vault, AWS Secrets Manager) in production
- Rotate Redis passwords regularly

Author: Backend Lead Developer
"""

from pydantic_settings import BaseSettings
from typing import List, Optional


class TerminalConfig(BaseSettings):
    """
    Production-grade configuration for terminal service.
    
    Configuration Sources (priority order):
    1. Environment variables
    2. .env file
    3. Default values
    """
    
    # Redis Configuration
    redis_cluster_urls: List[str] = ["redis://localhost:6379"]
    redis_password: Optional[str] = None
    redis_db: int = 0
    session_ttl_seconds: int = 3600  # 1 hour
    output_buffer_size: int = 1500  # Messages per session
    
    # Security & Authentication
    oauth2_issuer_url: Optional[str] = None
    oauth2_audience: str = "terminal-service"
    jwt_algorithm: str = "RS256"
    jwt_public_key_path: Optional[str] = None
    
    # Performance Tuning
    max_concurrent_sessions_per_user: int = 10
    pty_buffer_size: int = 16384  # 16KB
    websocket_max_message_size: int = 65536  # 64KB
    websocket_ping_interval: float = 15.0  # seconds
    websocket_ping_timeout: float = 3.0  # seconds
    
    # Flow Control
    flow_control_window_size: int = 262144  # 256KB
    flow_control_pause_threshold: float = 0.8  # 80%
    
    # Observability
    otel_exporter_endpoint: Optional[str] = None
    otel_service_name: str = "terminal-service"
    prometheus_port: int = 9090
    log_level: str = "INFO"
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8080
    grpc_port: int = 50051
    workers: int = 1  # Single worker (asyncio event loop)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global configuration instance
config = TerminalConfig()
