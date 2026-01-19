"""
Observability Module - Prometheus Metrics and Health Monitoring.

Production-grade metrics collection for monitoring and alerting.

Metrics Collected:
- Session metrics (created, active, terminated)
- WebSocket metrics (connections, messages, latency)
- PTY metrics (process spawns, failures)
- Redis metrics (operations, errors)
- HTTP metrics (requests, response times)

Author: Backend Lead Developer
"""

from prometheus_client import Counter, Gauge, Histogram, Summary, Info
import time
from functools import wraps

# Session Metrics
sessions_created_total = Counter(
    'terminal_sessions_created_total',
    'Total number of terminal sessions created'
)

sessions_active = Gauge(
    'terminal_sessions_active',
    'Number of currently active terminal sessions'
)

sessions_terminated_total = Counter(
    'terminal_sessions_terminated_total',
    'Total number of terminal sessions terminated'
)

session_duration_seconds = Histogram(
    'terminal_session_duration_seconds',
    'Duration of terminal sessions in seconds',
    buckets=(60, 300, 600, 1800, 3600, 7200, 14400)  # 1m to 4h
)

# WebSocket Metrics
websocket_connections_total = Counter(
    'terminal_websocket_connections_total',
    'Total number of WebSocket connections'
)

websocket_active_connections = Gauge(
    'terminal_websocket_active_connections',
    'Number of active WebSocket connections'
)

websocket_messages_sent_total = Counter(
    'terminal_websocket_messages_sent_total',
    'Total number of WebSocket messages sent',
    ['message_type']
)

websocket_messages_received_total = Counter(
    'terminal_websocket_messages_received_total',
    'Total number of WebSocket messages received',
    ['message_type']
)

websocket_message_latency_seconds = Histogram(
    'terminal_websocket_message_latency_seconds',
    'WebSocket message latency in seconds',
    buckets=(.001, .005, .01, .025, .05, .1, .25, .5, 1.0)
)

websocket_reconnections_total = Counter(
    'terminal_websocket_reconnections_total',
    'Total number of WebSocket reconnections'
)

# PTY Metrics
pty_spawns_total = Counter(
    'terminal_pty_spawns_total',
    'Total number of PTY processes spawned',
    ['shell_type']
)

pty_failures_total = Counter(
    'terminal_pty_failures_total',
    'Total number of PTY spawn failures',
    ['shell_type', 'error_type']
)

pty_active_processes = Gauge(
    'terminal_pty_active_processes',
    'Number of active PTY processes'
)

# Redis Metrics
redis_operations_total = Counter(
    'terminal_redis_operations_total',
    'Total number of Redis operations',
    ['operation']
)

redis_errors_total = Counter(
    'terminal_redis_errors_total',
    'Total number of Redis errors',
    ['operation']
)

redis_latency_seconds = Histogram(
    'terminal_redis_latency_seconds',
    'Redis operation latency in seconds',
    buckets=(.001, .005, .01, .025, .05, .1)
)

# Connection Stability Metrics
connection_health_score = Gauge(
    'terminal_connection_health_score',
    'Connection health score (0.0-1.0)',
    ['session_id']
)

connection_rtt_milliseconds = Histogram(
    'terminal_connection_rtt_milliseconds',
    'Round-trip time in milliseconds',
    buckets=(5, 10, 25, 50, 100, 250, 500, 1000)
)

connection_packet_loss_rate = Gauge(
    'terminal_connection_packet_loss_rate',
    'Packet loss rate (0.0-1.0)',
    ['session_id']
)

# Flow Control Metrics
flow_control_pauses_total = Counter(
    'terminal_flow_control_pauses_total',
    'Total number of flow control PAUSE events'
)

flow_control_window_utilization = Gauge(
    'terminal_flow_control_window_utilization',
    'Flow control window utilization (0.0-1.0)',
    ['session_id']
)

# Output Buffer Metrics
output_buffer_messages_total = Counter(
    'terminal_output_buffer_messages_total',
    'Total messages buffered'
)

output_buffer_replays_total = Counter(
    'terminal_output_buffer_replays_total',
    'Total output buffer replays on reconnection'
)

output_buffer_size_bytes = Gauge(
    'terminal_output_buffer_size_bytes',
    'Output buffer size in bytes',
    ['session_id']
)

# Service Info
service_info = Info(
    'terminal_service',
    'Terminal service information'
)

service_info.info({
    'version': '1.0.0',
    'architecture': '6-layer-connection-stability'
})


# Metric Decorators

def track_redis_operation(operation_name: str):
    """Decorator to track Redis operations."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            redis_operations_total.labels(operation=operation_name).inc()
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                latency = time.time() - start_time
                redis_latency_seconds.observe(latency)
                return result
            except Exception as e:
                redis_errors_total.labels(operation=operation_name).inc()
                raise
        return wrapper
    return decorator


def track_websocket_message(message_type: str, direction: str = 'sent'):
    """Decorator to track WebSocket messages."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            result = await func(*args, **kwargs)
            
            latency = time.time() - start_time
            websocket_message_latency_seconds.observe(latency)
            
            if direction == 'sent':
                websocket_messages_sent_total.labels(message_type=message_type).inc()
            else:
                websocket_messages_received_total.labels(message_type=message_type).inc()
            
            return result
        return wrapper
    return decorator


# Helper Functions

def record_session_created():
    """Record session creation."""
    sessions_created_total.inc()
    sessions_active.inc()


def record_session_terminated(duration_seconds: float):
    """Record session termination."""
    sessions_terminated_total.inc()
    sessions_active.dec()
    session_duration_seconds.observe(duration_seconds)


def record_websocket_connection():
    """Record WebSocket connection."""
    websocket_connections_total.inc()
    websocket_active_connections.inc()


def record_websocket_disconnection():
    """Record WebSocket disconnection."""
    websocket_active_connections.dec()


def record_pty_spawn(shell_type: str, success: bool = True, error_type: str = None):
    """Record PTY spawn."""
    if success:
        pty_spawns_total.labels(shell_type=shell_type).inc()
        pty_active_processes.inc()
    else:
        pty_failures_total.labels(shell_type=shell_type, error_type=error_type or 'unknown').inc()


def record_pty_termination():
    """Record PTY process termination."""
    pty_active_processes.dec()


def update_connection_metrics(session_id: str, health_score: float, rtt_ms: float, packet_loss: float):
    """Update connection quality metrics."""
    connection_health_score.labels(session_id=session_id).set(health_score)
    connection_rtt_milliseconds.observe(rtt_ms)
    connection_packet_loss_rate.labels(session_id=session_id).set(packet_loss)


def record_flow_control_pause():
    """Record flow control pause event."""
    flow_control_pauses_total.inc()


def update_flow_control_window(session_id: str, utilization: float):
    """Update flow control window utilization."""
    flow_control_window_utilization.labels(session_id=session_id).set(utilization)


def record_output_buffer_message():
    """Record message added to output buffer."""
    output_buffer_messages_total.inc()


def record_output_replay():
    """Record output buffer replay."""
    output_buffer_replays_total.inc()


__all__ = [
    # Metrics
    'sessions_created_total',
    'sessions_active',
    'websocket_connections_total',
    'websocket_active_connections',
    
    # Decorators
    'track_redis_operation',
    'track_websocket_message',
    
    # Helper Functions
    'record_session_created',
    'record_session_terminated',
    'record_websocket_connection',
    'record_websocket_disconnection',
    'record_pty_spawn',
    'record_pty_termination',
    'update_connection_metrics',
    'record_flow_control_pause',
    'update_flow_control_window',
    'record_output_buffer_message',
    'record_output_replay',
]
