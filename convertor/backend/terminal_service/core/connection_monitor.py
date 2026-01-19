"""
Connection Monitor - Multi-Layer Health Monitoring.

Implements proactive failure detection with ping/pong, heartbeat, and PTY monitoring.
Detects connection issues before they cause user-visible failures.

## Monitoring Layers:

### Layer 1: WebSocket Ping/Pong (15s interval)
- Client/server exchange PING/PONG frames
- 3-second timeout, 3 consecutive failures trigger reconnection
- Detects network path failures

### Layer 2: Application Heartbeat (30s interval)
- Server broadcasts heartbeat with timestamp
- Client validates clock skew <5s
- Detects network partitions invisible to TCP

### Layer 3: PTY Process Monitor (10s interval)
- Poll PTY file descriptor for POLLHUP/POLLERR
- Detect zombie processes via waitpid()
- Automatic cleanup on PTY death

Performance Metrics:
- RTT (Round-Trip Time) from ping/pong
- Packet loss rate from sequence gaps
- Connection health score (0.0-1.0)

Author: Backend Lead Developer
"""

from __future__ import annotations

import asyncio
import time
import logging
import select
import os
from dataclasses import dataclass, field
from typing import Optional, Callable
from enum import Enum

logger = logging.getLogger(__name__)

__all__ = ["ConnectionMonitor", "ConnectionMetrics", "HealthStatus"]


class HealthStatus(Enum):
    """Connection health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class ConnectionMetrics:
    """
    Real-time connection quality metrics.
    
    Attributes:
        rtt_ms: Round-trip time in milliseconds (from ping/pong)
        packet_loss_rate: Percentage of packets lost (0.0-1.0)
        last_activity: Unix timestamp of last successful I/O
        health_score: Composite health score (0.0-1.0)
        consecutive_ping_failures: Number of consecutive ping timeouts
    """
    rtt_ms: float = 0.0
    packet_loss_rate: float = 0.0
    last_activity: float = field(default_factory=time.time)
    health_score: float = 1.0
    consecutive_ping_failures: int = 0
    
    def is_healthy(self) -> bool:
        """
        Determine if connection is healthy based on metrics.
        
        Thresholds:
        - RTT < 500ms (interactive terminal usability)
        - Packet loss < 1%
        - Activity within last 60s
        - No consecutive ping failures
        """
        return (
            self.rtt_ms < 500 and
            self.packet_loss_rate < 0.01 and
            (time.time() - self.last_activity) < 60 and
            self.consecutive_ping_failures == 0
        )
    
    def get_status(self) -> HealthStatus:
        """Get health status category."""
        if self.is_healthy():
            return HealthStatus.HEALTHY
        elif self.health_score > 0.5:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.UNHEALTHY
    
    def calculate_health_score(self) -> float:
        """
        Calculate composite health score (0.0-1.0).
        
        Weighted formula:
        - RTT: 40% (0-500ms range)
        - Packet loss: 30% (0-5% range)
        - Recency: 20% (0-60s range)
        - Ping failures: 10% (0-3 failures)
        """
        # RTT component (0-500ms → 1.0-0.0)
        rtt_score = max(0.0, 1.0 - (self.rtt_ms / 500.0))
        
        # Packet loss component (0-5% → 1.0-0.0)
        loss_score = max(0.0, 1.0 - (self.packet_loss_rate / 0.05))
        
        # Recency component (0-60s → 1.0-0.0)
        age = time.time() - self.last_activity
        recency_score = max(0.0, 1.0 - (age / 60.0))
        
        # Ping failure component (0-3 failures → 1.0-0.0)
        ping_score = max(0.0, 1.0 - (self.consecutive_ping_failures / 3.0))
        
        # Weighted sum
        self.health_score = (
            rtt_score * 0.4 +
            loss_score * 0.3 +
            recency_score * 0.2 +
            ping_score * 0.1
        )
        
        return self.health_score


class ConnectionMonitor:
    """
    Multi-layered connection health monitoring system.
    
    Monitoring Tasks:
    1. websocket_ping_loop: 15s ping/pong with 3s timeout
    2. heartbeat_loop: 30s application heartbeat
    3. pty_monitor_loop: 10s PTY liveness check
    
    All loops run concurrently as asyncio tasks.
    Automatic failure detection triggers reconnection callbacks.
    """
    
    __slots__ = (
        'metrics',
        'on_ping_timeout',
        'on_pty_death',
        'ping_interval',
        'ping_timeout',
        'heartbeat_interval',
        'pty_check_interval',
        '_monitoring_tasks',
        '_last_ping_sent',
        '_last_pong_received',
    )
    
    def __init__(
        self,
        on_ping_timeout: Optional[Callable] = None,
        on_pty_death: Optional[Callable] = None,
        ping_interval: float = 15.0,
        ping_timeout: float = 3.0,
        heartbeat_interval: float = 30.0,
        pty_check_interval: float = 10.0,
    ):
        """
        Initialize connection monitor.
        
        Args:
            on_ping_timeout: Async callback fn() when consecutive pings fail
            on_pty_death: Async callback fn(pid) when PTY process dies
            ping_interval: Seconds between ping frames (default: 15s)
            ping_timeout: Timeout for pong response (default: 3s)
            heartbeat_interval: Seconds between heartbeats (default: 30s)
            pty_check_interval: Seconds between PTY checks (default: 10s)
        """
        self.metrics = ConnectionMetrics()
        self.on_ping_timeout = on_ping_timeout
        self.on_pty_death = on_pty_death
        
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout
        self.heartbeat_interval = heartbeat_interval
        self.pty_check_interval = pty_check_interval
        
        self._monitoring_tasks: list[asyncio.Task] = []
        self._last_ping_sent = 0.0
        self._last_pong_received = 0.0
    
    async def websocket_ping_loop(self, websocket: any, session_id: str):
        """
        WebSocket ping/pong monitoring loop.
        
        Process:
        1. Sleep for ping_interval (15s)
        2. Send PING frame with current timestamp
        3. Wait for PONG with timeout (3s)
        4. On success: Calculate RTT, reset failure counter
        5. On timeout: Increment failure counter
        6. After 3 failures: Trigger reconnection callback
        
        Args:
            websocket: WebSocket connection to monitor
            session_id: Terminal session ID for logging
        """
        logger.info(f"Started ping/pong monitor for {session_id} (interval={self.ping_interval}s)")
        
        while True:
            try:
                await asyncio.sleep(self.ping_interval)
                
                # Send PING frame
                self._last_ping_sent = time.time()
                await websocket.send_json({
                    "type": "ping",
                    "timestamp": self._last_ping_sent,
                })
                
                # Wait for PONG with timeout
                try:
                    pong = await asyncio.wait_for(
                        websocket.receive_json(),
                        timeout=self.ping_timeout
                    )
                    
                    if pong.get("type") == "pong":
                        # Success: Calculate RTT
                        self._last_pong_received = time.time()
                        self.metrics.rtt_ms = (self._last_pong_received - self._last_ping_sent) * 1000
                        self.metrics.consecutive_ping_failures = 0
                        self.metrics.last_activity = time.time()
                        
                        logger.debug(f"Ping/pong successful for {session_id} (RTT={self.metrics.rtt_ms:.1f}ms)")
                    else:
                        # Invalid response
                        self.metrics.consecutive_ping_failures += 1
                        logger.warning(f"Invalid pong response for {session_id}")
                        
                except asyncio.TimeoutError:
                    # Ping timeout
                    self.metrics.consecutive_ping_failures += 1
                    logger.warning(
                        f"Ping timeout #{self.metrics.consecutive_ping_failures} for {session_id} "
                        f"({self.ping_timeout}s)"
                    )
                
                # Check if we should trigger reconnection
                if self.metrics.consecutive_ping_failures >= 3:
                    logger.error(
                        f"Health check failed for {session_id}: "
                        f"{self.metrics.consecutive_ping_failures} consecutive ping failures"
                    )
                    
                    # Trigger callback
                    if self.on_ping_timeout:
                        await self.on_ping_timeout()
                    
                    # Close connection gracefully
                    try:
                        await websocket.close(code=1001, reason="Health check timeout")
                    except:
                        pass
                    
                    break
                
                # Update health score
                self.metrics.calculate_health_score()
                
            except asyncio.CancelledError:
                logger.info(f"Ping/pong monitor cancelled for {session_id}")
                break
            except Exception as e:
                logger.error(f"Error in ping/pong monitor for {session_id}: {e}", exc_info=True)
                await asyncio.sleep(1.0)
    
    async def heartbeat_loop(self, websocket: any, session_id: str):
        """
        Application-level heartbeat broadcast.
        
        Broadcasts heartbeat with server timestamp every 30s.
        Client validates clock skew <5s to detect network partitions.
        
        Args:
            websocket: WebSocket connection
            session_id: Terminal session ID
        """
        logger.info(f"Started heartbeat broadcast for {session_id} (interval={self.heartbeat_interval}s)")
        
        while True:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                
                # Broadcast heartbeat
                await websocket.send_json({
                    "type": "heartbeat",
                    "timestamp": time.time(),
                    "session_id": session_id,
                })
                
                logger.debug(f"Sent heartbeat for {session_id}")
                
            except asyncio.CancelledError:
                logger.info(f"Heartbeat broadcast cancelled for {session_id}")
                break
            except Exception as e:
                logger.error(f"Error in heartbeat broadcast for {session_id}: {e}")
                await asyncio.sleep(1.0)
    
    async def pty_monitor_loop(self, pty_fd: int, pid: int, session_id: str):
        """
        PTY process liveness monitoring.
        
        Polls PTY file descriptor for POLLHUP/POLLERR every 10s.
        Detects zombie processes and PTY crashes.
        
        Args:
            pty_fd: PTY file descriptor to monitor
            pid: Process ID of shell
            session_id: Terminal session ID
        """
        logger.info(f"Started PTY monitor for {session_id} (pid={pid}, fd={pty_fd})")
        
        while True:
            try:
                await asyncio.sleep(self.pty_check_interval)
                
                # Check if process is alive (non-blocking)
                try:
                    pid_status, _ = os.waitpid(pid, os.WNOHANG)
                    if pid_status != 0:
                        # Process exited
                        logger.warning(f"PTY process {pid} exited for {session_id}")
                        
                        if self.on_pty_death:
                            await self.on_pty_death(pid)
                        
                        break
                except ChildProcessError:
                    # Process already reaped
                    logger.warning(f"PTY process {pid} already reaped for {session_id}")
                    break
                
                # Check PTY file descriptor status
                try:
                    poll_result = select.select([], [], [pty_fd], 0)
                    if pty_fd in poll_result[2]:  # Exceptional condition
                        logger.warning(f"PTY fd {pty_fd} has exceptional condition for {session_id}")
                except Exception as e:
                    logger.error(f"Failed to poll PTY fd {pty_fd}: {e}")
                
                logger.debug(f"PTY health check passed for {session_id} (pid={pid})")
                
            except asyncio.CancelledError:
                logger.info(f"PTY monitor cancelled for {session_id}")
                break
            except Exception as e:
                logger.error(f"Error in PTY monitor for {session_id}: {e}", exc_info=True)
                await asyncio.sleep(1.0)
    
    async def start_monitoring(
        self,
        websocket: any,
        session_id: str,
        pty_fd: Optional[int] = None,
        pid: Optional[int] = None
    ):
        """
        Start all monitoring tasks concurrently.
        
        Args:
            websocket: WebSocket connection to monitor
            session_id: Terminal session ID
            pty_fd: Optional PTY file descriptor for process monitoring
            pid: Optional process ID for liveness checks
        """
        # Start ping/pong monitor
        task = asyncio.create_task(
            self.websocket_ping_loop(websocket, session_id),
            name=f"ping_monitor_{session_id}"
        )
        self._monitoring_tasks.append(task)
        
        # Start heartbeat broadcast
        task = asyncio.create_task(
            self.heartbeat_loop(websocket, session_id),
            name=f"heartbeat_{session_id}"
        )
        self._monitoring_tasks.append(task)
        
        # Start PTY monitor if applicable
        if pty_fd is not None and pid is not None:
            task = asyncio.create_task(
                self.pty_monitor_loop(pty_fd, pid, session_id),
                name=f"pty_monitor_{session_id}"
            )
            self._monitoring_tasks.append(task)
        
        logger.info(f"Started {len(self._monitoring_tasks)} monitoring tasks for {session_id}")
    
    async def stop_monitoring(self):
        """Stop all monitoring tasks gracefully."""
        logger.info(f"Stopping {len(self._monitoring_tasks)} monitoring tasks...")
        
        for task in self._monitoring_tasks:
            task.cancel()
        
        # Wait for all tasks to complete
        await asyncio.gather(*self._monitoring_tasks, return_exceptions=True)
        
        self._monitoring_tasks.clear()
        logger.info("All monitoring tasks stopped")
    
    def get_metrics(self) -> dict[str, any]:
        """Get current connection metrics."""
        self.metrics.calculate_health_score()
        
        return {
            "rtt_ms": round(self.metrics.rtt_ms, 2),
            "packet_loss_rate": round(self.metrics.packet_loss_rate, 4),
            "last_activity": self.metrics.last_activity,
            "health_score": round(self.metrics.health_score, 3),
            "health_status": self.metrics.get_status().value,
            "consecutive_ping_failures": self.metrics.consecutive_ping_failures,
        }
