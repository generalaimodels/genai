"""
Unit Tests for Reconnection Manager.

Tests exponential backoff, infinite retry logic, and state recovery.

Test Coverage:
- Exponential backoff calculation
- Jitter randomization
- State machine transitions
- Redis distributed locking
- Infinite retry loop
"""

import asyncio
import pytest
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from terminal_service.core.reconnection_manager import (
    ReconnectionManager,
    ReconnectionPolicy,
    ConnectionState,
    SessionExpiredError,
)


class TestReconnectionPolicy:
    """Test suite for reconnection backoff policy."""
    
    def test_initial_delay(self):
        """Test first retry has base delay."""
        policy = ReconnectionPolicy(base_delay=0.1)
        
        delay = policy.next_delay(attempt=0)
        
        # Should be ~100ms ±25ms jitter
        assert 0.075 <= delay <= 0.125
    
    def test_exponential_growth(self):
        """Test delay grows exponentially."""
        policy = ReconnectionPolicy(base_delay=0.1, jitter_factor=0.0)
        
        # attempt=0: 100ms
        assert policy.next_delay(0) == pytest.approx(0.1, rel=0.01)
        
        # attempt=1: 200ms
        assert policy.next_delay(1) == pytest.approx(0.2, rel=0.01)
        
        # attempt=2: 400ms
        assert policy.next_delay(2) == pytest.approx(0.4, rel=0.01)
        
        # attempt=3: 800ms
        assert policy.next_delay(3) == pytest.approx(0.8, rel=0.01)
    
    def test_max_delay_cap(self):
        """Test delay is capped at max_delay."""
        policy = ReconnectionPolicy(
            base_delay=0.1,
            max_delay=5.0,
            jitter_factor=0.0
        )
        
        # attempt=10: Would be 102.4s, but capped at 5s
        delay = policy.next_delay(10)
        assert delay == 5.0
    
    def test_jitter_prevents_thundering_herd(self):
        """Test jitter adds randomization to prevent thundering herd."""
        policy = ReconnectionPolicy(base_delay=1.0, jitter_factor=0.25)
        
        # Generate multiple delays for same attempt
        delays = [policy.next_delay(5) for _ in range(100)]
        
        # All should be different (due to randomization)
        assert len(set(delays)) > 50  # At least 50% unique
        
        # All should be within expected range
        # 2^5 = 32s, with jitter of ±25% 
        # Range should be 32 * (1 - 0.25) to 32 * (1 + 0.25) = 24s to 40s
        # But jitter uses random.uniform which can go slightly outside
        for delay in delays:
            assert 23.0 <= delay <= 41.0  # Slightly wider range for tolerance
    
    def test_custom_backoff_multiplier(self):
        """Test custom backoff multiplier."""
        policy = ReconnectionPolicy(
            base_delay=1.0,
            backoff_multiplier=3.0,  # Triple each time
            jitter_factor=0.0
        )
        
        assert policy.next_delay(0) == 1.0  # 1 * 3^0
        assert policy.next_delay(1) == 3.0  # 1 * 3^1
        assert policy.next_delay(2) == 9.0  # 1 * 3^2


@pytest.mark.asyncio
class TestReconnectionManager:
    """Test suite for reconnection manager."""
    
    async def test_initial_state(self):
        """Test manager starts in DISCONNECTED state."""
        manager = ReconnectionManager()
        
        assert manager.state == ConnectionState.DISCONNECTED
        assert manager.attempt_count == 0
        assert manager.last_seq == 0
    
    async def test_successful_reconnection(self):
        """Test successful reconnection flow."""
        # Mock dependencies
        mock_redis = AsyncMock()
        mock_redis.set.return_value = True  # Lock acquired
        mock_redis.exists.return_value = True  # Session exists
        mock_redis.delete.return_value = True
        
        mock_ws_factory = AsyncMock()
        mock_ws = Mock()
        mock_ws_factory.return_value = mock_ws
        
        # Create manager with fast policy
        policy = ReconnectionPolicy(base_delay=0.01, max_delay=0.05)
        manager = ReconnectionManager(policy=policy)
        
        # Attempt reconnection
        result = await manager.handle_disconnect(
            session_id="test123",
            last_seq=100,
            redis_client=mock_redis,
            websocket_factory=mock_ws_factory
        )
        
        # Verify successful reconnection
        assert result == mock_ws
        assert manager.state == ConnectionState.CONNECTED
        assert manager.attempt_count == 0  # Reset on success
    
    async def test_session_expired_error(self):
        """Test behavior when session no longer exists."""
        mock_redis = AsyncMock()
        mock_redis.set.return_value = True  # Lock acquired
        mock_redis.exists.return_value = False  # Session does NOT exist
        
        manager = ReconnectionManager()
        
        with pytest.raises(SessionExpiredError):
            await manager.handle_disconnect(
                session_id="expired",
                last_seq=0,
                redis_client=mock_redis,
                websocket_factory=AsyncMock()
            )
        
        assert manager.state == ConnectionState.DISCONNECTED
    
    async def test_retry_on_failure(self):
        """Test infinite retry loop on transient failures."""
        mock_redis = AsyncMock()
        mock_redis.set.return_value = True
        mock_redis.exists.return_value = True
        mock_redis.delete.return_value = True
        
        mock_ws_factory = AsyncMock()
        # Fail twice, then succeed
        mock_ws_factory.side_effect = [
            ConnectionError("Network error"),
            ConnectionError("Still failing"),
            Mock()  # Success on 3rd attempt
        ]
        
        policy = ReconnectionPolicy(base_delay=0.01, max_delay=0.02)
        manager = ReconnectionManager(policy=policy)
        
        result = await manager.handle_disconnect(
            "test",
            0,
            mock_redis,
            mock_ws_factory
        )
        
        # Should eventually succeed
        assert result is not None
        assert manager.state == ConnectionState.CONNECTED
    
    async def test_distributed_lock_prevents_duplicate_reconnections(self):
        """Test Redis lock prevents multiple reconnection attempts."""
        mock_redis = AsyncMock()
        mock_redis.set.return_value = False  # Lock NOT acquired
        
        policy = ReconnectionPolicy(base_delay=0.01, max_delay=0.02)
        manager = ReconnectionManager(policy=policy)
        
        # Start reconnection task
        task = asyncio.create_task(
            manager.handle_disconnect(
                "test",
                0,
                mock_redis,
                AsyncMock()
            )
        )
        
        # Wait a bit to allow lock attempts
        await asyncio.sleep(0.1)
        
        # Cancel the task
        task.cancel()
        
        try:
            await task
        except asyncio.CancelledError:
            pass
        
        # Verify multiple lock attempts were made
        assert mock_redis.set.call_count > 1
    
    async def test_reconnection_callback(self):
        """Test on_reconnect callback is invoked."""
        callback = AsyncMock()
        
        mock_redis = AsyncMock()
        mock_redis.set.return_value = True
        mock_redis.exists.return_value = True
        mock_redis.delete.return_value = True
        
        mock_ws = Mock()
        mock_ws_factory = AsyncMock(return_value=mock_ws)
        
        policy = ReconnectionPolicy(base_delay=0.01)
        manager = ReconnectionManager(policy=policy, on_reconnect=callback)
        
        await manager.handle_disconnect(
            "test",
            42,
            mock_redis,
            mock_ws_factory
        )
        
        # Verify callback was called with correct arguments
        callback.assert_called_once_with("test", 42, mock_ws)
    
    def test_get_stats(self):
        """Test statistics retrieval."""
        manager = ReconnectionManager()
        manager.session_id = "test123"
        manager.attempt_count = 5
        manager.last_seq = 999
        manager.state = ConnectionState.RECONNECTING
        
        stats = manager.get_stats()
        
        assert stats["session_id"] == "test123"
        assert stats["attempt_count"] == 5
        assert stats["last_sequence"] == 999
        assert stats["state"] == "reconnecting"
