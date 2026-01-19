"""
Unit Tests for Session Manager.

Tests Redis-backed session state management with ACID transactions.

Test Coverage:
- Session creation with transactions
- Session retrieval
- Session updates (activity, sequence)
- Session listing and filtering
- Session deletion with cleanup
- TTL management
- Error handling
"""

import pytest
import time
from unittest.mock import AsyncMock, MagicMock, Mock

from terminal_service.core.session_manager import (
    SessionManager,
    TerminalSession,
    SessionNotFoundError
)


@pytest.fixture
def mock_redis():
    """Create mock Redis client."""
    redis = AsyncMock()
    
    # Mock pipeline as async context manager
    mock_pipe = AsyncMock()
    mock_pipe.__aenter__ = AsyncMock(return_value=mock_pipe)
    mock_pipe.__aexit__ = AsyncMock(return_value=None)
    mock_pipe.execute = AsyncMock(return_value=[])
    
    redis.pipeline = Mock(return_value=mock_pipe)
    
    return redis


@pytest.fixture
def session_manager(mock_redis):
    """Create session manager with mocked Redis."""
    return SessionManager(mock_redis, default_ttl=3600)


@pytest.fixture
def sample_session():
    """Create sample terminal session."""
    return TerminalSession(
        session_id="test-session-123",
        shell_type="bash",
        pid=12345,
        master_fd=10,
        rows=24,
        cols=80,
        created_at=time.time(),
        last_activity=time.time(),
        user_id="user-123",
        env={"TEST": "value"},
        output_sequence=0
    )


@pytest.mark.asyncio
class TestSessionManager:
    """Test suite for session manager."""
    
    async def test_create_session_success(self, session_manager, sample_session, mock_redis):
        """Test successful session creation."""
        mock_redis.exists.return_value = False
        
        result = await session_manager.create_session(sample_session)
        
        assert result.session_id == sample_session.session_id
        assert session_manager.stats['sessions_created'] == 1
        
        # Verify Redis operations
        mock_redis.exists.assert_called_once()
        mock_redis.pipeline.assert_called_once_with(transaction=True)
    
    async def test_create_session_already_exists(self, session_manager, sample_session, mock_redis):
        """Test creating session that already exists."""
        mock_redis.exists.return_value = True
        
        with pytest.raises(ValueError, match="already exists"):
            await session_manager.create_session(sample_session)
    
    async def test_get_session_success(self, session_manager, mock_redis):
        """Test retrieving existing session."""
        # Mock Redis response
        mock_redis.hgetall.return_value = {
            b"session_id": b"test-123",
            b"shell_type": b"bash",
            b"pid": b"12345",
            b"master_fd": b"10",
            b"rows": b"24",
            b"cols": b"80",
            b"created_at": b"1706432921.0",
            b"last_activity": b"1706432921.0",
            b"output_sequence": b"0"
        }
        
        session = await session_manager.get_session("test-123")
        
        assert session.session_id == "test-123"
        assert session.shell_type == "bash"
        assert session.pid == 12345
        assert session.rows == 24
    
    async def test_get_session_not_found(self, session_manager, mock_redis):
        """Test retrieving non-existent session."""
        mock_redis.hgetall.return_value = {}
        
        with pytest.raises(SessionNotFoundError):
            await session_manager.get_session("nonexistent")
    
    async def test_update_activity(self, session_manager, mock_redis):
        """Test updating session activity timestamp."""
        await session_manager.update_activity("test-123")
        
        mock_redis.hset.assert_called_once()
        assert session_manager.stats['total_activity_updates'] == 1
    
    async def test_update_output_sequence(self, session_manager, mock_redis):
        """Test updating output sequence number."""
        await session_manager.update_output_sequence("test-123", 42)
        
        mock_redis.hset.assert_called_once_with(
            "session:test-123",
            "output_sequence",
            42
        )
    
    async def test_list_all_sessions(self, session_manager, mock_redis):
        """Test listing all active sessions."""
        mock_redis.smembers.return_value = {b"session-1", b"session-2"}
        mock_redis.hgetall.side_effect = [
            {
                b"session_id": b"session-1",
                b"shell_type": b"bash",
                b"pid": b"100",
                b"master_fd": b"10",
                b"rows": b"24",
                b"cols": b"80",
                b"created_at": b"1706432921.0",
                b"last_activity": b"1706432921.0",
                b"output_sequence": b"0"
            },
            {
                b"session_id": b"session-2",
                b"shell_type": b"zsh",
                b"pid": b"200",
                b"master_fd": b"11",
                b"rows": b"30",
                b"cols": b"120",
                b"created_at": b"1706432922.0",
                b"last_activity": b"1706432922.0",
                b"output_sequence": b"5"
            }
        ]
        
        sessions = await session_manager.list_sessions()
        
        assert len(sessions) == 2
        assert sessions[0].session_id == "session-1"
        assert sessions[1].session_id == "session-2"
    
    async def test_list_sessions_by_user(self, session_manager, mock_redis):
        """Test listing sessions filtered by user."""
        mock_redis.smembers.return_value = {b"session-1"}
        mock_redis.hgetall.return_value = {
            b"session_id": b"session-1",
            b"shell_type": b"bash",
            b"pid": b"100",
            b"master_fd": b"10",
            b"rows": b"24",
            b"cols": b"80",
            b"created_at": b"1706432921.0",
            b"last_activity": b"1706432921.0",
            b"output_sequence": b"0",
            b"user_id": b"user-123"
        }
        
        sessions = await session_manager.list_sessions(user_id="user-123")
        
        assert len(sessions) == 1
        mock_redis.smembers.assert_called_with("user_sessions:user-123")
    
    async def test_delete_session_success(self, session_manager, mock_redis):
        """Test successful session deletion."""
        # Mock get_session
        mock_redis.hgetall.return_value = {
            b"session_id": b"test-123",
            b"shell_type": b"bash",
            b"pid": b"100",
            b"master_fd": b"10",
            b"rows": b"24",
            b"cols": b"80",
            b"created_at": b"1706432921.0",
            b"last_activity": b"1706432921.0",
            b"output_sequence": b"0",
            b"user_id": b"user-123"
        }
        
        result = await session_manager.delete_session("test-123")
        
        assert result is True
        assert session_manager.stats['sessions_deleted'] == 1
    
    async def test_delete_session_not_found(self, session_manager, mock_redis):
        """Test deleting non-existent session."""
        mock_redis.hgetall.return_value = {}
        
        result = await session_manager.delete_session("nonexistent")
        
        assert result is False
    
    async def test_extend_ttl(self, session_manager, mock_redis):
        """Test extending session TTL."""
        mock_redis.expire.return_value = 1
        
        result = await session_manager.extend_ttl("test-123", 7200)
        
        assert result is True
        mock_redis.expire.assert_called_once_with("session:test-123", 7200)
    
    async def test_get_stats(self, session_manager, mock_redis):
        """Test getting session manager statistics."""
        mock_redis.scard.return_value = 5
        
        stats = await session_manager.get_stats()
        
        assert stats['active_sessions'] == 5
        assert 'sessions_created' in stats
    
    async def test_cleanup_expired(self, session_manager, mock_redis):
        """Test cleanup of expired sessions."""
        # Mock session IDs in active set
        mock_redis.smembers.return_value = {b"session-1", b"session-2", b"session-3"}
        
        # Mock exists checks - session-2 expired
        async def mock_exists(key):
            if "session-2" in key:
                return False
            return True
        mock_redis.exists.side_effect = mock_exists
        
        cleaned = await session_manager.cleanup_expired()
        
        assert cleaned == 1
        # Verify session-2 was removed from active set
        assert mock_redis.srem.call_count == 1


class TestTerminalSession:
    """Test suite for TerminalSession model."""
    
    def test_to_dict_serialization(self):
        """Test session serialization to dictionary."""
        session = TerminalSession(
            session_id="test-123",
            shell_type="bash",
            pid=12345,
            master_fd=10,
            rows=24,
            cols=80,
            created_at=1706432921.0,
            last_activity=1706432922.0,
            user_id="user-123",
            env={"KEY": "value"},
            output_sequence=42
        )
        
        data = session.to_dict()
        
        assert data['session_id'] == "test-123"
        assert data['pid'] == 12345
        assert isinstance(data['env'], str)  # JSON serialized
    
    def test_from_dict_deserialization(self):
        """Test session deserialization from dictionary."""
        data = {
            "session_id": "test-123",
            "shell_type": "bash",
            "pid": "12345",
            "master_fd": "10",
            "rows": "24",
            "cols": "80",
            "created_at": "1706432921.0",
            "last_activity": "1706432922.0",
            "user_id": "user-123",
            "env": '{"KEY": "value"}',
            "output_sequence": "42"
        }
        
        session = TerminalSession.from_dict(data)
        
        assert session.session_id == "test-123"
        assert session.pid == 12345
        assert session.rows == 24
        assert isinstance(session.env, dict)
        assert session.env["KEY"] == "value"
