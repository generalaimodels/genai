"""
Unit Tests for Output Buffer Manager.

Tests Redis Streams circular buffer for guaranteed message delivery.

Test Coverage:
- Message appending
- Circular buffer trimming
- Range queries for replay
- Latest message retrieval
- Buffer statistics
"""

import asyncio
import base64
import pytest
from unittest.mock import AsyncMock, Mock, patch

from terminal_service.core.output_buffer_manager import (
    OutputBufferManager,
    OutputMessage,
)


@pytest.mark.asyncio
class TestOutputBufferManager:
    """Test suite for output buffer manager."""
    
    async def test_append_message(self):
        """Test appending message to buffer."""
        mock_redis = AsyncMock()
        mock_redis.xadd.return_value = b"1706432921000-0"
        mock_redis.expire.return_value = True
        
        manager = OutputBufferManager(mock_redis)
        
        data = b"Hello, terminal!"
        seq = await manager.append("session123", data, msg_type="output")
        
        # Verify sequence number extracted correctly
        assert seq == 1706432921000
        
        # Verify Redis XADD called correctly
        mock_redis.xadd.assert_called_once()
        call_args = mock_redis.xadd.call_args
        
        assert call_args[0][0] == "output:session123"  # Stream key
        assert "data" in call_args[0][1]  # Fields contain data
        assert call_args[1]["maxlen"] == 1500  # Max buffer size
        assert call_args[1]["approximate"] is True  # Performance optimization
    
    async def test_circular_buffer_trimming(self):
        """Test buffer automatically trims to max size."""
        mock_redis = AsyncMock()
        mock_redis.xadd.return_value = b"1000-0"
        
        manager = OutputBufferManager(mock_redis, max_buffer_size=100)
        
        await manager.append("test", b"data")
        
        # Verify MAXLEN parameter passed
        call_args = mock_redis.xadd.call_args
        assert call_args[1]["maxlen"] == 100
    
    async def test_get_range_for_replay(self):
        """Test retrieving range of messages for reconnection replay."""
        mock_redis = AsyncMock()
        
        # Simulate Redis Stream entries
        mock_redis.xrange.return_value = [
            (b"1000-0", {b"ts": b"1706432921.0", b"data": base64.b64encode(b"msg1"), b"type": b"output"}),
            (b"1001-0", {b"ts": b"1706432922.0", b"data": base64.b64encode(b"msg2"), b"type": b"output"}),
            (b"1002-0", {b"ts": b"1706432923.0", b"data": base64.b64encode(b"msg3"), b"type": b"output"}),
        ]
        
        manager = OutputBufferManager(mock_redis)
        
        messages = await manager.get_range("session123", start_seq=1000)
        
        # Verify correct number of messages
        assert len(messages) == 3
        
        # Verify message content
        assert messages[0].sequence == 1000
        assert messages[0].data == b"msg1"
        assert messages[1].sequence == 1001
        assert messages[2].sequence == 1002
        
        # Verify Redis XRANGE called correctly
        mock_redis.xrange.assert_called_once_with(
            "output:session123",
            min="1000-0",
            max="+",
            count=10000
        )
    
    async def test_get_range_with_end_sequence(self):
        """Test retrieving bounded range of messages."""
        mock_redis = AsyncMock()
        mock_redis.xrange.return_value = []
        
        manager = OutputBufferManager(mock_redis)
        
        await manager.get_range("test", start_seq=100, end_seq=200)
        
        # Verify end sequence used
        call_args = mock_redis.xrange.call_args
        assert call_args[1]["max"] == "200-9999"
    
    async def test_get_latest_messages(self):
        """Test retrieving most recent messages."""
        mock_redis = AsyncMock()
        mock_redis.xrevrange.return_value = [
            (b"2000-0", {b"ts": b"1706432925.0", b"data": base64.b64encode(b"latest"), b"type": b"output"}),
        ]
        
        manager = OutputBufferManager(mock_redis)
        
        messages = await manager.get_latest("session123", count=1)
        
        assert len(messages) == 1
        assert messages[0].sequence == 2000
        assert messages[0].data == b"latest"
        
        # Verify XREVRANGE used (reverse order)
        mock_redis.xrevrange.assert_called_once()
    
    async def test_clear_buffer(self):
        """Test clearing all buffered output."""
        mock_redis = AsyncMock()
        mock_redis.delete.return_value = 1  # 1 key deleted
        
        manager = OutputBufferManager(mock_redis)
        
        result = await manager.clear("session123")
        
        assert result is True
        mock_redis.delete.assert_called_once_with("output:session123")
    
    async def test_get_stats(self):
        """Test retrieving buffer statistics."""
        mock_redis = AsyncMock()
        mock_redis.xlen.return_value = 500
        mock_redis.ttl.return_value = 3600
        mock_redis.xrange.return_value = [(b"1000-0", {})]
        mock_redis.xrevrange.return_value = [(b"1499-0", {})]
        
        manager = OutputBufferManager(mock_redis)
        
        stats = await manager.get_stats("session123")
        
        assert stats["message_count"] == 500
        assert stats["ttl_seconds"] == 3600
        assert stats["first_sequence"] == 1000
        assert stats["last_sequence"] == 1499
        assert stats["sequence_range"] == 499
    
    async def test_ttl_update_on_append(self):
        """Test TTL is refreshed on each append."""
        mock_redis = AsyncMock()
        mock_redis.xadd.return_value = b"1000-0"
        
        manager = OutputBufferManager(mock_redis, ttl_seconds=7200)
        
        await manager.append("test", b"data")
        
        # Verify EXPIRE called with correct TTL
        mock_redis.expire.assert_called_once_with("output:test", 7200)
    
    async def test_base64_encoding(self):
        """Test binary data is base64 encoded."""
        mock_redis = AsyncMock()
        mock_redis.xadd.return_value = b"1000-0"
        
        manager = OutputBufferManager(mock_redis)
        
        binary_data = b"\x00\x01\x02\xff\xfe"
        await manager.append("test", binary_data)
        
        # Extract the data field that was passed to xadd
        call_args = mock_redis.xadd.call_args
        fields = call_args[0][1]
        
        # Verify data was base64 encoded
        encoded = fields["data"]
        decoded = base64.b64decode(encoded)
        assert decoded == binary_data


class TestOutputMessage:
    """Test suite for output message model."""
    
    def test_to_dict_serialization(self):
        """Test message serialization to dictionary."""
        msg = OutputMessage(
            sequence=12345,
            timestamp=1706432921.5,
            data=b"Test output",
            msg_type="output"
        )
        
        result = msg.to_dict()
        
        assert result["seq"] == 12345
        assert result["ts"] == 1706432921.5
        assert result["type"] == "output"
        
        # Verify data is base64 encoded
        decoded = base64.b64decode(result["data"])
        assert decoded == b"Test output"
    
    def test_from_redis_deserialization(self):
        """Test message deserialization from Redis."""
        stream_id = "1706432921000-0"
        fields = {
            b"ts": b"1706432921.5",
            b"data": base64.b64encode(b"Terminal data"),
            b"type": b"output"
        }
        
        msg = OutputMessage.from_redis(stream_id, fields)
        
        assert msg.sequence == 1706432921000
        assert msg.timestamp == 1706432921.5
        assert msg.data == b"Terminal data"
        assert msg.msg_type == "output"
