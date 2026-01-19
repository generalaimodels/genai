"""
Unit Tests for TCP Optimizer.

Tests socket configuration optimizations for connection stability.

Test Coverage:
- Socket option configuration
- Platform-specific keepalive settings
- Buffer size validation
- TCP_NODELAY verification
"""

import socket
import struct
import pytest
from unittest.mock import Mock, patch, MagicMock

from terminal_service.core.tcp_optimizer import optimize_socket, get_socket_info


class TestTCPOptimizer:
    """Test suite for TCP socket optimizer."""
    
    def test_optimize_socket_basic(self):
        """Test basic socket optimization applies all settings."""
        mock_sock = Mock(spec=socket.socket)
        
        optimize_socket(mock_sock)
        
        # Verify SO_KEEPALIVE enabled
        mock_sock.setsockopt.assert_any_call(
            socket.SOL_SOCKET,
            socket.SO_KEEPALIVE,
            1
        )
        
        # Verify TCP_NODELAY enabled
        mock_sock.setsockopt.assert_any_call(
            socket.IPPROTO_TCP,
            socket.TCP_NODELAY,
            1
        )
    
    def test_optimize_socket_buffer_sizes(self):
        """Test socket buffer sizes are set correctly."""
        mock_sock = Mock(spec=socket.socket)
        buffer_size = 262144  # 256KB
        
        optimize_socket(mock_sock, buffer_size=buffer_size)
        
        # Verify send buffer
        mock_sock.setsockopt.assert_any_call(
            socket.SOL_SOCKET,
            socket.SO_SNDBUF,
            buffer_size
        )
        
        # Verify receive buffer
        mock_sock.setsockopt.assert_any_call(
            socket.SOL_SOCKET,
            socket.SO_RCVBUF,
            buffer_size
        )
    
    @patch('socket.socket')
    def test_optimize_socket_linger(self, mock_socket_class):
        """Test SO_LINGER configuration."""
        mock_sock = Mock(spec=socket.socket)
        
        optimize_socket(mock_sock)
        
        # Verify linger struct (l_onoff=1, l_linger=5)
        expected_linger = struct.pack('ii', 1, 5)
        mock_sock.setsockopt.assert_any_call(
            socket.SOL_SOCKET,
            socket.SO_LINGER,
            expected_linger
        )
    
    @patch('socket.TCP_KEEPIDLE', 4, create=True)
    @patch('socket.TCP_KEEPINTVL', 5, create=True)
    @patch('socket.TCP_KEEPCNT', 6, create=True)
    def test_optimize_socket_keepalive_unix(self):
        """Test keepalive configuration on Unix/Linux systems."""
        mock_sock = Mock(spec=socket.socket)
        
        optimize_socket(mock_sock)
        
        # Verify TCP_KEEPIDLE (60 seconds)
        mock_sock.setsockopt.assert_any_call(
            socket.IPPROTO_TCP,
            4,  # TCP_KEEPIDLE
            60
        )
        
        # Verify TCP_KEEPINTVL (10 seconds)
        mock_sock.setsockopt.assert_any_call(
            socket.IPPROTO_TCP,
            5,  # TCP_KEEPINTVL
            10
        )
        
        # Verify TCP_KEEPCNT (3 probes)
        mock_sock.setsockopt.assert_any_call(
            socket.IPPROTO_TCP,
            6,  # TCP_KEEPCNT
            3
        )
    
    def test_optimize_socket_error_handling(self):
        """Test optimizer handles socket errors gracefully."""
        mock_sock = Mock(spec=socket.socket)
        mock_sock.setsockopt.side_effect = OSError("Socket error")
        
        # Should not raise exception
        optimize_socket(mock_sock)
    
    def test_get_socket_info(self):
        """Test socket info retrieval."""
        mock_sock = Mock(spec=socket.socket)
        mock_sock.getsockopt.side_effect = [
            1,  # SO_KEEPALIVE
            1,  # TCP_NODELAY
            262144,  # SO_SNDBUF
            262144,  # SO_RCVBUF
        ]
        
        info = get_socket_info(mock_sock)
        
        assert info['keepalive'] == 1
        assert info['nodelay'] == 1
        assert info['sndbuf'] == 262144
        assert info['rcvbuf'] == 262144
    
    def test_custom_buffer_size(self):
        """Test custom buffer size parameter."""
        mock_sock = Mock(spec=socket.socket)
        custom_size = 524288  # 512KB
        
        optimize_socket(mock_sock, buffer_size=custom_size)
        
        mock_sock.setsockopt.assert_any_call(
            socket.SOL_SOCKET,
            socket.SO_SNDBUF,
            custom_size
        )
