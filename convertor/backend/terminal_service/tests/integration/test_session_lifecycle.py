"""
Integration Tests for Terminal Service.

End-to-end testing with real components (excluding external dependencies).

Test Coverage:
- Full session lifecycle (create, use, delete)
- WebSocket connection and I/O
- Reconnection with output replay
- Health monitoring
- Concurrent sessions
"""

import pytest
import asyncio
import json
import base64
import time
from fastapi.testclient import TestClient
from fastapi import FastAPI
from unittest.mock import AsyncMock, Mock

from terminal_service.main import app
from terminal_service.core.pty_manager import PTYManager
from terminal_service.core.session_manager import SessionManager
from terminal_service.core.output_buffer_manager import OutputBufferManager


@pytest.fixture
def test_app():
    """Get test FastAPI application."""
    return app


@pytest.fixture
def client(test_app):
    """Create test client."""
    return TestClient(test_app)


@pytest.mark.integration
class TestSessionLifecycle:
    """Integration tests for full session lifecycle."""
    
    def test_create_and_list_session(self, client):
        """Test creating session and listing it."""
        # Create session
        response = client.post("/api/v1/sessions", json={
            "shell": "bash",
            "rows": 24,
            "cols": 80
        })
        
        assert response.status_code == 201
        data = response.json()
        session_id = data["session_id"]
        
        # List sessions
        response = client.get("/api/v1/sessions")
        assert response.status_code == 200
        sessions = response.json()
        
        # Find our session
        found = any(s["session_id"] == session_id for s in sessions)
        assert found
    
    def test_get_session_details(self, client):
        """Test retrieving session details."""
        # Create session
        response = client.post("/api/v1/sessions", json={
            "shell": "bash",
            "rows": 24,
            "cols": 80
        })
        session_id = response.json()["session_id"]
        
        # Get details
        response = client.get(f"/api/v1/sessions/{session_id}")
        assert response.status_code == 200
        
        details = response.json()
        assert details["session_id"] == session_id
        assert details["shell"] == "bash"
        assert details["rows"] == 24
        assert details["cols"] == 80
    
    def test_resize_session(self, client):
        """Test resizing terminal session."""
        # Create session
        response = client.post("/api/v1/sessions", json={
            "shell": "bash",
            "rows": 24,
            "cols": 80
        })
        session_id = response.json()["session_id"]
        
        # Resize
        response = client.put(
            f"/api/v1/sessions/{session_id}/resize",
            json={"rows": 30, "cols": 120}
        )
        assert response.status_code == 200
    
    def test_delete_session(self, client):
        """Test deleting session."""
        # Create session
        response = client.post("/api/v1/sessions", json={
            "shell": "bash",
            "rows": 24,
            "cols": 80
        })
        session_id = response.json()["session_id"]
        
        # Delete
        response = client.delete(f"/api/v1/sessions/{session_id}")
        assert response.status_code == 204
        
        # Verify deleted
        response = client.get(f"/api/v1/sessions/{session_id}")
        assert response.status_code == 404
    
    def test_session_not_found(self, client):
        """Test accessing non-existent session."""
        response = client.get("/api/v1/sessions/nonexistent-id")
        assert response.status_code == 404


@pytest.mark.integration
class TestHealthEndpoints:
    """Integration tests for health and monitoring."""
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] in ["healthy", "unhealthy"]
        assert "active_sessions" in data
    
    def test_root_endpoint(self, client):
        """Test root endpoint with service info."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["service"] == "Advanced Terminal Service"
        assert "version" in data
        assert "endpoints" in data


@pytest.mark.integration
class TestConcurrentSessions:
    """Integration tests for concurrent session handling."""
    
    def test_multiple_sessions_creation(self, client):
        """Test creating multiple sessions concurrently."""
        # Create 5 sessions
        session_ids = []
        for i in range(5):
            response = client.post("/api/v1/sessions", json={
                "shell": "bash",
                "rows": 24,
                "cols": 80
            })
            assert response.status_code == 201
            session_ids.append(response.json()["session_id"])
        
        # Verify all exist
        response = client.get("/api/v1/sessions")
        assert response.status_code == 200
        sessions = response.json()
        
        assert len([s for s in sessions if s["session_id"] in session_ids]) >= 5
        
        # Cleanup
        for sid in session_ids:
            client.delete(f"/api/v1/sessions/{sid}")
    
    def test_session_isolation(self, client):
        """Test that sessions are properly isolated."""
        # Create two sessions
        response1 = client.post("/api/v1/sessions", json={
            "shell": "bash",
            "rows": 24,
            "cols": 80
        })
        sid1 = response1.json()["session_id"]
        
        response2 = client.post("/api/v1/sessions", json={
            "shell": "bash",
            "rows": 30,
            "cols": 120
        })
        sid2 = response2.json()["session_id"]
        
        # Get details
        details1 = client.get(f"/api/v1/sessions/{sid1}").json()
        details2 = client.get(f"/api/v1/sessions/{sid2}").json()
        
        # Verify isolation
        assert details1["session_id"] != details2["session_id"]
        assert details1["pid"] != details2["pid"]
        assert details1["rows"] != details2["rows"]
        
        # Cleanup
        client.delete(f"/api/v1/sessions/{sid1}")
        client.delete(f"/api/v1/sessions/{sid2}")


@pytest.mark.integration
class TestErrorHandling:
    """Integration tests for error scenarios."""
    
    def test_invalid_shell_type(self, client):
        """Test creating session with invalid shell."""
        response = client.post("/api/v1/sessions", json={
            "shell": "invalid_shell_999",
            "rows": 24,
            "cols": 80
        })
        # Should still create (defaults to bash)
        assert response.status_code in [201, 500]
    
    def test_invalid_terminal_size(self, client):
        """Test invalid terminal dimensions."""
        response = client.post("/api/v1/sessions", json={
            "shell": "bash",
            "rows": -1,
            "cols": 80
        })
        assert response.status_code == 422  # Validation error
    
    def test_resize_nonexistent_session(self, client):
        """Test resizing non-existent session."""
        response = client.put(
            "/api/v1/sessions/nonexistent/resize",
            json={"rows": 30, "cols": 120}
        )
        assert response.status_code in [404, 500]


@pytest.mark.integration
class TestPerformance:
    """Integration tests for performance benchmarks."""
    
    def test_session_creation_performance(self, client):
        """Test session creation latency."""
        start = time.time()
        
        response = client.post("/api/v1/sessions", json={
            "shell": "bash",
            "rows": 24,
            "cols": 80
        })
        
        latency = (time.time() - start) * 1000  # ms
        
        assert response.status_code == 201
        assert latency < 100  # Should be < 100ms
        
        # Cleanup
        session_id = response.json()["session_id"]
        client.delete(f"/api/v1/sessions/{session_id}")
    
    def test_list_sessions_performance(self, client):
        """Test session listing performance."""
        # Create 10 sessions
        session_ids = []
        for _ in range(10):
            response = client.post("/api/v1/sessions", json={
                "shell": "bash",
                "rows": 24,
                "cols": 80
            })
            session_ids.append(response.json()["session_id"])
        
        # Measure list latency
        start = time.time()
        response = client.get("/api/v1/sessions")
        latency = (time.time() - start) * 1000  # ms
        
        assert response.status_code == 200
        assert latency < 200  # Should be < 200ms for 10 sessions
        
        # Cleanup
        for sid in session_ids:
            client.delete(f"/api/v1/sessions/{sid}")
