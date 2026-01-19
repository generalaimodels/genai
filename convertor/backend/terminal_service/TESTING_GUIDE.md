# Terminal Service - Testing Guide

## Prerequisites

### 1. Install Redis
```powershell
# Windows (via Docker)
docker run -d -p 6379:6379 redis:latest

# Or install Redis on Windows
# Download from: https://github.com/microsoftarchive/redis/releases
```

### 2. Install Python Dependencies
```powershell
cd backend/terminal_service
pip install -e .
pip install -r tests/requirements-test.txt
```

## Running the Service

### Start Terminal Service
```powershell
# Make sure Redis is running first
cd backend/terminal_service
python main.py
```

Expected output:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8080
```

## CLI Testing

### Run All Tests
```powershell
cd tests
python cli_test.py --test all
```

### Run Specific Tests

**Health Check**:
```powershell
python cli_test.py --test health
```

**Create Session**:
```powershell
python cli_test.py --test create_session
```

**List Sessions**:
```powershell
python cli_test.py --test list_sessions
```

**Stress Test** (10 concurrent sessions):
```powershell
python cli_test.py --test stress --sessions 10
```

## Manual API Testing

### Using curl

**Create Session**:
```powershell
curl -X POST http://localhost:8080/api/v1/sessions `
  -H "Content-Type: application/json" `
  -d '{"shell": "powershell", "rows": 24, "cols": 80}'
```

**List Sessions**:
```powershell
curl http://localhost:8080/api/v1/sessions
```

**Get Session Details**:
```powershell
curl http://localhost:8080/api/v1/sessions/{session_id}
```

**Delete Session**:
```powershell
curl -X DELETE http://localhost:8080/api/v1/sessions/{session_id}
```

**Resize Terminal**:
```powershell
curl -X PUT http://localhost:8080/api/v1/sessions/{session_id}/resize `
  -H "Content-Type: application/json" `
  -d '{"rows": 30, "cols": 120}'
```

## Testing WebSocket Connection

### Using Python Script
```python
import asyncio
import websockets
import json
import base64

async def test_websocket():
    uri = "ws://localhost:8080/ws/terminal/{session_id}"
    
    async with websockets.connect(uri) as websocket:
        # Send command
        await websocket.send(json.dumps({
            "type": "input",
            "data": base64.b64encode(b"echo 'Hello'\n").decode()
        }))
        
        # Receive output
        for _ in range(5):
            msg = await websocket.recv()
            data = json.dumps(msg)
            print(data)

asyncio.run(test_websocket())
```

## Integration Testing

### Test Connection Stability

**1. Reconnection Test**:
- Create session
- Connect via WebSocket
- Send commands
- Disconnect
- Reconnect with `?recover_from=<seq>` parameter
- Verify output replay

**2. Flow Control Test**:
- Send large amounts of data
- Monitor flow control messages (PAUSE/RESUME)
- Verify no data loss

**3. Health Monitoring Test**:
- Send PING messages
- Receive PONG responses
- Measure RTT

## Expected Test Results

### All Tests Passing
```
════════════════════════════════════════════════════════════
TEST SUMMARY
════════════════════════════════════════════════════════════
Test                          Result                        
create_session                ✓ PASS
list_sessions                 ✓ PASS
get_metadata                  ✓ PASS
websocket_io                  ✓ PASS
ping_pong                     ✓ PASS
reconnection                  ✓ PASS
resize                        ✓ PASS
delete_session                ✓ PASS
health_check                  ✓ PASS

Total: 9  |  Passed: 9  |  Failed: 0
Success Rate: 100.0%
```

## Troubleshooting

### Service Won't Start

**Error**: `redis.exceptions.ConnectionError`
- **Fix**: Make sure Redis is running on port 6379

**Error**: `ModuleNotFoundError`
- **Fix**: Install dependencies: `pip install -e .`

### WebSocket Connection Failed

**Error**: `Connection refused`
- **Fix**: Verify service is running on port 8080

**Error**: `Session not found`
- **Fix**: Create session via REST API first

### Tests Timing Out

**Issue**: WebSocket operations timeout
- **Fix**: Increase timeout in cli_test.py
- **Fix**: Check if Redis is responding slowly

## Performance Benchmarks

Expected performance metrics:

- **Session Creation**: < 50ms P95
- **WebSocket Latency**: < 10ms P95
- **Throughput**: 10MB/s per connection
- **Concurrent Sessions**: 100+ per instance
- **Reconnection Time**: < 100ms

## Next Steps

After CLI testing passes:
1. Integration testing with frontend
2. Load testing with multiple concurrent sessions
3. Chaos engineering (network failures, Redis failures)
4. Security testing
5. Production deployment

## Monitoring

While tests run, monitor:
- Redis memory usage
- Active WebSocket connections
- Session count
- Error logs

Access metrics at: `http://localhost:8080/api/v1/health`
