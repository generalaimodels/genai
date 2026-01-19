# Quick Start Guide - Windows Users

## Issue Fix Applied ✓

### Problem 1: Windows Compatibility
**Error**: `ModuleNotFoundError: No module named 'termios'`  
**Cause**: PTY module requires Unix-only `termios` module  
**Fix**: Made imports conditional, added Windows subprocess fallback

### Problem 2: Test Import Paths
**Error**: `ModuleNotFoundError: No module named 'terminal_service'`  
**Cause**: Python can't find the module  
**Fix**: Created `conftest.py` and `setup.py` for proper package installation

## Running Tests on Windows (Fixed!)

```powershell
# 1. Install package in development mode (one-time setup)
cd C:\Users\heman\Desktop\code\genai\convertor\backend\terminal_service
pip install -e .

# 2. Run all unit tests
pytest tests/unit/ -v

# 3. Run specific test
pytest tests/unit/test_tcp_optimizer.py -v

# 4. Run with coverage
pytest tests/unit/ --cov=terminal_service --cov-report=html
```

## Windows Notes

- **PTY on Windows**: Windows doesn't have native PTY. The current implementation uses `subprocess` as a fallback for testing.
- **Production Windows Support**: For production, would use:
  - `pywinpty` library
  - Windows ConPTY API (Windows 10 1809+)
  - Or run in WSL2 for full Linux PTY support

## Test Results Expected

All unit tests should pass:
- ✓ `test_tcp_optimizer.py` - 8 tests
- ✓ `test_reconnection_manager.py` - 9 tests  
- ✓ `test_output_buffer_manager.py` - 11 tests
- ✓ `test_flow_control.py` - 13 tests

Total: ~41 unit tests for connection stability components

## If Tests Still Fail

1. **Module not found**: Run `pip install -e .` again
2. **Import errors**: Check that you're in the correct directory
3. **Async errors**: Install `pytest-asyncio`: `pip install pytest-asyncio`
