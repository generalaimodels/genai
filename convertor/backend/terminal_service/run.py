"""
Simple runner script for terminal service.
Handles imports properly without requiring package installation.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

if __name__ == "__main__":
    import uvicorn
    
    print("Starting Advanced Terminal Service...")
    print("Note: Make sure Redis is running on localhost:6379")
    print()
    
    uvicorn.run(
        "terminal_service.main:app",
        host="0.0.0.0",
        port=8081,
        reload=True,
        log_level="info"
    )
