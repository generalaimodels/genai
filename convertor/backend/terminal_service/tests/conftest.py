"""
Test configuration and fixtures.

Fixes import paths and provides shared test fixtures.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
backend_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_dir))
