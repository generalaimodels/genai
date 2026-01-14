import sys
from pathlib import Path
import os

# Ensure backend can resolve core
sys.path.append(str(Path.cwd() / "backend"))

try:
    from core.parser import MarkdownParser
    print("Core Parser Import: Success")
    p = MarkdownParser()
    print("Parser Init: Success")
except Exception as e:
    print(f"Core Parser Import/Init Failed: {e}")

try:
    from git_pipeline.service import GitService
    print("GitService Import: Success")
except Exception as e:
    print(f"GitService Import Failed: {e}")
