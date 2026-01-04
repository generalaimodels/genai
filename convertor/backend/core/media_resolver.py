"""
SOTA Media Path Resolver.

Generalized utility for resolving relative media paths to absolute URLs.
Includes security checks, type validation, and performance optimizations.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional
from urllib.parse import quote


class MediaPathResolver:
    """
    Production-ready media path resolver.
    
    Features:
    - Resolves relative paths to API URLs
    - Handles attachment: protocol
    - Supports nested directories
    - Validates file existence
    - Path traversal protection
    - URL encoding for special characters
    - Serves ALL file types (no extension filtering)
    """
    
    def __init__(self, data_dir: Path, media_url_prefix: str = "/api/media"):
        """
        Initialize resolver.
        
        Args:
            data_dir: Root data directory (for validation)
            media_url_prefix: URL prefix for media endpoint
        """
        self.data_dir = data_dir.resolve()
        self.media_url_prefix = media_url_prefix.rstrip('/')
    
    def resolve_html_paths(self, html: str, document_path: str) -> str:
        """
        Resolve all media paths in HTML to absolute URLs.
        
        Args:
            html: HTML content with potentially relative paths
            document_path: Document's path relative to data_dir
            
        Returns:
            HTML with resolved absolute URLs
        """
        doc_dir = self._get_document_dir(document_path)
        
        # Pattern 1: src="..." (images, videos, audio)
        html = re.sub(
            r'src=(["\'])([^"\']+)\1',
            lambda m: f'src={m.group(1)}{self._resolve_path(m.group(2), doc_dir)}{m.group(1)}',
            html,
            flags=re.IGNORECASE
        )
        
        # Pattern 2: href="..." (for downloadable media)
        html = re.sub(
            r'href=(["\'])([^"\']+)\1',
            lambda m: f'href={m.group(1)}{self._resolve_if_media(m.group(2), doc_dir)}{m.group(1)}',
            html,
            flags=re.IGNORECASE
        )
        
        return html
    
    def _get_document_dir(self, document_path: str) -> str:
        """Get document's directory path."""
        doc_path = Path(document_path)
        if doc_path.suffix:
            # Remove filename, keep directory
            return doc_path.parent.as_posix()
        return doc_path.as_posix()
    
    def _resolve_path(self, url: str, doc_dir: str) -> str:
        """
        Resolve single URL to absolute path.
        
        Args:
            url: Original URL from HTML
            doc_dir: Document's directory
            
        Returns:
            Absolute URL or original if not applicable
        """
        # Skip if already absolute
        if self._is_absolute_url(url):
            return url
        
        # Handle attachment: protocol (Jupyter notebooks)
        if url.startswith('attachment:'):
            url = url.replace('attachment:', '', 1)
        
        # Strip leading ./ if present
        if url.startswith('./'):
            url = url[2:]
        
        # Build full path
        if doc_dir and doc_dir != '.':
            full_url = f"{doc_dir}/{url}"
        else:
            full_url = url
        
        # Normalize path (resolve ../, ./, etc.) and clean up
        # Use PurePosixPath for consistent forward slashes
        from pathlib import PurePosixPath
        normalized = str(PurePosixPath(full_url))
        
        # Remove any leading ./ that might remain after normalization
        while normalized.startswith('./'):
            normalized = normalized[2:]
        
        # URL encode special characters (but keep slashes and basic chars)
        encoded = quote(normalized, safe='/-._~')
        
        # Return API URL
        return f"{self.media_url_prefix}/{encoded}"
    
    def _resolve_if_media(self, url: str, doc_dir: str) -> str:
        """
        Resolve ALL file paths, not just media.
        Simplified: serve any file from data directory.
        """
        if self._is_absolute_url(url):
            return url
        
        # Resolve all files, not just specific extensions
        return self._resolve_path(url, doc_dir)
    
    @staticmethod
    def _is_absolute_url(url: str) -> bool:
        """Check if URL is already absolute."""
        return url.startswith((
            'http://', 'https://', 'data:', 'blob:',
            '/', '//', 'ftp://', 'mailto:', '#'
        ))
    
    def validate_media_path(self, media_path: str) -> Optional[Path]:
        """
        Validate and resolve media path to filesystem.
        
        Args:
            media_path: Requested media path (from URL)
            
        Returns:
            Resolved Path if valid, None if invalid/unsafe
        """
        try:
            # Resolve to absolute path
            full_path = (self.data_dir / media_path).resolve()
            
            # Security: ensure path is within data_dir
            # use relative_to for robust check (handles case sensitivity on Windows)
            try:
                full_path.relative_to(self.data_dir)
            except ValueError:
                return None
            
            # Verify file exists and is a file (not directory)
            if not full_path.exists() or not full_path.is_file():
                return None
            
            return full_path
            
        except (ValueError, OSError):
            return None
