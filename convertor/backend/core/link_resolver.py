"""
Link Resolver for Cross-Document References.

Handles:
- MDX component imports → resolve to actual files
- RD \\seealso{} and \\link{} → resolve R package docs
- RST :doc: and :ref: → resolve Sphinx cross-references
- Markdown [text](path) → resolve relative paths

SOTA: O(1) link resolution with hash table lookup
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional
from urllib.parse import unquote


class LinkResolver:
    """
    Resolves cross-document links for all formats.
    
    Complexity: O(1) lookups with document index hash table
    """
    
    def __init__(self, document_root: Path):
        """
        Initialize link resolver.
        
        Args:
            document_root: Root directory of documentation
        """
        self.document_root = Path(document_root)
        self.document_index: dict[str, str] = {}  # filename -> full_path
        self._build_index()
    
    def _build_index(self) -> None:
        """Build document index for O(1) lookups."""
        # Index all documentation files
        extensions = {'.md', '.mdx', '.rst', '.rd', '.rdx', '.ipynb'}
        
        for ext in extensions:
            for filepath in self.document_root.rglob(f'*{ext}'):
                rel_path = filepath.relative_to(self.document_root)
                
                # Index by filename (for quick lookup)
                self.document_index[filepath.name] = str(rel_path)
                
                # Index by stem (without extension)
                self.document_index[filepath.stem] = str(rel_path)
                
                # Index by full relative path
                self.document_index[str(rel_path)] = str(rel_path)
    
    def resolve_link(self, link: str, current_doc: Optional[str] = None) -> str:
        """
        Resolve a link to its full document path.
        
        FIXED: Uses frontend hash routing (/#/) instead of /docs/
        
        Args:
            link: Link to resolve (can be relative, filename, or ref)
            current_doc: Current document path for relative resolution
            
        Returns:
            Resolved document path or original link if not found
            
        Complexity: O(1) with hash table lookup
        """
        # Clean link
        link = unquote(link).strip()
        
        # Remove anchor
        link_without_anchor = link.split('#')[0] if '#' in link else link
        anchor = '#' + link.split('#')[1] if '#' in link else ''
        
        # Skip external links
        if link.startswith(('http://', 'https://', 'mailto:', 'ftp://')):
            return link
        
        # Try direct index lookup
        if link_without_anchor in self.document_index:
            # FIXED: Use frontend hash routing
            return f"/#/{self.document_index[link_without_anchor]}{anchor}"
        
        # Try relative path resolution
        if current_doc and not link.startswith('/'):
            current_dir = Path(current_doc).parent
            resolved_path = (current_dir / link_without_anchor).resolve()
            
            try:
                rel_to_root = resolved_path.relative_to(self.document_root)
                if str(rel_to_root) in self.document_index:
                    # FIXED: Use frontend hash routing
                    return f"/#/{rel_to_root}{anchor}"
            except ValueError:
                pass
        
        # Return original if not resolved
        return link
    
    def resolve_mdx_import(self, import_path: str) -> Optional[str]:
        """
        Resolve MDX import statement to actual file.
        
        Example: import { Button } from '@/components/Button'
        → resolves to: /#/components/Button.mdx
        """
        # Remove quotes and clean
        import_path = import_path.strip().strip('"\'')
        
        # Handle @/ alias (common in React/Next.js)
        if import_path.startswith('@/'):
            import_path = import_path[2:]
        
        # Try with .mdx extension
        for candidate in [
            f"{import_path}.mdx",
            f"{import_path}/index.mdx",
            import_path
        ]:
            if candidate in self.document_index:
                # FIXED: Use frontend hash routing
                return f"/#/{self.document_index[candidate]}"
        
        return None
    
    def resolve_rst_role(self, role_type: str, target: str) -> str:
        """
        Resolve RST role references.
        
        Handles:
        - :doc:`target` → document link
        - :ref:`target` → reference link
        - :meth:`target` → method reference
        """
        if role_type in ('doc', 'ref'):
            # Extract link text and target
            if '<' in target and '>' in target:
                # Format: text <target>
                text = target.split('<')[0].strip()
                link = target.split('<')[1].split('>')[0].strip()
            else:
                text = target
                link = target
            
            # Resolve link
            resolved = self.resolve_link(link)
            return f"[{text}]({resolved})"
        
        # For other roles, return as code
        return f"`{target}`"
    
    def resolve_rd_link(self, link_target: str) -> str:
        """
        Resolve R documentation \\link{} references.
        
        Example: \\link{mean} → /#/stats/mean.Rd
        """
        # Try to find in index
        candidates = [
            f"{link_target}.Rd",
            f"{link_target}.rd",
            link_target
        ]
        
        for candidate in candidates:
            if candidate in self.document_index:
                # FIXED: Use frontend hash routing
                return f"/#/{self.document_index[candidate]}"
        
        # Return as text if not found
        return link_target
    
    def process_html_links(self, html: str, current_doc: Optional[str] = None) -> str:
        """
        Process all links in HTML content.
        
        Finds <a href="..."> tags and resolves their paths.
        """
        def replace_link(match):
            href = match.group(1)
            resolved = self.resolve_link(href, current_doc)
            return f'href="{resolved}"'
        
        # Replace all href attributes
        html = re.sub(r'href="([^"]+)"', replace_link, html)
        html = re.sub(r"href='([^']+)'", replace_link, html)
        
        return html


# Global instance (initialized by scanner)
_resolver: Optional[LinkResolver] = None


def get_link_resolver() -> Optional[LinkResolver]:
    """Get global link resolver instance."""
    return _resolver


def set_link_resolver(document_root: Path) -> LinkResolver:
    """Initialize global link resolver."""
    global _resolver
    _resolver = LinkResolver(document_root)
    return _resolver
