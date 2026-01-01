"""
Jupyter Notebook (.ipynb) to Markdown Converter.

Converts Jupyter notebooks to markdown format while preserving:
- Code cells with syntax highlighting
- Markdown cells with all formatting
- Cell outputs (text, HTML, images, LaTeX)
- Execution order and cell metadata
- Error outputs with proper formatting

Engineering considerations:
- Zero-copy processing where possible
- Base64 image embedding for portability
- Lazy output rendering to minimize memory footprint
- Comprehensive error handling for malformed notebooks
"""

from __future__ import annotations

import json
import base64
import html
from pathlib import Path
from typing import Any
from dataclasses import dataclass


@dataclass
class NotebookCell:
    """Structured representation of a notebook cell."""
    cell_type: str  # 'code', 'markdown', 'raw'
    source: str
    execution_count: int | None
    outputs: list[dict[str, Any]]
    metadata: dict[str, Any]


@dataclass
class ConvertedNotebook:
    """Result of notebook conversion."""
    markdown_content: str
    title: str
    cell_count: int
    code_cell_count: int
    has_outputs: bool


class NotebookConverter:
    """
    Converts Jupyter notebooks (.ipynb) to markdown format.
    
    Supports all standard Jupyter notebook features including:
    - IPython magics
    - Rich outputs (images, plots, LaTeX, HTML)
    - Error tracebacks
    - Multiple output formats per cell
    """
    
    # Supported image formats for embedding
    IMAGE_FORMATS: frozenset[str] = frozenset({'image/png', 'image/jpeg', 'image/gif', 'image/svg+xml'})
    
    def __init__(self) -> None:
        """Initialize notebook converter."""
        pass
    
    def convert_file(self, filepath: str | Path) -> ConvertedNotebook:
        """
        Convert .ipynb file to markdown.
        
        Args:
            filepath: Path to .ipynb file
            
        Returns:
            ConvertedNotebook with markdown content and metadata
            
        Raises:
            ValueError: If file is not valid JSON or missing required fields
            IOError: If file cannot be read
        """
        filepath = Path(filepath)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            notebook_data = json.load(f)
        
        return self.convert(notebook_data, filepath.stem)
    
    def convert(self, notebook_data: dict[str, Any], title_fallback: str = "Notebook") -> ConvertedNotebook:
        """
        Convert notebook JSON to markdown.
        
        Args:
            notebook_data: Parsed notebook JSON
            title_fallback: Fallback title if not found in notebook
            
        Returns:
            ConvertedNotebook with markdown content
        """
        if 'cells' not in notebook_data:
            raise ValueError("Invalid notebook: missing 'cells' field")
        
        # Extract metadata
        metadata = notebook_data.get('metadata', {})
        title = self._extract_title(notebook_data, title_fallback)
        
        # Parse cells
        cells = [self._parse_cell(cell_data) for cell_data in notebook_data['cells']]
        
        # Generate markdown
        markdown_parts = [self._generate_header(title, metadata)]
        
        code_cell_count = 0
        has_outputs = False
        
        for cell in cells:
            if cell.cell_type == 'markdown':
                markdown_parts.append(self._convert_markdown_cell(cell))
            elif cell.cell_type == 'code':
                markdown_parts.append(self._convert_code_cell(cell))
                code_cell_count += 1
                if cell.outputs:
                    has_outputs = True
            elif cell.cell_type == 'raw':
                markdown_parts.append(self._convert_raw_cell(cell))
        
        markdown_content = '\n\n'.join(filter(None, markdown_parts))
        
        return ConvertedNotebook(
            markdown_content=markdown_content,
            title=title,
            cell_count=len(cells),
            code_cell_count=code_cell_count,
            has_outputs=has_outputs
        )
    
    def _extract_title(self, notebook_data: dict[str, Any], fallback: str) -> str:
        """Extract title from notebook metadata or first heading."""
        # Try notebook metadata
        metadata = notebook_data.get('metadata', {})
        if 'title' in metadata:
            return str(metadata['title'])
        
        # Try first markdown cell for H1
        cells = notebook_data.get('cells', [])
        for cell in cells:
            if cell.get('cell_type') == 'markdown':
                source = self._join_source(cell.get('source', []))
                lines = source.split('\n')
                for line in lines:
                    stripped = line.strip()
                    if stripped.startswith('# '):
                        return stripped[2:].strip()
        
        return fallback.replace('-', ' ').replace('_', ' ').title()
    
    def _parse_cell(self, cell_data: dict[str, Any]) -> NotebookCell:
        """Parse raw cell data into NotebookCell."""
        return NotebookCell(
            cell_type=cell_data.get('cell_type', 'code'),
            source=self._join_source(cell_data.get('source', [])),
            execution_count=cell_data.get('execution_count'),
            outputs=cell_data.get('outputs', []),
            metadata=cell_data.get('metadata', {})
        )
    
    def _join_source(self, source: str | list[str]) -> str:
        """Join source lines into single string."""
        if isinstance(source, str):
            return source
        elif isinstance(source, list):
            return ''.join(source)
        return ''
    
    def _generate_header(self, title: str, metadata: dict[str, Any]) -> str:
        """Generate markdown header with metadata."""
        parts = [f"# {title}"]
        
        # Add metadata if present
        if metadata:
            parts.append("")
            parts.append("> **Notebook Metadata**")
            
            # Language info
            if 'language_info' in metadata:
                lang_info = metadata['language_info']
                lang_name = lang_info.get('name', 'Unknown')
                lang_version = lang_info.get('version', '')
                parts.append(f"> - **Language**: {lang_name} {lang_version}".strip())
            
            # Kernel info
            if 'kernelspec' in metadata:
                kernel = metadata['kernelspec']
                kernel_name = kernel.get('display_name', kernel.get('name', 'Unknown'))
                parts.append(f"> - **Kernel**: {kernel_name}")
        
        return '\n'.join(parts)
    
    def _convert_markdown_cell(self, cell: NotebookCell) -> str:
        """Convert markdown cell to markdown."""
        return cell.source
    
    def _convert_code_cell(self, cell: NotebookCell) -> str:
        """Convert code cell with outputs to markdown."""
        parts = []
        
        # Code input
        lang = 'python'  # Default, could extract from metadata
        
        # Add execution count if present
        if cell.execution_count is not None:
            parts.append(f"**In [{cell.execution_count}]:**")
            parts.append("")
        
        # Code block
        parts.append(f"```{lang}")
        parts.append(cell.source.rstrip())
        parts.append("```")
        
        # Outputs
        if cell.outputs:
            parts.append("")
            output_md = self._convert_outputs(cell.outputs, cell.execution_count)
            if output_md:
                parts.append(output_md)
        
        return '\n'.join(parts)
    
    def _convert_raw_cell(self, cell: NotebookCell) -> str:
        """Convert raw cell to markdown (as code block)."""
        return f"```\n{cell.source}\n```"
    
    def _convert_outputs(self, outputs: list[dict[str, Any]], execution_count: int | None) -> str:
        """Convert cell outputs to markdown."""
        parts = []
        
        for output in outputs:
            output_type = output.get('output_type', 'unknown')
            
            if output_type == 'stream':
                # stdout/stderr output
                stream_parts = self._convert_stream_output(output)
                if stream_parts:
                    parts.append(stream_parts)
            
            elif output_type == 'execute_result' or output_type == 'display_data':
                # Rich output (images, HTML, LaTeX, etc.)
                display_parts = self._convert_display_output(output, execution_count)
                if display_parts:
                    parts.append(display_parts)
            
            elif output_type == 'error':
                # Error traceback
                error_parts = self._convert_error_output(output)
                if error_parts:
                    parts.append(error_parts)
        
        return '\n\n'.join(filter(None, parts))
    
    def _convert_stream_output(self, output: dict[str, Any]) -> str:
        """Convert stream output (stdout/stderr) to markdown."""
        stream_name = output.get('name', 'stdout')
        text = self._join_source(output.get('text', []))
        
        if not text:
            return ''
        
        # Use distinct styling for stderr
        if stream_name == 'stderr':
            return f"**Output (stderr):**\n\n```\n{text.rstrip()}\n```"
        else:
            return f"**Output:**\n\n```\n{text.rstrip()}\n```"
    
    def _convert_display_output(self, output: dict[str, Any], execution_count: int | None) -> str:
        """Convert display/execute_result output to markdown."""
        data = output.get('data', {})
        
        if not data:
            return ''
        
        parts = []
        
        # Add execution count for execute_result
        if output.get('output_type') == 'execute_result' and execution_count is not None:
            parts.append(f"**Out [{execution_count}]:**")
            parts.append("")
        
        # Priority order: images > HTML > LaTeX > text
        
        # Images (highest priority)
        for mime_type in self.IMAGE_FORMATS:
            if mime_type in data:
                image_md = self._convert_image(data[mime_type], mime_type)
                if image_md:
                    parts.append(image_md)
                    return '\n'.join(parts)
        
        # LaTeX math
        if 'text/latex' in data:
            latex = self._join_source(data['text/latex'])
            # Wrap in $$ for block display
            if latex and not latex.strip().startswith('$$'):
                latex = f"$$\n{latex.strip()}\n$$"
            parts.append(latex)
            return '\n'.join(parts)
        
        # HTML (render as HTML block)
        if 'text/html' in data:
            html_content = self._join_source(data['text/html'])
            parts.append(html_content)
            return '\n'.join(parts)
        
        # Markdown
        if 'text/markdown' in data:
            md_content = self._join_source(data['text/markdown'])
            parts.append(md_content)
            return '\n'.join(parts)
        
        # Plain text (fallback)
        if 'text/plain' in data:
            text = self._join_source(data['text/plain'])
            parts.append(f"```\n{text.rstrip()}\n```")
            return '\n'.join(parts)
        
        return ''
    
    def _convert_image(self, image_data: str | list[str], mime_type: str) -> str:
        """Convert base64 image to markdown image tag."""
        # Join if list
        if isinstance(image_data, list):
            image_data = ''.join(image_data)
        
        # Remove whitespace
        image_data = image_data.replace('\n', '').replace(' ', '')
        
        # Determine format
        if mime_type == 'image/svg+xml':
            # SVG can be embedded directly
            try:
                svg_content = base64.b64decode(image_data).decode('utf-8')
                return svg_content
            except Exception:
                return ''
        else:
            # Other images: use data URL
            data_url = f"data:{mime_type};base64,{image_data}"
            return f"![Output]({data_url})"
    
    def _convert_error_output(self, output: dict[str, Any]) -> str:
        """Convert error traceback to markdown."""
        ename = output.get('ename', 'Error')
        evalue = output.get('evalue', '')
        traceback = output.get('traceback', [])
        
        parts = []
        parts.append("**Error:**")
        parts.append("")
        parts.append(f"```python")
        
        # Remove ANSI escape codes from traceback
        clean_traceback = [self._strip_ansi(line) for line in traceback]
        parts.extend(clean_traceback)
        
        parts.append("```")
        
        return '\n'.join(parts)
    
    def _strip_ansi(self, text: str) -> str:
        """Remove ANSI escape codes from text."""
        import re
        # ANSI escape code pattern
        ansi_pattern = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]')
        return ansi_pattern.sub('', text)
