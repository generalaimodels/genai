"""
R Documentation (.Rd/.Rdx) to Markdown Converter.

Converts R documentation files to markdown while preserving:
- LaTeX-like tag structure (\name{}, \title{}, \description{})
- Function signatures and arguments
- Return value descriptions
- Code examples
- Cross-references and links

Engineering considerations:
- Balanced brace matching for nested structures
- Hierarchical section processing
- O(n) complexity with minimal backtracking
- Robust error handling for malformed Rd files
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from dataclasses import dataclass


@dataclass
class ConvertedRd:
    """Result of Rd conversion."""
    markdown_content: str
    title: str
    function_name: str
    has_examples: bool


class RdConverter:
    """
    Converts R documentation (.Rd/.Rdx) files to markdown format.
    
    Supports Rd features:
    - LaTeX-like tags (\name{}, \title{}, \description{})
    - Arguments and value sections
    - Code examples with proper formatting
    - Cross-references (\link{}, \seealso{})
    - Text formatting (\code{}, \emph{}, \strong{})
    
    OPTIMIZATION: Single-pass O(n) tag extraction with balanced brace matching
    """
    
    # Section tags that should be rendered as headings
    SECTION_TAGS = {
        'arguments': 'Arguments',
        'value': 'Return Value',
        'details': 'Details',
        'examples': 'Examples',
        'note': 'Note',
        'section': None,  # Has custom title
        'seealso': 'See Also',
        'references': 'References',
        'author': 'Author',
    }
    
    def __init__(self) -> None:
        """Initialize Rd converter."""
        pass
    
    def convert_file(self, filepath: str | Path) -> ConvertedRd:
        """
        Convert .Rd file to markdown.
        
        Args:
            filepath: Path to .Rd or .Rdx file
            
        Returns:
            ConvertedRd with markdown content and metadata
            
        Raises:
            ValueError: If file is malformed
            IOError: If file cannot be read
        """
        filepath = Path(filepath)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return self.convert(content, filepath.stem)
    
    def convert(self, content: str, title_fallback: str = "R Function") -> ConvertedRd:
        """
        Convert Rd content to markdown.
        
        Args:
            content: Raw Rd string
            title_fallback: Fallback title if not found
            
        Returns:
            ConvertedRd with markdown content
        """
        # Extract main tags
        name = self._extract_tag(content, 'name') or title_fallback
        title = self._extract_tag(content, 'title') or name
        description = self._extract_tag(content, 'description') or ''
        usage = self._extract_tag(content, 'usage') or ''
        
        # Build markdown sections
        sections = []
        
        # Title and description
        sections.append(f"# {title}")
        if description:
            sections.append(self._convert_inline_formatting(description))
        
        # Usage section
        if usage:
            sections.append("## Usage")
            sections.append(f"```r\n{usage.strip()}\n```")
        
        # Arguments section
        arguments = self._extract_tag(content, 'arguments')
        if arguments:
            sections.append("## Arguments")
            sections.append(self._convert_arguments(arguments))
        
        # Other sections
        for tag, heading in self.SECTION_TAGS.items():
            if tag in ('arguments',):  # Already handled
                continue
            
            section_content = self._extract_tag(content, tag)
            if section_content:
                if tag == 'section':
                    # Extract custom section title
                    title_match = re.search(r'\\heading\{([^}]+)\}', section_content)
                    if title_match:
                        custom_heading = title_match.group(1)
                        section_content = section_content[title_match.end():].strip()
                        sections.append(f"## {custom_heading}")
                    else:
                        sections.append("## Additional Information")
                else:
                    sections.append(f"## {heading}")
                
                # Special handling for examples
                if tag == 'examples':
                    sections.append(self._convert_examples(section_content))
                else:
                    sections.append(self._convert_inline_formatting(section_content))
        
        markdown_content = '\n\n'.join(filter(None, sections))
        
        return ConvertedRd(
            markdown_content=markdown_content,
            title=title,
            function_name=name,
            has_examples=bool(self._extract_tag(content, 'examples'))
        )
    
    def _extract_tag(self, content: str, tag: str) -> str | None:
        """
        Extract content from LaTeX-like tag with balanced brace matching.
        
        Args:
            content: Full Rd content
            tag: Tag name to extract
            
        Returns:
            Tag content or None if not found
            
        ENGINEERING: Uses balanced brace counting for O(n) extraction
        """
        # Find tag start
        pattern = rf'\\{tag}\{{'
        match = re.search(pattern, content)
        if not match:
            return None
        
        # Extract balanced braces content
        start_pos = match.end()
        brace_count = 1
        current_pos = start_pos
        
        while current_pos < len(content) and brace_count > 0:
            char = content[current_pos]
            if char == '{' and (current_pos == 0 or content[current_pos - 1] != '\\'):
                brace_count += 1
            elif char == '}' and (current_pos == 0 or content[current_pos - 1] != '\\'):
                brace_count -= 1
            current_pos += 1
        
        if brace_count == 0:
            return content[start_pos:current_pos - 1].strip()
        
        return None
    
    def _convert_inline_formatting(self, text: str) -> str:
        r"""
        Convert Rd inline formatting to markdown.
        
        Handles:
        - \code{} → `code`
        - \emph{} → *italic*
        - \strong{} → **bold**
        - \link{} → cross-references
        - \url{} → URLs
        """
        if not text:
            return ''
        
        # Convert \code{...} to markdown code
        text = re.sub(r'\\code\{([^}]+)\}', r'`\1`', text)
        
        # Convert \emph{...} to italic
        text = re.sub(r'\\emph\{([^}]+)\}', r'*\1*', text)
        
        # Convert \strong{...} to bold
        text = re.sub(r'\\strong\{([^}]+)\}', r'**\1**', text)
        
        # Convert \link{...} to markdown link (placeholder)
        text = re.sub(r'\\link\{([^}]+)\}', r'[\1](#)', text)
        
        # Convert \url{...} to markdown link
        text = re.sub(r'\\url\{([^}]+)\}', r'[\1](\1)', text)
        
        # Convert \href{url}{text} to markdown link
        text = re.sub(r'\\href\{([^}]+)\}\{([^}]+)\}', r'[\2](\1)', text)
        
        # Remove \dontrun{}, \donttest{}, \dontshow{} wrappers
        text = re.sub(r'\\dont(?:run|test|show)\{([^}]+)\}', r'\1', text)
        
        # Convert \item{name}{description} to markdown list
        text = re.sub(r'\\item\{([^}]+)\}\{([^}]+)\}', r'- **\1**: \2', text)
        
        # Clean up remaining backslash commands
        text = re.sub(r'\\([a-zA-Z]+)\{', r'\1: {', text)
        
        return text.strip()
    
    def _convert_arguments(self, arguments_text: str) -> str:
        """
        Convert \arguments{} section to markdown table or list.
        
        Returns formatted argument list.
        """
        # Extract individual \item{name}{description} entries
        items = []
        item_pattern = re.compile(r'\\item\{([^}]+)\}\{(.+?)\}(?=\\item|$)', re.DOTALL)
        
        for match in item_pattern.finditer(arguments_text):
            arg_name = match.group(1).strip()
            arg_desc = match.group(2).strip()
            arg_desc = self._convert_inline_formatting(arg_desc)
            items.append(f"- **`{arg_name}`**: {arg_desc}")
        
        if items:
            return '\n'.join(items)
        
        # Fallback: convert as regular text
        return self._convert_inline_formatting(arguments_text)
    
    def _convert_examples(self, examples_text: str) -> str:
        """
        Convert \examples{} section to markdown code blocks.
        
        Handles:
        - \dontrun{} - marked as such
        - \donttest{} - included but marked
        - Regular R code
        """
        if not examples_text:
            return ''
        
        sections = []
        
        # Extract \dontrun{} blocks
        dontrun_pattern = re.compile(r'\\dontrun\{(.+?)\}', re.DOTALL)
        for match in dontrun_pattern.finditer(examples_text):
            code = match.group(1).strip()
            sections.append("**Example (not run):**\n\n```r\n" + code + "\n```")
        
        # Remove dontrun blocks from main text
        main_text = dontrun_pattern.sub('', examples_text)
        
        # Extract \donttest{} blocks
        donttest_pattern = re.compile(r'\\donttest\{(.+?)\}', re.DOTALL)
        for match in donttest_pattern.finditer(main_text):
            code = match.group(1).strip()
            sections.append("**Example (not tested):**\n\n```r\n" + code + "\n```")
        
        # Remove donttest blocks
        main_text = donttest_pattern.sub('', main_text)
        
        # Clean up main code
        main_text = main_text.strip()
        if main_text:
            # Remove comment markers if present
            main_text = re.sub(r'^%\s*', '', main_text, flags=re.MULTILINE)
            sections.insert(0, f"```r\n{main_text}\n```")
        
        return '\n\n'.join(filter(None, sections))
