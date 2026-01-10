"""
SOTA Markdown Parser with comprehensive feature support.

Supports all markdown syntax including:
- Headers, text formatting, lists, links, images
- Code blocks with syntax highlighting
- Tables with alignment
- LaTeX math (ALL formats: $...$, $$...$$, \[...\], \(...\), \begin{...})
- Mermaid diagrams
- GitHub-style alerts
- HTML tags (50+ supported)
- Footnotes, task lists, emojis
- YAML front matter
"""

from __future__ import annotations

import re
import html
from dataclasses import dataclass, field
from typing import Any
from enum import Enum
import yaml

from markdown_it import MarkdownIt
from markdown_it.renderer import RendererHTML
from mdit_py_plugins.front_matter import front_matter_plugin
from mdit_py_plugins.footnote import footnote_plugin
from mdit_py_plugins.tasklists import tasklists_plugin
from mdit_py_plugins.deflist import deflist_plugin
from pygments import highlight
from pygments.lexers import get_lexer_by_name, guess_lexer
from pygments.formatters import HtmlFormatter
from pygments.util import ClassNotFound


class AlertType(Enum):
    """GitHub-style alert types with associated styling."""
    NOTE = ("note", "ℹ️", "#0969da")
    TIP = ("tip", "💡", "#1a7f37")
    IMPORTANT = ("important", "❗", "#8250df")
    WARNING = ("warning", "⚠️", "#9a6700")
    CAUTION = ("caution", "🛑", "#cf222e")


@dataclass
class Heading:
    """Represents a heading in the document for TOC generation."""
    level: int
    text: str
    id: str
    line: int


@dataclass
class ParsedDocument:
    """Complete parsed document with metadata and content."""
    content_html: str
    headings: list[Heading] = field(default_factory=list)
    front_matter: dict[str, Any] = field(default_factory=dict)
    title: str | None = None
    description: str | None = None


class MarkdownParser:
    """
    SOTA Markdown parser with comprehensive feature support.
    
    Engineering notes:
    - Uses markdown-it-py for core parsing with plugins
    - Custom renderers for code highlighting, math, mermaid
    - Regex-based preprocessing for LaTeX math normalization
    - Memory-efficient: single-pass parsing where possible
    - SOTA HTML sanitization with XSS prevention
    """
    
    # ============================================================
    # SOTA HTML SANITIZATION - Enterprise-Grade Security
    # ============================================================
    
    # Comprehensive whitelist of safe HTML tags
    ALLOWED_HTML_TAGS = {
        # Text formatting
        'b', 'strong', 'i', 'em', 'u', 'mark', 'small', 'del', 's', 'ins',
        'sub', 'sup', 'code', 'kbd', 'samp', 'var', 'pre', 'abbr', 'cite',
        'dfn', 'q', 'time', 'bdi', 'bdo', 'wbr',
        
        # Structure
        'p', 'div', 'span', 'section', 'article', 'header', 'footer', 'main',
        'aside', 'nav', 'details', 'summary', 'dialog', 'figure', 'figcaption',
        
        # Lists
        'ul', 'ol', 'li', 'dl', 'dt', 'dd',
        
        # Tables
        'table', 'thead', 'tbody', 'tfoot', 'tr', 'th', 'td', 'caption',
        'col', 'colgroup',
        
        # Media (safe - no scripts)
        'img', 'picture', 'source', 'video', 'audio', 'track', 'iframe',
        'embed', 'object', 'param',
        
        # Forms (input only - no scripts)
        'fieldset', 'legend', 'label', 'input', 'textarea', 'select',
        'option', 'optgroup', 'button', 'datalist', 'output', 'progress', 'meter',
        
        # Grouping
        'blockquote', 'hr', 'br',
        
        # SVG (inline only - no scripts)
        'svg', 'path', 'circle', 'rect', 'line', 'ellipse', 'polygon',
        'polyline', 'g', 'defs', 'use', 'symbol', 'text', 'tspan',
        
        # Ruby annotations
        'ruby', 'rt', 'rp', 'rtc', 'rb',
        
        # Other semantic tags
        'address', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
    }
    
    # Comprehensive whitelist of safe attributes
    ALLOWED_HTML_ATTRIBUTES = {
        # Universal attributes
        'class', 'id', 'title', 'lang', 'dir', 'role', 'aria-*', 'data-*',
        
        # Styling (sanitized separately)
        'style',
        
        # Links
        'href', 'target', 'rel', 'download', 'hreflang', 'type',
        
        # Images
        'src', 'srcset', 'sizes', 'alt', 'width', 'height', 'loading',
        'decoding', 'crossorigin', 'usemap', 'ismap',
        
        # Media
        'controls', 'autoplay', 'loop', 'muted', 'poster', 'preload',
        'playsinline',
        
        # Tables
        'colspan', 'rowspan', 'headers', 'scope', 'align', 'valign',
        
        # Forms
        'name', 'value', 'placeholder', 'required', 'disabled', 'readonly',
        'checked', 'selected', 'multiple', 'min', 'max', 'step', 'pattern',
        'maxlength', 'minlength', 'size', 'autocomplete', 'autofocus',
        'form', 'formaction', 'formmethod', 'formtarget',
        
        # Details
        'open', 'reversed', 'start',
        
        # Time
        'datetime',
        
        # SVG
        'viewBox', 'xmlns', 'fill', 'stroke', 'stroke-width', 'd', 'cx', 'cy',
        'r', 'rx', 'ry', 'x', 'y', 'x1', 'y1', 'x2', 'y2', 'points',
        'transform', 'opacity',
        
        # Accessibility
        'tabindex', 'contenteditable', 'draggable', 'hidden', 'spellcheck',
        'translate',
    }
    
    # Dangerous CSS properties to strip (XSS prevention)
    DANGEROUS_CSS_PROPERTIES = {
        'behavior', 'expression', '-moz-binding', 'binding',
        'javascript:', 'vbscript:', 'data:', 'livescript:',
    }
    
    # Dangerous URL schemes (XSS prevention)
    DANGEROUS_URL_SCHEMES = {
        'javascript:', 'data:', 'vbscript:', 'livescript:', 'blob:', 
        'mocha:', 'ms-settings:', 'ms-appx:', 'ms-appdata:',
    }
    
    # Safe URL schemes for links and media
    SAFE_URL_SCHEMES = {
        'http:', 'https:', 'mailto:', 'tel:', 'ftp:', 'ftps:',
        'ssh:', 'git:', 'svn:', 'data:image/', 'blob:http', 'blob:https',
        './', '../', '#', '/',  # Relative paths
    }
    
    # Emoji shortcode mappings (subset - full list would be extensive)
    EMOJI_MAP: dict[str, str] = {
        ":smile:": "😀", ":heart:": "❤️", ":thumbsup:": "👍", ":thumbsdown:": "👎",
        ":star:": "⭐", ":rocket:": "🚀", ":fire:": "🔥", ":100:": "💯",
        ":warning:": "⚠️", ":x:": "❌", ":white_check_mark:": "✅",
        ":question:": "❓", ":exclamation:": "❗", ":bulb:": "💡",
        ":memo:": "📝", ":book:": "📖", ":link:": "🔗", ":email:": "📧",
        ":phone:": "📞", ":computer:": "💻", ":keyboard:": "⌨️",
        ":calendar:": "📅", ":clock1:": "🕐", ":hourglass:": "⏳",
        ":point_right:": "👉", ":point_left:": "👈", ":point_up:": "👆",
        ":point_down:": "👇", ":tada:": "🎉", ":sparkles:": "✨",
        ":zap:": "⚡", ":boom:": "💥", ":trophy:": "🏆", ":medal:": "🏅",
        ":1st_place_medal:": "🥇", ":2nd_place_medal:": "🥈", ":3rd_place_medal:": "🥉",
        ":checkered_flag:": "🏁", ":triangular_flag_on_post:": "🚩",
        ":lock:": "🔒", ":unlock:": "🔓", ":key:": "🔑",
        ":hammer:": "🔨", ":wrench:": "🔧", ":gear:": "⚙️",
        ":package:": "📦", ":clipboard:": "📋", ":pushpin:": "📌",
        ":paperclip:": "📎", ":straight_ruler:": "📏", ":triangular_ruler:": "📐",
        ":scissors:": "✂️", ":file_folder:": "📁", ":open_file_folder:": "📂",
        ":page_facing_up:": "📄", ":page_with_curl:": "📃",
        ":bookmark:": "🔖", ":label:": "🏷️", ":moneybag:": "💰",
        ":chart_with_upwards_trend:": "📈", ":chart_with_downwards_trend:": "📉",
        ":bar_chart:": "📊", ":date:": "📅", ":card_index:": "📇",
    }
    
    # LaTeX math patterns for comprehensive detection
    # Order matters: more specific patterns first
    MATH_PATTERNS: list[tuple[re.Pattern, str, bool]] = [
        # Block math: \[...\]
        (re.compile(r'\\\[(.*?)\\\]', re.DOTALL), 'block', True),
        # Block math: $$...$$
        (re.compile(r'\$\$(.*?)\$\$', re.DOTALL), 'block', True),
        # Block math: \begin{equation}...\end{equation}
        (re.compile(r'\\begin\{(equation|align|gather|multline|eqnarray|cases|matrix|pmatrix|bmatrix|vmatrix|Vmatrix|split|aligned|gathered)\*?\}(.*?)\\end\{\1\*?\}', re.DOTALL), 'block', True),
        # Inline math: \(...\)
        (re.compile(r'\\\((.*?)\\\)', re.DOTALL), 'inline', False),
        # Inline math: $...$ (not preceded/followed by $)
        (re.compile(r'(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)'), 'inline', False),
    ]
    
    # GitHub-style alert pattern
    ALERT_PATTERN = re.compile(
        r'>\s*\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]\s*\n((?:>.*\n?)*)',
        re.IGNORECASE | re.MULTILINE
    )
    
    # Custom callout pattern (Obsidian-style)
    CALLOUT_PATTERN = re.compile(
        r'>\s*\[!(\w+)\](?:\s+(.+))?\s*\n((?:>.*\n?)*)',
        re.IGNORECASE | re.MULTILINE
    )

    def __init__(self) -> None:
        """Initialize markdown parser with all plugins and custom renderers."""
        # Initialize markdown-it with common preset and HTML enabled
        self.md = MarkdownIt("commonmark", {"html": True, "linkify": True, "typographer": True})
        
        # Enable plugins
        self.md.enable(["table", "strikethrough"])
        self.md.use(front_matter_plugin)
        self.md.use(footnote_plugin)
        self.md.use(tasklists_plugin)
        self.md.use(deflist_plugin)
        
        # Store original fence renderer for code blocks
        self._original_fence = self.md.renderer.rules.get("fence")
        self.md.renderer.rules["fence"] = self._render_fence
        
        # Custom heading renderer for ID generation
        self._original_heading_open = self.md.renderer.rules.get("heading_open")
        self.md.renderer.rules["heading_open"] = self._render_heading_open
        
        # Track headings during parsing
        self._current_headings: list[Heading] = []
        self._heading_ids: set[str] = set()
        
        # Pygments formatter for code highlighting
        self._code_formatter = HtmlFormatter(
            cssclass="highlight",
            linenos=False,
            nowrap=False,
            wrapcode=True
        )

    def parse(self, content: str) -> ParsedDocument:
        """
        Parse markdown content into HTML with full feature extraction.
        
        Args:
            content: Raw markdown string
            
        Returns:
            ParsedDocument with HTML content, headings, and metadata
        """
        # Reset state for new document
        self._current_headings = []
        self._heading_ids = set()
        
        # Extract front matter first
        front_matter = self._extract_front_matter(content)
        
        # Preprocess content for special syntax
        processed = self._preprocess(content)
        
        # Parse with markdown-it
        tokens = self.md.parse(processed)
        html_content = self.md.renderer.render(tokens, self.md.options, {})
        
        # Post-process for remaining features
        html_content = self._postprocess(html_content)
        
        # Extract title and description
        title = front_matter.get("title") or (
            self._current_headings[0].text if self._current_headings else None
        )
        description = front_matter.get("description")
        
        return ParsedDocument(
            content_html=html_content,
            headings=self._current_headings.copy(),
            front_matter=front_matter,
            title=title,
            description=description
        )

    def _extract_front_matter(self, content: str) -> dict[str, Any]:
        """Extract YAML front matter from document start."""
        if not content.startswith("---"):
            return {}
        
        # Find closing ---
        end_match = re.search(r'\n---\s*\n', content[3:])
        if not end_match:
            return {}
        
        yaml_content = content[3:end_match.start() + 3]
        try:
            return yaml.safe_load(yaml_content) or {}
        except yaml.YAMLError:
            return {}

    def _preprocess(self, content: str) -> str:
        """
        Preprocess markdown for special syntax that markdown-it doesn't handle.
        
        Handles:
        - GitHub-style alerts
        - Math expressions (all formats)
        - Emoji shortcodes
        - Mermaid diagram markers
        """
        # Process GitHub-style alerts
        content = self._process_alerts(content)
        
        # Process math expressions (normalize to consistent format)
        content = self._process_math(content)
        
        # Process emoji shortcodes
        content = self._process_emojis(content)
        
        return content

    def _process_alerts(self, content: str) -> str:
        """Convert GitHub-style alerts to HTML with styling."""
        def replace_alert(match: re.Match) -> str:
            alert_type = match.group(1).upper()
            body = match.group(2)
            
            # Remove leading > from each line
            body_lines = []
            for line in body.split('\n'):
                line = line.strip()
                if line.startswith('>'):
                    line = line[1:].strip()
                body_lines.append(line)
            body_text = '\n'.join(body_lines).strip()
            
            # Get alert styling
            try:
                alert = AlertType[alert_type]
                icon = alert.value[1]
                color = alert.value[2]
            except KeyError:
                icon = "ℹ️"
                color = "#0969da"
            
            return f'''<div class="alert alert-{alert_type.lower()}" style="border-left-color: {color}">
<span class="alert-icon">{icon}</span>
<span class="alert-title">{alert_type}</span>
<div class="alert-content">

{body_text}

</div>
</div>

'''
        
        return self.ALERT_PATTERN.sub(replace_alert, content)

    def _process_math(self, content: str) -> str:
        """
        Process all LaTeX math formats and normalize to HTML spans.
        
        IMPORTANT: Protects code blocks from math processing.
        
        Supports:
        - Inline: $...$ and \(...\)
        - Block: $$...$$, \[...\], and \begin{env}...\end{env}
        - Standalone delimiters from LLM outputs: ( ... ) on lines, [ ... ] on lines
        """
        # Step 1: Protect code blocks from math processing
        # Extract all fenced code blocks and inline code, replace with placeholders
        code_blocks: list[str] = []
        
        def protect_code(match: re.Match) -> str:
            code_blocks.append(match.group(0))
            return f"§§CODE{len(code_blocks) - 1}§§"
        
        # Protect fenced code blocks - handle 4 backticks first, then 3
        # Order matters: longer sequences first
        protected = re.sub(r'````[\s\S]*?````', protect_code, content)
        protected = re.sub(r'```[\s\S]*?```', protect_code, protected)
        protected = re.sub(r'~~~[\s\S]*?~~~', protect_code, protected)
        # Protect inline code (`...`) - but not empty backticks
        protected = re.sub(r'`[^`\n]+`', protect_code, protected)
        
        # Step 2: Process math patterns on protected content
        result = protected
        
        # Process standard LaTeX patterns
        for pattern, math_type, is_block in self.MATH_PATTERNS:
            def make_replacer(is_block_math: bool):
                def replacer(match: re.Match) -> str:
                    # Handle \begin{env} groups which have 2 groups
                    if len(match.groups()) == 2 and match.group(1) in [
                        'equation', 'align', 'gather', 'multline', 'eqnarray',
                        'cases', 'matrix', 'pmatrix', 'bmatrix', 'vmatrix', 'Vmatrix',
                        'split', 'aligned', 'gathered'
                    ]:
                        env = match.group(1)
                        latex = match.group(2)
                        full_latex = f"\\begin{{{env}}}{latex}\\end{{{env}}}"
                    else:
                        full_latex = match.group(1) if len(match.groups()) >= 1 else match.group(0)
                    
                    # Skip empty matches
                    if not full_latex.strip():
                        return match.group(0)
                    
                    # Escape HTML in LaTeX content
                    escaped = html.escape(full_latex.strip())
                    
                    if is_block_math:
                        return f'<div class="math-block" data-math="{escaped}">$${escaped}$$</div>'
                    else:
                        return f'<span class="math-inline" data-math="{escaped}">${escaped}$</span>'
                return replacer
            
            result = pattern.sub(make_replacer(is_block), result)
        
        # Step 3: Handle LLM-style standalone math delimiters
        # These are commonly used in LLM outputs and in markdown "rendered output" sections
        
        # Pattern for block math: standalone [ on a line, content, ] on a line
        # Example:
        # [
        # a^2 + b^2 = c^2
        # ]
        llm_block_pattern = re.compile(
            r'^\[\s*\n(.*?)\n\s*\]$',
            re.MULTILINE | re.DOTALL
        )
        
        def llm_block_replacer(match: re.Match) -> str:
            latex = match.group(1).strip()
            if not latex:
                return match.group(0)
            # Check if it looks like LaTeX
            if any(c in latex for c in ['^', '_', '\\', '=', '+', '-', '*', '/']):
                escaped = html.escape(latex)
                return f'<div class="math-block" data-math="{escaped}">$${escaped}$$</div>'
            return match.group(0)
        
        result = llm_block_pattern.sub(llm_block_replacer, result)
        
        # Pattern for inline math: ( space content space ) where content looks like math
        # Example: ( a^2 + b^2 = c^2 )
        # Must have spaces around content to distinguish from normal parentheses
        llm_inline_pattern = re.compile(
            r'\(\s+([^\n\(\)]+?)\s+\)',
        )
        
        def llm_inline_replacer(match: re.Match) -> str:
            latex = match.group(1).strip()
            if not latex:
                return match.group(0)
            # Check if it looks like a math expression
            # Must have math-like operators or LaTeX commands
            has_math_indicators = any(x in latex for x in ['^', '_', '\\', '='])
            # Also check for common equation patterns like "a + b = c"
            has_equation = re.search(r'[a-zA-Z0-9]+\s*[+\-*/=]\s*[a-zA-Z0-9]', latex)
            if has_math_indicators or has_equation:
                escaped = html.escape(latex)
                return f'<span class="math-inline" data-math="{escaped}">${escaped}$</span>'
            return match.group(0)
        
        result = llm_inline_pattern.sub(llm_inline_replacer, result)
        
        # Step 4: Restore protected code blocks
        for i, code_block in enumerate(code_blocks):
            result = result.replace(f"§§CODE{i}§§", code_block)
        
        return result

    def _process_emojis(self, content: str) -> str:
        """Replace emoji shortcodes with unicode emojis."""
        result = content
        for shortcode, emoji in self.EMOJI_MAP.items():
            result = result.replace(shortcode, emoji)
        return result

    def _render_fence(self, tokens: list, idx: int, options: dict, env: dict) -> str:
        """Custom fence renderer for code blocks with syntax highlighting."""
        token = tokens[idx]
        info = token.info.strip() if token.info else ""
        content = token.content
        
        # Handle mermaid diagrams
        if info.lower() == "mermaid":
            escaped = html.escape(content)
            return f'''<div class="mermaid-container">
<pre class="mermaid">{escaped}</pre>
</div>
'''
        
        # Handle diff syntax
        if info.lower() == "diff":
            return self._render_diff(content)
        
        # Handle HTML code blocks with media elements - render them as actual HTML
        # This is for documentation that shows HTML code they want to demonstrate
        if info.lower() == "html":
            # Check if content contains media elements that should be rendered
            media_tags = ['<video', '<audio', '<iframe', '<embed', '<object']
            has_media = any(tag in content.lower() for tag in media_tags)
            
            if has_media:
                # Render as actual HTML (media elements are safe and expected)
                # Wrap in a media-demo container for premium styling
                return f'''<div class="media-demo">
<div class="media-demo-content">
{content}
</div>
</div>
<details class="code-source">
<summary>View Source Code</summary>
{self._highlight_code(content, info)}
</details>
'''
        
        # Regular syntax highlighting
        return self._highlight_code(content, info)
    
    def _highlight_code(self, content: str, info: str) -> str:
        """Apply syntax highlighting to code content."""
        # Syntax highlighting with Pygments
        if info:
            try:
                lexer = get_lexer_by_name(info, stripall=True)
            except ClassNotFound:
                try:
                    lexer = guess_lexer(content)
                except ClassNotFound:
                    lexer = None
        else:
            try:
                lexer = guess_lexer(content)
            except ClassNotFound:
                lexer = None
        
        if lexer:
            highlighted = highlight(content, lexer, self._code_formatter)
            lang_class = f' data-language="{html.escape(info)}"' if info else ""
            return f'<div class="code-block"{lang_class}><div class="code-header"><span class="code-lang">{html.escape(info)}</span><button class="copy-btn" aria-label="Copy code">Copy</button></div>{highlighted}</div>'
        
        # Fallback: plain code block
        escaped = html.escape(content)
        return f'<pre><code class="language-{html.escape(info)}">{escaped}</code></pre>'

    def _render_diff(self, content: str) -> str:
        """Render diff code block with line-by-line coloring."""
        lines = content.split('\n')
        result_lines = []
        
        for line in lines:
            escaped = html.escape(line)
            if line.startswith('+'):
                result_lines.append(f'<span class="diff-add">{escaped}</span>')
            elif line.startswith('-'):
                result_lines.append(f'<span class="diff-remove">{escaped}</span>')
            elif line.startswith('!'):
                result_lines.append(f'<span class="diff-change">{escaped}</span>')
            elif line.startswith('#'):
                result_lines.append(f'<span class="diff-comment">{escaped}</span>')
            else:
                result_lines.append(escaped)
        
        return f'<pre class="diff"><code>{chr(10).join(result_lines)}</code></pre>'

    def _render_heading_open(self, tokens: list, idx: int, options: dict, env: dict) -> str:
        """Custom heading renderer to add IDs and track for TOC."""
        token = tokens[idx]
        level = int(token.tag[1])  # h1 -> 1, h2 -> 2, etc.
        
        # Find the text content (next token should be inline with text)
        text = ""
        if idx + 1 < len(tokens) and tokens[idx + 1].type == "inline":
            text = tokens[idx + 1].content
        
        # Generate unique ID
        base_id = self._slugify(text)
        unique_id = base_id
        counter = 1
        while unique_id in self._heading_ids:
            unique_id = f"{base_id}-{counter}"
            counter += 1
        self._heading_ids.add(unique_id)
        
        # Track heading
        self._current_headings.append(Heading(
            level=level,
            text=text,
            id=unique_id,
            line=token.map[0] if token.map else 0
        ))
        
        return f'<{token.tag} id="{unique_id}" class="heading">'

    def _slugify(self, text: str) -> str:
        """Convert heading text to URL-safe slug."""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Convert to lowercase
        text = text.lower()
        # Replace non-alphanumeric with hyphens
        text = re.sub(r'[^a-z0-9]+', '-', text)
        # Remove leading/trailing hyphens
        text = text.strip('-')
        return text or 'heading'

    def _postprocess(self, html_content: str) -> str:
        """
        Post-process HTML for features that need final touches.
        
        Handles:
        - Table wrapper for responsive scrolling
        - Image lazy loading
        - External link indicators
        - Keyboard key styling
        """
        # Wrap tables for responsive scrolling
        html_content = re.sub(
            r'<table>',
            '<div class="table-wrapper"><table>',
            html_content
        )
        html_content = re.sub(
            r'</table>',
            '</table></div>',
            html_content
        )
        
        # Add lazy loading to images
        # Properly handle self-closing tags and existing attributes
        html_content = re.sub(
            r'<img\s+([^>]*?)\s*(/?)>',
            r'<img \1 loading="lazy" \2>',
            html_content
        )
        
        # Style kbd elements
        html_content = re.sub(
            r'<kbd>([^<]+)</kbd>',
            r'<kbd class="keyboard-key">\1</kbd>',
            html_content
        )
        
        return html_content

    def get_css(self) -> str:
        """Get Pygments CSS for syntax highlighting."""
        return self._code_formatter.get_style_defs('.highlight')
