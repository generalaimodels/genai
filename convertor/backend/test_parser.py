"""Quick test for the markdown parser."""
from core.parser import MarkdownParser

def test_parser():
    parser = MarkdownParser()
    
    # Test basic markdown
    md = """# Test Document

This is a test with **bold** and *italic* text.

## Math Examples

Inline math: $E = mc^2$

Block math:
$$
\\int_0^1 x\\,dx = \\frac{1}{2}
$$

Also block with brackets:
\\[
\\sum_{i=1}^n i = \\frac{n(n+1)}{2}
\\]

## Code Block

```python
def hello():
    print("Hello, World!")
```

## Mermaid

```mermaid
graph TD
    A --> B
```

> [!NOTE]
> This is a note alert.

- [ ] Task 1
- [x] Task 2
"""
    
    result = parser.parse(md)
    
    print("=" * 60)
    print("Parsed successfully!")
    print(f"HTML length: {len(result.content_html)}")
    print(f"Headings found: {len(result.headings)}")
    for h in result.headings:
        print(f"  - H{h.level}: {h.text} (#{h.id})")
    print("=" * 60)
    print("\nHTML Preview (first 1000 chars):")
    print(result.content_html[:1000])
    
if __name__ == "__main__":
    test_parser()
