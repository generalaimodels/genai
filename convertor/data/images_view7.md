
# Comprehensive Markdown → HTML Converter Test Suite

This document is designed to **exhaustively validate Markdown-to-HTML conversion** across structural, inline, block, and edge-case scenarios.

---

## 1. Headings (All Levels)
# H1 Heading
## H2 Heading
### H3 Heading
#### H4 Heading
##### H5 Heading
###### H6 Heading

---

## 2. Text Formatting
**Bold Text**  
*Italic Text*  
***Bold + Italic***  
~~Strikethrough~~  
`Inline Code`  
==Highlight (non-standard)==

---

## 3. Paragraphs & Line Breaks
This is a paragraph.

This is another paragraph separated by a blank line.  
This line uses an explicit line break.

---

## 4. Blockquotes
> Single-level blockquote
>> Nested blockquote
>>> Deeply nested blockquote

---

## 5. Lists

### Unordered List
- Item A
  - Sub Item A1
    - Sub Item A1.1
- Item B

### Ordered List
1. First
2. Second
   1. Nested Second-1
   2. Nested Second-2
3. Third

### Task List
- [x] Completed task
- [ ] Pending task

---

## 6. Links
[Relative Link](./docs/readme.md)  
[Absolute Link](https://example.com)  
<https://example.com>

---

## 7. Images
![Background](./Images/background.png)
![Photo](./Images/my_photo.png)
![Face](./Images/only_face.png)

---

## 8. Tables
| Column A | Column B | Column C |
|----------|----------|----------|
| A1       | B1       | C1       |
| A2       | B2       | C2       |

Alignment Test:

| Left | Center | Right |
|:-----|:------:|------:|
| L1   | C1     | R1    |

---

## 9. Code Blocks

### Fenced Code (Language Specified)
```python
def markdown_test():
    return "Markdown to HTML validation"
````

### Fenced Code (No Language)

```
Plain text code block
Line 2
```

### Indented Code

```
Indented code line
Another line
```

---

## 10. Horizontal Rules

---

---

---

---

## 11. HTML Inside Markdown

<div style="border:1px solid #ccc; padding:10px;">
  <strong>Embedded HTML Block</strong><br>
  Should render correctly.
</div>

---

## 12. Escaping Characters

*Not Italic*
# Not a heading
`Not code`

---

## 13. Footnotes

This is a sentence with a footnote.[^1]

[^1]: This is the footnote content.

---

## 14. Definition List (Non-standard)

Term 1
: Definition 1

Term 2
: Definition 2

---

## 15. Emojis (Unicode)

✅ ❌ ⚠️ 🚀

---

## 16. Mathematical Expressions (If Supported)

Inline math: $E = mc^2$

Block math:
$$
\int_0^\infty e^{-x} dx = 1
$$

---

## 17. Nested Mixed Content

1. **Bold Item**

   * *Italic Sub-item*

     * `Code Sub-item`

       > Blockquote inside list

---

## 18. Edge Cases

Empty line below:

Multiple     spaces     test.

---

## 19. Two-Column Layout (HTML + Markdown)

<style>
.columns { display: flex; gap: 20px; }
.col { width: 50%; }
</style>

<div class="columns">
  <div class="col">

### Left Column

* Text
* Image
* Code

  </div>
  <div class="col">

### Right Column

> Quote
> `Inline code`

  </div>
</div>

---

## 20. End of Test Suite

If all sections render correctly, the Markdown-to-HTML converter
**passes functional and structural validation**.

```

If required, I can also:
- Produce **GitHub-flavored Markdown–only** (no HTML)
- Generate **expected HTML output** for diff-based testing
- Create **automated regression test cases**
- Split this into **unit-test-level Markdown files**

Specify the target renderer and compliance level.





# Comprehensive Markdown → HTML Converter Test Suite

This document is designed to **exhaustively validate Markdown-to-HTML conversion** across structural, inline, block, and edge-case scenarios.


## 1. Headings (All Levels)
# H1 Heading
## H2 Heading
### H3 Heading
#### H4 Heading
##### H5 Heading
###### H6 Heading



## 2. Text Formatting
**Bold Text**  
*Italic Text*  
***Bold + Italic***  
~~Strikethrough~~  
`Inline Code`  
==Highlight (non-standard)==



## 3. Paragraphs & Line Breaks
This is a paragraph.

This is another paragraph separated by a blank line.  
This line uses an explicit line break.


## 4. Blockquotes
> Single-level blockquote
>> Nested blockquote
>>> Deeply nested blockquote


## 5. Lists

### Unordered List
- Item A
  - Sub Item A1
    - Sub Item A1.1
- Item B

### Ordered List
1. First
2. Second
   1. Nested Second-1
   2. Nested Second-2
3. Third

### Task List
- [x] Completed task
- [ ] Pending task

---

## 6. Links
[Relative Link](./docs/readme.md)  
[Absolute Link](https://example.com)  
<https://example.com>

---

## 7. Images
![Background](./Images/background.png)
![Photo](./Images/my_photo.png)
![Face](./Images/only_face.png)

---

## 8. Tables
| Column A | Column B | Column C |
|----------|----------|----------|
| A1       | B1       | C1       |
| A2       | B2       | C2       |

Alignment Test:

| Left | Center | Right |
|:-----|:------:|------:|
| L1   | C1     | R1    |


## 9. Code Blocks

### Fenced Code (Language Specified)
```python
def markdown_test():
    return "Markdown to HTML validation"
```

### Fenced Code (No Language)

```
Plain text code block
Line 2
```

### Indented Code

```
Indented code line
Another line
```

---

## 10. Horizontal Rules

---

---

---

---

## 11. HTML Inside Markdown

<div style="border:1px solid #ccc; padding:10px;">
  <strong>Embedded HTML Block</strong><br>
  Should render correctly.
</div>

---

## 12. Escaping Characters

*Not Italic*
# Not a heading
`Not code`

---

## 13. Footnotes

This is a sentence with a footnote.[^1]

[^1]: This is the footnote content.

---

## 14. Definition List (Non-standard)

Term 1
: Definition 1

Term 2
: Definition 2

---

## 15. Emojis (Unicode)

✅ ❌ ⚠️ 🚀

---

## 16. Mathematical Expressions (If Supported)

Inline math: $E = mc^2$

Block math:
$$
\int_0^\infty e^{-x} dx = 1
$$

---

## 17. Nested Mixed Content

1. **Bold Item**

   * *Italic Sub-item*

     * `Code Sub-item`

       > Blockquote inside list

---

## 18. Edge Cases

Empty line below:

Multiple     spaces     test.

---

## 19. Two-Column Layout (HTML + Markdown)

<style>
.columns { display: flex; gap: 20px; }
.col { width: 50%; }
</style>

<div class="columns">
  <div class="col">

### Left Column

* Text
* Image
* Code

  </div>
  <div class="col">

### Right Column

> Quote
> `Inline code`

  </div>
</div>

---

## 20. End of Test Suite

If all sections render correctly, the Markdown-to-HTML converter
**passes functional and structural validation**.

```

If required, I can also:
- Produce **GitHub-flavored Markdown–only** (no HTML)
- Generate **expected HTML output** for diff-based testing
- Create **automated regression test cases**
- Split this into **unit-test-level Markdown files**

Specify the target renderer and compliance level.
