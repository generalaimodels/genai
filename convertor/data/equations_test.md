



## 1. Inline Mathematical Equations

Use **single dollar symbols** `$ ... $` for inline math.

```markdown
This is an inline equation: $a^2 + b^2 = c^2$
```

**Rendered output:**

This is an inline equation: ( a^2 + b^2 = c^2 )

---

## 2. Block (Display) Mathematical Equations

Use **double dollar symbols** `$$ ... $$` for standalone equations.

```markdown
$$
a^2 + b^2 = c^2
$$
```

**Rendered output:**

[
a^2 + b^2 = c^2
]

---

## 3. Fractions

```markdown
Inline: $\frac{a}{b}$

Block:
$$
\frac{x+1}{y-1}
$$
```

---

## 4. Powers and Subscripts

```markdown
Powers: $x^2$, $a^{n+1}$

Subscripts: $x_1$, $a_{ij}$
```

---

## 5. Square Roots and n-th Roots

```markdown
$\sqrt{x}$

$\sqrt[n]{x}$
```

---

## 6. Summation, Product, Limits

```markdown
Summation:
$$
\sum_{i=1}^{n} i^2
$$

Product:
$$
\prod_{k=1}^{n} k
$$

Limit:
$$
\lim_{x \to 0} \frac{\sin x}{x}
$$
```

---

## 7. Integrals

```markdown
Indefinite:
$$
\int x^2 \, dx
$$

Definite:
$$
\int_{0}^{1} x^2 \, dx
$$
```

---

## 8. Matrices

```markdown
$$
\begin{bmatrix}
1 & 2 \\
3 & 4
\end{bmatrix}
$$
```

Other matrix styles:

* `matrix`
* `pmatrix` (round brackets)
* `bmatrix` (square brackets)
* `vmatrix` (single bar)
* `Vmatrix` (double bar)

---

## 9. Greek Letters

```markdown
$\alpha \beta \gamma \Delta \lambda \theta \pi \Sigma$
```

---

## 10. Comparison and Logic Symbols

```markdown
$\le \ge \neq \approx \equiv$

$\forall x \in \mathbb{R}$

$\exists y \Rightarrow x = y^2$
```

---

## 11. Vectors and Dot / Cross Products

```markdown
Vector: $\vec{v}$

Dot product: $\vec{a} \cdot \vec{b}$

Cross product: $\vec{a} \times \vec{b}$
```

---

## 12. Piecewise Functions

```markdown
$$
f(x) =
\begin{cases}
x^2, & x \ge 0 \\
-x, & x < 0
\end{cases}
$$
```

---

## 13. Alignment (Multiple Equations)

```markdown
$$
\begin{aligned}
a + b &= c \\
x + y &= z
\end{aligned}
$$
```

---

## 14. Probability & Statistics

```markdown
$P(A|B)$

$$
E[X] = \sum x \cdot P(x)
$$

$$
\sigma^2 = \frac{1}{n}\sum (x_i - \mu)^2
$$
```

---

## 15. Code Block vs Math Block (Important Difference)

❌ **Do NOT use code blocks for math rendering**

````markdown
```math
a^2 + b^2 = c^2
````

````

✅ **Correct**
```markdown
$$
a^2 + b^2 = c^2
$$
````

---

## 16. Platforms That Support Markdown Math

| Platform                   | Math Support |
| -------------------------- | ------------ |
| GitHub README              | ❌ (limited)  |
| GitHub Issues              | ❌            |
| GitHub Pages (MathJax)     | ✅            |
| Jupyter Notebook           | ✅            |
| Obsidian                   | ✅            |
| Notion                     | ✅            |
| Markdown + MathJax / KaTeX | ✅            |

---

## Quick Summary (Telugu-style English)

* Inline math → `$ ... $`
* Big equation → `$$ ... $$`
* Fractions → `\frac{}{}`
* Power → `^`
* Subscript → `_`
* Matrix → `\begin{bmatrix} ... \end{bmatrix}`
* Complex math ki **MathJax / KaTeX mandatory**

---

Kavali ante:

* **Cheat sheet one-page**
* **GitHub-compatible workaround**
* **LaTeX vs Markdown comparison**
* **ML / AI equations template**

