Formal model of prompting

Definition
Prompting is the construction of a conditioning context that steers a pretrained model’s conditional distribution over outputs to satisfy task, format, and safety constraints.

Mathematical formulation
- Model and channels:
  - Language model $p_\phi(y \mid c)$ with parameters $\phi$ and context $c$ (tokens).
  - Decompose $c = [S \parallel I \parallel H \parallel K \parallel X]$ where:
    - $S$: system/role message(s)
    - $I$: instructions
    - $H$: dialogue history
    - $K$: retrieved/contextual knowledge
    - $X$: task-specific input
- A prompting technique is a transformation $\tau$ that maps base inputs to a context:
$$
  c = \tau(S,I,H,K,X;\theta), \quad y \sim p_\phi(\,\cdot \mid c)
$$
where $\theta$ are prompt hyperparameters.
- Utility and constraints:
$$
\max_{\tau \in \mathcal{T}} \mathbb{E}_{(X,Y^*) \sim \mathcal{D}} \big[ U(y, Y^*) \big]
\quad \text{s.t.} \quad
y \in \mathcal{L}(G),\; C(y) \le B,\; S_{\text{safety}}(y)=1
$$

where $G$ is a grammar (format), $C$ cost, $B$ budget.

Importance
- Provides a principled view of prompt design as constrained distribution shaping.
- Unifies reasoning, retrieval, and formatting as channel-specific control of $c$.


Core prompt engineering techniques

1) Role-based prompting

- Definition
  Assigns explicit roles (system, user, assistant) to condition the model via channel-specific priors over behavior.

- Mathematical formulation
  $$
  p_\phi(y \mid c) = p_\phi\big(y \mid [\texttt{<system>} S; \texttt{<user>} X; \texttt{<assistant>} \varnothing]\big)
  $$
  Role tokens alter the attention prior and decoding policy; let $\rho$ denote role embeddings, then logits shift $\Delta \ell = f_\phi(\rho, S)$.

- Detailed explanation
  Roles separate long-horizon constraints (system), task content (user), and generation policy (assistant). This reduces instruction–content interference and improves adherence.

- Importance
  In multi-turn, multi-agent, and safety-critical settings, role separation increases instruction-following fidelity and reduces prompt injection.

- Example prompt
```bash
  System:
  """
  You are a meticulous scientific writing assistant. Always cite sources if used and output valid JSON as specified.
  """
  User:
  """
  Task: Summarize the methodology section into 3 bullet points.
  Text: <paper> ... </paper>
  Output schema: {"bullets": string[3]}
  """
```
2) Instructional clarity

- Definition
  Explicit, unambiguous task specifications with success criteria and constraints.

- Mathematical formulation
  Replace vague $I$ by constraint set $\mathcal{C}$ and validator $V$: enforce $V(y; \mathcal{C})=1$.
  $$
  \min_{y} \mathbb{H}[Y \mid S,I,X] \quad \text{via } I \text{ that prunes infeasible outputs}
  $$

- Detailed explanation
  Precise verbs, scope, inputs, outputs, and constraints reduce entropy of the target distribution and decoding variance.

- Importance
  Improves reproducibility and reduces retry/parse failures.

- Example prompt
```bash
  """
  Instruction: Extract all function names and their docstrings from the code.
  Constraints: 
  - Return JSON with keys: "functions": [{ "name": string, "doc": string }]
  - Exclude private functions (prefix "_")
  Data: ```python ... ```
  Validation: JSON must parse.
  """
```
3) Few-shot prompting

- Definition
  Provide $k$ input–output exemplars to induce in-context learning of the mapping.

- Mathematical formulation
  $$
  c = [I; (x_1,y_1), \dots, (x_k,y_k); x_{\text{test}}], \quad y \sim p_\phi(\cdot \mid c)
  $$
  Acts as Bayesian updating with pseudo-observations; improves posterior $p(y\mid x)$ alignment.

- Detailed explanation
  Examples instantiate task format, style, edge cases, and implicit constraints; transformers can implement gradient-free function induction via attention over exemplars.

- Importance
  Reduces the need for fine-tuning; critical for niche formats and domain-specific normalization.

- Example prompt
```bash
  """
  Task: Convert English to SQL for a PostgreSQL schema.

  Example 1
  Q: "Top 5 customers by total spend last year"
  A: SELECT customer_id, SUM(amount) AS total FROM orders WHERE order_date >= '2024-01-01' AND order_date < '2025-01-01' GROUP BY 1 ORDER BY total DESC LIMIT 5;

  Example 2
  Q: "Monthly signup counts in 2023"
  A: SELECT DATE_TRUNC('month', created_at) AS m, COUNT(*) FROM users WHERE created_at >= '2023-01-01' AND created_at < '2024-01-01' GROUP BY 1 ORDER BY 1;

  Q: "Average basket size by weekday"
  A:
  """
```

4) Zero-shot prompting

- Definition
  Query without task-specific exemplars, relying on pretrained priors and instruction clarity.

- Mathematical formulation
  $$
  c = [I; X], \quad y \sim p_\phi(\cdot \mid c)
  $$

- Detailed explanation
  Depends on model’s generalization; robust when the task distribution aligns with pretraining or RLHF priors.

- Importance
  Fastest path in production; useful for well-known tasks or with strict schemas.

- Example prompt
```bash
  """
  Instruction: Provide a 2-sentence abstract of the article below. Do not include citations. 
  Text: <article> ... </article>
  """
```
5) Chain-of-thought prompting (CoT)

- Definition
  Elicit explicit intermediate reasoning steps before the final answer.

- Mathematical formulation
  Introduce latent $z$ (rationale) and encourage $p(y,z \mid c)$ factorization:
  $$
  p(y \mid c) = \sum_{z} p(y \mid z,c) p(z \mid c),\quad \text{prompt: "Think step-by-step."}
  $$
  Optimize by widening support over informative $z$.

- Detailed explanation
  Externalizes internal computation into a scratchpad, increasing effective compute and enabling error localization.

- Importance
  Significant gains on arithmetic, logic, and multi-hop QA; supports verifiability.

- Example prompt
```bash
  """
  Solve and show your reasoning, then state the final answer on the last line as "Answer: <value>".
  Problem: A train leaves at 9:10 traveling 72 km/h... 
"""
```

6) Self-consistency prompting

- Definition
  Sample multiple CoT paths, aggregate the final answers to reduce variance.

- Mathematical formulation
  $$
  \hat{y} = \arg\max_{y} \sum_{i=1}^{N} \mathbb{1}\{\mathrm{ans}(z^{(i)})=y\}, \quad z^{(i)} \sim p(z \mid c)
  $$
  Approximates marginalization over $z$ by majority/plurality vote.

- Detailed explanation
  Reduces brittleness to spurious reasoning trajectories; improves calibration.

- Importance
  Critical for high-stakes reasoning; trades tokens for accuracy.

- Example prompt
  Controller:
```bash
  """
  Generate 10 independent solutions with different reasoning paths. Return only the final answers list.
  """
  
```
Base prompt (from CoT) repeated with temperature > 0.
7) ReAct (Reason + Act)

- Definition
  Interleave free-form reasoning with tool actions and observations.

- Mathematical formulation
  POMDP with states $s_t$, actions $a_t \in \{\text{tool}_i(x)\}$, observations $o_t$:
  $$
  \pi(a_t, r_t \mid h_t) \quad \text{with} \quad h_t=(x,r_{1:t-1},a_{1:t-1},o_{1:t-1})
  $$
  where $r_t$ is textual reasoning.

- Detailed explanation
  Alternates “Thought:” and “Action:” steps, executes tools (search, DB, code), then updates context with “Observation:”.

- Importance
  Enables grounded reasoning and real-time data access; reduces hallucination.

- Example prompt
```bash 
"""
  You may use TOOLS: search(query), lookup(id).
  Use the format:
  Thought: ...
  Action: search["..."]
  Observation: ...
  ... repeat ...
  Final: <answer>
  Question: What is the latest GDP figure for Japan (source + date)? 
"""
```
8) Prompt chaining

- Definition
  Decompose a task into sequential prompts whose outputs feed subsequent inputs.

- Mathematical formulation
  $$
  y_1 = f_1(c_1),\; y_2 = f_2(c_2 \oplus y_1),\; \dots,\; y_n = f_n(c_n \oplus y_{n-1})
  $$
  with $f_i$ being model calls or tools.

- Detailed explanation
  Stages: plan → gather → analyze → verify → format. Each stage narrows scope and constraints.

- Importance
  Improves reliability, modularity, and observability; aligns with workflow orchestration.

- Example prompt chain
```bash 
  1) Planner: “List sub-questions and required data sources for …”
  2) Retriever: “For each sub-question, query and extract facts …”
  3) Synthesizer: “Combine facts into a 150-word summary with citations …”
  4) Verifier: “Check consistency; list contradictions …”
  5) Formatter: “Return JSON: {summary, citations, checks}”
```
9) Instruction + context separation

- Definition
  Provide instructions in a dedicated channel and data separately with clear delimiters.

- Mathematical formulation
  $$
  c = [S; I; \langle\text{CONTEXT}\rangle K \langle/\text{CONTEXT}\rangle; X]
  $$
  Minimizes cross-attention leakage between $I$ and $K$.

- Detailed explanation
  Prevents user content from overriding instructions; improves robustness against prompt injection.

- Importance
  Essential for enterprise/RAG security.

- Example prompt
```bash  """
  Instruction: Answer only from the CONTEXT. If insufficient, say "Not found."
  CONTEXT:
  <<< ... retrieved documents ... >>>
  Question: ...
  """
```
10) Output formatting control

- Definition
  Constrain outputs to a schema (JSON, Markdown, tables) for machine-readability.

- Mathematical formulation
  Grammar-constrained decoding:
  $$
  y \in \mathcal{L}(G_{\text{JSON}}), \quad \text{decode with } \mathrm{constrain}(G)
  $$
  or validate $V(y)=1$.

- Detailed explanation
  Prompts specify exact keys, types, and examples; optionally use JSON mode or EBNF grammars.

- Importance
  Eliminates parse errors and enables deterministic downstream processing.

- Example prompt
```bash  """
  Return strictly valid JSON:
  {
    "title": string,
    "bullets": string[3],
    "confidence": number in [0,1]
  }
  Content: ...
  """
```

Advanced prompting strategies

11) Role-playing / persona prompting

- Definition
  Condition the model to emulate a domain expert persona with specific priors and style.

- Mathematical formulation
  Prior injection: $p(y \mid c) \propto p(y \mid c, \pi)$ where $\pi$ encodes persona embeddings.

- Detailed explanation
  Persona calibrates tone, vocabulary, and domain assumptions; can gate internal knowledge clusters.

- Importance
  Improves domain adherence and style control.

- Example prompt
  System:
```bash
  """
  You are a board-certified cardiologist. Use ACC/AHA terminology; cite guideline years.
  """
```
12) Delimiters for clarity

- Definition
  Use explicit markers to separate instructions, data, and constraints.

- Mathematical formulation
  Boundary tokens reduce unintended attention:
  $$
  c = [I; \langle D \rangle K \langle /D \rangle],\quad \text{mask cross-region copying}
  $$

- Detailed explanation
  Distinct segments decrease the chance of instruction contamination by user-provided text.

- Importance
  Critical in RAG and code execution prompts.

- Example prompt
```bash
"""
  Rules:
  - Answer only using DATA.
  DATA:
  """
  <BEGIN>
  ...
  <END>
  """
```
13) Iterative refinement

- Definition
  Generate, critique, and improve in multiple passes.

- Mathematical formulation
  Two-stage decoding:
  $$
  y^{(0)} \sim p(\cdot \mid c);\quad \epsilon = \kappa(y^{(0)},\mathcal{C});\quad y^{(1)} \sim p(\cdot \mid c \oplus \text{feedback}(\epsilon))
  $$

- Detailed explanation
  A critic or verifier (self or separate model) highlights violations and drives a repair step.

- Importance
  Boosts adherence to specs and correctness, especially with strict schemas.

- Example prompt
```bash
  1) Generate:
  """
  Draft a 200-word abstract; constraints: ≤ 2 passive sentences; include 1 citation.
  """
  2) Critique:
  """
  Critique the draft vs constraints; list fixes.
  """
  3) Revise:
  """
  Apply fixes; return final abstract with a single [1]-style citation.
  """
```
14) Constraint prompting

- Definition
  Impose explicit hard/soft constraints (length, tone, vocabulary, steps).

- Mathematical formulation
  Hard constraints as grammar/length caps; soft constraints via penalty:
  $$
  \arg\max_y \log p(y \mid c) - \lambda \,\Omega(y;\mathcal{C})
  $$

- Detailed explanation
  Specifies structural or stylistic limits; can be enforced by decoding or post-validation.

- Importance
  Ensures compliance for regulatory or UI requirements.

- Example prompt
  """
  Produce exactly 5 bullet points, each ≤ 12 words, imperative mood, no adverbs.
  """

15) Safety and refusal shaping

- Definition
  Encode boundaries for disallowed content and graceful refusal behavior.

- Mathematical formulation
  Constrained set $\mathcal{Y}_{\text{safe}}$ with classifier $S(y)=1$:
  $$
  \max_{y \in \mathcal{Y}_{\text{safe}}} p(y \mid c)
  $$

- Detailed explanation
  Prompts define forbidden topics, require disclaimers, and prescribe safe alternatives.

- Importance
  Mandatory for compliance, trust, and harm mitigation.

- Example prompt
```bash  """
  If asked for medical diagnosis, refuse and suggest consulting a physician. Provide general wellness information only.
  """
```
16) Meta-prompting

- Definition
  Ask the model to outline its solution strategy before solving.

- Mathematical formulation
  Introduce plan variable $\pi$:
  $$
  \pi \sim p(\pi \mid c);\quad y \sim p(y \mid c,\pi)
  $$

- Detailed explanation
  Planning focuses subsequent decoding and improves coherence; plan can be hidden or shown.

- Importance
  Enhances performance on complex, multi-step tasks.

- Example prompt
```bash  
"""
  First produce a high-level plan (3 steps) labeled PLAN:. Then execute the plan under EXECUTION: with citations.
"""
```
17) Multi-agent prompting

- Definition
  Coordinate specialized prompts/agents with aggregation.

- Mathematical formulation
  $$
  y_i \sim p_i(\cdot \mid c_i),\; \hat{y} = \mathrm{Agg}(y_1,\dots,y_m)
  $$
  where $\mathrm{Agg}$ may be voting, verifier selection, or ranker.

- Detailed explanation
  Decompose tasks into planner–solver–verifier roles; exchange intermediate artifacts with typed prompts.

- Importance
  Improves robustness and specialization at scale.

- Example prompt
```bash
  - Planner:
    """
    Decompose into subtasks with required tools and expected outputs per subtask.
    """
  - Solver:
    """
    For each subtask, act with ReAct format; return evidence.
    """
  - Verifier:
    """
    Check factuality and format; return pass/fail with fixes.
    """
```

Practical tools and best practices

18) Pin model snapshots

- Definition
  Fix the model/version to ensure reproducible behavior.

- Mathematical formulation
  Use $p_{\phi_v}$ with version $v$, keeping $\phi$ constant across runs.

- Detailed explanation
  Eliminates drift from backend updates; enables fair A/B testing.

- Importance
  Production stability and auditability.

- Example
```bash
  """
  meta: { model: "gpt-5-2025-08-07", temperature: 0.2 }
  """
```
19) Prompt evaluations (evals)

- Definition
  Systematic testing of prompts against datasets with metrics.

- Mathematical formulation
$$
\text{score}(\tau) = \frac{1}{N} \sum_{i=1}^N M\!\Big(y_i = \mathcal{A}(x_i;\tau),\, y_i^*\Big)
$$

with metrics $M$ (exact match, BLEU, factuality, parse rate).

- Detailed explanation
  Automates regression detection and prompt optimization.

- Importance
  Empirical grounding for prompt changes.

- Example
  - Dataset: 500 SQL tasks
  - Metrics: exact match, execution accuracy, JSON parse success
  - Variants: zero-shot vs few-shot vs CoT

20) Template libraries

- Definition
  Reusable, parameterized prompt templates with versioning.

- Mathematical formulation
  Template $T(\alpha)$ with parameters $\alpha$; prompt $c = T(\alpha) \oplus X$.

- Detailed explanation
  Standardizes style and constraints; reduces code duplication.

- Importance
  Consistency and maintainability across teams.

- Example
```bash
  """
  TEMPLATE v1.2: Summarization
  Instruction: Summarize in {n} bullets; ≤ {w} words each; audience: {aud}.
  """
```
21) Guardrails + validation

- Definition
  Automatic validation and repair of outputs.

- Mathematical formulation
  $$
  y \leftarrow \begin{cases}
  y & V(y)=1 \\
  \mathrm{repair}(y) & V(y)=0
  \end{cases}
  $$
  or constrained decoding $y \in \mathcal{L}(G)$.

- Detailed explanation
  JSON schema validation, regex/EBNF grammars, type-checking, unit tests for code.

- Importance
  Reduces runtime failures and supports automation.

- Example
```bash
  - Prompt requires JSON; post-validate with schema; on failure, send repair prompt:
  """
  Repair to satisfy schema: <schema>. Original: <y>.
  """
```

22) Progressive disclosure

- Definition
  Reveal information incrementally to focus attention and reduce distraction.

- Mathematical formulation
  Stage-wise conditioning $c_t = [I_t; K_t; y_{t-1}]$ with $K_t \subset K$.

- Detailed explanation
  Start with task specification, then provide minimal context per step; avoids context-window overload and injection.

- Importance
  Improves relevance and reduces hallucination in long contexts.

- Example
```bash
  - Step 1: “Summarize section 1 only: <text1>”
  - Step 2: “Now summarize section 2 only: <text2>”
  - Step 3: “Synthesize a global summary from summaries 1–2.”
```

Why these techniques work (cross-cutting mathematical intuition)

- Entropy reduction
  $$
  \mathbb{H}[Y \mid S,I,X] \downarrow \quad \text{via constraints and exemplars}
  $$
  yielding lower-variance decoding.

- Latent computation externalization (CoT, Meta, ReAct)
  Introduce auxiliary variables $z,\pi$ to expand hypothesis space and permit approximate marginalization or tool-grounded updates.

- Variance reduction (Self-consistency, Multi-agent)
  Monte Carlo averaging over independent trajectories:
  $$
  \hat{y} = \arg\max_y \sum_i \mathbb{1}\{y_i=y\}
  $$

- Constraint satisfaction (Formatting, Safety)
  Grammar/validator restricts search to feasible region $\mathcal{L}(G)$, increasing parse success and compliance probability.

- Decomposition (Chaining)
  Solve $f = f_n \circ \cdots \circ f_1$ to reduce per-stage complexity and improve local verification.


Implementation notes (operationalization)

- Decoding controls: temperature, nucleus $p$, max_tokens tuned per technique (higher for CoT/self-consistency).
- Separation with delimiters and roles to prevent prompt injection; always place rules in system channel.
- For ReAct, define a minimal, typed tool API and enforce tool call budgets.
- For formatting, prefer grammar-constrained decoding over post-hoc parsing when available.
- For evals, track: task accuracy, parse rate, refusal rate, latency, token cost, and calibration (ECE).


Compact reference: example scaffolds

- Zero-shot + formatting
  """
  System: Output valid JSON only.
  User: Extract risk factors from the text. Schema: {"factors": string[]}. Text: """..."""
  """

- Few-shot + constraint
```bash
  """
  System: You write concise release notes (≤80 words).
  User: 
  Example:
  Input: Fixed crash when opening large files.
  Output: - Resolve large-file opening crash improving stability.
  ---
  Input: Added offline mode and sync fixes.
  Output: - Introduce offline mode; improve sync reliability.
  ---
  Input: <new changes>
  Output:
  """
```

- CoT + self-consistency
```bash
  Controller: sample N=7 with temperature=0.8; majority vote the final “Answer:”.
  Base prompt:
  """
  Show step-by-step reasoning. Finally output: Answer: <value>.
  Problem: ...
  """
```
- ReAct with search
```bash
  """
  Tools: search(q), fetch(url).
  Thought: identify query
  Action: search["..."]
  Observation: ...
  Thought: ...
  Final: ...
  """
```