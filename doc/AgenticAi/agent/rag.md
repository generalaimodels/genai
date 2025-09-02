# Retrieval-Augmented Generation (RAG): Variants, Mathematics, and Practice

## Core Formalism

- Definition
  - Retrieval-Augmented Generation (RAG) is a class of architectures that condition a generator (typically an LLM) on external evidence retrieved from a corpus or the web to ground outputs, improve factuality, and extend knowledge beyond parametric memory.

- Mathematical formulation
  - Given a query $q$, corpus $\mathcal{D}$, retriever $R_\theta$, and generator $G_\phi$:
    $$
    C = R_\theta(q, \mathcal{D}), \quad p_\phi(y \mid q, C) = \prod_{t=1}^T p_\phi(y_t \mid y_{<t}, q, C)
    $$
  - Training (approximate marginalization over retrieved contexts):
    $$
    \max_{\theta,\phi} \sum_{(q,y)} \log p_\phi(y \mid q, C_{top\text{-}k}), \quad C_{top\text{-}k} = \operatorname*{arg\,top\text{-}k}_{d \in \mathcal{D}} s_\theta(q,d)
    $$
  - Contrastive retrieval learning:
    $$
    \mathcal{L}_{ret} = -\log \frac{e^{s_\theta(q, d^+)}}{e^{s_\theta(q, d^+)} + \sum_{d^- \in \mathcal{N}(q)} e^{s_\theta(q, d^-)}}
    $$
  - Hybrid scoring (dense + sparse):
    $$
    s(q,d) = \lambda \langle f_\theta(q), g_\theta(d) \rangle + (1-\lambda)\,\mathrm{BM25}(q,d)
    $$

- Conceptual explanation
  - Index documents, retrieve top-$k$ contexts, optionally re-rank/compress, condition the LLM on $q$ and $C$, generate $y$ with citations/attribution.

- Importance
  - Improves factuality and recency, reduces hallucinations, enables domain adaptation without full model retraining, and provides auditable provenance.


---

## 1) Naive (Standard) RAG

- Definition
  - A single-shot pipeline: retrieve top-$k$ text chunks and condition the LLM on concatenated contexts to generate the answer.

- Mathematical formulation
  - One-pass retrieval and generation:
    $$
    C = \{d_1,\dots,d_k\}, \quad y^* = \arg\max_y p_\phi(y \mid q, C)
    $$

- Detailed conceptual explanation
  - Steps:
    1) Chunk and index $\mathcal{D}$ using BM25 or dense embeddings.
    2) Retrieve top-$k$ by similarity.
    3) Concatenate $C$ with $q$.
    4) Generate $y$; optionally add simple citation via nearest source spans.

- Importance and role
  - Establishes a minimal grounding baseline, low complexity and cost, sufficient for many FAQ/KB and open-domain QA tasks with stable corpora.

- Improvements and advantages
  - Fast to implement, low operational overhead, good baseline factuality gains versus parametric-only LLMs.


---

## 2) Advanced RAG

- Definition
  - Enhances naive RAG with hybrid retrieval, query rewriting, cross-encoder re-ranking, context compression, multi-step retrieval, and uncertainty-aware routing.

- Mathematical formulation
  - Mixture-of-retrievers with learned weights:
    $$
    s(q,d) = \sum_{m=1}^M w_m\, s_m(q,d), \quad \sum_m w_m = 1
    $$
  - Cross-encoder re-ranking:
    $$
    r(d \mid q) = \mathrm{BERT}_{ce}([q;d])
    $$
  - Context compression operator $S$:
    $$
    \tilde{C} = S(C) \ \text{with} \ |\tilde{C}| \ll |C|
    $$

- Detailed conceptual explanation
  - Steps:
    1) Hybrid retrieve (dense+sparse+BM25).
    2) Cross-encoder re-rank top-$k$ to maximize precision@k.
    3) Query rewriting/expansion using the LLM or PRF: $q' = \mathrm{Rewrite}(q, C)$.
    4) Context compression via extractive summarization or salience filtering.
    5) Uncertainty estimation and routing to web search or second retrieval pass if needed.

- Importance and role
  - Stronger precision/recall, fewer irrelevant tokens, robust to noisy corpora, better use of context length.

- Improvements and advantages
  - Higher answer accuracy, lower context cost, improved stability and controllability, better long-form attribution.


---

## 3) Modular RAG

- Definition
  - A system design pattern where retrieval, re-ranking, compression, planning, verification, and citation are decoupled into interchangeable modules with stable interfaces.

- Mathematical formulation
  - Compositional operators:
    $$
    x_{i+1} = T_i(x_i), \quad x_0=q,\; x_1=R(q),\; x_2=\mathrm{ReRank}(x_1),\; \dots,\; y=G(x_n)
    $$
  - System-level objective (utility with latency/cost regularization):
    $$
    \max \ \mathbb{E}[\mathrm{Acc}(y)] - \alpha\,\mathbb{E}[\mathrm{Latency}] - \beta\,\mathbb{E}[\mathrm{Cost}]
    $$

- Detailed conceptual explanation
  - Steps:
    1) Define typed interfaces: Retriever, Ranker, Compressor, Router, Verifier, Generator.
    2) Implement monitoring and A/B testing per module.
    3) Enable domain-specific stacks via routing (legal, code, medical).
    4) Support hot-swapping and continuous improvement per module.

- Importance and role
  - Maintainability, rapid iteration, multi-domain scaling, safe deployment with guardrails and observability.

- Improvements and advantages
  - Easier experimentation, clearer accountability of failure modes, scalable engineering.


---

## 4) Agentic RAG

- Definition
  - An agent performs iterative tool use—planning, multi-hop retrieval, evidence evaluation, and self-correction—to synthesize answers.

- Mathematical formulation
  - POMDP:
    $$
    \mathcal{M}=(\mathcal{S},\mathcal{A},T,R,\Omega,O),\quad \pi_\psi(a_t \mid b_t)
    $$
    with belief $b_t$ over latent state; optimize
    $$
    \max_\psi \ \mathbb{E}\left[\sum_{t=1}^T R(s_t,a_t)\right]
    $$
  - Action set $\mathcal{A}$ includes: retrieve, browse, read, query-rewrite, verify, stop.

- Detailed conceptual explanation
  - Steps:
    1) Decompose the query and plan a tool-use sequence.
    2) Retrieve-hop → reason → retrieve-next (self-ask, chain-of-thought constrained).
    3) Evidence scoring and deduplication.
    4) Draft, verify with targeted retrieval, correct, and finalize with citations.

- Importance and role
  - Excels at multi-hop, complex synthesis, weakly structured web data, and tool-rich environments.

- Improvements and advantages
  - Higher coverage and reliability on complex tasks, better calibration via active verification, interpretable action traces.


---

## 5) Corrective Retrieval-Augmented Generation (Corrective RAG)

- Definition
  - A RAG pipeline with explicit detection of inadequate evidence or hallucination and corrective mechanisms (re-retrieval, editing, or selective abstention).

- Mathematical formulation
  - Confidence-gated correction:
    $$
    \hat{y} = \arg\max_y p_\phi(y \mid q, C), \quad c = \mathrm{Conf}(\hat{y}, C)
    $$
    If $c < \tau$:
    $$
    C' = R_\theta(q,\hat{y}), \quad y^* = \arg\max_y p_\phi(y \mid q, C \cup C', \hat{y})
    $$
  - Mixture with parametric knowledge via gating $g(c)$:
    $$
    p(y\mid q) = g(c)\,p_\phi(y\mid q) + (1-g(c))\,p_\phi(y\mid q,C)
    $$

- Detailed conceptual explanation
  - Steps:
    1) Generate with evidence and compute faithfulness/entailment score.
    2) If low confidence, trigger corrective retrieval (targeted queries from claims).
    3) Edit or regenerate; optionally use a verifier (NLI or fact-checker).
    4) Optionally abstain with calibrated uncertainty.

- Importance and role
  - Reduces hallucinations, increases factual precision, improves safety in high-stakes domains.

- Improvements and advantages
  - Significant gains in factuality with minimal extra cost; principled fallback behavior.


---

## 6) Knowledge-Intensive RAG (KI-RAG)

- Definition
  - RAG tailored for tasks requiring extensive, multi-hop, or specialized knowledge with coverage and compositional reasoning guarantees.

- Mathematical formulation
  - Iterative evidence accumulation:
    $$
    E_1 = R(q), \quad q_{t+1} = \mathrm{Reason}(q, E_{\le t}), \quad E_{t+1} = R(q_{t+1})
    $$
    Supervised evidence training:
    $$
    \mathcal{L} = -\log p_\phi(y^* \mid q, E^*) + \lambda \sum_t \mathcal{L}_{ret}(q_t, e_t^+)
    $$

- Detailed conceptual explanation
  - Steps:
    1) Decompose $q$ into sub-queries/entities.
    2) Retrieve and link supporting evidence per hop.
    3) Enforce consistency and coverage via entailment/rationale constraints.
    4) Generate with explicit evidence chains.

- Importance and role
  - Open-domain QA, scientific/biomedical synthesis, legal analyses, and cross-lingual knowledge tasks.

- Improvements and advantages
  - Superior recall and reasoning depth, traceable multi-evidence answers.


---

## 7) Multimodal RAG

- Definition
  - RAG where retrieval spans text, images, audio, video, or other modalities, and the generator conditions on multimodal contexts.

- Mathematical formulation
  - Cross-modal similarity:
    $$
    s(q^{(m)}, d^{(n)}) = \langle f_q^{(m)}(q^{(m)}),\, f_d^{(n)}(d^{(n)}) \rangle
    $$
  - Multimodal generation:
    $$
    p_\phi(y \mid q^{(m)}, C^{(1:N)}) = \prod_t p_\phi(y_t \mid y_{<t}, \{z^{(n)}\}_{n=1}^N, q^{(m)})
    $$
    with encodings $z^{(n)} = \mathrm{Enc}^{(n)}(C^{(n)})$.

- Detailed conceptual explanation
  - Steps:
    1) Train/fine-tune modality-specific encoders to a shared embedding space (e.g., CLIP-style).
    2) Build multimodal index and retrieval fusion.
    3) Align temporal segments for video/audio.
    4) Generate with modality adapters and cite frames/clips.

- Importance and role
  - Vision-language QA, medical imaging reports, media search, and grounded creative assistance.

- Improvements and advantages
  - Access to visual/acoustic evidence, higher grounding fidelity for non-textual tasks.


---

## 8) Memory-Augmented RAG (Memory RAG)

- Definition
  - RAG with a dynamic, user/task-specific memory that accumulates and curates facts across interactions, enabling personalization and long-horizon grounding.

- Mathematical formulation
  - Memory update:
    $$
    m_t = \mathrm{Extract}(q_t, y_t, C_t), \quad M_{t+1} = \mathrm{Dedup}\big(M_t \cup \{m_t\}\big)
    $$
  - Time-decayed retrieval:
    $$
    s_M(q, m) = \alpha \langle f(q), f(m) \rangle + \beta\,\mathrm{BM25}(q,m) + \gamma\,e^{-\delta\,\mathrm{age}(m)}
    $$

- Detailed conceptual explanation
  - Steps:
    1) Extract salient memory items (facts, preferences, entities).
    2) Summarize/deduplicate and attach metadata (time, source, certainty).
    3) Retrieve from $M$ with decay and importance priors.
    4) Merge with global retrieval as needed.

- Importance and role
  - Personalized assistants, longitudinal workflows, project memory, enterprise copilots.

- Improvements and advantages
  - Better continuity, fewer repeated queries, reduced context length via selective recall.


---

## 9) Meta-Learning / Few-Shot or Zero-Shot RAG

- Definition
  - Retrieval of demonstrations, instructions, or task programs to support in-context learning rather than (or in addition to) factual evidence.

- Mathematical formulation
  - Example subset selection:
    $$
    \mathcal{E}^*(q) = \arg\max_{\mathcal{E}:\,|\mathcal{E}|=k} \ \mathbb{E}_{(q,y)} \big[\log p_\phi(y \mid q, \mathcal{E})\big]
    $$
  - Learning an example retriever:
    $$
    \min_\theta \ \mathbb{E}_{(q,y)}\left[-\log p_\phi(y \mid q, \mathcal{E}_\theta(q))\right]
    $$

- Detailed conceptual explanation
  - Steps:
    1) Build an example bank with inputs, outputs, rationales.
    2) Retrieve diverse yet similar exemplars with anti-collision and coverage constraints.
    3) Order/format demonstrations for maximal transfer (e.g., instruction-first).
    4) Optionally compress demonstrations to fit context.

- Importance and role
  - Rapid task adaptation, code synthesis styles, mathematical reasoning patterns, domain-specific formats.

- Improvements and advantages
  - Boosts performance without fine-tuning; robust across tasks; reusable demonstration libraries.


---

## 10) Graph Retrieval-Augmented Generation (Graph-RAG)

- Definition
  - Retrieval over structured knowledge graphs (KGs) or heterogeneous graphs (text nodes + entities), enabling path-based reasoning and explicit relations.

- Mathematical formulation
  - Subgraph extraction:
    $$
    G_q = (V_q,E_q) = \mathrm{KG\text{-}Retrieve}(q)
    $$
  - Graph representation and conditioning:
    $$
    H = \mathrm{GNN}(G_q), \quad p_\phi(y \mid q, G_q) = \prod_t p_\phi(y_t \mid y_{<t}, q, \mathrm{linearize}(G_q), H)
    $$
  - Path ranking (Personalized PageRank):
    $$
    \pi = (1-\alpha)P^\top \pi + \alpha e_q
    $$

- Detailed conceptual explanation
  - Steps:
    1) Entity/relation linking from the query.
    2) Graph traversal/multi-hop expansion with path scoring.
    3) Optionally retrieve attached text for nodes/edges (heterogeneous RAG).
    4) Generate with explicit triple paths and evidence.

- Importance and role
  - High-precision, interpretable reasoning; schema-aware QA; enterprise knowledge integration.

- Improvements and advantages
  - Better compositionality, disambiguation, and verifiability; strong for multi-hop factual queries.


---

## 11) Knowledge-Augmented Generation (KAG)

- Definition
  - Generation augmented by pre-injected knowledge (triples, curated summaries) via fine-tuning or in-context scaffolds, not necessarily performing retrieval at inference.

- Mathematical formulation
  - Knowledge infusion with PEFT (LoRA):
    $$
    W' = W + A B, \quad A \in \mathbb{R}^{d\times r}, B \in \mathbb{R}^{r\times d}, \ r \ll d
    $$
  - Objective:
    $$
    \min_\phi \ \mathbb{E}_{(k, y_k)}[-\log p_\phi(y_k \mid k)]
    $$

- Detailed conceptual explanation
  - Steps:
    1) Curate/normalize knowledge (triples, schemas, summaries).
    2) Inject via fine-tuning, prefix/prompt-tuning, or static context templates.
    3) Optionally combine with lightweight retrieval for freshness.

- Importance and role
  - Reduces runtime dependency on retrieval infrastructure; strong for stable, high-value knowledge.

- Improvements and advantages
  - Lower latency and cost; consistent outputs; improved recall for domain-critical priors.


---

## 12) Cache-Augmented Generation (CAG)

- Definition
  - Augments generation with caches of prior queries, outputs, retrieval results, or model KV states to reduce latency and cost and stabilize repeated workloads.

- Mathematical formulation
  - Nearest-neighbor answer reuse:
    $$
    (q_i,y_i) = \arg\max_i s(q, q_i), \quad \text{if } s(q,q_i)>\tau \Rightarrow \hat{y}=y_i
    $$
  - KV-state cache reuse: reuse $(K,V)$ for repeated prefixes $x_{1:t}$ to skip recomputation.

- Detailed conceptual explanation
  - Steps:
    1) Maintain semantic cache over Q/A and retrieved contexts with TTL/invalidation.
    2) Gate by similarity and confidence; optionally re-verify before reuse.
    3) Cache cross-encoder ranks and compressed snippets for hot topics.
    4) Exploit server-side KV caching for common system prompts and boilerplate.

- Importance and role
  - High-QPS applications, helpdesks, code search, and dashboards where queries repeat.

- Improvements and advantages
  - Large latency/cost reductions; operational stability; smooths load spikes.


---

## 13) Zero-Indexing Internet Search-Augmented Generation

- Definition
  - Retrieval is performed directly via live web search and ad-hoc crawling without pre-building local indices.

- Mathematical formulation
  - Web retrieval operator:
    $$
    C = S_{web}(q) = \{ \mathrm{Summarize}(\mathrm{Fetch}(u_i)) \}_{i=1}^k
    $$
  - Latency budget:
    $$
    t_{tot} \approx \sum_{i=1}^k (t_{dns,i}+t_{net,i}+t_{parse,i}) + t_{LLM}
    $$

- Detailed conceptual explanation
  - Steps:
    1) Query rewriting for search engines, choose engine mix (news, scholarly, general).
    2) Fetch and parse pages; de-duplicate and source-rank (authority, freshness).
    3) Compress/segment content for context.
    4) Generate with citations and provenance.

- Importance and role
  - Freshness-critical tasks, real-time events, long-tail web knowledge without ingestion overhead.

- Improvements and advantages
  - Maximum coverage and recency; no indexing pipeline; flexible domain reach.


---

## Cross-Cutting Improvements

- Query optimization: decomposition, expansion, paraphrasing to improve recall.
- Reranking: cross-encoders and LLM-as-ranker for high precision@k.
- Context optimization: salience extraction, deduplication, discourse-aware compression.
- Uncertainty and verification: entailment, self-consistency, debate, external fact-checkers.
- Attribution: span alignment, citation grounding scores.
- Cost/latency control: hybrid retrieval, caches, early-exit, adaptive $k$.
- Training: joint retriever–generator fine-tuning, negative mining, RL for tool policies.
- Evaluation: Recall@k, NDCG, precision/coverage, faithfulness (NLI), attribution F1, answer EM/F1, calibration (ECE), cost/latency metrics.


---

## Summary Table: Improvements, Advantages, and When to Use

| RAG Type | Core Idea | Key Improvements over Naive | Advantages | When to Use |
|---|---|---|---|---|
| Naive (Standard) RAG | One-shot retrieve-and-generate | Baseline grounding | Simple, low-cost, fast | Stable KBs, FAQs, internal docs with moderate complexity |
| Advanced RAG | Hybrid retrieval, re-ranking, compression, routing | Higher precision/recall, fewer tokens | Better accuracy, lower context cost | Medium–large corpora, noisy sources, long-form responses |
| Modular RAG | Pluggable components with stable interfaces | Engineering scalability and observability | Rapid iteration, safe deployment | Multi-domain systems, enterprise stacks, ongoing optimization |
| Agentic RAG | Iterative planning, multi-hop tool use | Complex reasoning and coverage | High reliability on hard tasks | Multi-hop QA, synthesis, research assistants, tool-rich workflows |
| Corrective RAG | Confidence-gated re-retrieval/editing | Hallucination detection and repair | Higher factual precision | Safety-critical outputs, compliance, high-trust answers |
| KI-RAG | Evidence-coverage for knowledge-intensive tasks | Supervised evidence chains | Strong multi-hop and breadth | Open-domain QA, scientific/legal/biomedical |
| Multimodal RAG | Retrieve across text/image/audio/video | Cross-modal grounding | Visual/acoustic evidence usage | VQA, medical imaging, media search, AV logs |
| Memory RAG | Persistent dynamic memory store | Personalization and continuity | Long-horizon grounding | Assistants, project memory, longitudinal workflows |
| Meta-/Few-/Zero-Shot RAG | Retrieve demonstrations for ICL | Task adaptation without training | Broad generalization | New tasks, code styles, math reasoning, formatting |
| Graph-RAG | KG/graph retrieval and reasoning | Structured multi-hop compositionality | Interpretability, disambiguation | Enterprise KGs, ontologies, regulated domains |
| Knowledge-Augmented Generation (KAG) | Pre-injected knowledge via tuning/prompts | Runtime independence from retrieval | Low latency, stable priors | Stable domains, on-device, cost-sensitive |
| Cache-Augmented Generation (CAG) | Reuse Q/A, retrieval, and KV states | Major latency/cost reductions | Operational stability | High-QPS, repetitive queries, dashboards |
| Zero-Indexing Web RAG | Live web search without local index | Max freshness and coverage | No ingestion pipeline | Real-time events, news, long-tail web knowledge |

----
# Technical Processing Methods for RAG

## 1) Sparse, Dense, and Attentional Representations (e.g., SPLADE)

- Definition
  - Sparse representations encode texts as high-dimensional lexical vectors (e.g., BM25, learned lexical expansion).
  - Dense representations encode texts as low-dimensional continuous embeddings learned end-to-end.
  - Attentional sparse representations use transformer attention/MLM logits to induce sparse lexical expansions (e.g., SPLADE).

- Mathematical formulation
  - BM25 scoring:
    $$
    \mathrm{BM25}(q,d) = \sum_{t \in q} \mathrm{IDF}(t)\,\frac{f(t,d)\,(k_1+1)}{f(t,d)+k_1\left(1-b+b\frac{|d|}{\mathrm{avgdl}}\right)}
    $$
  - Dense similarity:
    $$
    s(q,d)=\langle f_\theta(q), g_\theta(d) \rangle \quad \text{or} \quad \frac{f_\theta(q)^\top g_\theta(d)}{\|f_\theta(q)\|\|g_\theta(d)\|}
    $$
  - SPLADE document/query vector over the vocabulary $V$:
    $$
    r_x[v] = \log\!\left(1 + \sum_{i \in x} \mathrm{ReLU}\big(z_{i,v}\big)\right), \quad z_{i,\cdot}=\text{MLMHead}(\text{Transformer}(x))_i
    $$
    with sparsity regularization:
    $$
    \mathcal{L}_{\text{sparse}} = \lambda_q \|r_q\|_1 + \lambda_d \|r_d\|_1
    $$

- Detailed conceptual explanation
  - Sparse: exact term matching and IDF-driven weighting; strong precision and interpretability.
  - Dense: semantic matching beyond exact tokens; robust to paraphrase/synonymy; efficient sub-linear ANN search.
  - Attentional sparse (SPLADE): expands queries/documents to related terms via MLM logits; preserves sparsity and indexability while capturing semantics.

- Importance and role
  - Sparse excels in precision, legal/compliance traceability.
  - Dense excels in recall and semantic robustness.
  - Attentional sparse bridges lexical and semantic matching with scalable inverted indices.


## 2) Dot-Product Similarity, ANN Search, Late Interactions, Hybrid Vector Approaches

- Definition
  - Dot-product/cosine scores compare dense vectors.
  - Approximate Nearest Neighbor (ANN) indexes enable sub-linear retrieval (e.g., HNSW, IVF-PQ).
  - Late interactions score token-level matches post-retrieval (e.g., ColBERT).
  - Hybrid vector approaches fuse multiple dense spaces and/or sparse-dense signals.

- Mathematical formulation
  - InfoNCE training for bi-encoders:
    $$
    \mathcal{L}_{\text{NCE}} = -\log \frac{\exp(s(q,d^+)/\tau)}{\exp(s(q,d^+)/\tau)+\sum_{d^-}\exp(s(q,d^-)/\tau)}
    $$
  - ANN product quantization (IVF-PQ) approximation:
    $$
    \hat{v} = [c_1+\Delta_1;\dots;c_M+\Delta_M], \quad \|v-\hat{v}\|^2 \approx \sum_{m=1}^M \|v_m - (c_m+\Delta_m)\|^2
    $$
  - Late interaction (ColBERT) scoring:
    $$
    s(q,d) = \sum_{i=1}^{|q|} \max_{j \le |d|} \frac{q_i^\top d_j}{\|q_i\|\,\|d_j\|}
    $$
  - Hybrid fusion (dense + sparse + multi-dense):
    $$
    s_{\text{hyb}}(q,d) = \sum_m w_m s_m(q,d), \quad \sum_m w_m=1
    $$

- Detailed conceptual explanation
  - Dot product with vector normalization stabilizes scale and enables efficient ANN.
  - ANN methods: HNSW (navigable small-world graphs), IVF-PQ (coarse quantization + PQ), ScaNN (asymmetric hashing with anisotropic quantization), DiskANN (disk-backed).
  - Late interactions index per-token embeddings with compression; preserve fine-grained matching while avoiding cross-encoder cost.
  - Hybrid vectors combine generalist and domain-specific encoders; optionally include sparse scores via calibrated fusion or RRF.

- Importance and role
  - Core to dense retrieval scalability and accuracy.
  - Late interactions approach cross-encoder precision at index-time cost.
  - Hybrid vectors mitigate domain shift and calibrate recall/precision.


## 3) Inverse Cloze Task (ICT) Pre-Training for Retrievers

- Definition
  - Self-supervised pre-training where a sentence is removed from a passage; the sentence is the query and the remainder is the positive document.

- Mathematical formulation
  - Construct $(q, d^+)$ from passage $p$: $q=s$, $d^+=p\setminus s$.
  - Contrastive objective:
    $$
    \mathcal{L}_{\text{ICT}} = -\log \frac{\exp(\langle f(q), g(d^+)\rangle/\tau)}{\sum_{d \in \{d^+\}\cup \mathcal{N}} \exp(\langle f(q), g(d)\rangle/\tau)}
    $$

- Detailed conceptual explanation
  - Encourages $f$ and $g$ to capture discourse coherence and topicality.
  - Scales without labels; negatives drawn in-batch or via hard negative mining (e.g., BM25, ANCE).

- Importance and role
  - Provides strong initialization for dense retrievers; improves sample efficiency and out-of-domain robustness.


## 4) DRAGON: Diverse Augmentation for Generalizable Dense Retrieval

- Definition
  - Training with a mixture of diverse query-generation and perturbation strategies to improve domain generalization and robustness.

- Mathematical formulation
  - Mixture-of-augmentations contrastive training:
    $$
    \mathcal{A}=\{\text{doc2query},\,\text{backtranslation},\,\text{random spans},\,\text{title},\,\text{entity-queries},\dots\}
    $$
    $$
    \mathcal{L}_{\text{DRAGON}} = \sum_{a \in \mathcal{A}} \alpha_a\,\mathbb{E}_{(q_a,d)}\!\left[-\log \frac{e^{s(f(q_a),g(d))/\tau}}{\sum_{d'\in \mathcal{N}} e^{s(f(q_a),g(d'))/\tau}}\right] + \beta\,\mathcal{L}_{\text{inv}}
    $$
    Invariance regularization across views of the same document:
    $$
    \mathcal{L}_{\text{inv}} = \sum_{(a,b),\,a\neq b} \|f(q_a) - f(q_b)\|_2^2
    $$

- Detailed conceptual explanation
  - Generate multiple heterogeneous queries per document; train with hard negatives across domains.
  - Encourage representation consistency across query views; optionally distill from a cross-encoder on mixed domains.

- Importance and role
  - Reduces overfitting to annotation artifacts; improves zero-shot retrieval on unseen domains and query styles.


## 5) Supervised Retriever Optimization via KL Divergence (Distillation)

- Definition
  - Distill a bi-encoder retriever from a teacher (e.g., cross-encoder) by matching soft relevance distributions.

- Mathematical formulation
  - Teacher distribution over a candidate set $\mathcal{D}_q$:
    $$
    P_T(d \mid q)=\frac{\exp(r_T(q,d)/\tau_T)}{\sum_{d'\in \mathcal{D}_q}\exp(r_T(q,d')/\tau_T)},\quad
    P_S(d \mid q)=\frac{\exp(s_S(q,d)/\tau_S)}{\sum_{d'\in \mathcal{D}_q}\exp(s_S(q,d')/\tau_S)}
    $$
    $$
    \mathcal{L}_{\text{KL}} = \sum_{q} \mathrm{KL}\big(P_T(\cdot \mid q)\, \|\, P_S(\cdot \mid q)\big)
    $$

- Detailed conceptual explanation
  - Use candidate pools from recalls (BM25/dense); compute teacher scores; optimize student to mimic listwise soft labels.
  - Combine with hard-negative InfoNCE for stability.

- Importance and role
  - Approaches cross-encoder quality with bi-encoder latency; improves ranking calibration and reduces false positives.


## 6) Reranking Techniques

- Definition
  - Precision-enhancing models that re-score a small candidate set from first-stage retrieval.

- Mathematical formulation
  - Cross-encoder scoring $r(q,d)$ with pairwise loss:
    $$
    \mathcal{L}_{\text{pair}} = \log\big(1+\exp(-\gamma (r(q,d^+)-r(q,d^-)))\big)
    $$
  - Listwise softmax:
    $$
    \mathcal{L}_{\text{list}} = -\sum_{i} y_i \log \frac{\exp(r(q,d_i))}{\sum_j \exp(r(q,d_j))}
    $$
  - Reciprocal Rank Fusion (rank-level fusion):
    $$
    \mathrm{RRF}(d) = \sum_m \frac{1}{k + \mathrm{rank}_m(d)}
    $$

- Detailed conceptual explanation
  - Cross-encoders jointly encode [q;d] for maximal interaction (e.g., monoBERT/monoT5).
  - Generative rerankers (monoT5) cast relevance as likelihood of “true”.
  - LLM-as-reranker can provide long-document, instruction-aware judgment with pairwise prompting.

- Importance and role
  - Substantially increases precision@k; critical in noisy corpora and for long-form synthesis quality.


## 7) RETRO and RETRO++

- Definition
  - RETRO integrates external nearest-neighbor chunks into a transformer via cross-attention during training/inference; RETRO++ refines retrieval, conditioning, and efficiency to improve scaling and latency.

- Mathematical formulation
  - Retrieval at step $t$:
    $$
    \mathcal{N}_t=\operatorname{NN}_k\big(h_t, \mathcal{I}\big), \quad h_t = \mathrm{Enc}(x_{1:t})
    $$
  - Cross-attention augmentation:
    $$
    h_t' = h_t + \mathrm{Attn}\big(h_t, K(\mathcal{N}_t), V(\mathcal{N}_t)\big), \quad p(y_t\mid \cdot)=\mathrm{Softmax}(W h_t')
    $$
  - Gated retrieval (RETRO++-style):
    $$
    h_t' = h_t + \sigma(g^\top h_t)\,\mathrm{Attn}(\cdot)
    $$

- Detailed conceptual explanation
  - Build a chunk index; at each step retrieve neighbors and inject via cross-attention; train end-to-end.
  - RETRO++ adds dynamic retrieval gating, improved negative sampling, hierarchical/coarse-to-fine retrieval, and memory/compression for inference efficiency.

- Importance and role
  - Lowers perplexity without scaling model parameters; offers controllable grounding through explicit neighbors; strong for language modeling with external corpora.


## 8) Chunking Strategies

- Definition
  - Methods to segment documents into retrievable units that balance semantic coherence and retrieval granularity.

- Mathematical formulation
  - Fixed-length with overlap: length $L$, stride $s=L-o$; for document length $T$:
    $$
    N=\left\lceil \frac{\max(0,T-L)}{s} \right\rceil + 1
    $$
  - Syntax-based segmentation objective (maximize coherence):
    $$
    \max_{\{c_i\}} \sum_i \mathrm{Coh}(c_i) - \lambda \sum_i \mathbb{I}\{|c_i|>L_{\max}\}
    $$
  - Format-aware chunking (e.g., code AST nodes, table rows/columns) modeled as partition over structure graph $G_s$.

- Detailed conceptual explanation
  - Fixed-length with overlap ensures boundary robustness for retrievers with limited receptive fields.
  - Syntax-aware (sentence/paragraph/heading, RST) preserves discourse signals; reduces irrelevant tokens.
  - File-format-aware: code via AST/function scope, markdown sections, PDF layout blocks, tables by row/column groups, spreadsheets via header propagation.

- Importance and role
  - Directly impacts recall/precision, index size, and context efficiency; format-aware chunking is essential for code, tables, and scientific PDFs.


## 9) Knowledge Graph Integration (GraphRAG)

- Definition
  - Retrieval and reasoning over structured graphs (entities, relations) and hybrid text-graph corpora.

- Mathematical formulation
  - Entity/relation linking and subgraph extraction:
    $$
    G_q=(V_q,E_q)=\mathrm{Expand}\big(\mathrm{Link}(q), k\text{-hop}\big)
    $$
  - Path scoring and PPR:
    $$
    \pi = (1-\alpha) P^\top \pi + \alpha e_q, \quad s(q,d)=\sum_{p \in \mathcal{P}_{q \to d}} w(p)
    $$
  - Conditioning with linearized triples and GNN encodings:
    $$
    H=\mathrm{GNN}(G_q),\; p(y\mid q,G_q)=\prod_t p(y_t \mid y_{<t}, \mathrm{linearize}(G_q), H)
    $$

- Detailed conceptual explanation
  - Link query mentions to KG nodes; expand along high-scoring relations; attach textual evidence to nodes/edges; re-rank paths and generate with explicit chains.

- Importance and role
  - Improves compositional reasoning and disambiguation; provides interpretable, auditable evidence chains in enterprise and scientific settings.


## 10) Hybrid Search (Vector + Traditional Lexical)

- Definition
  - Fusion of sparse lexical retrieval (e.g., BM25/SPLADE) and dense semantic retrieval to maximize recall and precision.

- Mathematical formulation
  - Score-level fusion with calibration:
    $$
    s(q,d) = \lambda\,\tilde{s}_{\text{dense}}(q,d) + (1-\lambda)\,\tilde{s}_{\text{BM25}}(q,d)
    $$
    where $\tilde{s}$ are z-score or min–max normalized.
  - Rank-level fusion:
    $$
    \mathrm{RRF}(d)= \sum_{m \in \{\text{dense},\text{sparse}\}} \frac{1}{k + \mathrm{rank}_m(d)}
    $$

- Detailed conceptual explanation
  - Run both retrievers; fuse scores or ranks; optionally cascade: lexical-first recall → dense re-rank or vice versa; learn $\lambda$ via validation or learning-to-rank.

- Importance and role
  - Robust under query mismatch and OOD vocabulary; strong default for production RAG pipelines.


# Practical “When to Use” Table

| Method | When to Use | Notes |
|---|---|---|
| Sparse (BM25) | High-precision, transparent matching; legal/compliance; short queries | Cheap, deterministic, interpretable |
| Dense Bi-Encoder + Dot Product | Semantic paraphrase robustness; large corpora with ANN | Train with InfoNCE/ICT; use FAISS/HNSW/ScaNN |
| Attentional Sparse (SPLADE) | Need semantic expansion but inverted-index scalability | L1-regularized MLM logits; strong lexical-semantic bridge |
| ANN (HNSW, IVF-PQ, ScaNN, DiskANN) | Sub-linear retrieval at scale; memory/latency constraints | Choose index by recall–latency–memory tradeoffs |
| Late Interactions (ColBERT) | Near cross-encoder precision with scalable retrieval | Token-level MaxSim; higher index size; compress aggressively |
| Hybrid Vector Approaches | Multi-domain corpora; ensemble robustness | Calibrate weights; include sparse as another signal |
| ICT Pre-Training | Label-scarce regimes; initialize retrievers | Combine with hard negatives for best results |
| DRAGON | OOD/generalization across query styles and domains | Diverse query views + invariance; distill if possible |
| KL-Distilled Retrievers | Want cross-encoder quality at bi-encoder speed | Listwise distillation with temperature smoothing |
| Rerankers (monoBERT/monoT5/LLM) | Need high precision@k on small candidate set | Pairwise/listwise training; consider LLM-as-reranker for long docs |
| RETRO / RETRO++ | Language modeling with external memory; end-to-end retrieval conditioning | Cross-attend to neighbors; use dynamic gating/compression |
| Chunking: Fixed+Overlap | Baseline; noisy/heterogeneous text | Tune length/stride; ensure boundary resilience |
| Chunking: Syntax-Based | Preserve coherence; long-form and QA | Segment by sentences/sections; improves re-ranking |
| Chunking: Format-Based | Code, tables, PDFs, spreadsheets | Use AST/layout graphs; align with task granularity |
| GraphRAG | Multi-hop, schema-aware reasoning; enterprise KGs | Link–expand–rank paths; combine with text evidence |
| Hybrid Search (Vector+Lexical) | Production default under query diversity | Score or rank fusion; learn calibration parameters |