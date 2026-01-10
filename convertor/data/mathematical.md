# Mastery Plan: Compute-Optimal Training + 1000+ GPU/TPU Training & Inference Systems (DDP → FSDP2 → ZeRO-3, Kernels → Comms, Training → Serving)

## Scope
### Definition
End-to-end large-scale model development spanning:
- **Compute-optimal training** (scaling laws, dataset/parameter/compute allocation).
- **Distributed training systems** (from single-node DDP to multi-node 3D parallelism + ZeRO/FSDP2).
- **GPU/TPU performance engineering** (kernels, memory hierarchy, NCCL/XLA collectives, overlap).
- **Production inference** (quantization, KV cache, compilation, batching, routing, reliability).

### Core Objective
Design and run training/inference stacks that are:
- **Compute-efficient** (max loss reduction per FLOP),
- **Memory-efficient** (fit 7B → 1T+),
- **Communication-efficient** (scale to 1000+ accelerators),
- **Operationally reliable** (fault-tolerant, observable, SLO-driven).

---

## Table of Contents
- [0. Prerequisites](#0-prerequisites)
- [1. Scaling Laws & Compute-Optimal Training](#1-scaling-laws--compute-optimal-training)
- [2. Transformer Training Mechanics](#2-transformer-training-mechanics)
- [3. Numerical Precision at Scale (FP16/BF16/FP8)](#3-numerical-precision-at-scale-fp16bf16fp8)
- [4. GPU/TPU Architecture for AI Systems](#4-gputpu-architecture-for-ai-systems)
- [5. Distributed Communication Fundamentals](#5-distributed-communication-fundamentals)
- [6. Parallelism Strategies (DP/TP/PP/SP)](#6-parallelism-strategies-dptpppsp)
- [7. Memory Optimization (Checkpointing, Sharding, ZeRO 0-3, Offload)](#7-memory-optimization-checkpointing-sharding-zero-0-3-offload)
- [8. PyTorch Distributed: DDP → FSDP → FSDP2](#8-pytorch-distributed-ddp--fsdp--fsdp2)
- [9. Megascale Training: DeepSpeed, Megatron-LM, TPU/XLA](#9-megascale-training-deepspeed-megatron-lm-tpuxla)
- [10. Scheduling: Pipeline Bubbles, Zero-Bubble, Overlap](#10-scheduling-pipeline-bubbles-zero-bubble-overlap)
- [11. Hyperparameter Tuning at Scale (ASHA/PBT/BO)](#11-hyperparameter-tuning-at-scale-ashapbtbo)
- [12. Checkpointing, Fault Tolerance, Recovery](#12-checkpointing-fault-tolerance-recovery)
- [13. Debugging: Loss Spikes, Divergence, Silent Corruption](#13-debugging-loss-spikes-divergence-silent-corruption)
- [14. Inference Systems: Throughput, Latency, KV Cache](#14-inference-systems-throughput-latency-kv-cache)
- [15. Inference Optimizations: Quantization, Compilation, SpecDec](#15-inference-optimizations-quantization-compilation-specdec)
- [16. Serving & Reliability: Multi-Tenant GPU, SLOs, Observability](#16-serving--reliability-multi-tenant-gpu-slos-observability)
- [17. Capstone Projects](#17-capstone-projects)
- [References](#references)

---

## 0. Prerequisites
### Definition
Foundational skills required to reason about compute/memory/communication trade-offs in training and inference.

### Mathematics
- Linear algebra: matrix multiply, norms, conditioning.
- Probability/statistics: cross-entropy, sampling, variance.
- Optimization: SGD/AdamW dynamics.

### Systems
- Linux, networking basics, filesystems/object stores.
- CUDA basics (threads/blocks/warps), profiling (Nsight Systems/Compute).
- PyTorch internals (autograd, parameter storage, optimizer states).

---

## 1. Scaling Laws & Compute-Optimal Training
### Definition
**Scaling laws** describe how loss decreases as a power law with increased **model size**, **data**, and **compute**. **Compute-optimal training** chooses model size and dataset tokens to minimize loss for a fixed compute budget.

### Core Equations
A common empirical form (compute-scaling):
$$
\mathcal{L}(C) \approx \mathcal{L}_\infty + a C^{-\alpha}
$$
where:
- $\mathcal{L}(C)$ is loss after training with compute $C$ (FLOPs),
- $\mathcal{L}_\infty$ is irreducible loss,
- $a,\alpha$ are fitted constants.

Token/model trade-off (compute-optimal heuristics; “Chinchilla-style”):
- Compute budget scales roughly with:
$$
C \propto N \cdot D
$$
where $N$ = parameters, $D$ = trained tokens (to first order; actual constants depend on architecture/training recipe).
- Compute-optimal regimes often imply allocating **more tokens per parameter** than earlier “parameter-heavy” regimes.

### What Researchers Must Master
- Fit scaling curves:
  - Log-log regression on $(C, \mathcal{L})$.
  - Separate effects of $N$ and $D$ with controlled sweeps.
- Translate scaling to planning:
  - Given budget $C$, pick $(N,D)$ and batch/sequence lengths.
- Decide target:
  - Pretraining loss, downstream quality proxy, or capability metric.

### Deliverables
- A reproducible scaling-law notebook:
  - sweeps over $N$, $D$, and context length,
  - fitted parameters $(a,\alpha,\mathcal{L}_\infty)$,
  - compute-optimal recommendation for a new budget.

---

## 2. Transformer Training Mechanics
### Definition
Training is repeated evaluation of loss and gradients over token sequences to update parameters. Systems choices (parallelism, precision, sharding) alter **throughput**, **stability**, and **cost**.

### Core Equations
Cross-entropy over tokens:
$$
\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T}\log p_\theta(x_t \mid x_{<t})
$$

AdamW update (per parameter):
$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t,\quad
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
$$
$$
\theta_{t+1} = \theta_t - \eta \frac{m_t}{\sqrt{v_t}+\epsilon} - \eta \lambda \theta_t
$$

Global batch with data parallel:
$$
B_{\text{global}} = B_{\text{micro}} \cdot G_{\text{accum}} \cdot W_{\text{dp}}
$$
where $G_{\text{accum}}$ = gradient accumulation steps, $W_{\text{dp}}$ = DP world size.

### Critical Concepts
- Optimizer states: $m_t, v_t$ (often dominant memory).
- Gradient accumulation: simulates large batch without increasing per-step activation memory.
- Sequence length effects:
  - Attention compute roughly scales as:
$$
\mathcal{O}(L^2 d)
$$
for sequence length $L$ and hidden size $d$ (dense attention).

---

## 3. Numerical Precision at Scale (FP16/BF16/FP8)
### Definition
Mixed precision uses low-precision arithmetic to increase throughput and reduce memory, while maintaining stability using scaling, accumulation, and careful casting.

### Core Equations
Dynamic loss scaling (conceptual):
$$
\tilde{\mathcal{L}} = s \cdot \mathcal{L},\quad
\tilde{g} = \nabla_\theta \tilde{\mathcal{L}} = s \cdot g,\quad
g = \tilde{g}/s
$$
Choose scale $s$ to avoid underflow/overflow in FP16.

FP8 training uses per-tensor/per-channel scaling:
$$
x_{\text{fp8}} = \operatorname{quantize}(x / s_x),\quad
x \approx s_x \cdot \operatorname{dequantize}(x_{\text{fp8}})
$$

### Required Mastery
- When BF16 is “easy mode” vs FP16 requiring loss scaling.
- FP8 constraints:
  - format selection (e.g., E4M3 vs E5M2),
  - amax/scale tracking,
  - accumulation in FP16/BF16/FP32.

### Deliverables
- A stability report:
  - overflow/underflow counters,
  - gradient norm monitoring,
  - loss-scale schedule,
  - convergence comparisons FP32 vs BF16 vs FP8.

---

## 4. GPU/TPU Architecture for AI Systems
### Definition
Performance is determined by how efficiently kernels use compute units and how well memory/communication is overlapped.

### Key Model
Time per step:
$$
T_{\text{step}} \approx \max\Big(T_{\text{compute}},\; T_{\text{memory}},\; T_{\text{comm}}\Big) + T_{\text{overhead}}
$$

### GPU Concepts to Master (H100/A100 class)
- Memory hierarchy:
  - registers → shared memory/L1 → L2 → HBM.
- Kernel execution:
  - warps, occupancy, instruction mix.
- Asynchrony:
  - CUDA streams for overlap (compute vs NCCL comm).
- Interconnects:
  - NVLink/NVSwitch intra-node,
  - InfiniBand/RoCE inter-node.

### TPU Concepts (DeepMind-style stacks)
- SPMD execution model and collective-heavy programming (XLA).
- High-bandwidth interconnect and compiler-driven fusion.

### Deliverables
- Nsight Systems trace showing:
  - overlapped compute+comm,
  - kernel hotspots,
  - memory stalls,
  - host overhead sources.

---

## 5. Distributed Communication Fundamentals
### Definition
Distributed training is dominated by collective communication: all-reduce, reduce-scatter, all-gather, broadcast.

### Communication Cost Model
For message size $S$ across $p$ ranks (ring-based intuition):
$$
T_{\text{comm}} \approx \alpha \cdot \log p + \beta \cdot S
$$
where $\alpha$ = latency term, $\beta$ = inverse bandwidth term (effective).

### Key Collectives
- **All-reduce**: sum gradients across DP ranks.
- **Reduce-scatter + all-gather**: used by ZeRO/FSDP for sharded gradients/params.

### Deliverables
- Microbenchmarks:
  - NCCL all-reduce bandwidth vs message size,
  - topology-aware comparison (intra-node vs inter-node),
  - tuning notes (bucket sizes, stream priorities).

---

## 6. Parallelism Strategies (DP/TP/PP/SP)
### Definition
Parallelism decomposes work across devices:
- **Data Parallel (DP)**: replicate model, shard data.
- **Tensor Parallel (TP)**: split matrix multiplications across GPUs.
- **Pipeline Parallel (PP)**: split layers across stages.
- **Sequence Parallel (SP)**: split sequence dimension for specific ops to reduce activation memory/comm.

### Core Equations
If total devices $P$:
$$
P = P_{\text{dp}} \cdot P_{\text{tp}} \cdot P_{\text{pp}}
$$

DP gradient synchronization volume per step (approx):
$$
V_{\text{dp}} \approx |\nabla \theta|
$$

TP communication often scales with activation size (varies by partition):
$$
V_{\text{tp}} \propto \text{activations exchanged per layer}
$$

Pipeline bubble utilization (simple model, $m$ microbatches, $p$ stages):
$$
U \approx \frac{m}{m + p - 1}
$$

### Decision Framework
- If comm-bound on gradients → consider sharding (FSDP/ZeRO) or reduce DP degree.
- If compute-bound on matmuls → increase TP to use more SMs efficiently.
- If memory-bound by params/acts → PP + checkpointing + ZeRO-3.

### Deliverables
- A parallelism search table:
  - candidate $(P_{\text{dp}},P_{\text{tp}},P_{\text{pp}})$,
  - predicted memory per GPU,
  - predicted comm volumes,
  - measured throughput tokens/s.

---

## 7. Memory Optimization (Checkpointing, Sharding, ZeRO 0-3, Offload)
### Definition
Memory is consumed by:
- Parameters,
- Gradients,
- Optimizer states,
- Activations (dominant with long context),
- KV cache (inference).

### Memory Accounting (training, per parameter)
Let parameter count be $N$ and bytes per element depend on dtype.
- Parameters: $\approx N \cdot b_{\theta}$
- Gradients: $\approx N \cdot b_{g}$
- Adam states: $\approx 2N \cdot b_{m/v}$

Total (replicated, naive):
$$
M_{\text{naive}} \approx N(b_\theta + b_g + 2b_{m/v}) + M_{\text{acts}}
$$

### ZeRO Stages (conceptual)
- **Stage 0**: replicate everything (DDP-style).
- **Stage 1**: shard optimizer states across DP ranks.
- **Stage 2**: shard optimizer states + gradients.
- **Stage 3**: shard optimizer states + gradients + parameters (parameters are gathered just-in-time).

If DP world size is $P_{\text{dp}}$, ideal sharding reduces those components by $\approx 1/P_{\text{dp}}$ (ignoring overhead):
$$
M_{\text{sharded}} \approx \frac{N(\text{sharded components})}{P_{\text{dp}}} + M_{\text{unsharded}} + M_{\text{acts}} + M_{\text{overhead}}
$$

### Activation Checkpointing
Trade compute for memory by recomputing activations:
- Save only selected checkpoints; recompute others in backward.
Compute overhead factor depends on policy; conceptually:
$$
T_{\text{step}}' \approx T_{\text{step}} + T_{\text{recompute}}
$$

### Offloading (CPU/NVMe)
Moves states/params off GPU:
- Helps fit models but risks PCIe/NVMe bandwidth bottlenecks:
$$
T_{\text{offload}} \approx \frac{S_{\text{moved}}}{\text{BW}_{\text{link}}}
$$

### Deliverables
- A memory budget sheet:
  - exact bytes for params/grads/optimizer/acts,
  - effect of ZeRO stage selection,
  - checkpointing policy vs recompute cost.

---

## 8. PyTorch Distributed: DDP → FSDP → FSDP2
### Definition
PyTorch provides increasingly memory-efficient distributed strategies:
- **DDP**: replicate parameters, all-reduce gradients.
- **FSDP**: shard parameters/gradients/optimizer states with gather/scatter.
- **FSDP2**: newer sharding + composability improvements (more explicit control of state, resharding, better integration patterns).

### DDP Mechanics
Gradient all-reduce per bucket:
$$
g \leftarrow \frac{1}{P_{\text{dp}}}\sum_{i=1}^{P_{\text{dp}}} g_i
$$

### FSDP/ZeRO-3 Mechanics (conceptual)
Per layer:
- all-gather sharded params → compute forward/backward → reduce-scatter grads → optionally reshard params.

### Mastery Checklist
- Bucket sizing and overlap:
  - tune gradient bucket size to overlap comm with backprop.
- Parameter flattening vs per-parameter overhead.
- State dict:
  - sharded checkpointing for fast save/load at scale.

### Deliverables
- 3 training runs (same model):
  - DDP baseline,
  - FSDP with full_shard,
  - FSDP2 configuration,
  comparing throughput, peak memory, and failure modes.

---

## 9. Megascale Training: DeepSpeed, Megatron-LM, TPU/XLA
### Definition
Megascale stacks provide integrated 3D parallelism, optimized kernels, and memory sharding to train 10B–1T+ models.

### Components
- **DeepSpeed**: ZeRO, offload, pipeline parallel, monitoring.
- **Megatron-LM**: tensor/pipeline parallel transformer kernels and schedules.
- **TPU/XLA**: SPMD partitioning + compiler fusion + collectives orchestration.

### What “1T+ Parameters” Forces
- 3D parallelism:
  - TP for matmul scaling,
  - PP for memory partitioning,
  - DP for statistical efficiency,
  - ZeRO/FSDP for optimizer/param sharding.
- Fault tolerance:
  - node failures are normal; recovery must be routine.

### Deliverables
- A reference configuration for:
  - 7B on 8 GPUs,
  - 70B on 64–256 GPUs,
  - 200B+ on 512–2048 GPUs,
  with explicit parallelism degrees and memory math.

---

## 10. Scheduling: Pipeline Bubbles, Zero-Bubble, Overlap
### Definition
Pipeline parallelism introduces **bubbles** (idle periods) due to pipeline fill/drain. “Zero-bubble” aims to eliminate or minimize idle time via scheduling and interleaving.

### Core Model
Utilization for simple GPipe-style schedule:
$$
U \approx \frac{m}{m+p-1}
$$
Increase microbatches $m$ to reduce bubble, but $m$ is constrained by memory and optimizer behavior.

### Techniques
- **1F1B** scheduling (common in Megatron-style PP).
- **Interleaved pipeline** (multiple model chunks per stage).
- **Overlap**:
  - overlap reduce-scatter/all-gather with compute using dedicated CUDA streams.

### Deliverables
- Pipeline schedule report:
  - measured idle time,
  - microbatch tuning results,
  - comm/compute overlap percentage from traces.

---

## 11. Hyperparameter Tuning at Scale (ASHA/PBT/BO)
### Definition
At scale, tuning must allocate compute adaptively to promising runs.

### Key Algorithms
- Random/grid search (baseline).
- Bayesian optimization (sample-efficient).
- Population-based training (PBT): exploit/explore with weight inheritance.
- ASHA / successive halving: early-stop poor performers.

### Resource Allocation Model
Given $n$ trials, budget per trial $r$, and halving factor $\eta$, ASHA allocates progressively:
$$
n_{k+1} \approx \frac{n_k}{\eta},\quad r_{k+1} = \eta r_k
$$

### Deliverables
- An “at-scale tuning” playbook:
  - what to tune (LR, warmup, weight decay, grad clip, batch, seq len),
  - stopping metrics (loss slope, divergence detectors),
  - reproducibility constraints (seeds, data order, checkpoint resume).

---

## 12. Checkpointing, Fault Tolerance, Recovery
### Definition
Checkpointing preserves training state to recover from faults and to enable experimentation (branching, evaluation, finetuning).

### What Must Be Checkpointed
- Model parameters (sharded).
- Optimizer states ($m,v$ for AdamW).
- RNG states (CPU/GPU) for determinism.
- Data loader position / dataset shard state.
- Scheduler state (LR, warmup counters).

### Recovery Correctness Condition
After resume at step $t$, training should continue such that:
$$
(\theta_t, \text{opt}_t, \text{rng}_t, \text{data}_t) \;\text{match}\;\text{the uninterrupted run}
$$
(up to known nondeterminism sources).

### Deliverables
- A fault-injection test:
  - kill random ranks,
  - validate exact/near-exact loss continuity,
  - measure checkpoint save/load throughput.

---

## 13. Debugging: Loss Spikes, Divergence, Silent Corruption
### Definition
At large scale, training failures often come from numerical issues, data issues, or distributed desynchronization.

### Primary Failure Modes
- Numerical overflow (FP16/FP8), underflow, NaNs.
- Bad batches (corrupt samples, extreme token distributions).
- Optimizer instability (LR too high, insufficient warmup).
- Distributed mismatch:
  - inconsistent parameter broadcasts,
  - uneven gradient accumulation,
  - incorrect reshard/gather ordering.

### Diagnostics (must be instrumented)
- Gradient norms and per-layer statistics:
$$
\|g\|_2 = \sqrt{\sum_i g_i^2}
$$
- Activation stats (mean/variance, max).
- Overflow counters, loss scale trajectory.
- Data quality metrics (dedup rate, language/domain mix, token entropy).

### Deliverables
- A “loss spike runbook”:
  - triage tree (data vs numeric vs comm),
  - minimal reproduction strategy,
  - mitigations (grad clip, LR drop, longer warmup, sanitize batches).

---

## 14. Inference Systems: Throughput, Latency, KV Cache
### Definition
Inference performance is dominated by:
- Prefill (prompt) compute,
- Decode loop (one token at a time),
- KV cache memory bandwidth and management.

### Latency Model
Per request latency:
$$
T \approx T_{\text{queue}} + T_{\text{prefill}} + \sum_{t=1}^{T_{\text{out}}} T_{\text{decode}}(t)
$$

### KV Cache Memory
For $L$ layers, heads $H$, head dim $d_h$, sequence length $S$:
$$
M_{\text{KV}} \propto L \cdot S \cdot H \cdot d_h \cdot b_{\text{kv}} \cdot 2
$$
(2 for K and V; exact constants depend on layout).

### Deliverables
- A profiling report separating:
  - prefill vs decode time,
  - KV cache bandwidth stalls,
  - batching efficiency vs tail latency.

---

## 15. Inference Optimizations: Quantization, Compilation, SpecDec
### Definition
Inference optimization reduces cost/latency via lower precision, better kernels, and smarter decoding.

### Quantization Basics
Uniform affine quantization (conceptual):
$$
q = \operatorname{clip}\Big(\operatorname{round}(x/s) + z,\; q_{\min}, q_{\max}\Big),\quad
\hat{x} = s(q - z)
$$
where $s$ = scale, $z$ = zero-point.

### Key Techniques
- Weight-only INT8/INT4 for matmuls.
- Mixed precision KV caches (lower-bit KV).
- Paged attention / KV paging (vLLM-style) to reduce fragmentation.
- Prefix caching (shared prompt reuse).
- Speculative decoding:
  - draft model proposes tokens; target model verifies.

### Compilation / Runtimes
- TensorRT-LLM, ONNX Runtime, TVM, custom CUDA kernels.
- Python overhead elimination and fused operators.

### Deliverables
- Quantization pipeline:
  - calibration set design,
  - perplexity + task metrics regression tests,
  - deployment with monitoring for quality drift.

---

## 16. Serving & Reliability: Multi-Tenant GPU, SLOs, Observability
### Definition
Production serving is a distributed system that must satisfy SLOs (latency/availability) while maximizing GPU utilization.

### SLO Model
Example latency SLO:
$$
P(T_{\text{resp}} \le \tau) \ge 0.99
$$
for threshold $\tau$ (e.g., 1s).

### Core Architecture
- API gateway → router → model servers → GPU workers.
- Multi-tenancy:
  - route by model size, priority, SLA tier.
- Batching:
  - continuous batching for decode efficiency.
- Isolation:
  - MIG/MPS (GPU partitioning/sharing) where applicable.

### Observability (must-have)
- Metrics:
  - tokens/s, queue depth, GPU util, KV cache hit rate, OOM rate.
- Tracing:
  - request path timing (queue/prefill/decode/postprocess).
- Logging:
  - structured events for errors and regressions.

### Deliverables
- A production-readiness checklist:
  - autoscaling policy,
  - rollback strategy,
  - incident playbooks,
  - load tests and chaos tests.

---

## 17. Capstone Projects
### Project A: 1000-GPU Pretraining Reference Stack (PyTorch)
**Goal**: Train a transformer at scale with DDP/FSDP2 + TP/PP, stable mixed precision, fault recovery.
- Outputs:
  - full config system (parallelism degrees, bucket sizes, checkpointing),
  - throughput scaling curve (1 → 8 → 64 → 512 → 1024 GPUs),
  - failure injection + recovery report.

### Project B: Zero-3 vs FSDP2 Comparative Study
**Goal**: Same model, compare memory, comm, throughput, stability, and engineering complexity.
- Outputs:
  - memory math validated by profiler,
  - NCCL traces and overlap analysis,
  - recommendations by model size regime.

### Project C: LLM Serving Stack (vLLM/TensorRT-LLM + K8s)
**Goal**: Multi-tenant inference with batching, prefix caching, quantization, and SLO compliance.
- Outputs:
  - routing policy (cost-aware),
  - quantized deployment with quality gates,
  - full observability dashboard + incident drills.

---

## References
### Scaling Laws / Compute-Optimal
- Kaplan et al., “Scaling Laws for Neural Language Models” (2020). https://arxiv.org/abs/2001.08361
- Hoffmann et al., “Training Compute-Optimal Large Language Models” (Chinchilla, 2022). https://arxiv.org/abs/2203.15556

### Parallelism / Systems
- PyTorch Distributed (DDP/FSDP). https://pytorch.org/docs/stable/distributed.html
- PyTorch FSDP. https://pytorch.org/docs/stable/fsdp.html
- DeepSpeed + ZeRO. https://www.deepspeed.ai/
- ZeRO paper: Rajbhandari et al. (2020). https://arxiv.org/abs/1910.02054
- Megatron-LM. https://github.com/NVIDIA/Megatron-LM
- NCCL. https://developer.nvidia.com/nccl
- CUDA C++ Programming Guide (streams, memory). https://docs.nvidia.com/cuda/cuda-c-programming-guide/

### Inference
- vLLM (PagedAttention). https://github.com/vllm-project/vllm
- TensorRT-LLM. https://github.com/NVIDIA/TensorRT-LLM
- ONNX Runtime. https://onnxruntime.ai/
- Speculative decoding: Chen et al. (2023). https://arxiv.org/abs/2302.01318

### Stanford CS336 (LLM systems + training from scratch)
- CS336 course page (verify current term materials). https://cs336.stanford.edu/

### TPU / XLA
- XLA documentation. https://www.tensorflow.org/xla
- JAX distributed. https://jax.readthedocs.io/en/latest/multi_process.html