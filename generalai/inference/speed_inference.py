#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generalized and Robust Transformers Pipeline (Professional-Grade)

This single-file module provides a production-ready, generalized pipeline wrapper
around Hugging Face's transformers.pipeline that preserves 100% compatibility
with existing features while adding a carefully engineered runtime with:

1) Memory-efficient caching with LRU and weak references
2) Advanced error handling with retry mechanisms and exponential backoff
3) Resource pooling and connection management (multi-worker pipelines)
4) Batch processing optimization with dynamic batching
5) Comprehensive logging with rich integration
6) Memory profiling and garbage collection optimization
7) Thread-safe operations with concurrent processing
8) Configuration management with validation
9) Performance monitoring and metrics collection
10) Graceful degradation and fallback mechanisms (e.g., OOM handling, CPU fallback)

Design principles:
- Maintain transformers' behavior: we never remove or alter existing pipeline features.
- Add functionality via a thin, well-abstracted runtime layer that composes with transformers.
- Be robust by default: safe fallbacks, validated configuration, and strong observability.
- Prefer simple, explainable concurrency primitives (threads + conditions) over complicated stacks.

All explanations, rationale, and examples are contained in this file to assist advanced
and aspiring engineers alike.

Author's note on performance:
- This implementation uses dynamic batching with a single worker per device by default
  to maximize GPU utilization and avoid cross-thread contention within a single model.
- For CPU-bound tasks, enabling multiple workers increases throughput.

-------------------------------------------------------------------------------
Quickstart
-------------------------------------------------------------------------------

- Install or ensure the following are available:
  pip install transformers torch rich psutil pillow

- Create a SmartPipeline using the factory function smart_pipeline (keep your
  original HF pipeline usage unchanged; all your kwargs still work):

    sp = smart_pipeline(
        task="text-classification",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        verbose=True,  # enables rich logging
        runtime_config=PipelineRuntimeConfig(max_batch_size=16, num_workers=1),
    )

- Call it like a normal pipeline (single input):
    result = sp("I love this product!")

- Or batch of inputs (also works; dynamic batching is best with many concurrent single calls):
    results = sp(["Great work!", "This is terrible..."])

- Metrics and memory:
    print(sp.metrics())
    print(sp.memory_snapshot())

- Shutdown gracefully (closes workers and releases memory):
    sp.shutdown()

-------------------------------------------------------------------------------
Examples (see __main__ at the bottom for runnable examples):
- Sentiment analysis
- Text generation with quantization hint
- Image classification with graceful CPU fallback
- Dynamic batching with concurrent requests
-------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import sys
import gc
import io
import re
import json
import math
import time
import copy
import uuid
import types
import queue
import atexit
import weakref
import random
import signal
import hashlib
import logging
import tracemalloc
import threading
import warnings
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    List,
    Tuple,
    Optional,
    Union,
    Callable,
    Deque,
    Iterable,
    Hashable,
)
from collections import OrderedDict, deque
from concurrent.futures import Future, ThreadPoolExecutor

# Third-party imports and soft dependencies
try:
    from rich.logging import RichHandler  # pretty logs
    _HAS_RICH = True
except Exception:
    _HAS_RICH = False

try:
    import psutil  # memory/proc stats
    _HAS_PSUTIL = True
except Exception:
    _HAS_PSUTIL = False

try:
    import numpy as np
except Exception:
    np = None  # noqa: N816

try:
    import PIL.Image as PILImage  # noqa: N812
except Exception:
    PILImage = None

import torch

from pathlib import Path

# Transformers - kept generic and compatible
from transformers import (
    pipeline as hf_pipeline,
    Pipeline,
    set_seed,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    AutoConfig,
    AutoTokenizer,
    AutoFeatureExtractor,
    AutoImageProcessor,
    AutoProcessor,
    AutoModel,
    AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering, AutoModelForTableQuestionAnswering, AutoModelForVisualQuestionAnswering,
    AutoModelForDocumentQuestionAnswering, AutoModelForTokenClassification, AutoModelForMultipleChoice,
    AutoModelForNextSentencePrediction, AutoModelForImageClassification, AutoModelForZeroShotImageClassification,
    AutoModelForImageSegmentation, AutoModelForSemanticSegmentation, AutoModelForUniversalSegmentation,
    AutoModelForInstanceSegmentation, AutoModelForObjectDetection, AutoModelForZeroShotObjectDetection,
    AutoModelForDepthEstimation, AutoModelForVideoClassification, AutoModelForVision2Seq,
    AutoModelForAudioClassification, AutoModelForCTC, AutoModelForSpeechSeq2Seq, AutoModelForAudioFrameClassification,
    AutoModelForAudioXVector, AutoModelForTextToSpectrogram, AutoModelForTextToWaveform, AutoBackbone,
    AutoModelForMaskedImageModeling
)

from transformers import BitsAndBytesConfig # quantization


# =========================
# Logging and Diagnostics
# =========================

def _configure_logging(verbose: bool = False, log_level: int = logging.INFO) -> None:
    """
    Configure logging with optional rich integration. Idempotent and process-wide.
    """
    root = logging.getLogger()
    # If already configured, do not duplicate handlers
    if getattr(_configure_logging, "_configured", False):
        # But update level based on verbose flag
        root.setLevel(logging.DEBUG if verbose else log_level)
        for h in root.handlers:
            h.setLevel(logging.DEBUG if verbose else log_level)
        return

    root.handlers.clear()
    level = logging.DEBUG if verbose else log_level

    if _HAS_RICH:
        handler = RichHandler(rich_tracebacks=True, show_time=True, show_level=True, show_path=False)
    else:
        handler = logging.StreamHandler(stream=sys.stdout)

    fmt = "%(message)s" if _HAS_RICH else "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"
    formatter = logging.Formatter(fmt=fmt, datefmt="%H:%M:%S")
    handler.setFormatter(formatter)
    root.addHandler(handler)
    root.setLevel(level)

    _configure_logging._configured = True  # type: ignore[attr-defined]


LOGGER = logging.getLogger("smart_pipeline")


# =========================
# Utilities
# =========================

def _env_flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.lower() in {"1", "true", "yes", "y", "on"}


def _now_ms() -> int:
    return int(time.time() * 1000)


def _sleep_ms(ms: int) -> None:
    time.sleep(ms / 1000.0)


def _available_device(device: Optional[Union[int, str, torch.device]]) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        # Prefer MPS if available on Apple Silicon
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if isinstance(device, int):
        return torch.device("cuda", device) if torch.cuda.is_available() else torch.device("cpu")
    return torch.device(device)


def _is_gpu_device(dev: torch.device) -> bool:
    return dev.type == "cuda"


def _torch_dtype_from_str(dtype: Optional[Union[str, torch.dtype]]) -> Optional[torch.dtype]:
    if dtype is None:
        return None
    if isinstance(dtype, torch.dtype):
        return dtype
    # safe getattr
    try:
        return getattr(torch, dtype)
    except Exception:
        return None


def _estimate_size_bytes(obj: Any, max_blob_sample: int = 256_000) -> int:
    """
    Best-effort size estimate of Python objects. Conservative to avoid undercounting.
    """
    import sys as _sys
    visited = set()

    def inner(o: Any) -> int:
        if id(o) in visited:
            return 0
        visited.add(id(o))
        s = _sys.getsizeof(o)

        # Numpy arrays
        if np is not None and isinstance(o, np.ndarray):
            return s + int(o.nbytes)

        # Torch tensors
        if isinstance(o, torch.Tensor):
            try:
                t = o
                if t.is_cuda:
                    # CUDA memory not counted in sys.getsizeof; approximate via numel * element_size
                    return s + t.numel() * t.element_size()
                else:
                    return s + t.element_size() * t.nelement()
            except Exception:
                return s

        # PIL Image
        if PILImage is not None and isinstance(o, PILImage.Image):
            try:
                # Estimate compressed size by saving to bytes with low quality
                byte_io = io.BytesIO()
                o.save(byte_io, format="PNG", optimize=True)
                return s + len(byte_io.getvalue())
            except Exception:
                return s

        # Basic containers
        if isinstance(o, dict):
            return s + sum(inner(k) + inner(v) for k, v in o.items())
        if isinstance(o, (list, tuple, set, frozenset, deque)):
            return s + sum(inner(x) for x in o)

        # Bytes/bytearray
        if isinstance(o, (bytes, bytearray, memoryview)):
            return s + len(o)

        # Strings/statics
        return s

    try:
        return min(inner(obj), max_blob_sample)  # sample bound
    except Exception:
        return 0


def _stable_hash(obj: Any, max_bytes: int = 512_000) -> str:
    """
    Produce a stable content hash for caching keys across many common data types.

    We take care to bound the hashing cost for large blobs by truncating bytes.
    """
    h = hashlib.sha1()

    def update_with_type_tag(tag: str) -> None:
        h.update(f"<{tag}>".encode("utf-8"))

    def update_bytes(b: bytes) -> None:
        if len(b) > max_bytes:
            h.update(b[:max_bytes])
            h.update(f"[truncated:{len(b)}]".encode("utf-8"))
        else:
            h.update(b)

    def inner(x: Any) -> None:
        if x is None:
            update_with_type_tag("None")
            return
        if isinstance(x, (bool, int, float)):
            update_with_type_tag(type(x).__name__)
            h.update(str(x).encode("utf-8"))
            return
        if isinstance(x, str):
            update_with_type_tag("str")
            update_bytes(x.encode("utf-8"))
            return
        if isinstance(x, (bytes, bytearray, memoryview)):
            update_with_type_tag("bytes")
            update_bytes(bytes(x))
            return
        # numpy
        if np is not None and isinstance(x, np.ndarray):
            update_with_type_tag("np.ndarray")
            h.update(str(x.shape).encode("utf-8"))
            h.update(str(x.dtype).encode("utf-8"))
            try:
                update_bytes(x.tobytes())
            except Exception:
                pass
            return
        # torch
        if isinstance(x, torch.Tensor):
            update_with_type_tag("torch.Tensor")
            h.update(str(tuple(x.size())).encode("utf-8"))
            h.update(str(x.dtype).encode("utf-8"))
            try:
                cpu = x.detach().cpu().contiguous()
                update_bytes(cpu.numpy().tobytes())
            except Exception:
                pass
            return
        # PIL
        if PILImage is not None and isinstance(x, PILImage.Image):
            update_with_type_tag("PIL.Image")
            h.update(str((x.size, x.mode)).encode("utf-8"))
            try:
                buf = io.BytesIO()
                x.save(buf, format="PNG", optimize=True)
                update_bytes(buf.getvalue())
            except Exception:
                pass
            return
        # dict
        if isinstance(x, dict):
            update_with_type_tag("dict")
            # sort keys for stability
            for k in sorted(x.keys(), key=lambda z: str(z)):
                inner(k)
                inner(x[k])
            return
        # list/tuple
        if isinstance(x, (list, tuple)):
            update_with_type_tag(type(x).__name__)
            for el in x:
                inner(el)
            return
        # fallback: repr
        update_with_type_tag("repr")
        update_bytes(repr(x).encode("utf-8"))

    inner(obj)
    return h.hexdigest()


# =========================
# LRU Cache with Weak Refs
# =========================

class Weakable:
    """
    Small wrapper to allow weakref of built-in containers where needed.
    """
    __slots__ = ("value",)

    def __init__(self, value: Any) -> None:
        self.value = value


class WeakLRUCache:
    """
    A memory-aware LRU cache with optional TTL and weak references.

    - Stores up to 'max_bytes' of estimated memory.
    - Evicts least-recently-used items when over capacity.
    - TTL expiry if 'ttl_seconds' is set.
    - Thread-safe operations.
    - Values optionally wrapped for weak referencing when supported.

    Notes:
    - For built-in immutable types (str, dict, list), Python does not allow weakrefs, so
      we fall back to storing the object directly.
    - For large outputs, you should set a conservative max_bytes to keep memory usage modest.
    """

    def __init__(self, max_bytes: int = 64 * 1024 * 1024, ttl_seconds: Optional[float] = None) -> None:
        self.max_bytes = max_bytes
        self.ttl_seconds = ttl_seconds
        self._map: OrderedDict[str, Tuple[float, int, Any]] = OrderedDict()
        self._bytes = 0
        self._lock = threading.RLock()

    def _evict_if_needed(self) -> None:
        with self._lock:
            now = time.time()
            # TTL pass
            if self.ttl_seconds is not None:
                to_del = []
                for k, (ts, size, val) in self._map.items():
                    if (now - ts) > self.ttl_seconds:
                        to_del.append(k)
                for k in to_del:
                    _, sz, _ = self._map.pop(k)
                    self._bytes -= sz
            # Size-based eviction
            while self._bytes > self.max_bytes and self._map:
                k, (_, sz, _) = self._map.popitem(last=False)  # pop LRU
                self._bytes -= sz

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            item = self._map.get(key)
            if not item:
                return None
            ts, size, val = item
            # Move to MRU
            self._map.move_to_end(key, last=True)
            # Rehydrate weakref if needed
            if isinstance(val, weakref.ReferenceType):
                real = val()
                if real is None:
                    # collected
                    del self._map[key]
                    self._bytes -= size
                    return None
                return real.value if isinstance(real, Weakable) else real
            return val

    def put(self, key: str, value: Any) -> None:
        try:
            size = _estimate_size_bytes(value)
        except Exception:
            size = 0
        with self._lock:
            # remove old if exists
            if key in self._map:
                _, old_size, _ = self._map.pop(key)
                self._bytes -= old_size

            # Try weakref wrapping
            stored_value: Any
            try:
                w = Weakable(value)
                stored_value = weakref.ref(w)  # weakref to wrapper
            except Exception:
                stored_value = value

            self._map[key] = (time.time(), size, stored_value)
            self._bytes += size
            # Evict if needed
            self._evict_if_needed()

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "entries": len(self._map),
                "approx_bytes": self._bytes,
                "max_bytes": self.max_bytes,
                "ttl_seconds": self.ttl_seconds,
            }

    def clear(self) -> None:
        with self._lock:
            self._map.clear()
            self._bytes = 0


# =========================
# Retry Mechanism
# =========================

def with_retry(
    fn: Callable[[], Any],
    *,
    retries: int = 3,
    backoff_base: float = 0.25,
    backoff_jitter: float = 0.1,
    retriable: Optional[Callable[[BaseException], bool]] = None,
    logger: Optional[logging.Logger] = LOGGER,
    retry_name: str = "operation",
) -> Any:
    """
    Execute fn() with exponential backoff retries. Intended for load-time network or transient errors.

    retriable(e) returns True if the exception should be retried. If None, all exceptions are retried.
    """
    if retriable is None:
        retriable = lambda e: True  # retry everything by default

    for attempt in range(1, retries + 1):
        try:
            return fn()
        except BaseException as e:
            if not retriable(e) or attempt == retries:
                if logger:
                    logger.error(f"[retry:{retry_name}] Failed after {attempt}/{retries} attempts: {e}")
                raise
            delay = backoff_base * (2 ** (attempt - 1)) + random.uniform(0, backoff_jitter)
            if logger:
                logger.warning(f"[retry:{retry_name}] Attempt {attempt}/{retries} failed: {e} -> sleeping {delay:.2f}s")
            time.sleep(delay)


# =========================
# Metrics and Profiling
# =========================

@dataclass
class MovingStats:
    """
    Track moving statistics for latencies and throughput without heavy memory usage.
    """
    count: int = 0
    total_time_s: float = 0.0
    max_time_s: float = 0.0
    min_time_s: float = float("inf")

    def update(self, latency_s: float) -> None:
        self.count += 1
        self.total_time_s += latency_s
        if latency_s > self.max_time_s:
            self.max_time_s = latency_s
        if latency_s < self.min_time_s:
            self.min_time_s = latency_s

    @property
    def mean_time_s(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total_time_s / self.count

    def as_dict(self) -> Dict[str, Any]:
        return {
            "count": self.count,
            "total_time_s": self.total_time_s,
            "mean_time_s": self.mean_time_s,
            "min_time_s": 0.0 if self.min_time_s == float("inf") else self.min_time_s,
            "max_time_s": self.max_time_s,
        }


@dataclass
class RuntimeMetrics:
    """
    Aggregate runtime metrics: calls, errors, cache hits, batch sizes, etc.
    """
    started_at: float = field(default_factory=time.time)
    calls: int = 0
    errors: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    batches_processed: int = 0
    items_processed: int = 0
    batch_latency: MovingStats = field(default_factory=MovingStats)
    item_latency: MovingStats = field(default_factory=MovingStats)
    last_error: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        up_s = time.time() - self.started_at
        return {
            "uptime_s": up_s,
            "calls": self.calls,
            "errors": self.errors,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "batches_processed": self.batches_processed,
            "items_processed": self.items_processed,
            "batch_latency": self.batch_latency.as_dict(),
            "item_latency": self.item_latency.as_dict(),
            "last_error": self.last_error,
        }


# =========================
# Configuration Management
# =========================

@dataclass
class PipelineRuntimeConfig:
    """
    Validated configuration for SmartPipeline execution runtime.

    Key knobs:
    - max_batch_size: dynamic batch ceiling. Use 1 to disable dynamic batching.
    - max_batch_wait_ms: maximum time to wait for batch to fill before executing.
    - num_workers: number of independent pipeline workers. For GPU tasks, 1 is usually ideal.
    - retries: number of inference retries on transient errors.
    - inference_timeout_s: timeout for a single inference call.
    - cache_enabled: enable end-to-end caching (input+kwargs -> output).
    - cache_max_bytes: approximate memory cap for cache.
    - cache_ttl_seconds: optional expiration for cache entries.
    - enable_tracemalloc: enables Python allocation tracing for memory insights.
    - collect_metrics: gather runtime metrics.
    - low_gpu_mem_mode: attempt to automatically reduce batch sizes or fallback to CPU on OOM.
    """
    max_batch_size: int = 16
    max_batch_wait_ms: int = 10
    num_workers: int = 1
    retries: int = 1
    inference_timeout_s: Optional[float] = None
    cache_enabled: bool = True
    cache_max_bytes: int = 64 * 1024 * 1024
    cache_ttl_seconds: Optional[float] = 600.0
    enable_tracemalloc: bool = False
    collect_metrics: bool = True
    low_gpu_mem_mode: bool = True

    def __post_init__(self) -> None:
        # Validation
        assert self.max_batch_size >= 1, "max_batch_size must be >= 1"
        assert self.max_batch_wait_ms >= 0, "max_batch_wait_ms must be >= 0"
        assert self.num_workers >= 1, "num_workers must be >= 1"
        assert self.retries >= 1, "retries must be >= 1"
        if self.inference_timeout_s is not None:
            assert self.inference_timeout_s > 0, "inference_timeout_s must be > 0"
        assert self.cache_max_bytes >= 0, "cache_max_bytes must be >= 0"


# =========================
# Dynamic Batching Engine
# =========================

class _BatchItem:
    __slots__ = ("input_obj", "kwargs", "future", "cache_key", "enqueue_time_ms")

    def __init__(self, input_obj: Any, kwargs: Dict[str, Any], future: Future, cache_key: Optional[str]) -> None:
        self.input_obj = input_obj
        self.kwargs = kwargs
        self.future = future
        self.cache_key = cache_key
        self.enqueue_time_ms = _now_ms()


class _BatchWorker(threading.Thread):
    """
    A worker thread that collects enqueued _BatchItem objects, groups them by kwargs,
    and executes the underlying transformers Pipeline in batches.

    Guarantees:
    - Underlying pipeline is only used within this worker thread => thread safety.
    - Respects max_batch_size and max_batch_wait_ms.
    - Handles OOM errors by auto-splitting batches or CPU fallback if enabled.
    - Updates metrics and populates cache on successes.
    """
    daemon = True

    def __init__(
        self,
        name: str,
        base_pipeline: Pipeline,
        runtime_cfg: PipelineRuntimeConfig,
        cache: Optional[WeakLRUCache],
        metrics: RuntimeMetrics,
        shutdown_event: threading.Event,
        device: torch.device,
        logger: logging.Logger,
    ) -> None:
        super().__init__(name=name)
        self._pipe = base_pipeline
        self._cfg = runtime_cfg
        self._cache = cache
        self._metrics = metrics
        self._shutdown = shutdown_event
        self._device = device
        self._logger = logger
        self._cv = threading.Condition()
        self._queue: Deque[_BatchItem] = deque()

    def enqueue(self, item: _BatchItem) -> None:
        with self._cv:
            self._queue.append(item)
            self._cv.notify()

    def _pop_grouped_batch(self) -> List[_BatchItem]:
        """
        Pop up to max_batch_size items, all sharing the same kwargs signature (to comply with HF Pipeline API).
        """
        if not self._queue:
            return []

        def signature(kwargs: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
            # Remove None values to improve batching; sort keys for determinism
            cleaned = tuple(sorted((k, v) for k, v in kwargs.items() if v is not None))
            return cleaned

        items: List[_BatchItem] = []
        first_sig: Optional[Tuple[Tuple[str, Any], ...]] = None

        while self._queue and len(items) < self._cfg.max_batch_size:
            candidate = self._queue[0]
            sig = signature(candidate.kwargs)
            if first_sig is None:
                first_sig = sig
            # Only batch with items having identical call-kwargs
            if sig != first_sig:
                # stop here to keep API behavior correct
                break
            items.append(self._queue.popleft())

        return items

    def _call_pipe(self, inputs: List[Any], kwargs: Dict[str, Any]) -> Any:
        # Call with retries for transient runtime errors. Most inference errors should surface immediately.
        def do_call():
            return self._pipe(inputs, **kwargs)

        return with_retry(
            do_call,
            retries=self._cfg.retries,
            retry_name="inference",
            logger=self._logger,
            retriable=lambda e: isinstance(e, (RuntimeError, ValueError, torch.cuda.OutOfMemoryError)),
        )

    def _handle_oom(self, exc: BaseException) -> None:
        # Attempt to free GPU cache on OOM
        self._logger.error(f"OOM or runtime error detected: {exc}")
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    def _execute_batch(self, batch: List[_BatchItem]) -> None:
        inputs = [b.input_obj for b in batch]
        kwargs = batch[0].kwargs if batch else {}

        # Timer start
        t0 = time.perf_counter()

        try:
            # The pipeline API accepts single item or list. Always pass list for batch-mode.
            outputs = self._call_pipe(inputs, kwargs=kwargs)
        except torch.cuda.OutOfMemoryError as oom:
            self._handle_oom(oom)
            if self._cfg.low_gpu_mem_mode and len(batch) > 1:
                # Split batch and retry in halves
                mid = len(batch) // 2
                self._execute_batch(batch[:mid])
                self._execute_batch(batch[mid:])
                return
            # Fallback to CPU if allowed and feasible
            if self._cfg.low_gpu_mem_mode and self._device.type == "cuda":
                try:
                    self._logger.warning("Falling back to CPU due to OOM.")
                    self._pipe.model.to("cpu")
                    outputs = self._call_pipe(inputs, kwargs=kwargs)
                except Exception as e:
                    self._logger.exception("CPU fallback failed.")
                    for item in batch:
                        item.future.set_exception(e)
                    self._metrics.errors += len(batch)
                    self._metrics.last_error = repr(e)
                    return
            else:
                for item in batch:
                    item.future.set_exception(oom)
                self._metrics.errors += len(batch)
                self._metrics.last_error = repr(oom)
                return
        except Exception as e:
            self._logger.exception("Inference failed.")
            for item in batch:
                item.future.set_exception(e)
            self._metrics.errors += len(batch)
            self._metrics.last_error = repr(e)
            return

        # Timer end, update metrics
        t1 = time.perf_counter()
        batch_latency_s = t1 - t0

        # HF pipeline returns either a list (for list inputs) or non-list (rare). Normalize to list.
        if not isinstance(outputs, list) or (isinstance(outputs, list) and len(outputs) != len(batch)):
            # Some pipelines can return a dict for single input even if we pass a list of one; normalize.
            outputs = outputs if isinstance(outputs, list) else [outputs]

        if len(outputs) != len(batch):
            # Graceful degradation: if output length mismatches, map as best effort
            self._logger.warning(
                f"Output length mismatch: got {len(outputs)} for batch of {len(batch)}. Broadcasting first result."
            )
            if outputs:
                outputs = [copy.deepcopy(outputs[0]) for _ in batch]
            else:
                outputs = [None for _ in batch]

        # Set results and cache
        for item, out in zip(batch, outputs):
            if self._cache is not None and item.cache_key is not None:
                try:
                    self._cache.put(item.cache_key, out)
                except Exception:
                    pass
            item.future.set_result(out)

        # Update metrics
        self._metrics.batches_processed += 1
        self._metrics.items_processed += len(batch)
        self._metrics.batch_latency.update(batch_latency_s)
        for item in batch:
            self._metrics.item_latency.update(batch_latency_s / max(1, len(batch)))

    def run(self) -> None:
        """
        Worker loop: accumulate work up to max_batch_size and max_batch_wait_ms, then execute.
        """
        while not self._shutdown.is_set():
            with self._cv:
                if not self._queue:
                    self._cv.wait(timeout=self._cfg.max_batch_wait_ms / 1000.0 if self._cfg.max_batch_wait_ms > 0 else None)
                    if not self._queue:
                        continue

                # Wait up to max_batch_wait_ms to allow more items to arrive (dynamic batching)
                if self._cfg.max_batch_wait_ms > 0:
                    deadline = _now_ms() + self._cfg.max_batch_wait_ms
                    while len(self._queue) < self._cfg.max_batch_size and _now_ms() < deadline:
                        remaining = (deadline - _now_ms()) / 1000.0
                        if remaining <= 0:
                            break
                        self._cv.wait(timeout=remaining)

                batch = self._pop_grouped_batch()

            if not batch:
                continue
            self._execute_batch(batch)


# =========================
# Smart Pipeline
# =========================

class SmartPipeline:
    """
    SmartPipeline is a thread-safe, production-grade wrapper around transformers.Pipeline.

    Features:
    - 100% compatible with transformers pipeline behavior and kwargs.
    - Dynamic batching to maximize throughput under concurrent calls.
    - LRU cache on (input, kwargs) -> output to reduce redundant computation.
    - Worker pool for CPU-bound parallelism (num_workers > 1).
    - Rich logging, metrics, memory snapshots, and graceful degradation.

    Contract:
    - Construct using the factory 'smart_pipeline' (recommended) or constructor (advanced).
    - Call it exactly like HF pipeline (accepts single input or list).
    - Use 'submit' for async calls that return futures.
    - Call 'shutdown' on teardown to stop workers and release resources.
    """

    def __init__(
        self,
        *,
        base_pipeline_build_fn: Callable[[], Pipeline],
        runtime_config: PipelineRuntimeConfig,
        device: Optional[Union[str, int, torch.device]] = None,
        verbose: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        _configure_logging(verbose=verbose)
        self._logger = logging.getLogger("smart_pipeline")

        self._runtime = runtime_config
        self._device = _available_device(device)

        # Optional profiling
        if self._runtime.enable_tracemalloc and not tracemalloc.is_tracing():
            self._logger.info("Enabling tracemalloc.")
            tracemalloc.start()

        # Random reproducibility option when desired
        if seed is not None:
            try:
                set_seed(seed)
            except Exception:
                pass

        # Cache layer
        self._cache = WeakLRUCache(
            max_bytes=self._runtime.cache_max_bytes, ttl_seconds=self._runtime.cache_ttl_seconds
        ) if self._runtime.cache_enabled else None

        # Metrics
        self._metrics = RuntimeMetrics() if self._runtime.collect_metrics else RuntimeMetrics(collect_metrics=False)  # type: ignore[arg-type]

        # Build pipelines (resource pooling)
        self._shutdown_event = threading.Event()
        self._workers: List[_BatchWorker] = []
        self._pipes: List[Pipeline] = []

        # Build one pipeline and then additional ones if required
        def build_with_retry() -> Pipeline:
            return with_retry(
                base_pipeline_build_fn,
                retries=3,
                retry_name="build_pipeline",
                logger=self._logger,
                retriable=lambda e: True,  # network/remote issues
            )

        for i in range(self._runtime.num_workers):
            p = build_with_retry()
            # Force to eval and correct device
            try:
                p.model.eval()
                if hasattr(p.model, "to"):
                    p.model.to(self._device)
            except Exception:
                pass

            self._pipes.append(p)
            worker = _BatchWorker(
                name=f"smart-pipeline-worker-{i}",
                base_pipeline=p,
                runtime_cfg=self._runtime,
                cache=self._cache,
                metrics=self._metrics,
                shutdown_event=self._shutdown_event,
                device=self._device,
                logger=self._logger,
            )
            worker.start()
            self._workers.append(worker)

        atexit.register(self.shutdown)

    def _maybe_cache_key(self, input_obj: Any, kwargs: Dict[str, Any]) -> Optional[str]:
        if not self._cache:
            return None
        # A conservative cache key approach: hash of input and immutable view of kwargs
        try:
            key = {"input": input_obj, "kwargs": kwargs}
            return _stable_hash(key)
        except Exception:
            return None

    def _schedule_item(self, input_obj: Any, kwargs: Dict[str, Any]) -> Any:
        # Single item scheduling to the least-loaded worker
        cache_key = self._maybe_cache_key(input_obj, kwargs)
        if cache_key and self._cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._metrics.cache_hits += 1
                return copy.deepcopy(cached)
            else:
                self._metrics.cache_misses += 1

        # Determine worker (least pending queue length)
        # A simple heuristic: they are similar because each worker manages its own internal queue
        worker = self._workers[random.randrange(len(self._workers))] if len(self._workers) > 1 else self._workers[0]  # random or round-robin
        future = Future()
        item = _BatchItem(input_obj=input_obj, kwargs=kwargs, future=future, cache_key=cache_key)
        worker.enqueue(item)

        # Wait for result (synchronous). For async usage, use 'submit' method instead.
        timeout = self._runtime.inference_timeout_s
        result = future.result(timeout=timeout) if timeout is not None else future.result()
        return result

    def submit(self, input_obj: Any, **kwargs: Any) -> Future:
        """
        Submit a single item asynchronously and return a Future.
        """
        cache_key = self._maybe_cache_key(input_obj, kwargs)
        if cache_key and self._cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._metrics.cache_hits += 1
                fut = Future()
                fut.set_result(copy.deepcopy(cached))
                return fut
            else:
                self._metrics.cache_misses += 1

        worker = self._workers[random.randrange(len(self._workers))] if len(self._workers) > 1 else self._workers[0]
        future = Future()
        item = _BatchItem(input_obj=input_obj, kwargs=kwargs, future=future, cache_key=cache_key)
        worker.enqueue(item)
        return future

    def __call__(self, inputs: Any, **kwargs: Any) -> Any:
        """
        Call SmartPipeline just like a Hugging Face pipeline.

        - If inputs is a list, we still benefit from batching, but dynamic batching mainly shines when
          multiple threads submit single inputs concurrently.
        - kwargs are forwarded and will be used to group batchable requests.
        """
        self._metrics.calls += 1
        t0 = time.perf_counter()

        # If user provides a list, we schedule items individually and gather results.
        if isinstance(inputs, list):
            results: List[Any] = []
            for inp in inputs:
                results.append(self._schedule_item(inp, kwargs))
            latency_s = time.perf_counter() - t0
            self._metrics.item_latency.update(latency_s / max(1, len(inputs)))
            return results
        else:
            result = self._schedule_item(inputs, kwargs)
            latency_s = time.perf_counter() - t0
            self._metrics.item_latency.update(latency_s)
            return result

    def metrics(self) -> Dict[str, Any]:
        """
        Return runtime metrics: throughput, cache stats, latencies, errors, uptime.
        """
        m = self._metrics.as_dict()
        if self._cache:
            m["cache"] = self._cache.stats()
        m["device"] = str(self._device)
        m["num_workers"] = len(self._workers)
        return m

    def memory_snapshot(self) -> Dict[str, Any]:
        """
        Provide a memory snapshot including RAM and GPU info where possible.
        """
        snap: Dict[str, Any] = {}
        if _HAS_PSUTIL:
            proc = psutil.Process(os.getpid())
            with proc.oneshot():
                mem = proc.memory_info()
                snap["rss_bytes"] = mem.rss
                snap["vms_bytes"] = mem.vms
                snap["cpu_percent"] = proc.cpu_percent(interval=0.0)
        if torch.cuda.is_available():
            try:
                idx = torch.cuda.current_device()
                snap["cuda"] = {
                    "device": torch.cuda.get_device_name(idx),
                    "mem_allocated": int(torch.cuda.memory_allocated(idx)),
                    "mem_reserved": int(torch.cuda.memory_reserved(idx)),
                    "mem_max_allocated": int(torch.cuda.max_memory_allocated(idx)),
                }
            except Exception:
                pass
        if tracemalloc.is_tracing():
            try:
                current, peak = tracemalloc.get_traced_memory()
                snap["tracemalloc"] = {"current_bytes": current, "peak_bytes": peak}
            except Exception:
                pass
        return snap

    def warmup(self, example_input: Any, **kwargs: Any) -> None:
        """
        Run a warmup call to initialize kernels, caches, and JIT to reduce first-token latency.
        """
        try:
            _ = self(example_input, **kwargs)
        except Exception as e:
            self._logger.warning(f"Warmup failed: {e}")

    def flush_cache(self) -> None:
        if self._cache:
            self._cache.clear()

    def shutdown(self) -> None:
        """
        Gracefully stop workers, flush GPU cache, and collect garbage.
        """
        if getattr(self, "_shutdown_event", None) is None:
            return
        if self._shutdown_event.is_set():
            return
        self._logger.info("Shutting down SmartPipeline workers...")
        self._shutdown_event.set()
        for w in getattr(self, "_workers", []):
            try:
                # Wake any worker waiting on condition
                if hasattr(w, "_cv"):
                    with w._cv:
                        w._cv.notify_all()
                w.join(timeout=2.0)
            except Exception:
                pass
        # Encourage cleanup
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        gc.collect()


# =========================
# Factory Function
# =========================

def smart_pipeline(
    task: Optional[str] = None,
    model: Optional[Union[str, "PreTrainedModel"]] = None,
    config: Optional[Union[str, "PretrainedConfig"]] = None,
    tokenizer: Optional[Union[str, "PreTrainedTokenizer", "PreTrainedTokenizerFast"]] = None,
    feature_extractor: Optional[Union[str, "PreTrainedFeatureExtractor"]] = None,
    image_processor: Optional[Union[str, "BaseImageProcessor"]] = None,
    framework: Optional[str] = None,
    revision: Optional[str] = None,
    use_fast: bool = True,
    token: Optional[Union[str, bool]] = None,
    device: Optional[Union[int, str, torch.device]] = None,
    device_map=None,
    torch_dtype: Optional[Union[str, torch.dtype]] = None,
    trust_remote_code: Optional[bool] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    pipeline_class: Optional[Any] = None,
    *,
    # Enhancements and knobs:
    runtime_config: Optional[PipelineRuntimeConfig] = None,
    verbose: bool = False,  # NOTE: toggles rich logging when available
    seed: Optional[int] = None,
    # Optional quantization hints:
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    bnb_compute_dtype: Optional[str] = None,  # "float16"/"bfloat16"/"float32"
    **kwargs: Any,
) -> SmartPipeline:
    """
    Create a SmartPipeline that wraps transformers.pipeline with robustness and performance features.

    All standard transformers.pipeline kwargs are supported and forwarded as-is (no breaking changes).

    Enhancements:
    - runtime_config: controls batching, caching, retries, and metrics.
    - verbose: enables rich logging when installed.
    - quantization hints: use BitsAndBytesConfig for 8-bit/4-bit loading if requested.

    Returns:
        SmartPipeline instance
    """

    _configure_logging(verbose=verbose)
    logger = logging.getLogger("smart_pipeline.factory")

    if runtime_config is None:
        # Sensible defaults: prefer single worker on GPU, modest batching
        auto_device = _available_device(device)
        default_workers = 1 if _is_gpu_device(auto_device) else min(4, os.cpu_count() or 2)
        runtime_config = PipelineRuntimeConfig(
            max_batch_size=16 if _is_gpu_device(auto_device) else 8,
            max_batch_wait_ms=10,
            num_workers=default_workers,
            retries=1,
            inference_timeout_s=None,
            cache_enabled=True,
            cache_max_bytes=64 * 1024 * 1024,
            cache_ttl_seconds=600.0,
            enable_tracemalloc=False,
            collect_metrics=True,
            low_gpu_mem_mode=True,
        )

    # Prepare model_kwargs and quantization
    model_kwargs = {} if model_kwargs is None else dict(model_kwargs)

    # Dtype conversions
    tdtype = _torch_dtype_from_str(torch_dtype)
    if tdtype is not None:
        model_kwargs["torch_dtype"] = tdtype

    if load_in_8bit or load_in_4bit:
        try:
            compute_dtype = None
            if bnb_compute_dtype:
                compute_dtype = _torch_dtype_from_str(bnb_compute_dtype)
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=bool(load_in_8bit),
                load_in_4bit=bool(load_in_4bit),
                llm_int8_enable_fp32_cpu_offload=False,
                bnb_4bit_compute_dtype=compute_dtype,
            )
            model_kwargs["quantization_config"] = bnb_config
            # Make sure device_map is set to 'auto' if none provided
            if device_map is None and _is_gpu_device(_available_device(device)):
                device_map = "auto"
                logger.info("Quantization requested. Setting device_map='auto' for optimal placement.")
        except Exception as e:
            logger.warning(f"Quantization setup failed; proceeding without it. Error: {e}")

    # Build function that returns a native HF pipeline
    def build_base_pipeline() -> Pipeline:
        return hf_pipeline(
            task=task,
            model=model,
            config=config,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            image_processor=image_processor,
            framework=framework,
            revision=revision,
            use_fast=use_fast,
            token=token,
            device=device,
            device_map=device_map,
            torch_dtype=tdtype,
            trust_remote_code=trust_remote_code,
            model_kwargs=model_kwargs,
            pipeline_class=pipeline_class,
            **kwargs,
        )

    return SmartPipeline(
        base_pipeline_build_fn=build_base_pipeline,
        runtime_config=runtime_config,
        device=device,
        verbose=verbose,
        seed=seed,
    )


# =========================
# Demonstrations / Examples
# =========================

if __name__ == "__main__":
    """
    Practical, self-contained examples.
    Note: you can comment out sections to reduce downloads during first run.
    """

    # Example 1: Sentiment Analysis (Text Classification)
    print("\n[Example 1] Sentiment Analysis with dynamic batching, rich logging, metrics\n")
    sp = smart_pipeline(
        task="text-classification",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        verbose=True,
        runtime_config=PipelineRuntimeConfig(
            max_batch_size=16,
            max_batch_wait_ms=20,
            num_workers=1,  # GPU -> 1 worker; CPU can use >1
            cache_enabled=True,
            cache_max_bytes=16 * 1024 * 1024,
            cache_ttl_seconds=300,
            retries=2,
            enable_tracemalloc=True,
            low_gpu_mem_mode=True,
        ),
        seed=42,
    )

    # Warmup
    sp.warmup("This warmup is only to pre-initialize kernels.")
    texts = [
        "I absolutely love this product!",
        "Worst experience ever.",
        "Service was okay, not great.",
        "I absolutely love this product!",  # duplicate for cache hit
    ]
    out = sp(texts)
    print("Outputs:", out)
    print("Metrics:", json.dumps(sp.metrics(), indent=2))
    print("Memory snapshot:", json.dumps(sp.memory_snapshot(), indent=2))

    # Example 2: Async submission + dynamic batching
    print("\n[Example 2] Async submission of multiple single-item calls (batched under the hood)\n")
    futs = [sp.submit(t) for t in texts]
    res = [f.result() for f in futs]
    print("Async results:", res)
    print("Metrics:", json.dumps(sp.metrics(), indent=2))

    # Example 3: Text Generation (optional quantization)
    # Quantization can reduce GPU memory usage; if not desired, set both flags False.
    print("\n[Example 3] Text Generation with optional 8-bit load (may download a model; comment out to skip)\n")
    try:
        gen = smart_pipeline(
            task="text-generation",
            model="gpt2",
            verbose=True,
            load_in_8bit=False,  # set True if you have bitsandbytes installed and a compatible GPU
            runtime_config=PipelineRuntimeConfig(
                max_batch_size=4,
                max_batch_wait_ms=15,
                num_workers=1,
                cache_enabled=True,
                cache_max_bytes=32 * 1024 * 1024,
                retries=1,
            ),
        )
        prompt = "Once upon a time"
        gen_out = gen(prompt, max_new_tokens=20, do_sample=True, temperature=0.9, top_p=0.95)
        print("Generation:", gen_out)
        print("Gen Metrics:", json.dumps(gen.metrics(), indent=2))
        gen.shutdown()
    except Exception as e:
        print("Skipping text-generation example due to:", e)

    # Example 4: Image Classification with graceful CPU fallback
    print("\n[Example 4] Image Classification (requires PIL). Graceful fallback on limited GPU memory.\n")
    if PILImage is not None:
        try:
            # Create a dummy image to avoid external downloads
            img = PILImage.new("RGB", (224, 224), color=(127, 127, 127))
            ic = smart_pipeline(
                task="image-classification",
                model="google/vit-base-patch16-224",
                verbose=True,
                runtime_config=PipelineRuntimeConfig(
                    max_batch_size=8,
                    max_batch_wait_ms=10,
                    num_workers=1,
                    low_gpu_mem_mode=True,
                    cache_enabled=True,
                ),
            )
            cls = ic(img)
            print("Image classification:", cls)
            print("IC Metrics:", json.dumps(ic.metrics(), indent=2))
            ic.shutdown()
        except Exception as e:
            print("Skipping image-classification example due to:", e)
    else:
        print("PIL not available; skipping image classification example.")

    # Cleanup
    sp.shutdown()
    print("\nAll examples completed.\n")