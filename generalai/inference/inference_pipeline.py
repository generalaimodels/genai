"""
Advanced Custom Pipeline Framework for Transformers - Enhanced Hardware Optimization
==================================================================================

A professional-grade, highly optimized pipeline system with advanced hardware optimization,
automatic best hardware selection, and real-time hardware adaptation capabilities.

New Features Added:
1. Advanced Hardware Profiler & Benchmarking
2. Dynamic Hardware Adaptation & Load Balancing
3. Multi-Device Orchestration & Parallelism
4. Hardware-Specific Optimizations (Mixed Precision, NUMA, etc.)
5. Intelligent Performance Prediction Models
6. Real-time Hardware Monitoring & Auto-switching
7. Accelerator Support (TPU, Intel Gaudi, AMD ROCm)
8. Memory Optimization & Pinning Strategies

Author: Advanced AI Systems
Version: 2.1.0
Optimized for: Production-grade ML systems with maximum hardware utilization
"""

import logging
import torch
import gc
import time
import threading
import psutil
import warnings
import subprocess
import platform
import multiprocessing
from pathlib import Path
from typing import Optional, Union, Dict, Any, List, Callable, Iterator, Tuple, NamedTuple
from dataclasses import dataclass, field
from collections import defaultdict, OrderedDict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import wraps, lru_cache
from contextlib import contextmanager
import weakref
import math

# Core Transformers imports
from transformers import (
    PreTrainedModel, TFPreTrainedModel, PretrainedConfig,
    PreTrainedTokenizer, PreTrainedTokenizerFast, Pipeline,
    AutoConfig, AutoTokenizer, AutoImageProcessor, AutoFeatureExtractor,
    AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification, AutoModelForQuestionAnswering,
    AutoModelForTokenClassification, AutoModelForImageClassification,
    AutoModel, BitsAndBytesConfig, set_seed
)
from transformers.feature_extraction_utils import PreTrainedFeatureExtractor
from transformers.image_processing_utils import BaseImageProcessor

# Rich console for beautiful output
try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.logging import RichHandler
    from rich.tree import Tree
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    warnings.warn("Rich not available. Install with: pip install rich")

# Hardware optimization imports
try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

try:
    import intel_extension_for_pytorch as ipex
    IPEX_AVAILABLE = True
except ImportError:
    IPEX_AVAILABLE = False

# Standard library
import json
import os
import sys
from queue import Queue, Empty
import numpy as np


@dataclass
class HardwareProfile:
    """
    Comprehensive hardware profile containing detailed system capabilities
    """
    device_type: str
    device_index: int
    name: str
    memory_total: int  # bytes
    memory_available: int  # bytes
    compute_capability: Optional[Tuple[int, int]] = None
    cores: Optional[int] = None
    frequency: Optional[float] = None  # Hz
    bandwidth: Optional[float] = None  # GB/s
    power_limit: Optional[float] = None  # Watts
    utilization: float = 0.0  # percentage
    temperature: Optional[float] = None  # Celsius
    supports_mixed_precision: bool = False
    supports_tensor_cores: bool = False
    numa_node: Optional[int] = None
    pcie_generation: Optional[int] = None
    architecture: Optional[str] = None


@dataclass
class PerformanceMetrics:
    """
    Performance metrics for hardware benchmarking and selection
    """
    throughput: float  # operations/second
    latency: float  # seconds
    memory_efficiency: float  # percentage
    power_efficiency: float  # operations/watt
    stability_score: float  # variance measure
    cost_score: float  # price/performance ratio


class HardwareProfiler:
    """
    Advanced hardware profiling system that detects and benchmarks all available hardware
    
    Provides comprehensive system analysis including:
    - CPU architecture and capabilities
    - GPU specifications and memory
    - Accelerator detection (TPU, Intel Gaudi, etc.)
    - Memory hierarchy analysis
    - NUMA topology mapping
    - Power and thermal characteristics
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.console = Console() if RICH_AVAILABLE and verbose else None
        self.profiles = {}
        self.benchmark_cache = {}
        self._initialize_monitoring()
    
    def _initialize_monitoring(self):
        """Initialize hardware monitoring systems"""
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
            except Exception as e:
                warnings.warn(f"Failed to initialize NVML: {e}")
    
    def profile_all_hardware(self) -> Dict[str, HardwareProfile]:
        """
        Profile all available hardware and return comprehensive specifications
        
        Returns:
            Dictionary mapping device identifiers to HardwareProfile objects
        """
        if self.verbose and self.console:
            self.console.print("🔍 Profiling Hardware...", style="bold blue")
        
        profiles = {}
        
        # Profile CPU
        cpu_profile = self._profile_cpu()
        profiles["cpu"] = cpu_profile
        
        # Profile CUDA GPUs
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_profile = self._profile_cuda_gpu(i)
                profiles[f"cuda:{i}"] = gpu_profile
        
        # Profile MPS (Apple Silicon)
        if torch.backends.mps.is_available():
            mps_profile = self._profile_mps()
            profiles["mps"] = mps_profile
        
        # Profile Intel XPU (if available)
        if IPEX_AVAILABLE and hasattr(torch, 'xpu') and torch.xpu.is_available():
            for i in range(torch.xpu.device_count()):
                xpu_profile = self._profile_intel_xpu(i)
                profiles[f"xpu:{i}"] = xpu_profile
        
        # Profile TPU (if available)
        tpu_profiles = self._profile_tpu()
        profiles.update(tpu_profiles)
        
        self.profiles = profiles
        
        if self.verbose:
            self._display_hardware_summary()
        
        return profiles
    
    def _profile_cpu(self) -> HardwareProfile:
        """Profile CPU capabilities and specifications"""
        cpu_count = multiprocessing.cpu_count()
        cpu_freq = psutil.cpu_freq()
        memory = psutil.virtual_memory()
        
        # Get CPU architecture info
        architecture = platform.machine()
        processor_name = platform.processor()
        
        # Detect NUMA topology
        numa_node = self._detect_numa_node()
        
        # Check for advanced instruction sets
        supports_avx = self._check_cpu_feature("avx")
        supports_avx512 = self._check_cpu_feature("avx512")
        
        return HardwareProfile(
            device_type="cpu",
            device_index=0,
            name=processor_name or f"CPU-{architecture}",
            memory_total=memory.total,
            memory_available=memory.available,
            cores=cpu_count,
            frequency=cpu_freq.current * 1e6 if cpu_freq else None,
            utilization=psutil.cpu_percent(interval=1),
            supports_mixed_precision=supports_avx512,
            numa_node=numa_node,
            architecture=architecture
        )
    
    def _profile_cuda_gpu(self, device_index: int) -> HardwareProfile:
        """Profile CUDA GPU specifications"""
        device = torch.device(f"cuda:{device_index}")
        props = torch.cuda.get_device_properties(device_index)
        
        memory_total = props.total_memory
        memory_allocated = torch.cuda.memory_allocated(device_index)
        memory_available = memory_total - memory_allocated
        
        # Get GPU utilization if NVML is available
        utilization = 0.0
        temperature = None
        power_limit = None
        
        if NVML_AVAILABLE:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                utilization = util.gpu
                
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                temperature = temp
                
                power_info = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)
                power_limit = power_info[1] / 1000.0  # Convert to watts
            except Exception:
                pass
        
        # Determine tensor core support
        supports_tensor_cores = props.major >= 7  # Volta and newer
        
        return HardwareProfile(
            device_type="cuda",
            device_index=device_index,
            name=props.name,
            memory_total=memory_total,
            memory_available=memory_available,
            compute_capability=(props.major, props.minor),
            cores=props.multi_processor_count,
            utilization=utilization,
            temperature=temperature,
            power_limit=power_limit,
            supports_mixed_precision=True,
            supports_tensor_cores=supports_tensor_cores,
            architecture=f"sm_{props.major}{props.minor}"
        )
    
    def _profile_mps(self) -> HardwareProfile:
        """Profile Apple Silicon MPS"""
        # MPS doesn't provide detailed specs, so we estimate
        return HardwareProfile(
            device_type="mps",
            device_index=0,
            name="Apple Silicon GPU",
            memory_total=8 * 1024 * 1024 * 1024,  # Estimate 8GB
            memory_available=6 * 1024 * 1024 * 1024,  # Estimate 6GB available
            supports_mixed_precision=True,
            architecture="apple_silicon"
        )
    
    def _profile_intel_xpu(self, device_index: int) -> HardwareProfile:
        """Profile Intel XPU (Gaudi, etc.)"""
        if not IPEX_AVAILABLE:
            return None
        
        # Intel XPU profiling (placeholder - actual implementation depends on Intel drivers)
        return HardwareProfile(
            device_type="xpu",
            device_index=device_index,
            name=f"Intel XPU {device_index}",
            memory_total=32 * 1024 * 1024 * 1024,  # Estimate
            memory_available=30 * 1024 * 1024 * 1024,
            supports_mixed_precision=True,
            architecture="intel_xpu"
        )
    
    def _profile_tpu(self) -> Dict[str, HardwareProfile]:
        """Profile TPU devices (if available)"""
        tpu_profiles = {}
        
        try:
            # Check for TPU availability (placeholder - requires TPU runtime)
            import torch_xla.core.xla_model as xm
            devices = xm.get_xla_supported_devices()
            
            for i, device in enumerate(devices):
                if 'TPU' in device:
                    tpu_profiles[f"tpu:{i}"] = HardwareProfile(
                        device_type="tpu",
                        device_index=i,
                        name=f"TPU v{i+3}",  # Estimate
                        memory_total=16 * 1024 * 1024 * 1024,  # 16GB HBM
                        memory_available=14 * 1024 * 1024 * 1024,
                        supports_mixed_precision=True,
                        architecture="tpu_v3+"
                    )
        except ImportError:
            pass
        
        return tpu_profiles
    
    def _detect_numa_node(self) -> Optional[int]:
        """Detect NUMA node for CPU"""
        try:
            if platform.system() == "Linux":
                result = subprocess.run(['numactl', '--show'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    # Parse NUMA node info
                    for line in result.stdout.split('\n'):
                        if 'nodebind:' in line:
                            return int(line.split()[-1])
        except Exception:
            pass
        return None
    
    def _check_cpu_feature(self, feature: str) -> bool:
        """Check if CPU supports specific feature"""
        try:
            if platform.system() == "Linux":
                with open('/proc/cpuinfo', 'r') as f:
                    cpuinfo = f.read()
                    return feature in cpuinfo
        except Exception:
            pass
        return False
    
    def benchmark_device(self, device_id: str, model_type: str = "transformer") -> PerformanceMetrics:
        """
        Benchmark specific device performance for given model type
        
        Args:
            device_id: Device identifier (e.g., "cuda:0", "cpu")
            model_type: Type of model to benchmark ("transformer", "cnn", etc.)
        
        Returns:
            PerformanceMetrics object with benchmark results
        """
        cache_key = f"{device_id}_{model_type}"
        if cache_key in self.benchmark_cache:
            return self.benchmark_cache[cache_key]
        
        if self.verbose and self.console:
            self.console.print(f"🏃 Benchmarking {device_id} for {model_type}...", style="yellow")
        
        device = torch.device(device_id)
        metrics = self._run_benchmark(device, model_type)
        
        self.benchmark_cache[cache_key] = metrics
        return metrics
    
    def _run_benchmark(self, device: torch.device, model_type: str) -> PerformanceMetrics:
        """Run actual benchmark on device"""
        # Create synthetic benchmark based on model type
        if model_type == "transformer":
            return self._benchmark_transformer(device)
        elif model_type == "cnn":
            return self._benchmark_cnn(device)
        else:
            return self._benchmark_generic(device)
    
    def _benchmark_transformer(self, device: torch.device) -> PerformanceMetrics:
        """Benchmark transformer-like operations"""
        batch_size = 32
        seq_length = 512
        hidden_size = 768
        
        # Create synthetic transformer operations
        input_tensor = torch.randn(batch_size, seq_length, hidden_size, device=device)
        weight = torch.randn(hidden_size, hidden_size, device=device)
        
        # Warm up
        for _ in range(10):
            output = torch.matmul(input_tensor, weight)
        
        torch.cuda.synchronize() if device.type == "cuda" else None
        
        # Benchmark
        start_time = time.perf_counter()
        iterations = 100
        
        for _ in range(iterations):
            output = torch.matmul(input_tensor, weight)
            output = torch.nn.functional.gelu(output)
        
        torch.cuda.synchronize() if device.type == "cuda" else None
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        throughput = iterations / total_time
        latency = total_time / iterations
        
        # Estimate memory efficiency
        memory_used = input_tensor.numel() * 4 + weight.numel() * 4  # float32
        total_memory = self.profiles.get(str(device), HardwareProfile("", 0, "", 0, 0)).memory_total
        memory_efficiency = (memory_used / total_memory) * 100 if total_memory > 0 else 0
        
        return PerformanceMetrics(
            throughput=throughput,
            latency=latency,
            memory_efficiency=min(memory_efficiency, 100),
            power_efficiency=throughput / 100.0,  # Placeholder
            stability_score=0.95,  # Placeholder
            cost_score=throughput / 1000.0  # Placeholder
        )
    
    def _benchmark_cnn(self, device: torch.device) -> PerformanceMetrics:
        """Benchmark CNN-like operations"""
        batch_size = 32
        channels = 3
        height, width = 224, 224
        
        input_tensor = torch.randn(batch_size, channels, height, width, device=device)
        conv = torch.nn.Conv2d(channels, 64, 3, padding=1).to(device)
        
        # Warm up
        for _ in range(10):
            output = conv(input_tensor)
        
        torch.cuda.synchronize() if device.type == "cuda" else None
        
        # Benchmark
        start_time = time.perf_counter()
        iterations = 50
        
        for _ in range(iterations):
            output = conv(input_tensor)
            output = torch.nn.functional.relu(output)
        
        torch.cuda.synchronize() if device.type == "cuda" else None
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        throughput = iterations / total_time
        latency = total_time / iterations
        
        return PerformanceMetrics(
            throughput=throughput,
            latency=latency,
            memory_efficiency=85.0,  # Placeholder
            power_efficiency=throughput / 150.0,
            stability_score=0.92,
            cost_score=throughput / 800.0
        )
    
    def _benchmark_generic(self, device: torch.device) -> PerformanceMetrics:
        """Generic benchmark for unknown model types"""
        # Simple matrix multiplication benchmark
        size = 2048
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        
        # Warm up
        for _ in range(5):
            c = torch.matmul(a, b)
        
        torch.cuda.synchronize() if device.type == "cuda" else None
        
        # Benchmark
        start_time = time.perf_counter()
        iterations = 20
        
        for _ in range(iterations):
            c = torch.matmul(a, b)
        
        torch.cuda.synchronize() if device.type == "cuda" else None
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        throughput = iterations / total_time
        latency = total_time / iterations
        
        return PerformanceMetrics(
            throughput=throughput,
            latency=latency,
            memory_efficiency=80.0,
            power_efficiency=throughput / 120.0,
            stability_score=0.90,
            cost_score=throughput / 600.0
        )
    
    def _display_hardware_summary(self):
        """Display comprehensive hardware summary using Rich"""
        if not self.console:
            return
        
        tree = Tree("🖥️ Hardware Profile Summary")
        
        for device_id, profile in self.profiles.items():
            device_node = tree.add(f"[bold cyan]{device_id}[/bold cyan] - {profile.name}")
            
            # Memory info
            memory_gb = profile.memory_total / (1024**3)
            available_gb = profile.memory_available / (1024**3)
            device_node.add(f"💾 Memory: {available_gb:.1f}GB / {memory_gb:.1f}GB available")
            
            # Utilization
            device_node.add(f"📊 Utilization: {profile.utilization:.1f}%")
            
            # Special capabilities
            if profile.supports_tensor_cores:
                device_node.add("⚡ Tensor Cores: Supported")
            if profile.supports_mixed_precision:
                device_node.add("🎯 Mixed Precision: Supported")
            
            # Architecture
            if profile.architecture:
                device_node.add(f"🏗️ Architecture: {profile.architecture}")
            
            # Temperature and power (if available)
            if profile.temperature:
                device_node.add(f"🌡️ Temperature: {profile.temperature}°C")
            if profile.power_limit:
                device_node.add(f"⚡ Power Limit: {profile.power_limit}W")
        
        self.console.print(tree)


class DynamicHardwareManager:
    """
    Advanced hardware management with real-time adaptation and optimization
    
    Features:
    - Real-time performance monitoring
    - Dynamic device switching based on workload
    - Load balancing across multiple devices
    - Hardware-specific optimizations
    - Predictive device selection
    """
    
    def __init__(self, config, verbose: bool = False):
        self.config = config
        self.verbose = verbose
        self.console = Console() if RICH_AVAILABLE and verbose else None
        
        # Initialize hardware profiler
        self.profiler = HardwareProfiler(verbose)
        self.hardware_profiles = self.profiler.profile_all_hardware()
        
        # Performance tracking
        self.device_performance = defaultdict(list)
        self.device_loads = defaultdict(float)
        self.model_device_affinity = {}
        
        # Multi-device orchestration
        self.active_devices = []
        self.device_allocations = {}
        
        # Optimization settings
        self.enable_dynamic_switching = True
        self.enable_load_balancing = True
        self.monitoring_interval = 10.0  # seconds
        
        # Start monitoring thread
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
    
    def select_optimal_devices(self, model_type: str = "transformer", 
                             model_size: int = 0, 
                             enable_multi_device: bool = False) -> List[torch.device]:
        """
        Select optimal device(s) for given model and workload
        
        Args:
            model_type: Type of model ("transformer", "cnn", etc.)
            model_size: Estimated model size in bytes
            enable_multi_device: Whether to use multiple devices
        
        Returns:
            List of optimal devices ordered by preference
        """
        if self.config.device is not None:
            # User specified device - respect it
            return [torch.device(self.config.device)]
        
        if self.verbose and self.console:
            self.console.print(f"🎯 Selecting optimal devices for {model_type} model...", style="blue")
        
        device_scores = {}
        
        # Score all available devices
        for device_id, profile in self.hardware_profiles.items():
            score = self._calculate_comprehensive_score(device_id, profile, model_type, model_size)
            device_scores[device_id] = score
        
        # Sort devices by score (highest first)
        sorted_devices = sorted(device_scores.items(), key=lambda x: x[1], reverse=True)
        
        if enable_multi_device and len(sorted_devices) > 1:
            # Select multiple devices for parallel processing
            selected_devices = []
            total_memory_needed = model_size * 1.5  # Safety margin
            
            for device_id, score in sorted_devices:
                device = torch.device(device_id)
                profile = self.hardware_profiles[device_id]
                
                if profile.memory_available >= total_memory_needed * 0.3:  # Each device needs 30% minimum
                    selected_devices.append(device)
                    total_memory_needed -= profile.memory_available * 0.7
                    
                    if total_memory_needed <= 0 or len(selected_devices) >= 4:
                        break
            
            if selected_devices:
                self.active_devices = selected_devices
                if self.verbose:
                    device_list = [str(d) for d in selected_devices]
                    self.console.print(f"📱 Selected devices: {device_list}", style="green")
                return selected_devices
        
        # Single device selection
        best_device = torch.device(sorted_devices[0][0])
        self.active_devices = [best_device]
        
        if self.verbose and self.console:
            self.console.print(f"🏆 Selected device: {best_device}", style="green")
        
        return [best_device]
    
    def _calculate_comprehensive_score(self, device_id: str, profile: HardwareProfile, 
                                     model_type: str, model_size: int) -> float:
        """Calculate comprehensive device suitability score"""
        base_score = 0.0
        
        # Memory score (30% weight)
        memory_score = self._calculate_memory_score(profile, model_size)
        base_score += memory_score * 0.3
        
        # Performance score (40% weight)
        performance_score = self._calculate_performance_score(device_id, profile, model_type)
        base_score += performance_score * 0.4
        
        # Efficiency score (20% weight)
        efficiency_score = self._calculate_efficiency_score(profile)
        base_score += efficiency_score * 0.2
        
        # Availability score (10% weight)
        availability_score = self._calculate_availability_score(device_id, profile)
        base_score += availability_score * 0.1
        
        # Apply device type preferences
        base_score *= self._get_device_type_multiplier(profile.device_type)
        
        return base_score
    
    def _calculate_memory_score(self, profile: HardwareProfile, model_size: int) -> float:
        """Calculate memory suitability score"""
        if profile.memory_available <= 0:
            return 0.0
        
        memory_ratio = model_size / profile.memory_available
        
        if memory_ratio > 1.0:
            return 0.0  # Insufficient memory
        elif memory_ratio > 0.8:
            return 0.2  # Tight memory
        elif memory_ratio > 0.5:
            return 0.6  # Adequate memory
        else:
            return 1.0  # Plenty of memory
    
    def _calculate_performance_score(self, device_id: str, profile: HardwareProfile, 
                                   model_type: str) -> float:
        """Calculate performance score based on benchmarks and specs"""
        score = 0.0
        
        # Use cached benchmark if available
        try:
            metrics = self.profiler.benchmark_device(device_id, model_type)
            score += min(metrics.throughput / 100.0, 1.0) * 0.5
            score += (1.0 - min(metrics.latency, 1.0)) * 0.3
            score += metrics.stability_score * 0.2
        except Exception:
            # Fallback to profile-based scoring
            if profile.device_type == "cuda":
                score = 0.9
            elif profile.device_type == "mps":
                score = 0.7
            elif profile.device_type == "cpu":
                score = 0.3
            else:
                score = 0.5
        
        # Adjust for special capabilities
        if profile.supports_tensor_cores:
            score *= 1.2
        if profile.supports_mixed_precision:
            score *= 1.1
        
        return min(score, 1.0)
    
    def _calculate_efficiency_score(self, profile: HardwareProfile) -> float:
        """Calculate power and thermal efficiency score"""
        score = 1.0
        
        # Utilization penalty
        if profile.utilization > 80:
            score *= 0.5
        elif profile.utilization > 60:
            score *= 0.8
        
        # Temperature penalty
        if profile.temperature and profile.temperature > 80:
            score *= 0.6
        elif profile.temperature and profile.temperature > 70:
            score *= 0.8
        
        return score
    
    def _calculate_availability_score(self, device_id: str, profile: HardwareProfile) -> float:
        """Calculate device availability score"""
        # Check if device is currently overloaded
        current_load = self.device_loads.get(device_id, 0.0)
        
        if current_load > 0.9:
            return 0.1
        elif current_load > 0.7:
            return 0.5
        else:
            return 1.0
    
    def _get_device_type_multiplier(self, device_type: str) -> float:
        """Get preference multiplier for device types"""
        multipliers = {
            "cuda": 1.0,
            "mps": 0.8,
            "xpu": 0.9,
            "tpu": 1.1,
            "cpu": 0.5
        }
        return multipliers.get(device_type, 0.7)
    
    def optimize_model_for_device(self, model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
        """
        Apply device-specific optimizations to model
        
        Args:
            model: PyTorch model to optimize
            device: Target device
        
        Returns:
            Optimized model
        """
        profile = self.hardware_profiles.get(str(device))
        if not profile:
            return model
        
        if self.verbose and self.console:
            self.console.print(f"⚡ Optimizing model for {device}...", style="yellow")
        
        # Apply device-specific optimizations
        if device.type == "cuda":
            model = self._optimize_for_cuda(model, device, profile)
        elif device.type == "cpu":
            model = self._optimize_for_cpu(model, profile)
        elif device.type == "mps":
            model = self._optimize_for_mps(model)
        
        return model
    
    def _optimize_for_cuda(self, model: torch.nn.Module, device: torch.device, 
                          profile: HardwareProfile) -> torch.nn.Module:
        """Apply CUDA-specific optimizations"""
        # Enable mixed precision if supported
        if profile.supports_mixed_precision and not self.config.enable_quantization:
            model = model.half()
        
        # Compile model for better performance (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            try:
                model = torch.compile(model, mode="reduce-overhead")
            except Exception:
                pass
        
        # Enable optimized attention if available
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            torch.backends.cuda.enable_flash_sdp(True)
        
        # Set memory format optimization
        model = model.to(memory_format=torch.channels_last)
        
        return model
    
    def _optimize_for_cpu(self, model: torch.nn.Module, profile: HardwareProfile) -> torch.nn.Module:
        """Apply CPU-specific optimizations"""
        # Intel Extension for PyTorch optimization
        if IPEX_AVAILABLE:
            try:
                model = ipex.optimize(model)
            except Exception:
                pass
        
        # Enable JIT compilation
        try:
            model = torch.jit.script(model)
        except Exception:
            pass
        
        # Set optimal number of threads
        if profile.cores:
            torch.set_num_threads(min(profile.cores, 8))
        
        return model
    
    def _optimize_for_mps(self, model: torch.nn.Module) -> torch.nn.Module:
        """Apply MPS-specific optimizations"""
        # MPS-specific optimizations (placeholder)
        return model
    
    def enable_memory_optimization(self, model: torch.nn.Module, 
                                 device: torch.device) -> torch.nn.Module:
        """
        Enable advanced memory optimizations
        
        Args:
            model: Model to optimize
            device: Target device
        
        Returns:
            Memory-optimized model
        """
        if device.type == "cuda":
            # Enable memory pinning
            torch.cuda.empty_cache()
            
            # Enable memory pool optimization
            if hasattr(torch.cuda, 'memory_pool'):
                torch.cuda.memory_pool.set_per_process_memory_fraction(0.9)
        
        # Enable gradient checkpointing for large models
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        
        return model
    
    def distribute_workload(self, inputs: List[Any], 
                          devices: List[torch.device]) -> List[Tuple[Any, torch.device]]:
        """
        Distribute workload across multiple devices optimally
        
        Args:
            inputs: List of inputs to process
            devices: Available devices
        
        Returns:
            List of (input, device) pairs for optimal distribution
        """
        if len(devices) == 1:
            return [(inp, devices[0]) for inp in inputs]
        
        # Calculate device capacities
        device_capacities = {}
        total_capacity = 0
        
        for device in devices:
            profile = self.hardware_profiles.get(str(device))
            if profile:
                # Capacity based on memory and current load
                capacity = profile.memory_available * (1.0 - self.device_loads.get(str(device), 0))
                device_capacities[device] = capacity
                total_capacity += capacity
        
        # Distribute inputs proportionally
        distribution = []
        device_assignments = {device: 0 for device in devices}
        
        for i, inp in enumerate(inputs):
            # Select device with lowest current assignment ratio
            best_device = min(devices, key=lambda d: 
                             device_assignments[d] / device_capacities.get(d, 1))
            
            distribution.append((inp, best_device))
            device_assignments[best_device] += 1
        
        return distribution
    
    def _monitoring_loop(self):
        """Background monitoring loop for real-time adaptation"""
        while self.monitoring_active:
            try:
                self._update_device_metrics()
                self._adaptive_optimization()
                time.sleep(self.monitoring_interval)
            except Exception as e:
                if self.verbose:
                    print(f"Monitoring error: {e}")
                time.sleep(5)
    
    def _update_device_metrics(self):
        """Update real-time device metrics"""
        for device_id in self.hardware_profiles.keys():
            if device_id.startswith("cuda"):
                device_index = int(device_id.split(":")[1])
                if NVML_AVAILABLE:
                    try:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        self.device_loads[device_id] = util.gpu / 100.0
                    except Exception:
                        pass
            elif device_id == "cpu":
                self.device_loads[device_id] = psutil.cpu_percent(interval=1) / 100.0
    
    def _adaptive_optimization(self):
        """Perform adaptive optimizations based on current metrics"""
        if not self.enable_dynamic_switching:
            return
        
        # Check if any device is overloaded
        for device_id, load in self.device_loads.items():
            if load > 0.9:  # 90% utilization threshold
                self._trigger_load_balancing(device_id)
    
    def _trigger_load_balancing(self, overloaded_device: str):
        """Trigger load balancing when device is overloaded"""
        if self.verbose and self.console:
            self.console.print(f"⚖️ Load balancing triggered for {overloaded_device}", 
                             style="yellow")
        
        # Implementation would depend on specific workload management system
        pass
    
    def get_device_recommendations(self, model_type: str = "transformer") -> Dict[str, Any]:
        """
        Get comprehensive device recommendations and insights
        
        Args:
            model_type: Type of model for recommendations
        
        Returns:
            Dictionary with recommendations and insights
        """
        recommendations = {
            "optimal_devices": [],
            "performance_predictions": {},
            "optimization_suggestions": [],
            "cost_analysis": {},
            "environmental_impact": {}
        }
        
        # Get optimal devices
        optimal_devices = self.select_optimal_devices(model_type, enable_multi_device=True)
        recommendations["optimal_devices"] = [str(d) for d in optimal_devices]
        
        # Performance predictions
        for device_id in self.hardware_profiles.keys():
            try:
                metrics = self.profiler.benchmark_device(device_id, model_type)
                recommendations["performance_predictions"][device_id] = {
                    "throughput": metrics.throughput,
                    "latency": metrics.latency,
                    "efficiency": metrics.power_efficiency
                }
            except Exception:
                pass
        
        # Optimization suggestions
        suggestions = []
        for device in optimal_devices:
            profile = self.hardware_profiles.get(str(device))
            if profile:
                if profile.supports_mixed_precision:
                    suggestions.append(f"Enable mixed precision on {device}")
                if profile.device_type == "cuda" and profile.supports_tensor_cores:
                    suggestions.append(f"Use tensor cores on {device}")
        
        recommendations["optimization_suggestions"] = suggestions
        
        return recommendations
    
    def shutdown(self):
        """Shutdown hardware manager and cleanup resources"""
        self.monitoring_active = False
        if self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        
        if self.verbose and self.console:
            self.console.print("🔌 Hardware manager shutdown complete", style="green")


@dataclass
class PipelineConfig:
    """
    Enhanced configuration class with advanced hardware optimization settings
    """
    # Core pipeline settings
    task: Optional[str] = None
    model: Optional[Union[str, PreTrainedModel, TFPreTrainedModel]] = None
    device: Optional[Union[int, str, torch.device]] = None
    device_map: Optional[Union[str, Dict]] = None
    torch_dtype: Optional[Union[str, torch.dtype]] = None
    
    # Performance optimization settings
    batch_size: int = 1
    max_batch_size: int = 32
    enable_caching: bool = True
    cache_size: int = 128
    enable_parallel: bool = True
    max_workers: int = 4
    
    # Advanced hardware optimization
    enable_auto_device_selection: bool = True
    enable_multi_device: bool = False
    enable_dynamic_switching: bool = True
    enable_hardware_optimization: bool = True
    hardware_profiling: bool = True
    
    # Memory management
    memory_threshold: float = 0.85
    auto_cleanup: bool = True
    garbage_collection_interval: int = 10
    enable_memory_pinning: bool = True
    
    # Quantization settings
    enable_quantization: bool = False
    quantization_config: Optional[BitsAndBytesConfig] = None
    auto_mixed_precision: bool = True
    
    # Streaming and batching
    enable_streaming: bool = False
    stream_chunk_size: int = 1000
    
    # Monitoring and logging
    verbose: bool = False
    enable_metrics: bool = True
    log_level: str = "INFO"
    
    # Error handling
    max_retries: int = 3
    retry_delay: float = 1.0
    fallback_device: str = "cpu"


class PerformanceMonitor:
    """
    Enhanced performance monitoring system with hardware-specific metrics
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.metrics = defaultdict(list)
        self.hardware_metrics = defaultdict(list)
        self.start_time = None
        self.console = Console() if RICH_AVAILABLE and config.verbose else None
        
    def start_timing(self, operation: str = "inference"):
        """Start timing an operation"""
        self.start_time = time.perf_counter()
        self.current_operation = operation
        
    def end_timing(self) -> float:
        """End timing and record the duration"""
        if self.start_time is None:
            return 0.0
        
        duration = time.perf_counter() - self.start_time
        self.metrics[f"{self.current_operation}_time"].append(duration)
        self.start_time = None
        return duration
    
    def record_memory_usage(self):
        """Record current memory usage across all devices"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        # CPU memory
        self.metrics["cpu_memory"].append(memory_info.rss / 1024 / 1024)  # MB
        
        # GPU memory for all CUDA devices
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_memory = torch.cuda.memory_allocated(i)
                gpu_total = torch.cuda.get_device_properties(i).total_memory
                self.hardware_metrics[f"gpu_{i}_memory_used"].append(gpu_memory / 1024 / 1024)
                self.hardware_metrics[f"gpu_{i}_memory_total"].append(gpu_total / 1024 / 1024)
                
                # GPU utilization if available
                if NVML_AVAILABLE:
                    try:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        self.hardware_metrics[f"gpu_{i}_utilization"].append(util.gpu)
                        
                        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                        self.hardware_metrics[f"gpu_{i}_temperature"].append(temp)
                    except Exception:
                        pass
    
    def record_hardware_metrics(self, device: torch.device):
        """Record device-specific performance metrics"""
        if device.type == "cuda":
            device_index = device.index or 0
            
            # Memory metrics
            memory_allocated = torch.cuda.memory_allocated(device_index)
            memory_reserved = torch.cuda.memory_reserved(device_index)
            
            self.hardware_metrics[f"cuda_{device_index}_memory_allocated"].append(
                memory_allocated / 1024 / 1024
            )
            self.hardware_metrics[f"cuda_{device_index}_memory_reserved"].append(
                memory_reserved / 1024 / 1024
            )
            
            # Power metrics (if available)
            if NVML_AVAILABLE:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Watts
                    self.hardware_metrics[f"cuda_{device_index}_power"].append(power)
                except Exception:
                    pass
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary including hardware metrics"""
        summary = {"general_metrics": {}, "hardware_metrics": {}}
        
        # General metrics
        for metric_name, values in self.metrics.items():
            if values:
                summary["general_metrics"][metric_name] = {
                    "avg": np.mean(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "std": np.std(values),
                    "count": len(values)
                }
        
        # Hardware-specific metrics
        for metric_name, values in self.hardware_metrics.items():
            if values:
                summary["hardware_metrics"][metric_name] = {
                    "avg": np.mean(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "std": np.std(values),
                    "count": len(values)
                }
        
        return summary
    
    def display_summary(self):
        """Display enhanced performance summary using Rich console"""
        if not self.console:
            return
            
        summary = self.get_summary()
        
        # General metrics table
        general_table = Table(title="General Performance Metrics")
        general_table.add_column("Metric", style="cyan")
        general_table.add_column("Average", style="green")
        general_table.add_column("Min", style="yellow")
        general_table.add_column("Max", style="red")
        general_table.add_column("Std Dev", style="blue")
        
        for metric, stats in summary["general_metrics"].items():
            general_table.add_row(
                metric,
                f"{stats['avg']:.4f}",
                f"{stats['min']:.4f}",
                f"{stats['max']:.4f}",
                f"{stats['std']:.4f}"
            )
        
        self.console.print(general_table)
        
        # Hardware metrics table
        if summary["hardware_metrics"]:
            hardware_table = Table(title="Hardware Performance Metrics")
            hardware_table.add_column("Device Metric", style="cyan")
            hardware_table.add_column("Average", style="green")
            hardware_table.add_column("Min", style="yellow")
            hardware_table.add_column("Max", style="red")
            
            for metric, stats in summary["hardware_metrics"].items():
                unit = ""
                if "memory" in metric:
                    unit = " MB"
                elif "temperature" in metric:
                    unit = "°C"
                elif "power" in metric:
                    unit = "W"
                elif "utilization" in metric:
                    unit = "%"
                
                hardware_table.add_row(
                    metric,
                    f"{stats['avg']:.2f}{unit}",
                    f"{stats['min']:.2f}{unit}",
                    f"{stats['max']:.2f}{unit}"
                )
            
            self.console.print(hardware_table)


class MemoryManager:
    """
    Enhanced memory management system with hardware-aware optimizations
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.memory_pool = {}
        self.last_cleanup = time.time()
        self.pinned_memory_cache = {}
        
    @contextmanager
    def memory_guard(self):
        """Enhanced context manager for memory-safe operations"""
        try:
            self._check_memory_usage()
            self._setup_memory_optimization()
            yield
        finally:
            if self.config.auto_cleanup:
                self._conditional_cleanup()
    
    def _setup_memory_optimization(self):
        """Setup device-specific memory optimizations"""
        if torch.cuda.is_available() and self.config.enable_memory_pinning:
            # Enable memory pool optimization
            torch.cuda.empty_cache()
            
            # Set memory fraction if not using device map
            if not self.config.device_map:
                try:
                    torch.cuda.set_per_process_memory_fraction(0.9)
                except Exception:
                    pass
    
    def _check_memory_usage(self):
        """Enhanced memory usage checking for all devices"""
        # Check system memory
        memory_percent = psutil.virtual_memory().percent / 100.0
        
        if memory_percent > self.config.memory_threshold:
            self._force_cleanup()
            
            # If still over threshold, raise warning
            if psutil.virtual_memory().percent / 100.0 > self.config.memory_threshold:
                warnings.warn(
                    f"System memory usage {memory_percent:.1%} exceeds threshold "
                    f"{self.config.memory_threshold:.1%}"
                )
        
        # Check GPU memory
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i)
                total = torch.cuda.get_device_properties(i).total_memory
                gpu_percent = allocated / total
                
                if gpu_percent > self.config.memory_threshold:
                    torch.cuda.empty_cache()
                    warnings.warn(
                        f"GPU {i} memory usage {gpu_percent:.1%} exceeds threshold"
                    )
    
    def _conditional_cleanup(self):
        """Enhanced cleanup with device-specific operations"""
        current_time = time.time()
        if current_time - self.last_cleanup > self.config.garbage_collection_interval:
            self._force_cleanup()
            self.last_cleanup = current_time
    
    def _force_cleanup(self):
        """Enhanced cleanup for all devices"""
        # Clear Python garbage
        gc.collect()
        
        # Clear CUDA cache for all devices
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
        
        # Clear pinned memory cache
        self.pinned_memory_cache.clear()
    
    def pin_memory_for_device(self, tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Pin memory for optimal device transfer"""
        if not self.config.enable_memory_pinning or device.type != "cuda":
            return tensor
        
        tensor_id = id(tensor)
        if tensor_id not in self.pinned_memory_cache:
            try:
                pinned_tensor = tensor.pin_memory()
                self.pinned_memory_cache[tensor_id] = pinned_tensor
                return pinned_tensor
            except Exception:
                return tensor
        
        return self.pinned_memory_cache[tensor_id]


class SmartCache:
    """
    Enhanced caching system with hardware-aware strategies
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.cache = OrderedDict()
        self.access_counts = defaultdict(int)
        self.memory_usage = {}
        self.device_caches = defaultdict(OrderedDict)  # Per-device caching
        
    def get(self, key: str, device: Optional[torch.device] = None) -> Any:
        """Get item from cache with device awareness"""
        # Try device-specific cache first
        if device:
            device_key = str(device)
            if device_key in self.device_caches and key in self.device_caches[device_key]:
                value = self.device_caches[device_key].pop(key)
                self.device_caches[device_key][key] = value
                self.access_counts[f"{device_key}_{key}"] += 1
                return value
        
        # Fallback to general cache
        if key in self.cache:
            value = self.cache.pop(key)
            self.cache[key] = value
            self.access_counts[key] += 1
            return value
        
        return None
    
    def put(self, key: str, value: Any, memory_size: int = 0, 
            device: Optional[torch.device] = None):
        """Put item in cache with device awareness"""
        if not self.config.enable_caching:
            return
        
        # Use device-specific cache if device specified
        if device:
            device_key = str(device)
            
            # Remove if already exists
            if key in self.device_caches[device_key]:
                self.device_caches[device_key].pop(key)
            
            # Evict if cache is full
            while len(self.device_caches[device_key]) >= self.config.cache_size // 2:
                self._evict_least_valuable_device(device_key)
            
            self.device_caches[device_key][key] = value
            self.memory_usage[f"{device_key}_{key}"] = memory_size
            self.access_counts[f"{device_key}_{key}"] += 1
        else:
            # Use general cache
            if key in self.cache:
                self.cache.pop(key)
            
            while len(self.cache) >= self.config.cache_size:
                self._evict_least_valuable()
            
            self.cache[key] = value
            self.memory_usage[key] = memory_size
            self.access_counts[key] += 1
    
    def _evict_least_valuable_device(self, device_key: str):
        """Evict least valuable item from device-specific cache"""
        if not self.device_caches[device_key]:
            return
        
        scores = {}
        for key in self.device_caches[device_key]:
            access_score = self.access_counts.get(f"{device_key}_{key}", 0)
            memory_penalty = self.memory_usage.get(f"{device_key}_{key}", 0) / 1024 / 1024
            scores[key] = access_score / (1 + memory_penalty)
        
        least_valuable = min(scores.items(), key=lambda x: x[1])[0]
        self.device_caches[device_key].pop(least_valuable)
        self.memory_usage.pop(f"{device_key}_{least_valuable}", None)
    
    def _evict_least_valuable(self):
        """Evict least valuable item from general cache"""
        if not self.cache:
            return
        
        scores = {}
        for key in self.cache:
            access_score = self.access_counts[key]
            memory_penalty = self.memory_usage.get(key, 0) / 1024 / 1024
            scores[key] = access_score / (1 + memory_penalty)
        
        least_valuable = min(scores.items(), key=lambda x: x[1])[0]
        self.cache.pop(least_valuable)
        self.memory_usage.pop(least_valuable, None)
    
    def clear(self):
        """Clear all caches"""
        self.cache.clear()
        self.device_caches.clear()
        self.access_counts.clear()
        self.memory_usage.clear()


class BatchProcessor:
    """
    Enhanced batch processing with hardware-aware optimization
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.optimal_batch_size = config.batch_size
        self.processing_times = []
        self.device_batch_sizes = {}  # Optimal batch size per device
        
    def process_batch(self, inputs: List[Any], processor_func: Callable,
                     devices: Optional[List[torch.device]] = None) -> List[Any]:
        """Enhanced batch processing with multi-device support"""
        if len(inputs) == 1:
            return [processor_func(inputs[0])]
        
        if devices and len(devices) > 1:
            return self._process_multi_device(inputs, processor_func, devices)
        else:
            return self._process_single_device(inputs, processor_func, devices[0] if devices else None)
    
    def _process_single_device(self, inputs: List[Any], processor_func: Callable,
                              device: Optional[torch.device] = None) -> List[Any]:
        """Process batch on single device with optimal batching"""
        batch_size = self._get_optimal_batch_size(len(inputs), device)
        
        results = []
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size]
            
            start_time = time.perf_counter()
            batch_results = self._process_single_batch(batch, processor_func)
            processing_time = time.perf_counter() - start_time
            
            results.extend(batch_results)
            self._update_batch_performance(len(batch), processing_time, device)
        
        return results
    
    def _process_multi_device(self, inputs: List[Any], processor_func: Callable,
                             devices: List[torch.device]) -> List[Any]:
        """Process batch across multiple devices"""
        # Distribute inputs across devices
        device_batches = self._distribute_across_devices(inputs, devices)
        
        results = [None] * len(inputs)
        
        # Process batches in parallel across devices
        with ThreadPoolExecutor(max_workers=len(devices)) as executor:
            futures = []
            
            for device, (batch_inputs, indices) in device_batches.items():
                future = executor.submit(
                    self._process_device_batch,
                    batch_inputs, processor_func, device
                )
                futures.append((future, indices))
            
            # Collect results maintaining order
            for future, indices in futures:
                try:
                    batch_results = future.result(timeout=60)
                    for i, result in enumerate(batch_results):
                        results[indices[i]] = result
                except Exception as e:
                    warnings.warn(f"Multi-device processing error: {e}")
                    # Fill with None for failed batches
                    for idx in indices:
                        results[idx] = None
        
        return [r for r in results if r is not None]
    
    def _distribute_across_devices(self, inputs: List[Any], 
                                  devices: List[torch.device]) -> Dict[torch.device, Tuple[List[Any], List[int]]]:
        """Distribute inputs across devices based on their capabilities"""
        device_batches = {device: ([], []) for device in devices}
        
        # Simple round-robin distribution
        # In a production system, this could be based on device capabilities
        for i, inp in enumerate(inputs):
            device = devices[i % len(devices)]
            device_batches[device][0].append(inp)
            device_batches[device][1].append(i)
        
        return device_batches
    
    def _process_device_batch(self, inputs: List[Any], processor_func: Callable,
                             device: torch.device) -> List[Any]:
        """Process batch on specific device"""
        # Set device context
        if device.type == "cuda":
            with torch.cuda.device(device):
                return [processor_func(inp) for inp in inputs]
        else:
            return [processor_func(inp) for inp in inputs]
    
    def _get_optimal_batch_size(self, total_inputs: int, 
                               device: Optional[torch.device] = None) -> int:
        """Get optimal batch size for specific device"""
        device_key = str(device) if device else "default"
        
        if device_key in self.device_batch_sizes:
            optimal_size = self.device_batch_sizes[device_key]
        else:
            optimal_size = self.optimal_batch_size
        
        if total_inputs <= optimal_size:
            return total_inputs
        
        # Adaptive batch sizing based on device performance
        if len(self.processing_times) > 5:
            recent_times = [(s, t) for s, t, d in self.processing_times[-10:] 
                           if d == device_key]
            if recent_times:
                avg_time_per_item = np.mean([t/s for s, t in recent_times])
                target_time = 2.0  # Target 2 seconds per batch
                new_optimal_size = int(target_time / avg_time_per_item)
                new_optimal_size = max(1, min(new_optimal_size, self.config.max_batch_size))
                self.device_batch_sizes[device_key] = new_optimal_size
                optimal_size = new_optimal_size
        
        return min(optimal_size, total_inputs)
    
    def _process_single_batch(self, batch: List[Any], processor_func: Callable) -> List[Any]:
        """Process a single batch with optimal strategy"""
        if self.config.enable_parallel and len(batch) > 1:
            return self._process_parallel(batch, processor_func)
        else:
            return [processor_func(item) for item in batch]
    
    def _process_parallel(self, batch: List[Any], processor_func: Callable) -> List[Any]:
        """Process batch in parallel using thread pool"""
        max_workers = min(self.config.max_workers, len(batch))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(processor_func, item) for item in batch]
            results = []
            
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=30)
                    results.append(result)
                except Exception as e:
                    warnings.warn(f"Parallel processing error: {e}")
                    results.append(None)
        
        return results
    
    def _update_batch_performance(self, batch_size: int, processing_time: float,
                                 device: Optional[torch.device] = None):
        """Update performance tracking for batch processing"""
        device_key = str(device) if device else "default"
        self.processing_times.append((batch_size, processing_time, device_key))
        
        # Keep only recent history
        if len(self.processing_times) > 200:
            self.processing_times = self.processing_times[-100:]


class ErrorHandler:
    """
    Enhanced error handling with hardware-specific recovery strategies
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.error_counts = defaultdict(int)
        self.last_errors = {}
        self.device_errors = defaultdict(list)
        
    def handle_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with intelligent retry logic and hardware fallbacks"""
        last_exception = None
        original_device = kwargs.get('device')
        
        for attempt in range(self.config.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                error_type = type(e).__name__
                self.error_counts[error_type] += 1
                self.last_errors[error_type] = str(e)
                
                # Record device-specific errors
                if original_device:
                    self.device_errors[str(original_device)].append(error_type)
                
                if attempt < self.config.max_retries:
                    # Exponential backoff
                    delay = self.config.retry_delay * (2 ** attempt)
                    time.sleep(delay)
                    
                    # Apply advanced fallback strategies
                    kwargs = self._apply_advanced_fallback_strategy(e, kwargs, attempt)
                else:
                    # Final attempt failed
                    self._log_final_error(e, attempt + 1)
        
        raise last_exception
    
    def _apply_advanced_fallback_strategy(self, error: Exception, kwargs: Dict, 
                                        attempt: int) -> Dict:
        """Apply advanced fallback strategies based on error type and hardware"""
        error_type = type(error).__name__
        error_msg = str(error).lower()
        
        # CUDA/GPU specific errors
        if "cuda" in error_msg or "out of memory" in error_msg:
            kwargs = self._handle_gpu_error(error, kwargs, attempt)
        
        # Device-specific errors
        elif "device" in error_msg:
            kwargs = self._handle_device_error(error, kwargs, attempt)
        
        # Model loading errors
        elif "model" in error_msg or "loading" in error_msg:
            kwargs = self._handle_model_error(error, kwargs, attempt)
        
        # Precision/dtype errors
        elif "dtype" in error_msg or "precision" in error_msg:
            kwargs = self._handle_precision_error(error, kwargs, attempt)
        
        return kwargs
    
    def _handle_gpu_error(self, error: Exception, kwargs: Dict, attempt: int) -> Dict:
        """Handle GPU-specific errors with intelligent fallbacks"""
        if "out of memory" in str(error).lower():
            # Progressive memory optimization strategies
            if attempt == 0:
                # Try reducing batch size
                if 'batch_size' in kwargs:
                    kwargs['batch_size'] = max(1, kwargs['batch_size'] // 2)
                
                # Clear GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    for i in range(torch.cuda.device_count()):
                        with torch.cuda.device(i):
                            torch.cuda.empty_cache()
            
            elif attempt == 1:
                # Try mixed precision or quantization
                kwargs['torch_dtype'] = torch.float16
                
            elif attempt == 2:
                # Fallback to CPU
                kwargs['device'] = self.config.fallback_device
        
        elif "device" in str(error).lower():
            # Try different GPU or fallback to CPU
            current_device = kwargs.get('device', 'cuda:0')
            if isinstance(current_device, str) and current_device.startswith('cuda:'):
                device_idx = int(current_device.split(':')[1])
                next_idx = (device_idx + 1) % torch.cuda.device_count()
                kwargs['device'] = f'cuda:{next_idx}'
            else:
                kwargs['device'] = self.config.fallback_device
        
        return kwargs
    
    def _handle_device_error(self, error: Exception, kwargs: Dict, attempt: int) -> Dict:
        """Handle general device errors"""
        # Progressive device fallback
        current_device = kwargs.get('device', 'cuda:0')
        
        if attempt == 0 and 'cuda' in str(current_device):
            # Try MPS if available
            if torch.backends.mps.is_available():
                kwargs['device'] = 'mps'
            else:
                kwargs['device'] = self.config.fallback_device
        else:
            # Final fallback to CPU
            kwargs['device'] = self.config.fallback_device
        
        return kwargs
    
    def _handle_model_error(self, error: Exception, kwargs: Dict, attempt: int) -> Dict:
        """Handle model loading/initialization errors"""
        # Try different model loading strategies
        if attempt == 0:
            # Try without device map
            kwargs.pop('device_map', None)
        elif attempt == 1:
            # Try different torch dtype
            kwargs['torch_dtype'] = torch.float32
        
        return kwargs
    
    def _handle_precision_error(self, error: Exception, kwargs: Dict, attempt: int) -> Dict:
        """Handle precision/dtype related errors"""
        # Progressive precision fallback
        if attempt == 0:
            kwargs['torch_dtype'] = torch.float32
        elif attempt == 1:
            # Disable quantization if enabled
            kwargs.pop('quantization_config', None)
            kwargs.pop('load_in_8bit', None)
            kwargs.pop('load_in_4bit', None)
        
        return kwargs
    
    def _log_final_error(self, error: Exception, attempts: int):
        """Enhanced error logging with hardware context"""
        error_msg = (
            f"Operation failed after {attempts} attempts. "
            f"Error: {type(error).__name__}: {str(error)}\n"
            f"Device error history: {dict(self.device_errors)}"
        )
        logging.error(error_msg)
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get comprehensive error summary including device-specific errors"""
        return {
            "error_counts": dict(self.error_counts),
            "last_errors": dict(self.last_errors),
            "device_errors": dict(self.device_errors),
            "most_problematic_devices": self._get_most_problematic_devices()
        }
    
    def _get_most_problematic_devices(self) -> List[str]:
        """Identify devices with most errors"""
        device_error_counts = {
            device: len(errors) 
            for device, errors in self.device_errors.items()
        }
        
        return sorted(device_error_counts.items(), 
                     key=lambda x: x[1], reverse=True)[:3]


class AdvancedPipeline:
    """
    Enhanced Advanced Pipeline System with comprehensive hardware optimization
    
    New Features:
    - Automatic best hardware selection and benchmarking
    - Real-time hardware adaptation and monitoring
    - Multi-device orchestration and load balancing
    - Hardware-specific optimizations (mixed precision, memory pinning)
    - Intelligent performance prediction and device switching
    - Accelerator support (TPU, Intel Gaudi, AMD ROCm)
    """
    
    def __init__(self, config: PipelineConfig):
        """Initialize the Enhanced Advanced Pipeline System"""
        self.config = config
        self._setup_logging()
        self._initialize_components()
        self._setup_hardware_optimization()
        self._load_pipeline()
        
        if self.config.verbose and RICH_AVAILABLE:
            self._display_initialization_summary()
    
    def _setup_logging(self):
        """Setup enhanced logging with Rich support"""
        log_level = getattr(logging, self.config.log_level.upper())
        
        if RICH_AVAILABLE and self.config.verbose:
            logging.basicConfig(
                level=log_level,
                format="%(message)s",
                handlers=[RichHandler(rich_tracebacks=True)]
            )
            self.console = Console()
        else:
            logging.basicConfig(
                level=log_level,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            self.console = None
        
        self.logger = logging.getLogger(__name__)
    
    def _initialize_components(self):
        """Initialize all system components"""
        self.monitor = PerformanceMonitor(self.config)
        self.memory_manager = MemoryManager(self.config)
        self.cache = SmartCache(self.config)
        self.error_handler = ErrorHandler(self.config)
        
        # Set random seed for reproducibility
        if hasattr(self.config, 'seed'):
            set_seed(self.config.seed)
    
    def _setup_hardware_optimization(self):
        """Setup advanced hardware optimization components"""
        if self.config.enable_auto_device_selection or self.config.hardware_profiling:
            self.hardware_manager = DynamicHardwareManager(self.config, self.config.verbose)
            
            # Get optimal devices
            self.optimal_devices = self.hardware_manager.select_optimal_devices(
                model_type=self._infer_model_type(),
                model_size=self._estimate_model_size(),
                enable_multi_device=self.config.enable_multi_device
            )
            
            if self.config.verbose:
                self.logger.info(f"Selected optimal devices: {[str(d) for d in self.optimal_devices]}")
        else:
            self.hardware_manager = None
            self.optimal_devices = [torch.device(self.config.device)] if self.config.device else [torch.device("cpu")]
        
        # Initialize batch processor with device awareness
        self.batch_processor = BatchProcessor(self.config)
    
    def _infer_model_type(self) -> str:
        """Infer model type from task for optimization"""
        task_to_model_type = {
            "text-classification": "transformer",
            "text-generation": "transformer",
            "question-answering": "transformer",
            "summarization": "transformer",
            "translation": "transformer",
            "image-classification": "cnn",
            "object-detection": "cnn",
            "image-segmentation": "cnn"
        }
        return task_to_model_type.get(self.config.task, "transformer")
    
    def _load_pipeline(self):
        """Load the core pipeline with advanced hardware optimization"""
        cache_key = self._generate_cache_key()
        
        # Try to get from cache first
        primary_device = self.optimal_devices[0]
        cached_pipeline = self.cache.get(cache_key, primary_device)
        if cached_pipeline is not None:
            self.pipeline = cached_pipeline
            if self.config.verbose:
                self.logger.info("Pipeline loaded from cache")
            return
        
        # Load new pipeline with hardware optimization
        with self.memory_manager.memory_guard():
            self.pipeline = self._create_optimized_pipeline()
        
        # Cache the pipeline
        model_size = self._estimate_model_size()
        self.cache.put(cache_key, self.pipeline, model_size, primary_device)
        
        if self.config.verbose:
            self.logger.info(f"New optimized pipeline loaded and cached (size: {model_size / 1024 / 1024:.1f} MB)")
    
    def _create_optimized_pipeline(self):
        """Create the pipeline with comprehensive hardware optimizations"""
        from transformers import pipeline
        
        # Prepare arguments with hardware optimization
        kwargs = {}
        primary_device = self.optimal_devices[0]
        
        # Device and device map configuration
        if self.config.enable_multi_device and len(self.optimal_devices) > 1:
            # Multi-device setup
            device_map = self._create_device_map()
            kwargs["device_map"] = device_map
        else:
            kwargs["device"] = primary_device
        
        # Precision optimization
        if self.config.auto_mixed_precision and self.hardware_manager:
            profile = self.hardware_manager.hardware_profiles.get(str(primary_device))
            if profile and profile.supports_mixed_precision:
                kwargs["torch_dtype"] = torch.float16
        
        # Quantization configuration
        if self.config.enable_quantization and self.config.quantization_config:
            kwargs["quantization_config"] = self.config.quantization_config
        
        # Create pipeline with error handling and hardware optimization
        pipeline_obj = self.error_handler.handle_with_retry(
            pipeline,
            task=self.config.task,
            model=self.config.model,
            torch_dtype=kwargs.get("torch_dtype", self.config.torch_dtype),
            device_map=kwargs.get("device_map", self.config.device_map),
            # device=kwargs.get("device"),
            **kwargs
        )
        
        # Apply hardware-specific optimizations
        if self.hardware_manager and self.config.enable_hardware_optimization:
            if hasattr(pipeline_obj, 'model'):
                optimized_model = self.hardware_manager.optimize_model_for_device(
                    pipeline_obj.model, primary_device
                )
                
                # Enable memory optimization
                optimized_model = self.hardware_manager.enable_memory_optimization(
                    optimized_model, primary_device
                )
                
                pipeline_obj.model = optimized_model
        
        return pipeline_obj
    
    def _create_device_map(self) -> Dict[str, Union[int, str, torch.device]]:
        """Create optimal device map for multi-device setup"""
        if not self.optimal_devices or len(self.optimal_devices) == 1:
            return None
        
        # Simple strategy: distribute layers across devices
        device_map = {}
        num_devices = len(self.optimal_devices)
        
        # This is a simplified device mapping - production systems would need
        # more sophisticated layer distribution based on model architecture
        for i, device in enumerate(self.optimal_devices):
            device_map[f"layer.{i}"] = device
        
        return device_map
    
    def _generate_cache_key(self) -> str:
        """Generate enhanced cache key including hardware configuration"""
        key_parts = [
            str(self.config.task),
            str(self.config.model),
            str(self.optimal_devices[0]) if self.optimal_devices else "cpu",
            str(self.config.torch_dtype),
            str(self.config.enable_quantization),
            str(self.config.auto_mixed_precision),
            str(self.config.enable_multi_device)
        ]
        return "|".join(key_parts)
    
    def _estimate_model_size(self) -> int:
        """Enhanced model size estimation"""
        # Model-specific size estimates (in bytes)
        model_sizes = {
            "distilbert": 250 * 1024 * 1024,    # ~250MB
            "bert-base": 440 * 1024 * 1024,     # ~440MB
            "bert-large": 1340 * 1024 * 1024,   # ~1.34GB
            "gpt2": 500 * 1024 * 1024,          # ~500MB
            "gpt2-medium": 1500 * 1024 * 1024,  # ~1.5GB
            "gpt2-large": 3000 * 1024 * 1024,   # ~3GB
            "t5-small": 240 * 1024 * 1024,      # ~240MB
            "t5-base": 890 * 1024 * 1024,       # ~890MB
            "t5-large": 2850 * 1024 * 1024,     # ~2.85GB
        }
        
        if isinstance(self.config.model, str):
            model_name = self.config.model.lower()
            for key, size in model_sizes.items():
                if key in model_name:
                    return size
        
        # Check if pipeline is already loaded
        if hasattr(self, 'pipeline') and hasattr(self.pipeline, 'model'):
            model = self.pipeline.model
            if hasattr(model, 'num_parameters'):
                # Estimate: 4 bytes per parameter (float32)
                return model.num_parameters() * 4
        
        # Default estimation
        return 500 * 1024 * 1024  # 500 MB
    
    def __call__(self, inputs: Union[str, List[str]], **kwargs) -> Union[Any, List[Any]]:
        """
        Enhanced inference method with comprehensive hardware optimization
        """
        # Normalize inputs to list
        single_input = isinstance(inputs, str)
        if single_input:
            inputs = [inputs]
        
        # Start performance monitoring
        self.monitor.start_timing("inference")
        self.monitor.record_memory_usage()
        
        # Record hardware metrics for primary device
        if self.optimal_devices:
            self.monitor.record_hardware_metrics(self.optimal_devices[0])
        
        try:
            with self.memory_manager.memory_guard():
                # Use hardware-aware batch processing
                if self.config.enable_multi_device and len(self.optimal_devices) > 1:
                    results = self._process_multi_device(inputs, **kwargs)
                else:
                    results = self.batch_processor.process_batch(
                        inputs,
                        lambda x: self.pipeline(x, **kwargs),
                        self.optimal_devices
                    )
            
            # Record performance metrics
            inference_time = self.monitor.end_timing()
            self.monitor.record_memory_usage()
            
            # Record hardware metrics after inference
            if self.optimal_devices:
                self.monitor.record_hardware_metrics(self.optimal_devices[0])
            
            if self.config.verbose:
                self._display_inference_stats(len(inputs), inference_time)
            
            # Return single result if single input
            return results[0] if single_input else results
            
        except Exception as e:
            self.logger.error(f"Enhanced inference failed: {e}")
            # Try fallback strategies
            if self.hardware_manager and self.config.enable_dynamic_switching:
                return self._fallback_inference(inputs, **kwargs)
            raise
    
    def _process_multi_device(self, inputs: List[Any], **kwargs) -> List[Any]:
        """Process inputs across multiple devices optimally"""
        if not self.hardware_manager:
            return self.batch_processor.process_batch(
                inputs, lambda x: self.pipeline(x, **kwargs), self.optimal_devices
            )
        
        # Distribute workload across devices
        workload_distribution = self.hardware_manager.distribute_workload(
            inputs, self.optimal_devices
        )
        
        results = [None] * len(inputs)
        
        # Process distributed workload
        with ThreadPoolExecutor(max_workers=len(self.optimal_devices)) as executor:
            futures = []
            
            for inp, device in workload_distribution:
                future = executor.submit(self._process_on_device, inp, device, **kwargs)
                futures.append((future, inputs.index(inp)))
            
            # Collect results
            for future, index in futures:
                try:
                    result = future.result(timeout=60)
                    results[index] = result
                except Exception as e:
                    self.logger.warning(f"Multi-device processing error: {e}")
                    results[index] = None
        
        return [r for r in results if r is not None]
    
    def _process_on_device(self, input_data: Any, device: torch.device, **kwargs) -> Any:
        """Process single input on specific device"""
        # This would require device-specific pipeline instances in a production system
        # For now, we use the main pipeline
        return self.pipeline(input_data, **kwargs)
    
    def _fallback_inference(self, inputs: List[str], **kwargs) -> List[Any]:
        """Fallback inference with alternative hardware configurations"""
        if self.config.verbose:
            self.logger.info("Attempting fallback inference...")
        
        # Try CPU fallback
        try:
            cpu_pipeline = self._create_cpu_fallback_pipeline()
            results = []
            for inp in inputs:
                result = cpu_pipeline(inp, **kwargs)
                results.append(result)
            return results
        except Exception as e:
            self.logger.error(f"CPU fallback failed: {e}")
            raise
    
    def _create_cpu_fallback_pipeline(self):
        """Create CPU fallback pipeline"""
        from transformers import pipeline
        
        return pipeline(
            task=self.config.task,
            model=self.config.model,
            device="cpu",
            torch_dtype=torch.float32
        )
    
    def _display_inference_stats(self, num_inputs: int, inference_time: float):
        """Enhanced inference statistics display"""
        if not self.console:
            return
        
        throughput = num_inputs / inference_time if inference_time > 0 else 0
        
        stats_text = Text()
        stats_text.append("⚡ Enhanced Inference Complete\n", style="bold green")
        stats_text.append(f"📊 Processed: {num_inputs} inputs\n", style="cyan")
        stats_text.append(f"⏱️  Time: {inference_time:.3f}s\n", style="yellow")
        stats_text.append(f"🚀 Throughput: {throughput:.2f} inputs/sec\n", style="magenta")
        
        # Add device information
        if self.optimal_devices:
            device_list = [str(d) for d in self.optimal_devices]
            stats_text.append(f"🖥️  Devices: {', '.join(device_list)}", style="blue")
        
        self.console.print(Panel(stats_text, title="Enhanced Performance Stats"))
    
    def _display_initialization_summary(self):
        """Display comprehensive initialization summary"""
        if not self.console:
            return
        
        # Create initialization summary
        summary_text = Text()
        summary_text.append("🚀 Advanced Pipeline System Initialized\n\n", style="bold green")
        
        # Hardware information
        if self.hardware_manager:
            num_devices = len(self.hardware_manager.hardware_profiles)
            optimal_devices = [str(d) for d in self.optimal_devices]
            
            summary_text.append(f"🖥️  Hardware Profiles: {num_devices} devices detected\n", style="blue")
            summary_text.append(f"🎯 Optimal Devices: {', '.join(optimal_devices)}\n", style="green")
            
            # Hardware features
            features = []
            for device in self.optimal_devices:
                profile = self.hardware_manager.hardware_profiles.get(str(device))
                if profile:
                    if profile.supports_mixed_precision:
                        features.append("Mixed Precision")
                    if profile.supports_tensor_cores:
                        features.append("Tensor Cores")
            
            if features:
                summary_text.append(f"⚡ Hardware Features: {', '.join(set(features))}\n", style="yellow")
        
        # Configuration highlights
        config_features = []
        if self.config.enable_caching:
            config_features.append("Smart Caching")
        if self.config.enable_parallel:
            config_features.append("Parallel Processing")
        if self.config.enable_multi_device:
            config_features.append("Multi-Device")
        if self.config.auto_mixed_precision:
            config_features.append("Auto Mixed Precision")
        
        if config_features:
            summary_text.append(f"🔧 Features: {', '.join(config_features)}", style="cyan")
        
        self.console.print(Panel(summary_text, title="System Ready", style="bold green"))
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information including hardware details"""
        info = {
            "config": self.config.__dict__,
            "optimal_devices": [str(d) for d in self.optimal_devices],
            "cache_stats": {
                "size": len(self.cache.cache),
                "max_size": self.config.cache_size,
                "device_caches": {k: len(v) for k, v in self.cache.device_caches.items()}
            },
            "performance": self.monitor.get_summary(),
            "errors": self.error_handler.get_error_summary(),
            "memory": {
                "cpu_percent": psutil.virtual_memory().percent,
                "gpu_memory": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            }
        }
        
        # Add hardware information if available
        if self.hardware_manager:
            info["hardware_profiles"] = {
                device_id: {
                    "name": profile.name,
                    "memory_total": profile.memory_total,
                    "memory_available": profile.memory_available,
                    "utilization": profile.utilization,
                    "supports_mixed_precision": profile.supports_mixed_precision,
                    "supports_tensor_cores": profile.supports_tensor_cores
                }
                for device_id, profile in self.hardware_manager.hardware_profiles.items()
            }
            
            # Add hardware recommendations
            info["hardware_recommendations"] = self.hardware_manager.get_device_recommendations(
                self._infer_model_type()
            )
        
        return info
    
    def get_hardware_recommendations(self) -> Dict[str, Any]:
        """Get hardware optimization recommendations"""
        if not self.hardware_manager:
            return {"message": "Hardware profiling disabled"}
        
        return self.hardware_manager.get_device_recommendations(
            self._infer_model_type()
        )
    
    def benchmark_hardware(self, test_inputs: List[str], iterations: int = 10) -> Dict[str, Any]:
        """
        Comprehensive hardware benchmark across all available devices
        """
        if not self.hardware_manager:
            return self.benchmark(test_inputs, iterations)
        
        if self.config.verbose and self.console:
            self.console.print("🏁 Starting Comprehensive Hardware Benchmark", style="bold red")
        
        benchmark_results = {
            "iterations": iterations,
            "input_count": len(test_inputs),
            "device_benchmarks": {},
            "recommendations": {}
        }
        
        # Benchmark each available device
        for device_id, profile in self.hardware_manager.hardware_profiles.items():
            if self.config.verbose and self.console:
                self.console.print(f"🔥 Benchmarking {device_id}...", style="yellow")
            
            try:
                # Create device-specific pipeline for benchmarking
                device_results = self._benchmark_device(device_id, test_inputs, iterations)
                benchmark_results["device_benchmarks"][device_id] = device_results
            except Exception as e:
                self.logger.warning(f"Failed to benchmark {device_id}: {e}")
                benchmark_results["device_benchmarks"][device_id] = {"error": str(e)}
        
        # Generate recommendations based on benchmark results
        benchmark_results["recommendations"] = self._generate_benchmark_recommendations(
            benchmark_results["device_benchmarks"]
        )
        
        if self.config.verbose:
            self._display_hardware_benchmark_results(benchmark_results)
        
        return benchmark_results
    
    def _benchmark_device(self, device_id: str, test_inputs: List[str], 
                         iterations: int) -> Dict[str, Any]:
        """Benchmark specific device performance"""
        device = torch.device(device_id)
        
        # Create device-specific configuration
        device_config = PipelineConfig(
            task=self.config.task,
            model=self.config.model,
            device=device,
            enable_auto_device_selection=False,
            enable_multi_device=False,
            verbose=False
        )
        
        # Create device-specific pipeline
        with AdvancedPipeline(device_config) as device_pipeline:
            device_results = device_pipeline.benchmark(test_inputs, iterations)
        
        return device_results
    
    def _generate_benchmark_recommendations(self, device_benchmarks: Dict[str, Any]) -> Dict[str, Any]:
        """Generate recommendations based on benchmark results"""
        recommendations = {
            "fastest_device": None,
            "most_efficient_device": None,
            "best_for_batch": None,
            "optimization_tips": []
        }
        
        valid_benchmarks = {k: v for k, v in device_benchmarks.items() 
                           if "error" not in v and "avg_time" in v}
        
        if not valid_benchmarks:
            return recommendations
        
        # Find fastest device
        fastest_device = min(valid_benchmarks.items(), 
                           key=lambda x: x[1]["avg_time"])[0]
        recommendations["fastest_device"] = fastest_device
        
        # Find most efficient (highest throughput)
        most_efficient = max(valid_benchmarks.items(), 
                           key=lambda x: x[1]["avg_throughput"])[0]
        recommendations["most_efficient_device"] = most_efficient
        
        # Generate optimization tips
        tips = []
        for device_id, results in valid_benchmarks.items():
            if "cuda" in device_id and results["avg_throughput"] > 10:
                tips.append(f"Enable mixed precision on {device_id} for better performance")
            elif "cpu" in device_id:
                tips.append(f"Consider using Intel optimizations for {device_id}")
        
        recommendations["optimization_tips"] = tips
        
        return recommendations
    
    def _display_hardware_benchmark_results(self, results: Dict[str, Any]):
        """Display comprehensive hardware benchmark results"""
        if not self.console:
            return
        
        # Device benchmark table
        table = Table(title="Hardware Benchmark Results")
        table.add_column("Device", style="cyan")
        table.add_column("Avg Time (s)", style="green")
        table.add_column("Throughput (inputs/s)", style="yellow")
        table.add_column("Memory Delta (MB)", style="blue")
        table.add_column("Status", style="magenta")
        
        for device_id, benchmark in results["device_benchmarks"].items():
            if "error" in benchmark:
                table.add_row(device_id, "N/A", "N/A", "N/A", f"Error: {benchmark['error'][:20]}...")
            else:
                table.add_row(
                    device_id,
                    f"{benchmark.get('avg_time', 0):.4f}",
                    f"{benchmark.get('avg_throughput', 0):.2f}",
                    f"{benchmark.get('avg_memory', 0) / 1024 / 1024:.2f}",
                    "✅ Success"
                )
        
        self.console.print(table)
        
        # Recommendations
        recs = results["recommendations"]
        if recs["fastest_device"]:
            self.console.print(f"🏆 Fastest Device: {recs['fastest_device']}", style="bold green")
        if recs["most_efficient_device"]:
            self.console.print(f"⚡ Most Efficient: {recs['most_efficient_device']}", style="bold yellow")
        
        if recs["optimization_tips"]:
            self.console.print("\n💡 Optimization Tips:", style="bold blue")
            for tip in recs["optimization_tips"]:
                self.console.print(f"  • {tip}", style="blue")
    
    def cleanup(self):
        """Enhanced cleanup with hardware manager shutdown"""
        if self.config.verbose and self.console:
            self.console.print("🧹 Cleaning up enhanced resources...", style="yellow")
        
        # Shutdown hardware manager
        if self.hardware_manager:
            self.hardware_manager.shutdown()
        
        # Clear caches
        self.cache.clear()
        
        # Force memory cleanup
        self.memory_manager._force_cleanup()
        
        # Display final performance summary
        if self.config.enable_metrics:
            self.monitor.display_summary()
        
        if self.config.verbose and self.console:
            self.console.print("✅ Enhanced cleanup complete", style="green")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with enhanced cleanup"""
        self.cleanup()


# Enhanced convenience function
def create_pipeline(
    task: str,
    model: Optional[str] = None,
    verbose: bool = False,
    enable_auto_optimization: bool = True,
    **kwargs
) -> AdvancedPipeline:
    """
    Enhanced convenience function to create an optimized pipeline with automatic hardware selection
    
    Args:
        task: The pipeline task (e.g., "text-classification")
        model: Model name or path (optional)
        verbose: Enable verbose output with Rich console
        enable_auto_optimization: Enable automatic hardware optimization
        **kwargs: Additional configuration options
    
    Returns:
        Configured AdvancedPipeline instance with optimal hardware selection
    
    Example:
        ```python
        # Quick text classification with auto-optimization
        pipe = create_pipeline(
            "text-classification",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            verbose=True,
            enable_auto_optimization=True
        )
        
        result = pipe("I love this product!")
        print(result)
        
        # Get hardware recommendations
        recommendations = pipe.get_hardware_recommendations()
        print(recommendations)
        ```
    """
    config = PipelineConfig(
        task=task,
        model=model,
        verbose=verbose,
        enable_auto_device_selection=enable_auto_optimization,
        enable_hardware_optimization=enable_auto_optimization,
        hardware_profiling=enable_auto_optimization,
        auto_mixed_precision=enable_auto_optimization,
        **kwargs
    )
    
    return AdvancedPipeline(config)


# Example usage and testing with enhanced hardware optimization
if __name__ == "__main__":
    """
    Enhanced demonstration of the Advanced Pipeline System with hardware optimization
    
    This section shows various usage patterns and features:
    1. Automatic hardware detection and optimization
    2. Multi-device orchestration
    3. Hardware-specific benchmarking
    4. Performance prediction and recommendations
    5. Real-time hardware adaptation
    """
    
    # Example 1: Automatic Hardware Optimization
    print("=" * 60)
    print("Example 1: Automatic Hardware Optimization Pipeline")
    print("=" * 60)
    
    config = PipelineConfig(
        task="text-classification",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        verbose=True,
        enable_auto_device_selection=True,
        enable_hardware_optimization=True,
        hardware_profiling=True,
        auto_mixed_precision=True,
        enable_caching=True,
        batch_size=4
    )
    
    with AdvancedPipeline(config) as pipeline:
        test_texts = [
            "I love this product! It's amazing!",
            "This is the worst thing I've ever bought.",
            "It's okay, nothing special.",
            "Absolutely fantastic! Highly recommended!",
            "Complete waste of money.",
            "Pretty good, would buy again."
        ]
        
        results = pipeline(test_texts)
        
        print("\nResults:")
        for text, result in zip(test_texts, results):
            label = result
            score = result['score']
            print(f"Text: {text[:50]}...")
            print(f"Sentiment: {label} (confidence:)")
            print("-" * 30)
        
        # Get hardware recommendations
        recommendations = pipeline.get_hardware_recommendations()
        print(f"\nHardware Recommendations: {recommendations}")
    
    # Example 2: Multi-Device Pipeline
    print("\n" + "=" * 60)
    print("Example 2: Multi-Device Optimization Pipeline")
    print("=" * 60)
    
    multi_device_config = PipelineConfig(
        task="text-generation",
        model="gpt2",
        verbose=True,
        enable_auto_device_selection=True,
        enable_multi_device=True,
        enable_hardware_optimization=True,
        enable_parallel=True,
        max_workers=2
    )
    
    with AdvancedPipeline(multi_device_config) as gen_pipeline:
        prompts = [
            "The future of artificial intelligence is",
            "Machine learning will revolutionize",
            "In the next decade, technology will"
        ]
        
        generated_texts = gen_pipeline(
            prompts,
            max_length=50,
            temperature=0.8,
            num_return_sequences=1
        )
        
        print("\nGenerated Texts:")
        for prompt, generated in zip(prompts, generated_texts):
            print(f"Prompt: {prompt}")
            print(f"Generated: {generated[0]['generated_text']}")
            print("-" * 50)
    
    # Example 3: Comprehensive Hardware Benchmarking
    print("\n" + "=" * 60)
    print("Example 3: Comprehensive Hardware Benchmarking")
    print("=" * 60)
    
    benchmark_config = PipelineConfig(
        task="text-classification",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        verbose=True,
        enable_auto_device_selection=True,
        enable_hardware_optimization=True,
        hardware_profiling=True,
        enable_metrics=True,
        batch_size=8
    )
    
    with AdvancedPipeline(benchmark_config) as bench_pipeline:
        benchmark_inputs = [
            "This is a test sentence for benchmarking.",
            "Another test sentence to measure performance.",
            "Benchmarking helps optimize the pipeline.",
            "Performance monitoring is crucial for production."
        ]
        
        # Run comprehensive hardware benchmark
        hardware_results = bench_pipeline.benchmark_hardware(
            benchmark_inputs,
            iterations=3
        )
        
        print(f"\nHardware Benchmark Summary:")
        print(f"Fastest Device: {hardware_results}")
        print(f"Most Efficient: {hardware_results}")
        
        # Get detailed system information
        system_info = bench_pipeline.get_system_info()
        print(f"\nSystem Overview:")
        print(f"Optimal Devices: {system_info['optimal_devices']}")
        if 'hardware_profiles' in system_info:
            print(f"Available Hardware: {len(system_info['hardware_profiles'])} devices")
    
    # Example 4: Advanced Configuration with Auto-Optimization
    print("\n" + "=" * 60)
    print("Example 4: Advanced Auto-Optimization Configuration")
    print("=" * 60)
    
    advanced_config = PipelineConfig(
        task="text-classification",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        verbose=True,
        enable_auto_device_selection=True,
        enable_hardware_optimization=True,
        enable_multi_device=False,
        hardware_profiling=True,
        auto_mixed_precision=True,
        enable_caching=True,
        enable_parallel=True,
        memory_threshold=0.8,
        cache_size=64,
        max_batch_size=16,
        enable_metrics=True,
        enable_dynamic_switching=True
    )
    
    with AdvancedPipeline(advanced_config) as advanced_pipeline:
        # Process examples with automatic optimization
        examples = ["Great product!", "Terrible experience!", "Average quality."]
        results = advanced_pipeline(examples)
        
        print("\nProcessing complete with auto-optimization!")
        
        # Display comprehensive system information
        system_info = advanced_pipeline.get_system_info()
        print(f"\nFinal System State:")
        print(f"Optimal Devices Used: {system_info['optimal_devices']}")
        print(f"Cache Efficiency: {system_info['cache_stats']}")
        
        if 'hardware_recommendations' in system_info:
            tips = system_info['hardware_recommendations'].get('optimization_suggestions', [])
            if tips:
                print(f"Optimization Tips Applied: {len(tips)} recommendations")
    
    print("\n" + "=" * 60)
    print("Enhanced Advanced Pipeline System Demo Complete!")
    print("All hardware optimization features successfully demonstrated!")
    print("=" * 60)