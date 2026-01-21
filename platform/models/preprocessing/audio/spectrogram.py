"""
Spectrogram Computation Module

High-performance spectrogram, mel-spectrogram, and MFCC extraction.
Triton-accelerated FFT and filterbank operations.
"""

from typing import Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import math

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


class WindowType(Enum):
    """Window function types."""
    HANN = "hann"
    HAMMING = "hamming"
    BLACKMAN = "blackman"
    BARTLETT = "bartlett"


@dataclass
class SpectrogramOutput:
    """Spectrogram output container."""
    spectrogram: "torch.Tensor"  # (freq_bins, time_frames) or (batch, freq_bins, time_frames)
    sample_rate: int
    n_fft: int
    hop_length: int
    n_mels: Optional[int] = None


# ============================================================================
# Triton Kernels
# ============================================================================

if HAS_TRITON:
    
    @triton.jit
    def mel_filterbank_kernel(
        spec_ptr,
        filterbank_ptr,
        out_ptr,
        n_freq,
        n_mels,
        n_frames,
        BLOCK_SIZE_MEL: tl.constexpr,
        BLOCK_SIZE_FREQ: tl.constexpr,
    ):
        """
        Apply mel filterbank to spectrogram.
        
        Matrix multiplication: mel_spec = filterbank @ spec
        Optimized for memory bandwidth with tiled access.
        """
        pid_mel = tl.program_id(0)
        pid_frame = tl.program_id(1)
        
        # Compute mel bin range for this block
        mel_start = pid_mel * BLOCK_SIZE_MEL
        mel_offsets = mel_start + tl.arange(0, BLOCK_SIZE_MEL)
        mel_mask = mel_offsets < n_mels
        
        # Accumulate over frequency bins
        acc = tl.zeros((BLOCK_SIZE_MEL,), dtype=tl.float32)
        
        for freq_block in range(0, n_freq, BLOCK_SIZE_FREQ):
            freq_offsets = freq_block + tl.arange(0, BLOCK_SIZE_FREQ)
            freq_mask = freq_offsets < n_freq
            
            # Load filterbank weights: (n_mels, n_freq)
            fb_idx = mel_offsets[:, None] * n_freq + freq_offsets[None, :]
            fb_weights = tl.load(
                filterbank_ptr + fb_idx,
                mask=mel_mask[:, None] & freq_mask[None, :],
                other=0.0,
            )
            
            # Load spectrogram values: (n_freq, n_frames)
            spec_idx = freq_offsets * n_frames + pid_frame
            spec_vals = tl.load(
                spec_ptr + spec_idx,
                mask=freq_mask,
                other=0.0,
            )
            
            # Accumulate
            acc += tl.sum(fb_weights * spec_vals[None, :], axis=1)
        
        # Store output: (n_mels, n_frames)
        out_idx = mel_offsets * n_frames + pid_frame
        tl.store(out_ptr + out_idx, acc, mask=mel_mask)
    
    
    @triton.jit
    def log_mel_kernel(
        x_ptr,
        out_ptr,
        n_elements,
        log_offset,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Compute log mel spectrogram with numerical stability.
        
        out = log(max(x, log_offset))
        """
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        x = tl.load(x_ptr + offsets, mask=mask)
        
        # Clamp for numerical stability
        x = tl.maximum(x, log_offset)
        out = tl.log(x)
        
        tl.store(out_ptr + offsets, out, mask=mask)


def _create_mel_filterbank(
    n_freq: int,
    n_mels: int,
    sample_rate: int,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    device: str = "cuda",
) -> "torch.Tensor":
    """
    Create mel filterbank matrix.
    
    Args:
        n_freq: Number of FFT frequency bins
        n_mels: Number of mel bins
        sample_rate: Audio sample rate
        fmin: Minimum frequency
        fmax: Maximum frequency
        device: Target device
        
    Returns:
        Filterbank tensor (n_mels, n_freq)
    """
    if fmax is None:
        fmax = sample_rate / 2.0
    
    # Mel scale conversion
    def hz_to_mel(hz):
        return 2595.0 * math.log10(1.0 + hz / 700.0)
    
    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
    
    # Mel points
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = torch.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = torch.tensor([mel_to_hz(m) for m in mel_points])
    
    # FFT bin frequencies
    fft_freqs = torch.linspace(0, sample_rate / 2, n_freq)
    
    # Create filterbank
    filterbank = torch.zeros(n_mels, n_freq)
    
    for i in range(n_mels):
        left = hz_points[i]
        center = hz_points[i + 1]
        right = hz_points[i + 2]
        
        # Left slope
        left_mask = (fft_freqs >= left) & (fft_freqs <= center)
        filterbank[i, left_mask] = (fft_freqs[left_mask] - left) / (center - left + 1e-8)
        
        # Right slope
        right_mask = (fft_freqs >= center) & (fft_freqs <= right)
        filterbank[i, right_mask] = (right - fft_freqs[right_mask]) / (right - center + 1e-8)
    
    return filterbank.to(device)


def _create_window(
    window_type: WindowType,
    length: int,
    device: str = "cuda",
) -> "torch.Tensor":
    """Create window function."""
    if window_type == WindowType.HANN:
        return torch.hann_window(length, device=device)
    elif window_type == WindowType.HAMMING:
        return torch.hamming_window(length, device=device)
    elif window_type == WindowType.BLACKMAN:
        return torch.blackman_window(length, device=device)
    elif window_type == WindowType.BARTLETT:
        return torch.bartlett_window(length, device=device)
    else:
        return torch.hann_window(length, device=device)


class SpectrogramComputer:
    """
    High-performance spectrogram computation.
    
    Features:
    - Triton-accelerated mel filterbank
    - STFT with configurable window
    - Log-mel spectrogram
    - MFCC extraction
    - Batch processing
    """
    
    def __init__(
        self,
        n_fft: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
        n_mfcc: int = 13,
        sample_rate: int = 16000,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
        window_type: WindowType = WindowType.HANN,
        power: float = 2.0,
        log_offset: float = 1e-6,
        device: str = "cuda",
        use_triton: bool = True,
    ):
        """
        Initialize spectrogram computer.
        
        Args:
            n_fft: FFT window size
            hop_length: STFT hop length
            n_mels: Number of mel filterbanks
            n_mfcc: Number of MFCCs
            sample_rate: Audio sample rate
            fmin: Minimum frequency for mel scale
            fmax: Maximum frequency for mel scale
            window_type: Window function type
            power: Spectrogram power (2.0 for power, 1.0 for magnitude)
            log_offset: Offset for log computation
            device: Target device
            use_triton: Use Triton kernels
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch required for SpectrogramComputer")
        
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.sample_rate = sample_rate
        self.fmin = fmin
        self.fmax = fmax or sample_rate / 2.0
        self.window_type = window_type
        self.power = power
        self.log_offset = log_offset
        self.device = device
        self.use_triton = use_triton and HAS_TRITON and device == "cuda"
        
        # Pre-compute window and filterbank
        self.window = _create_window(window_type, n_fft, device)
        self.mel_filterbank = _create_mel_filterbank(
            n_freq=n_fft // 2 + 1,
            n_mels=n_mels,
            sample_rate=sample_rate,
            fmin=fmin,
            fmax=self.fmax,
            device=device,
        )
        
        # DCT matrix for MFCC
        self.dct_matrix = self._create_dct_matrix(n_mels, n_mfcc, device)
    
    def _create_dct_matrix(
        self,
        n_mels: int,
        n_mfcc: int,
        device: str,
    ) -> "torch.Tensor":
        """Create DCT-II matrix for MFCC."""
        n = torch.arange(n_mels, dtype=torch.float32, device=device)
        k = torch.arange(n_mfcc, dtype=torch.float32, device=device)
        
        dct = torch.cos(math.pi / n_mels * (n[:, None] + 0.5) * k[None, :])
        
        # Orthonormal scaling
        dct[:, 0] *= 1.0 / math.sqrt(n_mels)
        dct[:, 1:] *= math.sqrt(2.0 / n_mels)
        
        return dct.T  # (n_mfcc, n_mels)
    
    def stft(
        self,
        waveform: "torch.Tensor",
        return_complex: bool = False,
    ) -> "torch.Tensor":
        """
        Compute Short-Time Fourier Transform.
        
        Args:
            waveform: Input waveform (samples,) or (batch, samples)
            return_complex: Return complex tensor
            
        Returns:
            STFT tensor (freq_bins, time_frames) or (batch, freq_bins, time_frames)
        """
        is_batched = waveform.dim() == 2
        if not is_batched:
            waveform = waveform.unsqueeze(0)
        
        # Pad to center
        pad_amount = self.n_fft // 2
        waveform = F.pad(waveform, (pad_amount, pad_amount), mode='reflect')
        
        # STFT
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=False,
            return_complex=True,
        )
        
        if not return_complex:
            # Power spectrogram
            stft = stft.abs().pow(self.power)
        
        if not is_batched:
            stft = stft.squeeze(0)
        
        return stft
    
    def spectrogram(self, waveform: "torch.Tensor") -> SpectrogramOutput:
        """
        Compute power spectrogram.
        
        Args:
            waveform: Input waveform tensor
            
        Returns:
            SpectrogramOutput with power spectrogram
        """
        waveform = waveform.to(self.device)
        spec = self.stft(waveform, return_complex=False)
        
        return SpectrogramOutput(
            spectrogram=spec,
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
    
    def mel_spectrogram(
        self,
        waveform: "torch.Tensor",
        log: bool = True,
    ) -> SpectrogramOutput:
        """
        Compute mel spectrogram.
        
        Args:
            waveform: Input waveform tensor
            log: Apply log transformation
            
        Returns:
            SpectrogramOutput with mel spectrogram
        """
        waveform = waveform.to(self.device)
        spec = self.stft(waveform, return_complex=False)
        
        is_batched = spec.dim() == 3
        if not is_batched:
            spec = spec.unsqueeze(0)
        
        # Apply mel filterbank
        if self.use_triton:
            mel_spec = self._triton_mel_filterbank(spec)
        else:
            # (batch, n_mels, n_freq) @ (batch, n_freq, time) 
            mel_spec = torch.matmul(
                self.mel_filterbank.unsqueeze(0),
                spec,
            )
        
        # Log transformation
        if log:
            if self.use_triton:
                mel_spec = self._triton_log(mel_spec)
            else:
                mel_spec = torch.log(torch.clamp(mel_spec, min=self.log_offset))
        
        if not is_batched:
            mel_spec = mel_spec.squeeze(0)
        
        return SpectrogramOutput(
            spectrogram=mel_spec,
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )
    
    def _triton_mel_filterbank(self, spec: "torch.Tensor") -> "torch.Tensor":
        """Apply mel filterbank using Triton kernel."""
        B, n_freq, n_frames = spec.shape
        out = torch.empty((B, self.n_mels, n_frames), device=spec.device, dtype=spec.dtype)
        
        for b in range(B):
            grid = lambda meta: (
                triton.cdiv(self.n_mels, meta['BLOCK_SIZE_MEL']),
                n_frames,
            )
            
            mel_filterbank_kernel[grid](
                spec[b].data_ptr(),
                self.mel_filterbank.data_ptr(),
                out[b].data_ptr(),
                n_freq,
                self.n_mels,
                n_frames,
                BLOCK_SIZE_MEL=32,
                BLOCK_SIZE_FREQ=64,
            )
        
        return out
    
    def _triton_log(self, x: "torch.Tensor") -> "torch.Tensor":
        """Apply log using Triton kernel."""
        out = torch.empty_like(x)
        n_elements = x.numel()
        
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        
        log_mel_kernel[grid](
            x.data_ptr(),
            out.data_ptr(),
            n_elements,
            self.log_offset,
            BLOCK_SIZE=1024,
        )
        
        return out
    
    def mfcc(self, waveform: "torch.Tensor") -> "torch.Tensor":
        """
        Compute Mel-Frequency Cepstral Coefficients.
        
        Args:
            waveform: Input waveform tensor
            
        Returns:
            MFCC tensor (n_mfcc, time_frames)
        """
        mel_output = self.mel_spectrogram(waveform, log=True)
        mel_spec = mel_output.spectrogram
        
        is_batched = mel_spec.dim() == 3
        if not is_batched:
            mel_spec = mel_spec.unsqueeze(0)
        
        # Apply DCT
        # (batch, n_mfcc, n_mels) @ (batch, n_mels, time)
        mfcc = torch.matmul(
            self.dct_matrix.unsqueeze(0),
            mel_spec,
        )
        
        if not is_batched:
            mfcc = mfcc.squeeze(0)
        
        return mfcc
    
    def __call__(
        self,
        waveform: "torch.Tensor",
        output_type: str = "mel",
    ) -> Union[SpectrogramOutput, "torch.Tensor"]:
        """
        Compute spectrogram.
        
        Args:
            waveform: Input waveform
            output_type: 'spectrogram', 'mel', 'log_mel', 'mfcc'
            
        Returns:
            Spectrogram output
        """
        if output_type == "spectrogram":
            return self.spectrogram(waveform)
        elif output_type == "mel":
            return self.mel_spectrogram(waveform, log=False)
        elif output_type in ("log_mel", "mel_log"):
            return self.mel_spectrogram(waveform, log=True)
        elif output_type == "mfcc":
            return self.mfcc(waveform)
        else:
            raise ValueError(f"Unknown output_type: {output_type}")
