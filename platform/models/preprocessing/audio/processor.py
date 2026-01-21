"""
Audio Processor Module

Unified audio preprocessing pipeline with:
- Multi-codec loading
- Spectrogram computation
- MFCC extraction
- Normalization
"""

from typing import Optional, Union, List, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from .loader import AudioLoader, AudioData
from .spectrogram import SpectrogramComputer, SpectrogramOutput, WindowType


@dataclass
class AudioProcessorOutput:
    """Output from audio processor."""
    input_features: "torch.Tensor"  # (features, time) or (batch, features, time)
    attention_mask: Optional["torch.Tensor"]
    sample_rates: List[int]
    durations: List[float]


class AudioProcessor:
    """
    Unified audio preprocessing pipeline.
    
    Features:
    - Any codec support
    - Configurable feature extraction
    - Triton-accelerated spectrogram
    - Batch processing
    - Duration limiting
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
        n_mfcc: int = 13,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
        feature_type: str = "log_mel",
        do_normalize: bool = True,
        max_duration: Optional[float] = 30.0,
        padding_value: float = 0.0,
        device: str = "cuda",
        use_triton: bool = True,
    ):
        """
        Initialize processor.
        
        Args:
            sample_rate: Target sample rate
            n_fft: FFT window size
            hop_length: STFT hop length
            n_mels: Number of mel filterbanks
            n_mfcc: Number of MFCCs
            fmin: Minimum frequency
            fmax: Maximum frequency
            feature_type: 'log_mel', 'mel', 'mfcc', 'spectrogram'
            do_normalize: Normalize features
            max_duration: Maximum audio duration
            padding_value: Padding value for batching
            device: Target device
            use_triton: Use Triton kernels
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch required for AudioProcessor")
        
        self.sample_rate = sample_rate
        self.feature_type = feature_type
        self.do_normalize = do_normalize
        self.max_duration = max_duration
        self.padding_value = padding_value
        self.device = device
        
        self.loader = AudioLoader(
            target_sample_rate=sample_rate,
            mono=True,
            max_duration=max_duration,
        )
        
        self.spectrogram = SpectrogramComputer(
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            n_mfcc=n_mfcc,
            sample_rate=sample_rate,
            fmin=fmin,
            fmax=fmax,
            device=device,
            use_triton=use_triton,
        )
    
    def _to_tensor(self, audio_data: AudioData) -> "torch.Tensor":
        """Convert AudioData to tensor."""
        waveform = audio_data.waveform
        
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform)
        
        return waveform.to(dtype=torch.float32, device=self.device)
    
    def _normalize_features(self, features: "torch.Tensor") -> "torch.Tensor":
        """Normalize features to zero mean, unit variance."""
        mean = features.mean()
        std = features.std() + 1e-8
        return (features - mean) / std
    
    def _extract_features(self, waveform: "torch.Tensor") -> "torch.Tensor":
        """Extract features from waveform."""
        if self.feature_type == "log_mel":
            output = self.spectrogram.mel_spectrogram(waveform, log=True)
            features = output.spectrogram
        elif self.feature_type == "mel":
            output = self.spectrogram.mel_spectrogram(waveform, log=False)
            features = output.spectrogram
        elif self.feature_type == "mfcc":
            features = self.spectrogram.mfcc(waveform)
        elif self.feature_type == "spectrogram":
            output = self.spectrogram.spectrogram(waveform)
            features = output.spectrogram
        else:
            raise ValueError(f"Unknown feature_type: {self.feature_type}")
        
        return features
    
    def process(
        self,
        audio: Union[str, Path, bytes, AudioData],
        return_tensors: bool = True,
    ) -> Union[AudioProcessorOutput, Dict[str, Any]]:
        """
        Process single audio.
        
        Args:
            audio: Input audio (path, URL, bytes, AudioData)
            return_tensors: Return tensor output
            
        Returns:
            AudioProcessorOutput or dict
        """
        # Load if needed
        if isinstance(audio, AudioData):
            audio_data = audio
        else:
            audio_data = self.loader.load(audio)
        
        # Convert to tensor
        waveform = self._to_tensor(audio_data)
        
        # Extract features
        features = self._extract_features(waveform)
        
        # Normalize
        if self.do_normalize:
            features = self._normalize_features(features)
        
        if return_tensors:
            return AudioProcessorOutput(
                input_features=features,
                attention_mask=None,
                sample_rates=[audio_data.sample_rate],
                durations=[audio_data.duration],
            )
        
        return {
            "input_features": features,
            "sample_rates": [audio_data.sample_rate],
            "durations": [audio_data.duration],
        }
    
    def process_batch(
        self,
        audios: List[Union[str, Path, bytes, AudioData]],
        return_tensors: bool = True,
        padding: bool = True,
    ) -> Union[AudioProcessorOutput, Dict[str, Any]]:
        """
        Process batch of audios.
        
        Args:
            audios: List of input audios
            return_tensors: Return tensor output
            padding: Pad to same length
            
        Returns:
            AudioProcessorOutput or dict with batched tensors
        """
        outputs = [self.process(a, return_tensors=True) for a in audios]
        
        features_list = [o.input_features for o in outputs]
        sample_rates = [o.sample_rates[0] for o in outputs]
        durations = [o.durations[0] for o in outputs]
        
        if padding:
            # Find max length
            max_len = max(f.shape[-1] for f in features_list)
            
            # Pad features
            padded_features = []
            attention_masks = []
            
            for features in features_list:
                pad_len = max_len - features.shape[-1]
                
                if pad_len > 0:
                    # Pad on time dimension
                    padded = torch.nn.functional.pad(
                        features, 
                        (0, pad_len), 
                        value=self.padding_value
                    )
                    mask = torch.cat([
                        torch.ones(features.shape[-1], device=self.device),
                        torch.zeros(pad_len, device=self.device),
                    ])
                else:
                    padded = features
                    mask = torch.ones(features.shape[-1], device=self.device)
                
                padded_features.append(padded)
                attention_masks.append(mask)
            
            # Stack
            input_features = torch.stack(padded_features, dim=0)
            attention_mask = torch.stack(attention_masks, dim=0)
        else:
            input_features = features_list
            attention_mask = None
        
        if return_tensors:
            return AudioProcessorOutput(
                input_features=input_features,
                attention_mask=attention_mask,
                sample_rates=sample_rates,
                durations=durations,
            )
        
        return {
            "input_features": input_features,
            "attention_mask": attention_mask,
            "sample_rates": sample_rates,
            "durations": durations,
        }
    
    def __call__(
        self,
        audios: Union[Any, List[Any]],
        return_tensors: bool = True,
        padding: bool = True,
    ) -> Union[AudioProcessorOutput, Dict[str, Any]]:
        """
        Process audio(s).
        
        Args:
            audios: Single audio or list of audios
            return_tensors: Return tensor output
            padding: Pad batch to same length
            
        Returns:
            AudioProcessorOutput or dict
        """
        if isinstance(audios, list):
            return self.process_batch(audios, return_tensors, padding)
        return self.process(audios, return_tensors)
