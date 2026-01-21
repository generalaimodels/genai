"""
Audio Preprocessing Triton Kernels

Optimized kernels for mel filterbank and spectrogram operations.
"""

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


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
        
        Optimized matrix multiplication: mel_spec = filterbank @ spec
        
        Memory access pattern optimized for HBM bandwidth.
        
        Args:
            spec_ptr: Power spectrogram (n_freq, n_frames)
            filterbank_ptr: Mel filterbank (n_mels, n_freq)
            out_ptr: Output mel spectrogram (n_mels, n_frames)
            n_freq: Number of frequency bins
            n_mels: Number of mel bins
            n_frames: Number of time frames
        """
        pid_mel = tl.program_id(0)
        pid_frame = tl.program_id(1)
        
        # Mel bin range for this block
        mel_start = pid_mel * BLOCK_SIZE_MEL
        mel_offsets = mel_start + tl.arange(0, BLOCK_SIZE_MEL)
        mel_mask = mel_offsets < n_mels
        
        # Accumulate over frequency bins
        acc = tl.zeros((BLOCK_SIZE_MEL,), dtype=tl.float32)
        
        for freq_block in range(0, n_freq, BLOCK_SIZE_FREQ):
            freq_offsets = freq_block + tl.arange(0, BLOCK_SIZE_FREQ)
            freq_mask = freq_offsets < n_freq
            
            # Load filterbank weights: (n_mels, n_freq)
            # Access pattern: filterbank[mel, freq]
            fb_idx = mel_offsets[:, None] * n_freq + freq_offsets[None, :]
            fb_weights = tl.load(
                filterbank_ptr + fb_idx,
                mask=mel_mask[:, None] & freq_mask[None, :],
                other=0.0,
            )
            
            # Load spectrogram values: (n_freq, n_frames)
            # Access pattern: spec[freq, frame]
            spec_idx = freq_offsets * n_frames + pid_frame
            spec_vals = tl.load(
                spec_ptr + spec_idx,
                mask=freq_mask,
                other=0.0,
            )
            
            # Matrix-vector product
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
        
        Args:
            x_ptr: Input mel spectrogram
            out_ptr: Output log mel spectrogram
            n_elements: Total elements
            log_offset: Minimum value for log (numerical stability)
        """
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        x = tl.load(x_ptr + offsets, mask=mask)
        
        # Clamp to minimum value for numerical stability
        x = tl.maximum(x, log_offset)
        out = tl.log(x)
        
        tl.store(out_ptr + offsets, out, mask=mask)
    
    
    @triton.jit
    def power_spectrogram_kernel(
        real_ptr,
        imag_ptr,
        out_ptr,
        n_elements,
        power,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Compute power spectrogram from complex STFT.
        
        out = (real^2 + imag^2)^(power/2)
        
        Args:
            real_ptr: Real component
            imag_ptr: Imaginary component
            out_ptr: Power spectrogram
            n_elements: Total elements
            power: Power exponent (2.0 for power, 1.0 for magnitude)
        """
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        real = tl.load(real_ptr + offsets, mask=mask)
        imag = tl.load(imag_ptr + offsets, mask=mask)
        
        # Compute magnitude squared
        mag_sq = real * real + imag * imag
        
        # Apply power
        if power == 2.0:
            out = mag_sq
        elif power == 1.0:
            out = tl.sqrt(mag_sq)
        else:
            out = tl.pow(mag_sq, power * 0.5)
        
        tl.store(out_ptr + offsets, out, mask=mask)
    
    
    @triton.jit
    def apply_window_kernel(
        x_ptr,
        window_ptr,
        out_ptr,
        frame_len,
        n_frames,
        hop_length,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Apply window function to audio frames.
        
        Args:
            x_ptr: Input waveform
            window_ptr: Window function (frame_len,)
            out_ptr: Output windowed frames (n_frames, frame_len)
            frame_len: Window/frame length
            n_frames: Number of frames
            hop_length: Hop length between frames
        """
        pid = tl.program_id(0)
        frame_idx = pid // tl.cdiv(frame_len, BLOCK_SIZE)
        block_idx = pid % tl.cdiv(frame_len, BLOCK_SIZE)
        
        if frame_idx >= n_frames:
            return
        
        # Sample offsets within frame
        sample_start = block_idx * BLOCK_SIZE
        sample_offsets = sample_start + tl.arange(0, BLOCK_SIZE)
        sample_mask = sample_offsets < frame_len
        
        # Input sample positions
        input_start = frame_idx * hop_length
        input_offsets = input_start + sample_offsets
        
        # Load input samples
        samples = tl.load(x_ptr + input_offsets, mask=sample_mask, other=0.0)
        
        # Load window values
        window = tl.load(window_ptr + sample_offsets, mask=sample_mask, other=0.0)
        
        # Apply window
        windowed = samples * window
        
        # Store to output
        out_idx = frame_idx * frame_len + sample_offsets
        tl.store(out_ptr + out_idx, windowed, mask=sample_mask)
    
    
    @triton.jit
    def dct_kernel(
        x_ptr,
        dct_matrix_ptr,
        out_ptr,
        n_input,
        n_output,
        n_frames,
        BLOCK_SIZE_OUT: tl.constexpr,
        BLOCK_SIZE_IN: tl.constexpr,
    ):
        """
        Discrete Cosine Transform for MFCC.
        
        out = dct_matrix @ x
        
        Args:
            x_ptr: Input log mel spectrogram (n_mels, n_frames)
            dct_matrix_ptr: DCT matrix (n_mfcc, n_mels)
            out_ptr: Output MFCCs (n_mfcc, n_frames)
            n_input: Number of mel bins
            n_output: Number of MFCCs
            n_frames: Number of time frames
        """
        pid_out = tl.program_id(0)
        pid_frame = tl.program_id(1)
        
        # Output coefficient range
        out_start = pid_out * BLOCK_SIZE_OUT
        out_offsets = out_start + tl.arange(0, BLOCK_SIZE_OUT)
        out_mask = out_offsets < n_output
        
        # Accumulate over input (mel bins)
        acc = tl.zeros((BLOCK_SIZE_OUT,), dtype=tl.float32)
        
        for in_block in range(0, n_input, BLOCK_SIZE_IN):
            in_offsets = in_block + tl.arange(0, BLOCK_SIZE_IN)
            in_mask = in_offsets < n_input
            
            # Load DCT weights: (n_output, n_input)
            dct_idx = out_offsets[:, None] * n_input + in_offsets[None, :]
            dct_weights = tl.load(
                dct_matrix_ptr + dct_idx,
                mask=out_mask[:, None] & in_mask[None, :],
                other=0.0,
            )
            
            # Load mel values: (n_input, n_frames)
            mel_idx = in_offsets * n_frames + pid_frame
            mel_vals = tl.load(
                x_ptr + mel_idx,
                mask=in_mask,
                other=0.0,
            )
            
            # Accumulate
            acc += tl.sum(dct_weights * mel_vals[None, :], axis=1)
        
        # Store output
        out_idx = out_offsets * n_frames + pid_frame
        tl.store(out_ptr + out_idx, acc, mask=out_mask)
    
    
    @triton.jit
    def normalize_audio_kernel(
        x_ptr,
        out_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Normalize audio to [-1, 1] range.
        
        This is a two-pass kernel (first pass finds max, second normalizes).
        For single-pass, use with pre-computed max.
        
        Args:
            x_ptr: Input audio
            out_ptr: Normalized audio
            n_elements: Number of samples
        """
        # This simplified version assumes max is precomputed
        # Full implementation would require atomic max reduction
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        x = tl.load(x_ptr + offsets, mask=mask)
        
        # Simple peak normalization (assumes max ~= 1.0)
        # For actual use, compute global max first
        out = x / (tl.abs(x).max() + 1e-8)
        
        tl.store(out_ptr + offsets, out, mask=mask)
