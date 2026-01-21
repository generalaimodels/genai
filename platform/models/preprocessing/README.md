# Multimodal Preprocessing Module

High-performance preprocessing infrastructure for text, image, video, audio modalities with Triton kernel acceleration.

## Installation

```bash
# Core dependencies
pip install torch>=2.0 triton>=2.1.0

# Optional dependencies
pip install pillow opencv-python librosa soundfile
```

## Quick Start

```python
from models.preprocessing import MultimodalProcessor, PreprocessingConfig

# Initialize
processor = MultimodalProcessor(device="cuda")

# Process multimodal input
output = processor({
    'role': 'user',
    'content': {
        'input_text': 'Hello world! 👋',
        'input_image': '/path/to/image.jpg',
    }
})

# Access outputs
print(output.input_ids)      # Token IDs
print(output.pixel_values)   # Image tensor
```

## Training Tokenizer

```python
from models.preprocessing.text import train_tokenizer

# From directory
tokenizer = train_tokenizer("./corpus/", vocab_size=32000)

# From HuggingFace dataset
tokenizer = train_tokenizer("wikitext", vocab_size=50000)

# Save
tokenizer.save("./tokenizer/")
```

## Input Schema

```python
{
    'role': str,  # 'user', 'assistant', 'system'
    'content': {
        'input_text': str,           # Unicode, emoji, multilingual
        'input_image': PathOrURL,    # Local path, HTTP/HTTPS
        'input_video': PathOrURL,    # Any video format
        'input_audio': PathOrURL,    # Any audio codec
    }
}
```

## Module Structure

```
preprocessing/
├── config.py          # Configuration dataclasses
├── processor.py       # Unified MultimodalProcessor
├── text/              # Text preprocessing
│   ├── tokenizer.py   # BPE tokenizer
│   ├── vocabulary.py  # Vocabulary management
│   ├── normalizer.py  # Unicode normalization
│   └── training.py    # Tokenizer training
├── image/             # Image preprocessing
│   ├── loader.py      # Multi-format loading
│   ├── transforms.py  # Triton transforms
│   └── processor.py   # Image processor
├── video/             # Video preprocessing
│   ├── extractor.py   # Frame extraction
│   ├── sampler.py     # Temporal sampling
│   └── processor.py   # Video processor
├── audio/             # Audio preprocessing
│   ├── loader.py      # Multi-codec loading
│   ├── spectrogram.py # Mel/MFCC extraction
│   └── processor.py   # Audio processor
└── kernels/triton/    # Triton kernels
    ├── text_kernels.py
    ├── image_kernels.py
    ├── audio_kernels.py
    └── autotune.py
```

## Model Integration

```python
from models import RSSMoDModel, RSSMoDConfig
from models.preprocessing import MultimodalProcessor

# Initialize model
config = RSSMoDConfig.base()
model = RSSMoDModel(config)

# Initialize processor with matching vocab_size
processor = MultimodalProcessor(device="cuda")

# Process input
output = processor({
    'role': 'user',
    'content': {'input_text': 'Example prompt'}
})

# Forward pass
model_output = model(
    input_ids=output.input_ids.unsqueeze(0),
    attention_mask=output.attention_mask.unsqueeze(0),
)
```

## Configuration

```python
from models.preprocessing import PreprocessingConfig

# Default config
config = PreprocessingConfig()

# LLM-optimized
config = PreprocessingConfig.for_llm()

# Vision-optimized
config = PreprocessingConfig.for_vision()

# Multimodal
config = PreprocessingConfig.for_multimodal()
```

## Triton Kernels

| Kernel | Operation | Optimization |
|--------|-----------|--------------|
| `bilinear_resize_kernel` | Image resize | Memory coalescing |
| `normalize_kernel` | Mean/std normalize | Fused operation |
| `mel_filterbank_kernel` | Mel spectrogram | Tiled matmul |
| `fused_tokenize_lookup_kernel` | Embedding lookup | Coalesced access |

## Performance

- **Text**: O(n log v) tokenization
- **Image**: Memory-coalesced resize/normalize
- **Audio**: Triton-accelerated mel filterbank
- **Video**: Efficient frame sampling

## License

MIT
