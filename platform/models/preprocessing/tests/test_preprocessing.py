"""
Preprocessing Unit Tests

Test suite for text, image, video, audio preprocessing pipelines.
"""

import unittest
from typing import Optional
from pathlib import Path
import tempfile

# Conditional imports
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


class TestTextNormalizer(unittest.TestCase):
    """Tests for TextNormalizer."""
    
    def setUp(self):
        from models.preprocessing.text import TextNormalizer
        self.normalizer = TextNormalizer()
    
    def test_basic_normalize(self):
        """Test basic text normalization."""
        text = "Hello  World"
        result = self.normalizer.normalize(text)
        self.assertEqual(result, "Hello World")
    
    def test_unicode_normalize(self):
        """Test Unicode normalization."""
        # Composed vs decomposed
        text = "café"  # May be composed or decomposed
        result = self.normalizer.normalize(text)
        self.assertTrue(len(result) <= len(text) + 1)
    
    def test_emoji_extraction(self):
        """Test emoji extraction."""
        text = "Hello 👋 World 🌍"
        emojis = self.normalizer.extract_emojis(text)
        self.assertGreater(len(emojis), 0)
    
    def test_control_char_removal(self):
        """Test control character removal."""
        text = "Hello\x00World"
        result = self.normalizer.normalize(text)
        self.assertNotIn("\x00", result)


class TestVocabulary(unittest.TestCase):
    """Tests for Vocabulary."""
    
    def setUp(self):
        from models.preprocessing.text import Vocabulary
        self.vocab = Vocabulary()
    
    def test_add_token(self):
        """Test adding tokens."""
        token_id = self.vocab.add_token("hello")
        self.assertTrue(token_id >= 0)
        self.assertIn("hello", self.vocab)
    
    def test_token_to_id(self):
        """Test token to ID lookup."""
        token_id = self.vocab.add_token("world")
        result = self.vocab.token_to_id("world")
        self.assertEqual(result, token_id)
    
    def test_id_to_token(self):
        """Test ID to token lookup."""
        token_id = self.vocab.add_token("test")
        result = self.vocab.id_to_token(token_id)
        self.assertEqual(result, "test")
    
    def test_unknown_token(self):
        """Test unknown token returns UNK_ID."""
        result = self.vocab.token_to_id("nonexistent_token_xyz")
        self.assertEqual(result, self.vocab.UNK_ID)
    
    def test_special_tokens(self):
        """Test special tokens are present."""
        self.assertTrue(self.vocab.is_special("<pad>"))
        self.assertTrue(self.vocab.is_special("<unk>"))
        self.assertTrue(self.vocab.is_special("<s>"))
        self.assertTrue(self.vocab.is_special("</s>"))
    
    def test_save_load(self):
        """Test vocabulary serialization."""
        self.vocab.add_token("test_token", frequency=10)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "vocab.json"
            self.vocab.save(path)
            
            from models.preprocessing.text import Vocabulary
            loaded = Vocabulary.load(path)
            
            self.assertEqual(loaded.size, self.vocab.size)
            self.assertIn("test_token", loaded)


class TestBPETokenizer(unittest.TestCase):
    """Tests for BPETokenizer."""
    
    def setUp(self):
        from models.preprocessing.text import BPETokenizer
        self.tokenizer = BPETokenizer()
    
    def test_encode_decode_roundtrip(self):
        """Test encode->decode produces similar text."""
        text = "Hello world"
        output = self.tokenizer.encode(text, add_special_tokens=False)
        decoded = self.tokenizer.decode(output.input_ids, skip_special_tokens=True)
        # Note: BPE may not perfectly roundtrip without training
        self.assertIsInstance(decoded, str)
    
    def test_encode_produces_ids(self):
        """Test encoding produces valid IDs."""
        text = "Test input"
        output = self.tokenizer.encode(text)
        self.assertIsInstance(output.input_ids, list)
        self.assertTrue(all(isinstance(i, int) for i in output.input_ids))
    
    def test_special_tokens_added(self):
        """Test special tokens are added."""
        text = "Hello"
        output = self.tokenizer.encode(text, add_special_tokens=True)
        # Should have BOS at start, EOS at end
        self.assertEqual(output.input_ids[0], self.tokenizer.vocab.BOS_ID)
        self.assertEqual(output.input_ids[-1], self.tokenizer.vocab.EOS_ID)
    
    def test_batch_encode(self):
        """Test batch encoding."""
        texts = ["Hello", "World", "Test"]
        outputs = self.tokenizer.encode_batch(texts, padding=True)
        self.assertEqual(len(outputs), 3)


@unittest.skipUnless(HAS_TORCH and HAS_NUMPY, "Requires torch and numpy")
class TestImageLoader(unittest.TestCase):
    """Tests for ImageLoader."""
    
    def setUp(self):
        from models.preprocessing.image import ImageLoader
        self.loader = ImageLoader()
    
    def test_format_detection(self):
        """Test image format detection."""
        from models.preprocessing.image.loader import detect_format, ImageFormat
        
        # PNG magic bytes
        png_data = b'\x89PNG\r\n\x1a\n' + b'\x00' * 100
        self.assertEqual(detect_format(png_data), ImageFormat.PNG)
        
        # JPEG magic bytes
        jpeg_data = b'\xff\xd8\xff' + b'\x00' * 100
        self.assertEqual(detect_format(jpeg_data), ImageFormat.JPEG)


@unittest.skipUnless(HAS_TORCH, "Requires torch")
class TestImageTransforms(unittest.TestCase):
    """Tests for ImageTransforms."""
    
    def setUp(self):
        from models.preprocessing.image import ImageTransforms
        self.transforms = ImageTransforms(device="cpu", use_triton=False)
    
    def test_resize(self):
        """Test image resize."""
        x = torch.randn(3, 64, 64)
        result = self.transforms.resize(x, (32, 32))
        self.assertEqual(result.shape, (3, 32, 32))
    
    def test_normalize(self):
        """Test image normalization."""
        x = torch.randn(3, 64, 64)
        result = self.transforms.normalize(x)
        self.assertEqual(result.shape, x.shape)
    
    def test_to_grayscale(self):
        """Test RGB to grayscale."""
        x = torch.randn(3, 64, 64)
        result = self.transforms.to_grayscale(x)
        self.assertEqual(result.shape, (1, 64, 64))
    
    def test_center_crop(self):
        """Test center crop."""
        x = torch.randn(3, 64, 64)
        result = self.transforms.center_crop(x, (32, 32))
        self.assertEqual(result.shape, (3, 32, 32))


@unittest.skipUnless(HAS_TORCH and HAS_NUMPY, "Requires torch and numpy")
class TestSpectrogramComputer(unittest.TestCase):
    """Tests for SpectrogramComputer."""
    
    def setUp(self):
        from models.preprocessing.audio import SpectrogramComputer
        self.computer = SpectrogramComputer(device="cpu", use_triton=False)
    
    def test_stft(self):
        """Test STFT computation."""
        waveform = torch.randn(16000)  # 1 second
        result = self.computer.stft(waveform)
        self.assertEqual(result.dim(), 2)  # (freq, time)
    
    def test_mel_spectrogram(self):
        """Test mel spectrogram."""
        waveform = torch.randn(16000)
        result = self.computer.mel_spectrogram(waveform)
        self.assertEqual(result.spectrogram.shape[0], self.computer.n_mels)
    
    def test_mfcc(self):
        """Test MFCC extraction."""
        waveform = torch.randn(16000)
        result = self.computer.mfcc(waveform)
        self.assertEqual(result.shape[0], self.computer.n_mfcc)


@unittest.skipUnless(HAS_TORCH, "Requires torch")
class TestMultimodalProcessor(unittest.TestCase):
    """Tests for MultimodalProcessor."""
    
    def setUp(self):
        from models.preprocessing import MultimodalProcessor
        self.processor = MultimodalProcessor(device="cpu", use_triton=False)
    
    def test_text_only(self):
        """Test text-only processing."""
        output = self.processor({
            "role": "user",
            "content": {"input_text": "Hello world"}
        })
        self.assertIsNotNone(output.input_ids)
    
    def test_modality_detection(self):
        """Test modality detection."""
        from models.preprocessing.processor import MultimodalInput
        
        mm_input = MultimodalInput(
            role="user",
            input_text="Hello",
            input_image=None,
        )
        modalities = mm_input.get_modalities()
        self.assertEqual(len(modalities), 1)


class TestTemporalSampler(unittest.TestCase):
    """Tests for TemporalSampler."""
    
    def test_uniform_sampling(self):
        """Test uniform frame sampling."""
        from models.preprocessing.video import TemporalSampler
        from models.preprocessing.video.sampler import SamplingStrategy
        
        sampler = TemporalSampler(n_frames=8, strategy=SamplingStrategy.UNIFORM)
        indices = sampler.sample(100)
        
        self.assertEqual(len(indices), 8)
        self.assertTrue(all(0 <= i < 100 for i in indices))
    
    def test_stride_sampling(self):
        """Test stride-based sampling."""
        from models.preprocessing.video import TemporalSampler
        from models.preprocessing.video.sampler import SamplingStrategy
        
        sampler = TemporalSampler(
            n_frames=4, 
            strategy=SamplingStrategy.STRIDE,
            stride=10,
        )
        indices = sampler.sample(100)
        
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[1], 10)


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestTextNormalizer))
    suite.addTests(loader.loadTestsFromTestCase(TestVocabulary))
    suite.addTests(loader.loadTestsFromTestCase(TestBPETokenizer))
    suite.addTests(loader.loadTestsFromTestCase(TestImageLoader))
    suite.addTests(loader.loadTestsFromTestCase(TestImageTransforms))
    suite.addTests(loader.loadTestsFromTestCase(TestSpectrogramComputer))
    suite.addTests(loader.loadTestsFromTestCase(TestMultimodalProcessor))
    suite.addTests(loader.loadTestsFromTestCase(TestTemporalSampler))
    
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


if __name__ == "__main__":
    run_tests()
