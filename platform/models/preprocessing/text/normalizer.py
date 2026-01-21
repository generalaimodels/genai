"""
Text Normalizer Module

Unicode normalization and text cleaning utilities.
Handles emoji, special characters, and multilingual text.
"""

import unicodedata
import re
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class NormForm(Enum):
    """Unicode normalization forms."""
    NFC = "NFC"
    NFKC = "NFKC"
    NFD = "NFD"
    NFKD = "NFKD"


# Emoji regex pattern (Unicode 15.0 compatible)
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F700-\U0001F77F"  # alchemical
    "\U0001F780-\U0001F7FF"  # geometric
    "\U0001F800-\U0001F8FF"  # arrows
    "\U0001F900-\U0001F9FF"  # supplemental
    "\U0001FA00-\U0001FA6F"  # chess
    "\U0001FA70-\U0001FAFF"  # symbols
    "\U00002702-\U000027B0"  # dingbats
    "\U0001F1E0-\U0001F1FF"  # flags
    "]+",
    flags=re.UNICODE
)

# Zero-width characters
ZERO_WIDTH_CHARS = frozenset([
    '\u200b',  # zero width space
    '\u200c',  # zero width non-joiner
    '\u200d',  # zero width joiner
    '\u2060',  # word joiner
    '\ufeff',  # byte order mark
])

# Control characters to remove
CONTROL_CHARS = frozenset([
    chr(i) for i in range(32) if chr(i) not in '\t\n\r'
] + [chr(127)])


@dataclass
class NormalizerStats:
    """Statistics from normalization pass."""
    original_length: int
    normalized_length: int
    emoji_count: int
    control_chars_removed: int
    zero_width_removed: int


class TextNormalizer:
    """
    High-performance text normalizer.
    
    Features:
    - Unicode normalization (NFC/NFKC/NFD/NFKD)
    - Emoji extraction and handling
    - Control character removal
    - Whitespace normalization
    - Case folding (optional)
    
    Optimized for batch processing with minimal allocations.
    """
    
    __slots__ = (
        '_norm_form', '_lowercase', '_remove_control',
        '_remove_zero_width', '_normalize_whitespace',
        '_emoji_cache', '_char_map'
    )
    
    def __init__(
        self,
        norm_form: NormForm = NormForm.NFC,
        lowercase: bool = False,
        remove_control: bool = True,
        remove_zero_width: bool = True,
        normalize_whitespace: bool = True,
    ):
        """
        Initialize normalizer.
        
        Args:
            norm_form: Unicode normalization form
            lowercase: Apply case folding
            remove_control: Remove control characters
            remove_zero_width: Remove zero-width characters
            normalize_whitespace: Collapse multiple whitespace
        """
        self._norm_form = norm_form
        self._lowercase = lowercase
        self._remove_control = remove_control
        self._remove_zero_width = remove_zero_width
        self._normalize_whitespace = normalize_whitespace
        
        # Pre-compute character mapping for fast substitution
        self._char_map = self._build_char_map()
        self._emoji_cache: Dict[str, List[str]] = {}
    
    def _build_char_map(self) -> Dict[int, Optional[str]]:
        """Build translation table for fast character mapping."""
        char_map = {}
        
        if self._remove_control:
            for c in CONTROL_CHARS:
                char_map[ord(c)] = None
        
        if self._remove_zero_width:
            for c in ZERO_WIDTH_CHARS:
                char_map[ord(c)] = None
        
        return char_map
    
    def normalize(self, text: str) -> str:
        """
        Normalize text string.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        # Unicode normalization
        text = unicodedata.normalize(self._norm_form.value, text)
        
        # Character-level cleanup via translation table
        if self._char_map:
            text = text.translate(self._char_map)
        
        # Whitespace normalization
        if self._normalize_whitespace:
            text = ' '.join(text.split())
        
        # Case folding
        if self._lowercase:
            text = text.lower()
        
        return text
    
    def normalize_batch(self, texts: List[str]) -> List[str]:
        """
        Batch normalize texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of normalized texts
        """
        return [self.normalize(t) for t in texts]
    
    def extract_emojis(self, text: str) -> List[str]:
        """
        Extract all emojis from text.
        
        Args:
            text: Input text
            
        Returns:
            List of emoji strings
        """
        # Check cache
        cache_key = text[:100]  # Use prefix for cache key
        if cache_key in self._emoji_cache and len(text) <= 100:
            return self._emoji_cache[cache_key]
        
        emojis = EMOJI_PATTERN.findall(text)
        
        # Cache small texts
        if len(text) <= 100:
            self._emoji_cache[cache_key] = emojis
        
        return emojis
    
    def replace_emojis(
        self,
        text: str,
        replacement: str = " <emoji> "
    ) -> str:
        """
        Replace emojis with placeholder token.
        
        Args:
            text: Input text
            replacement: Replacement string
            
        Returns:
            Text with emojis replaced
        """
        return EMOJI_PATTERN.sub(replacement, text)
    
    def segment_emojis(self, text: str) -> List[Tuple[str, bool]]:
        """
        Segment text into (substring, is_emoji) tuples.
        
        Args:
            text: Input text
            
        Returns:
            List of (text_segment, is_emoji) tuples
        """
        segments = []
        last_end = 0
        
        for match in EMOJI_PATTERN.finditer(text):
            start, end = match.span()
            
            # Add non-emoji text before this match
            if start > last_end:
                segments.append((text[last_end:start], False))
            
            # Add emoji
            segments.append((match.group(), True))
            last_end = end
        
        # Add remaining text
        if last_end < len(text):
            segments.append((text[last_end:], False))
        
        return segments
    
    def analyze(self, text: str) -> NormalizerStats:
        """
        Analyze text and return normalization statistics.
        
        Args:
            text: Input text
            
        Returns:
            NormalizerStats with counts
        """
        original_length = len(text)
        emoji_count = len(self.extract_emojis(text))
        
        control_count = sum(1 for c in text if c in CONTROL_CHARS)
        zero_width_count = sum(1 for c in text if c in ZERO_WIDTH_CHARS)
        
        normalized = self.normalize(text)
        
        return NormalizerStats(
            original_length=original_length,
            normalized_length=len(normalized),
            emoji_count=emoji_count,
            control_chars_removed=control_count,
            zero_width_removed=zero_width_count,
        )
    
    def is_valid_unicode(self, text: str) -> bool:
        """
        Check if text contains only valid Unicode.
        
        Args:
            text: Input text
            
        Returns:
            True if all characters are valid Unicode
        """
        try:
            text.encode('utf-8').decode('utf-8')
            return True
        except (UnicodeEncodeError, UnicodeDecodeError):
            return False
    
    def remove_accents(self, text: str) -> str:
        """
        Remove diacritical marks (accents) from text.
        
        Args:
            text: Input text
            
        Returns:
            Text with accents removed
        """
        # Decompose into base + combining chars
        decomposed = unicodedata.normalize('NFD', text)
        # Filter out combining marks
        return ''.join(
            c for c in decomposed
            if unicodedata.category(c) != 'Mn'
        )
