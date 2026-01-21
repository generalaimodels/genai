"""
Vocabulary Management Module

Efficient vocabulary storage and lookup with O(1) operations.
Supports special tokens, byte-fallback, and serialization.
"""

import json
from typing import Dict, List, Optional, Set, Iterator, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import struct


@dataclass
class VocabEntry:
    """Single vocabulary entry."""
    token: str
    id: int
    frequency: int = 0
    is_special: bool = False
    is_byte: bool = False


class Vocabulary:
    """
    High-performance vocabulary with O(1) lookup.
    
    Features:
    - Bidirectional mapping (token↔id)
    - Special token handling
    - Byte-fallback tokens (256 byte tokens)
    - Frequency tracking
    - Binary serialization for fast loading
    
    Memory-optimized with __slots__ and compact storage.
    """
    
    __slots__ = (
        '_token_to_id', '_id_to_token', '_special_tokens',
        '_byte_tokens', '_frequencies', '_size', '_frozen'
    )
    
    # Reserved ID ranges
    PAD_ID = 0
    UNK_ID = 1
    BOS_ID = 2
    EOS_ID = 3
    MASK_ID = 4
    BYTE_START_ID = 256  # Bytes occupy IDs 256-511
    
    def __init__(
        self,
        special_tokens: Optional[List[str]] = None,
        add_byte_tokens: bool = True,
    ):
        """
        Initialize vocabulary.
        
        Args:
            special_tokens: List of special tokens
            add_byte_tokens: Add 256 byte-fallback tokens
        """
        self._token_to_id: Dict[str, int] = {}
        self._id_to_token: Dict[int, str] = {}
        self._special_tokens: Set[str] = set()
        self._byte_tokens: Set[str] = set()
        self._frequencies: Dict[str, int] = {}
        self._size = 0
        self._frozen = False
        
        # Add default special tokens
        default_specials = special_tokens or [
            "<pad>", "<unk>", "<s>", "</s>", "<mask>"
        ]
        for token in default_specials:
            self._add_special_token(token)
        
        # Add byte tokens for fallback
        if add_byte_tokens:
            self._add_byte_tokens()
    
    def _add_special_token(self, token: str) -> int:
        """Add special token and return its ID."""
        if token in self._token_to_id:
            return self._token_to_id[token]
        
        token_id = self._size
        self._token_to_id[token] = token_id
        self._id_to_token[token_id] = token
        self._special_tokens.add(token)
        self._size += 1
        return token_id
    
    def _add_byte_tokens(self) -> None:
        """Add 256 byte-level tokens for fallback."""
        for byte_val in range(256):
            token = f"<0x{byte_val:02X}>"
            token_id = self._size
            self._token_to_id[token] = token_id
            self._id_to_token[token_id] = token
            self._byte_tokens.add(token)
            self._size += 1
    
    def add_token(self, token: str, frequency: int = 1) -> int:
        """
        Add token to vocabulary.
        
        Args:
            token: Token string
            frequency: Token frequency count
            
        Returns:
            Token ID
        """
        if self._frozen:
            raise RuntimeError("Vocabulary is frozen, cannot add tokens")
        
        if token in self._token_to_id:
            self._frequencies[token] = self._frequencies.get(token, 0) + frequency
            return self._token_to_id[token]
        
        token_id = self._size
        self._token_to_id[token] = token_id
        self._id_to_token[token_id] = token
        self._frequencies[token] = frequency
        self._size += 1
        return token_id
    
    def add_tokens(self, tokens: List[Tuple[str, int]]) -> List[int]:
        """
        Batch add tokens with frequencies.
        
        Args:
            tokens: List of (token, frequency) tuples
            
        Returns:
            List of token IDs
        """
        return [self.add_token(t, f) for t, f in tokens]
    
    def freeze(self) -> None:
        """Freeze vocabulary, preventing further additions."""
        self._frozen = True
    
    def unfreeze(self) -> None:
        """Unfreeze vocabulary, allowing additions."""
        self._frozen = False
    
    @property
    def size(self) -> int:
        """Get vocabulary size."""
        return self._size
    
    def __len__(self) -> int:
        return self._size
    
    def __contains__(self, token: str) -> bool:
        return token in self._token_to_id
    
    def __getitem__(self, token: str) -> int:
        return self._token_to_id.get(token, self.UNK_ID)
    
    def token_to_id(self, token: str) -> int:
        """
        Convert token to ID.
        
        Args:
            token: Token string
            
        Returns:
            Token ID (UNK_ID if not found)
        """
        return self._token_to_id.get(token, self.UNK_ID)
    
    def id_to_token(self, token_id: int) -> str:
        """
        Convert ID to token.
        
        Args:
            token_id: Token ID
            
        Returns:
            Token string (UNK token if not found)
        """
        return self._id_to_token.get(token_id, "<unk>")
    
    def tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Batch convert tokens to IDs."""
        return [self._token_to_id.get(t, self.UNK_ID) for t in tokens]
    
    def ids_to_tokens(self, ids: List[int]) -> List[str]:
        """Batch convert IDs to tokens."""
        return [self._id_to_token.get(i, "<unk>") for i in ids]
    
    def is_special(self, token: str) -> bool:
        """Check if token is special."""
        return token in self._special_tokens
    
    def is_byte(self, token: str) -> bool:
        """Check if token is byte-fallback."""
        return token in self._byte_tokens
    
    def get_frequency(self, token: str) -> int:
        """Get token frequency."""
        return self._frequencies.get(token, 0)
    
    def get_special_tokens(self) -> List[str]:
        """Get list of special tokens."""
        return list(self._special_tokens)
    
    def iter_tokens(self) -> Iterator[Tuple[str, int]]:
        """Iterate over (token, id) pairs."""
        for token, token_id in self._token_to_id.items():
            yield token, token_id
    
    def prune(self, min_frequency: int) -> int:
        """
        Remove tokens below frequency threshold.
        
        Args:
            min_frequency: Minimum frequency to keep
            
        Returns:
            Number of tokens removed
        """
        if self._frozen:
            raise RuntimeError("Vocabulary is frozen")
        
        to_remove = [
            token for token, freq in self._frequencies.items()
            if freq < min_frequency and token not in self._special_tokens
            and token not in self._byte_tokens
        ]
        
        for token in to_remove:
            token_id = self._token_to_id.pop(token)
            self._id_to_token.pop(token_id)
            self._frequencies.pop(token, None)
        
        # Reindex tokens
        self._reindex()
        
        return len(to_remove)
    
    def _reindex(self) -> None:
        """Reindex token IDs to be contiguous."""
        old_tokens = list(self._token_to_id.keys())
        self._token_to_id.clear()
        self._id_to_token.clear()
        self._size = 0
        
        # Re-add in order
        for token in old_tokens:
            if token in self._special_tokens:
                self._add_special_token(token)
            elif token in self._byte_tokens:
                token_id = self._size
                self._token_to_id[token] = token_id
                self._id_to_token[token_id] = token
                self._size += 1
            else:
                self.add_token(token, self._frequencies.get(token, 0))
    
    def save(self, path: Path) -> None:
        """
        Save vocabulary to file.
        
        Args:
            path: Output file path
        """
        path = Path(path)
        
        data = {
            "version": 1,
            "size": self._size,
            "tokens": [
                {
                    "token": token,
                    "id": token_id,
                    "frequency": self._frequencies.get(token, 0),
                    "is_special": token in self._special_tokens,
                    "is_byte": token in self._byte_tokens,
                }
                for token, token_id in self._token_to_id.items()
            ]
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "Vocabulary":
        """
        Load vocabulary from file.
        
        Args:
            path: Input file path
            
        Returns:
            Loaded Vocabulary instance
        """
        path = Path(path)
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        vocab = cls(special_tokens=[], add_byte_tokens=False)
        
        for entry in data["tokens"]:
            token = entry["token"]
            token_id = entry["id"]
            
            vocab._token_to_id[token] = token_id
            vocab._id_to_token[token_id] = token
            vocab._frequencies[token] = entry.get("frequency", 0)
            
            if entry.get("is_special", False):
                vocab._special_tokens.add(token)
            if entry.get("is_byte", False):
                vocab._byte_tokens.add(token)
        
        vocab._size = len(vocab._token_to_id)
        return vocab
    
    def save_binary(self, path: Path) -> None:
        """
        Save vocabulary in binary format for fast loading.
        
        Format:
        - 4 bytes: magic number
        - 4 bytes: version
        - 4 bytes: vocab size
        - For each token:
          - 4 bytes: token length
          - N bytes: token UTF-8
          - 4 bytes: token ID
          - 4 bytes: frequency
          - 1 byte: flags (is_special, is_byte)
        """
        path = Path(path)
        
        with open(path, 'wb') as f:
            # Header
            f.write(b'VOCB')  # magic
            f.write(struct.pack('<I', 1))  # version
            f.write(struct.pack('<I', self._size))  # size
            
            # Tokens
            for token, token_id in self._token_to_id.items():
                token_bytes = token.encode('utf-8')
                flags = 0
                if token in self._special_tokens:
                    flags |= 1
                if token in self._byte_tokens:
                    flags |= 2
                
                f.write(struct.pack('<I', len(token_bytes)))
                f.write(token_bytes)
                f.write(struct.pack('<I', token_id))
                f.write(struct.pack('<I', self._frequencies.get(token, 0)))
                f.write(struct.pack('<B', flags))
    
    @classmethod
    def load_binary(cls, path: Path) -> "Vocabulary":
        """Load vocabulary from binary format."""
        path = Path(path)
        
        vocab = cls(special_tokens=[], add_byte_tokens=False)
        
        with open(path, 'rb') as f:
            # Header
            magic = f.read(4)
            if magic != b'VOCB':
                raise ValueError(f"Invalid vocabulary file: bad magic {magic}")
            
            version = struct.unpack('<I', f.read(4))[0]
            if version != 1:
                raise ValueError(f"Unsupported vocabulary version: {version}")
            
            size = struct.unpack('<I', f.read(4))[0]
            
            # Tokens
            for _ in range(size):
                token_len = struct.unpack('<I', f.read(4))[0]
                token = f.read(token_len).decode('utf-8')
                token_id = struct.unpack('<I', f.read(4))[0]
                frequency = struct.unpack('<I', f.read(4))[0]
                flags = struct.unpack('<B', f.read(1))[0]
                
                vocab._token_to_id[token] = token_id
                vocab._id_to_token[token_id] = token
                vocab._frequencies[token] = frequency
                
                if flags & 1:
                    vocab._special_tokens.add(token)
                if flags & 2:
                    vocab._byte_tokens.add(token)
        
        vocab._size = len(vocab._token_to_id)
        return vocab
