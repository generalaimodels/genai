"""
Tokenizer Training Module

Train BPE tokenizer from scratch on custom corpus.
Supports text files, streaming datasets, and iterators.
"""

from typing import Iterator, List, Optional, Union
from pathlib import Path
from dataclasses import dataclass
import json
import os

from .tokenizer import BPETokenizer
from .vocabulary import Vocabulary
from .normalizer import TextNormalizer, NormForm


@dataclass
class TrainingConfig:
    """Tokenizer training configuration."""
    vocab_size: int = 32000
    min_frequency: int = 2
    special_tokens: List[str] = None
    normalize: bool = True
    lowercase: bool = False
    add_byte_tokens: bool = True
    show_progress: bool = True
    save_path: Optional[Path] = None
    
    def __post_init__(self):
        if self.special_tokens is None:
            self.special_tokens = ["<pad>", "<unk>", "<s>", "</s>", "<mask>"]


def text_file_iterator(
    file_paths: List[Union[str, Path]],
    encoding: str = "utf-8",
) -> Iterator[str]:
    """
    Iterate over lines from text files.
    
    Args:
        file_paths: List of file paths
        encoding: Text encoding
        
    Yields:
        Text lines
    """
    for path in file_paths:
        path = Path(path)
        if not path.exists():
            continue
        
        with open(path, "r", encoding=encoding, errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line


def jsonl_iterator(
    file_paths: List[Union[str, Path]],
    text_key: str = "text",
    encoding: str = "utf-8",
) -> Iterator[str]:
    """
    Iterate over text from JSONL files.
    
    Args:
        file_paths: List of file paths
        text_key: JSON key containing text
        encoding: Text encoding
        
    Yields:
        Text content
    """
    for path in file_paths:
        path = Path(path)
        if not path.exists():
            continue
        
        with open(path, "r", encoding=encoding, errors="ignore") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if text_key in data:
                        text = data[text_key]
                        if isinstance(text, str) and text.strip():
                            yield text.strip()
                except json.JSONDecodeError:
                    continue


def directory_iterator(
    directory: Union[str, Path],
    extensions: List[str] = None,
    recursive: bool = True,
) -> Iterator[str]:
    """
    Iterate over text from all files in directory.
    
    Args:
        directory: Directory path
        extensions: File extensions to include
        recursive: Search recursively
        
    Yields:
        Text content
    """
    extensions = extensions or [".txt", ".md", ".py", ".json", ".jsonl"]
    directory = Path(directory)
    
    if recursive:
        files = list(directory.rglob("*"))
    else:
        files = list(directory.glob("*"))
    
    for path in files:
        if path.is_file() and path.suffix in extensions:
            if path.suffix == ".jsonl":
                yield from jsonl_iterator([path])
            else:
                yield from text_file_iterator([path])


class TokenizerTrainer:
    """
    Train BPE tokenizer from scratch.
    
    Features:
    - Multiple input sources (files, directories, iterators)
    - Configurable vocabulary size
    - Progress tracking
    - Checkpoint saving
    """
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config or TrainingConfig()
        self.normalizer = TextNormalizer(
            norm_form=NormForm.NFC,
            lowercase=self.config.lowercase,
        )
    
    def train_from_files(
        self,
        file_paths: List[Union[str, Path]],
        file_type: str = "text",
    ) -> BPETokenizer:
        """
        Train tokenizer from files.
        
        Args:
            file_paths: List of file paths
            file_type: 'text' or 'jsonl'
            
        Returns:
            Trained BPETokenizer
        """
        if file_type == "jsonl":
            iterator = jsonl_iterator(file_paths)
        else:
            iterator = text_file_iterator(file_paths)
        
        return self.train(iterator)
    
    def train_from_directory(
        self,
        directory: Union[str, Path],
        extensions: List[str] = None,
        recursive: bool = True,
    ) -> BPETokenizer:
        """
        Train tokenizer from directory.
        
        Args:
            directory: Directory path
            extensions: File extensions
            recursive: Search recursively
            
        Returns:
            Trained BPETokenizer
        """
        iterator = directory_iterator(directory, extensions, recursive)
        return self.train(iterator)
    
    def train(self, texts: Iterator[str]) -> BPETokenizer:
        """
        Train tokenizer from text iterator.
        
        Args:
            texts: Iterator yielding text strings
            
        Returns:
            Trained BPETokenizer
        """
        # Initialize vocabulary
        vocab = Vocabulary(
            special_tokens=self.config.special_tokens,
            add_byte_tokens=self.config.add_byte_tokens,
        )
        
        # Initialize tokenizer
        tokenizer = BPETokenizer(
            vocab=vocab,
            normalizer=self.normalizer if self.config.normalize else None,
        )
        
        # Train
        tokenizer.train(
            texts=texts,
            vocab_size=self.config.vocab_size,
            min_frequency=self.config.min_frequency,
            show_progress=self.config.show_progress,
        )
        
        # Freeze vocabulary
        tokenizer.vocab.freeze()
        
        # Save if path provided
        if self.config.save_path:
            tokenizer.save(self.config.save_path)
        
        return tokenizer
    
    def train_from_huggingface(
        self,
        dataset_name: str,
        text_column: str = "text",
        split: str = "train",
        max_samples: Optional[int] = None,
    ) -> BPETokenizer:
        """
        Train tokenizer from HuggingFace dataset.
        
        Args:
            dataset_name: HuggingFace dataset name
            text_column: Column containing text
            split: Dataset split
            max_samples: Maximum samples to use
            
        Returns:
            Trained BPETokenizer
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Install datasets: pip install datasets")
        
        dataset = load_dataset(dataset_name, split=split)
        
        def iterator():
            for i, example in enumerate(dataset):
                if max_samples and i >= max_samples:
                    break
                text = example.get(text_column, "")
                if isinstance(text, str) and text.strip():
                    yield text.strip()
        
        return self.train(iterator())


def train_tokenizer(
    source: Union[str, Path, List[str], Iterator[str]],
    vocab_size: int = 32000,
    save_path: Optional[Union[str, Path]] = None,
    **kwargs,
) -> BPETokenizer:
    """
    Convenience function to train tokenizer.
    
    Args:
        source: Text source (file, directory, list, or iterator)
        vocab_size: Target vocabulary size
        save_path: Path to save tokenizer
        **kwargs: Additional TrainingConfig options
        
    Returns:
        Trained BPETokenizer
    
    Example:
        # From directory
        tokenizer = train_tokenizer("./corpus/", vocab_size=32000)
        
        # From files
        tokenizer = train_tokenizer(["train.txt", "valid.txt"])
        
        # From HuggingFace
        tokenizer = train_tokenizer("wikitext", vocab_size=50000)
    """
    config = TrainingConfig(
        vocab_size=vocab_size,
        save_path=Path(save_path) if save_path else None,
        **kwargs,
    )
    
    trainer = TokenizerTrainer(config)
    
    # Determine source type
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.is_dir():
            return trainer.train_from_directory(path)
        elif path.is_file():
            return trainer.train_from_files([path])
        else:
            # Assume HuggingFace dataset name
            return trainer.train_from_huggingface(str(source))
    
    elif isinstance(source, list):
        # List of files
        return trainer.train_from_files(source)
    
    else:
        # Iterator
        return trainer.train(source)
