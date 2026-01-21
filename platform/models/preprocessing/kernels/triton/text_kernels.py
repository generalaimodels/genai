"""
Text Preprocessing Triton Kernels

Optimized kernels for tokenization and encoding operations.
"""

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TRITON:
    
    @triton.jit
    def fused_tokenize_lookup_kernel(
        token_ids_ptr,
        vocab_table_ptr,
        output_ptr,
        seq_len,
        vocab_size,
        embedding_dim,
        BLOCK_SIZE_SEQ: tl.constexpr,
        BLOCK_SIZE_EMB: tl.constexpr,
    ):
        """
        Fused token ID to embedding lookup kernel.
        
        Performs vocabulary embedding lookup with memory coalescing.
        
        Args:
            token_ids_ptr: Input token IDs (seq_len,)
            vocab_table_ptr: Vocabulary embedding table (vocab_size, embedding_dim)
            output_ptr: Output embeddings (seq_len, embedding_dim)
            seq_len: Sequence length
            vocab_size: Vocabulary size
            embedding_dim: Embedding dimension
        """
        pid_seq = tl.program_id(0)
        pid_emb = tl.program_id(1)
        
        # Sequence offsets for this block
        seq_start = pid_seq * BLOCK_SIZE_SEQ
        seq_offsets = seq_start + tl.arange(0, BLOCK_SIZE_SEQ)
        seq_mask = seq_offsets < seq_len
        
        # Embedding offsets for this block
        emb_start = pid_emb * BLOCK_SIZE_EMB
        emb_offsets = emb_start + tl.arange(0, BLOCK_SIZE_EMB)
        emb_mask = emb_offsets < embedding_dim
        
        # Load token IDs
        token_ids = tl.load(token_ids_ptr + seq_offsets, mask=seq_mask, other=0)
        
        # Clamp to valid vocab range
        token_ids = tl.minimum(token_ids, vocab_size - 1)
        token_ids = tl.maximum(token_ids, 0)
        
        # Compute embedding table offsets
        # vocab_table[token_id, emb_idx]
        for i in range(BLOCK_SIZE_SEQ):
            if seq_start + i < seq_len:
                token_id = tl.load(token_ids_ptr + seq_start + i)
                token_id = tl.minimum(token_id, vocab_size - 1)
                token_id = tl.maximum(token_id, 0)
                
                # Load embedding row
                emb_idx = token_id * embedding_dim + emb_offsets
                embeddings = tl.load(vocab_table_ptr + emb_idx, mask=emb_mask, other=0.0)
                
                # Store to output
                out_idx = (seq_start + i) * embedding_dim + emb_offsets
                tl.store(output_ptr + out_idx, embeddings, mask=emb_mask)
    
    
    @triton.jit
    def batch_encode_kernel(
        input_ptr,
        output_ptr,
        offsets_ptr,
        batch_size,
        max_seq_len,
        pad_id,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Batch encoding with padding kernel.
        
        Pads sequences to uniform length for batching.
        
        Args:
            input_ptr: Input token IDs (variable length sequences)
            output_ptr: Output padded IDs (batch_size, max_seq_len)
            offsets_ptr: Start offsets for each sequence (batch_size + 1)
            batch_size: Number of sequences
            max_seq_len: Maximum sequence length
            pad_id: Padding token ID
        """
        pid = tl.program_id(0)
        batch_idx = pid // tl.cdiv(max_seq_len, BLOCK_SIZE)
        block_idx = pid % tl.cdiv(max_seq_len, BLOCK_SIZE)
        
        if batch_idx >= batch_size:
            return
        
        # Load sequence offsets
        seq_start = tl.load(offsets_ptr + batch_idx)
        seq_end = tl.load(offsets_ptr + batch_idx + 1)
        seq_len = seq_end - seq_start
        
        # Position offsets within sequence
        pos_start = block_idx * BLOCK_SIZE
        pos_offsets = pos_start + tl.arange(0, BLOCK_SIZE)
        pos_mask = pos_offsets < max_seq_len
        
        # Determine which positions have actual tokens vs padding
        valid_mask = pos_offsets < seq_len
        
        # Load tokens or use padding
        input_idx = seq_start + pos_offsets
        tokens = tl.load(input_ptr + input_idx, mask=valid_mask & pos_mask, other=pad_id)
        
        # For positions beyond sequence length, use pad_id
        tokens = tl.where(valid_mask, tokens, pad_id)
        
        # Store to output
        output_idx = batch_idx * max_seq_len + pos_offsets
        tl.store(output_ptr + output_idx, tokens, mask=pos_mask)
    
    
    @triton.jit
    def create_attention_mask_kernel(
        seq_lens_ptr,
        output_ptr,
        batch_size,
        max_seq_len,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Create attention mask from sequence lengths.
        
        Args:
            seq_lens_ptr: Sequence lengths (batch_size,)
            output_ptr: Output attention mask (batch_size, max_seq_len)
            batch_size: Number of sequences
            max_seq_len: Maximum sequence length
        """
        pid = tl.program_id(0)
        batch_idx = pid // tl.cdiv(max_seq_len, BLOCK_SIZE)
        block_idx = pid % tl.cdiv(max_seq_len, BLOCK_SIZE)
        
        if batch_idx >= batch_size:
            return
        
        # Load sequence length
        seq_len = tl.load(seq_lens_ptr + batch_idx)
        
        # Position offsets
        pos_start = block_idx * BLOCK_SIZE
        pos_offsets = pos_start + tl.arange(0, BLOCK_SIZE)
        pos_mask = pos_offsets < max_seq_len
        
        # Create mask: 1 for valid positions, 0 for padding
        mask_vals = tl.where(pos_offsets < seq_len, 1.0, 0.0)
        
        # Store
        output_idx = batch_idx * max_seq_len + pos_offsets
        tl.store(output_ptr + output_idx, mask_vals, mask=pos_mask)


if HAS_TORCH and HAS_TRITON:
    
    def fused_embedding_lookup(
        token_ids: "torch.Tensor",
        embedding_table: "torch.Tensor",
    ) -> "torch.Tensor":
        """
        Fused embedding lookup using Triton kernel.
        
        Args:
            token_ids: Input token IDs (seq_len,) or (batch, seq_len)
            embedding_table: Embedding table (vocab_size, embedding_dim)
            
        Returns:
            Embeddings (seq_len, embedding_dim) or (batch, seq_len, embedding_dim)
        """
        is_batched = token_ids.dim() == 2
        if not is_batched:
            token_ids = token_ids.unsqueeze(0)
        
        batch_size, seq_len = token_ids.shape
        vocab_size, embedding_dim = embedding_table.shape
        
        output = torch.empty(
            (batch_size, seq_len, embedding_dim),
            device=token_ids.device,
            dtype=embedding_table.dtype,
        )
        
        BLOCK_SIZE_SEQ = 32
        BLOCK_SIZE_EMB = 128
        
        for b in range(batch_size):
            grid = (
                triton.cdiv(seq_len, BLOCK_SIZE_SEQ),
                triton.cdiv(embedding_dim, BLOCK_SIZE_EMB),
            )
            
            fused_tokenize_lookup_kernel[grid](
                token_ids[b].data_ptr(),
                embedding_table.data_ptr(),
                output[b].data_ptr(),
                seq_len,
                vocab_size,
                embedding_dim,
                BLOCK_SIZE_SEQ=BLOCK_SIZE_SEQ,
                BLOCK_SIZE_EMB=BLOCK_SIZE_EMB,
            )
        
        if not is_batched:
            output = output.squeeze(0)
        
        return output
