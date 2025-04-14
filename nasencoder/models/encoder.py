import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np


class RotaryPositionalEncoding(nn.Module):
    """
    Rotary positional encoding (RoPE) as described in the paper
    "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    """
    def __init__(self, dim, max_seq_len=256):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Create frequency bands
        self.freqs = self._precompute_freqs_cis()
    
    def _precompute_freqs_cis(self):
        """Precompute the frequency tensor for complex exponentials (cos and sin)"""
        freqs = torch.exp(
            torch.arange(0, self.dim, 2) * -(math.log(10000) / self.dim)
        )
        
        # Create position tensor
        t = torch.arange(self.max_seq_len)
        
        # Outer product of position and frequencies
        freqs = torch.outer(t, freqs)
        
        # Get complex exponentials
        freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
        
        return freqs_complex
    
    def rotate_half(self, x):
        """Rotates half of the dimensions"""
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)
    
    def apply_rotary_emb(self, x, freqs):
        """Apply rotary embeddings to input tensor"""
        # Reshape for broadcasting
        freqs = freqs.view(*[1] * (len(x.shape) - 2), *freqs.shape)
        
        # Apply rotation using complex multiplication
        x_complex = torch.view_as_complex(
            x.float().reshape(*x.shape[:-1], -1, 2)
        )
        
        x_rotated = torch.view_as_real(
            x_complex * freqs
        ).reshape(*x.shape)
        
        return x_rotated
    
    def forward(self, x):
        """Forward pass applies rotary embeddings to input tensor"""
        # Make sure we don't exceed maximum sequence length
        seq_len = x.shape[1]
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum length {self.max_seq_len}")
        
        # Get the relevant frequencies
        freqs = self.freqs[:seq_len].to(x.device)
        
        # Apply rotary embeddings
        return self.apply_rotary_emb(x, freqs)


class MultiheadAttention(nn.Module):
    """
    Multihead Attention with Rotary Positional Encoding
    """
    def __init__(self, dim, num_heads=8, dropout=0.1, max_seq_len=256):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Linear projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Rotary positional encoding
        self.rope = RotaryPositionalEncoding(self.head_dim, max_seq_len=max_seq_len)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape to (batch_size, seq_len, num_heads, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Apply rotary encodings to q and k
        q = self.rope(q)
        k = self.rope(k)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # Compute attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, v)
        
        # Transpose and reshape
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.dim)
        
        # Final projection
        output = self.o_proj(output)
        
        return output


class FeedForward(nn.Module):
    """
    Feed Forward Network with GELU activation
    """
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class TransformerLayer(nn.Module):
    """
    Transformer layer with pre-normalization
    """
    def __init__(self, dim, num_heads, hidden_dim, dropout=0.1, max_seq_len=256):
        super().__init__()
        self.attention = MultiheadAttention(dim, num_heads, dropout, max_seq_len)
        self.feed_forward = FeedForward(dim, hidden_dim, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Pre-normalization and attention
        attention_output = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attention_output)
        
        # Pre-normalization and feed forward
        feed_forward_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout(feed_forward_output)
        
        return x


class SequenceEmbedding(nn.Module):
    """
    Embedding layer for DNA sequence using 1D convolutions
    """
    def __init__(self, dim, max_seq_len=256):
        super().__init__()
        self.conv1 = nn.Conv1d(4, dim // 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(dim // 4, dim // 2, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(dim // 2, dim, kernel_size=7, padding=3)
        self.norm = nn.LayerNorm(dim)
        self.activation = nn.GELU()
    
    def forward(self, x):
        # x is (batch_size, seq_len, 4) one-hot encoded sequence
        # Transpose to (batch_size, 4, seq_len) for 1D convolution
        x = x.transpose(1, 2)
        
        # Apply convolutions
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.conv3(x)
        
        # Transpose back to (batch_size, seq_len, dim)
        x = x.transpose(1, 2)
        
        # Apply layer normalization
        x = self.norm(x)
        
        return x


class TranscriptionEmbedding(nn.Module):
    """
    Embedding layer for transcription data (positive and negative strand coverage)
    """
    def __init__(self, dim, max_seq_len=256):
        super().__init__()
        self.conv1 = nn.Conv1d(2, dim // 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(dim // 4, dim // 2, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(dim // 2, dim, kernel_size=7, padding=3)
        self.norm = nn.LayerNorm(dim)
        self.activation = nn.GELU()
    
    def forward(self, pos_coverage, neg_coverage):
        # Combine positive and negative strand coverage
        # (batch_size, seq_len) -> (batch_size, 2, seq_len)
        x = torch.stack([pos_coverage, neg_coverage], dim=1)
        
        # Apply convolutions
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.conv3(x)
        
        # Transpose to (batch_size, seq_len, dim)
        x = x.transpose(1, 2)
        
        # Apply layer normalization
        x = self.norm(x)
        
        return x


class CellTypeEmbedding(nn.Module):
    """
    Embedding layer for cell type
    """
    def __init__(self, num_cell_types, dim):
        super().__init__()
        self.embedding = nn.Linear(num_cell_types, dim)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        # x is (batch_size, num_cell_types) one-hot encoded cell type
        x = self.embedding(x)
        x = self.norm(x)
        return x


class NascentEncoder(nn.Module):
    """
    Nascent Transcription Encoder Model
    
    A BERT-like masked autoencoder that encodes sequence, transcription, and cell type data.
    """
    def __init__(
        self,
        dim=384,
        num_layers=6,
        num_heads=6,
        hidden_dim=1536,
        num_cell_types=10,
        max_seq_len=256,
        dropout=0.1
    ):
        super().__init__()
        
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.num_cell_types = num_cell_types
        
        # Embedding layers
        self.sequence_embedding = SequenceEmbedding(dim, max_seq_len)
        self.transcription_embedding = TranscriptionEmbedding(dim, max_seq_len)
        self.cell_type_embedding = CellTypeEmbedding(num_cell_types, dim)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(dim, num_heads, hidden_dim, dropout, max_seq_len)
            for _ in range(num_layers)
        ])
        
        # Output heads
        self.sequence_head = nn.Linear(dim, 4)  # 4 nucleotides
        self.pos_coverage_head = nn.Linear(dim, 1)
        self.neg_coverage_head = nn.Linear(dim, 1)
        self.cell_type_head = nn.Linear(dim, num_cell_types)
        
        # Layer normalization for the final layer
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, batch_data):
        """
        Forward pass through the model.
        
        Args:
            batch_data (dict): Batch data dictionary containing:
                - masked_sequence: (batch_size, seq_len, 4) one-hot encoded masked sequence
                - masked_pos_coverage: (batch_size, seq_len) masked positive strand coverage
                - masked_neg_coverage: (batch_size, seq_len) masked negative strand coverage
                - cell_type: (batch_size, num_cell_types) one-hot encoded cell type
                
        Returns:
            dict: Output dictionary containing reconstructed data
        """
        # Get inputs
        masked_sequence = torch.tensor(batch_data['masked_sequence'], dtype=torch.float32)
        masked_pos_coverage = torch.tensor(batch_data['masked_pos_coverage'], dtype=torch.float32)
        masked_neg_coverage = torch.tensor(batch_data['masked_neg_coverage'], dtype=torch.float32)
        cell_type = torch.tensor(batch_data['cell_type'], dtype=torch.float32)
        
        # Move to device
        device = next(self.parameters()).device
        masked_sequence = masked_sequence.to(device)
        masked_pos_coverage = masked_pos_coverage.to(device)
        masked_neg_coverage = masked_neg_coverage.to(device)
        cell_type = cell_type.to(device)
        
        # Get embeddings
        seq_emb = self.sequence_embedding(masked_sequence)
        trans_emb = self.transcription_embedding(masked_pos_coverage, masked_neg_coverage)
        
        # Expand cell type embedding to match sequence length
        cell_emb = self.cell_type_embedding(cell_type)
        cell_emb = cell_emb.unsqueeze(1).expand(-1, self.max_seq_len, -1)
        
        # Combine embeddings in the residual stream
        x = seq_emb + trans_emb + cell_emb
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Apply final layer normalization
        x = self.norm(x)
        
        # Apply output heads
        seq_output = self.sequence_head(x)
        pos_output = self.pos_coverage_head(x).squeeze(-1)
        neg_output = self.neg_coverage_head(x).squeeze(-1)
        cell_output = self.cell_type_head(x.mean(dim=1))  # Pool over sequence length
        
        # Apply activation functions
        seq_output = F.softmax(seq_output, dim=-1)
        pos_output = F.relu(pos_output)
        neg_output = F.relu(neg_output)
        cell_output = F.softmax(cell_output, dim=-1)
        
        # Normalize coverage to sum to 1
        coverage_sum = pos_output.sum(dim=1, keepdim=True) + neg_output.sum(dim=1, keepdim=True)
        pos_output = pos_output / coverage_sum
        neg_output = neg_output / coverage_sum
        
        return {
            'pred_sequence': seq_output,
            'pred_pos_coverage': pos_output,
            'pred_neg_coverage': neg_output,
            'pred_cell_type': cell_output
        }


class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder for interpretable feature extraction
    """
    def __init__(self, input_dim, hidden_dim=256, sparsity_k=20):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sparsity_k = sparsity_k
        
        # Encoder and decoder
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=False)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights of the autoencoder"""
        nn.init.orthogonal_(self.encoder.weight)
        nn.init.orthogonal_(self.decoder.weight)
    
    def topk_sparsify(self, x):
        """Apply top-k sparsification"""
        # Determine the k value for each instance in the batch
        batch_size = x.shape[0]
        
        # Create a mask for the top-k values for each instance
        topk_values, topk_indices = torch.topk(x, self.sparsity_k, dim=1)
        
        # Create a sparse tensor with only the top-k values
        sparse_x = torch.zeros_like(x)
        for i in range(batch_size):
            sparse_x[i, topk_indices[i]] = x[i, topk_indices[i]]
        
        return sparse_x
    
    def forward(self, x):
        """Forward pass through the sparse autoencoder"""
        # Encode the input
        h = self.encoder(x)
        
        # Apply top-k sparsification
        h_sparse = self.topk_sparsify(h)
        
        # Decode
        x_recon = self.decoder(h_sparse)
        
        return {
            'hidden': h,
            'hidden_sparse': h_sparse,
            'reconstructed': x_recon
        }
    
    def extract_features(self, x, threshold=3.0):
        """
        Extract features that activate above a threshold.
        
        Args:
            x: Input tensor
            threshold: Number of standard deviations above the mean
            
        Returns:
            dict: Dictionary of feature activations and indices
        """
        # Encode the input
        h = self.encoder(x)
        
        # Calculate activation statistics
        mean = h.mean(dim=0)
        std = h.std(dim=0)
        
        # Find features that activate above threshold
        active_features = (h > mean + threshold * std).float()
        
        # Get the indices of active features for each example
        active_indices = [
            torch.nonzero(active_features[i]).squeeze(1).tolist()
            for i in range(active_features.shape[0])
        ]
        
        # Get the activation values for active features
        active_values = [
            h[i, active_indices[i]].tolist()
            for i in range(active_features.shape[0])
        ]
        
        return {
            'active_indices': active_indices,
            'active_values': active_values,
            'hidden': h
        }
    
    def get_attributions(self, x, feature_idx):
        """
        Get attributions for a specific feature.
        
        Args:
            x: Input tensor
            feature_idx: Index of the feature to get attributions for
            
        Returns:
            torch.Tensor: Attribution scores
        """
        # Get the feature's decoder weights
        feature_weights = self.decoder.weight[feature_idx]
        
        # Calculate attributions as the product of input and weights
        attributions = x * feature_weights
        
        return attributions