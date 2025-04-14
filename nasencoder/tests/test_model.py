import pytest
import torch
import numpy as np
from nasencoder.models.encoder import (
    RotaryPositionalEncoding,
    MultiheadAttention,
    FeedForward,
    TransformerLayer,
    SequenceEmbedding,
    TranscriptionEmbedding,
    CellTypeEmbedding,
    NascentEncoder,
    SparseAutoencoder
)
from nasencoder.models.loss import MixedLoss, SparseAutoencoderLoss


class TestModelComponents:
    """Tests for individual model components."""
    
    def test_rotary_positional_encoding(self):
        """Test the RotaryPositionalEncoding module."""
        batch_size = 2
        seq_len = 16
        dim = 8
        
        # Create input tensor
        x = torch.randn(batch_size, seq_len, dim)
        
        # Create module
        rope = RotaryPositionalEncoding(dim=dim, max_seq_len=seq_len)
        
        # Check forward pass
        output = rope(x)
        
        assert output.shape == x.shape
        assert not torch.allclose(output, x, atol=1e-5)  # Should be different
    
    def test_multihead_attention(self):
        """Test the MultiheadAttention module."""
        batch_size = 2
        seq_len = 16
        dim = 24
        num_heads = 3
        
        # Create input tensor
        x = torch.randn(batch_size, seq_len, dim)
        
        # Create module
        mha = MultiheadAttention(dim=dim, num_heads=num_heads)
        
        # Check forward pass
        output = mha(x)
        
        assert output.shape == x.shape
    
    def test_feed_forward(self):
        """Test the FeedForward module."""
        batch_size = 2
        seq_len = 16
        dim = 24
        hidden_dim = 96
        
        # Create input tensor
        x = torch.randn(batch_size, seq_len, dim)
        
        # Create module
        ff = FeedForward(dim=dim, hidden_dim=hidden_dim)
        
        # Check forward pass
        output = ff(x)
        
        assert output.shape == x.shape
    
    def test_transformer_layer(self):
        """Test the TransformerLayer module."""
        batch_size = 2
        seq_len = 16
        dim = 24
        num_heads = 3
        hidden_dim = 96
        
        # Create input tensor
        x = torch.randn(batch_size, seq_len, dim)
        
        # Create module
        layer = TransformerLayer(dim=dim, num_heads=num_heads, hidden_dim=hidden_dim)
        
        # Check forward pass
        output = layer(x)
        
        assert output.shape == x.shape
    
    def test_sequence_embedding(self):
        """Test the SequenceEmbedding module."""
        batch_size = 2
        seq_len = 16
        dim = 24
        
        # Create input tensor (one-hot encoded sequence)
        x = torch.zeros(batch_size, seq_len, 4)
        for i in range(batch_size):
            for j in range(seq_len):
                x[i, j, j % 4] = 1.0
        
        # Create module
        embedding = SequenceEmbedding(dim=dim)
        
        # Check forward pass
        output = embedding(x)
        
        assert output.shape == (batch_size, seq_len, dim)
    
    def test_transcription_embedding(self):
        """Test the TranscriptionEmbedding module."""
        batch_size = 2
        seq_len = 16
        dim = 24
        
        # Create input tensors
        pos_coverage = torch.rand(batch_size, seq_len)
        neg_coverage = torch.rand(batch_size, seq_len)
        
        # Create module
        embedding = TranscriptionEmbedding(dim=dim)
        
        # Check forward pass
        output = embedding(pos_coverage, neg_coverage)
        
        assert output.shape == (batch_size, seq_len, dim)
    
    def test_cell_type_embedding(self):
        """Test the CellTypeEmbedding module."""
        batch_size = 2
        num_cell_types = 3
        dim = 24
        
        # Create input tensor (one-hot encoded cell types)
        x = torch.zeros(batch_size, num_cell_types)
        for i in range(batch_size):
            x[i, i % num_cell_types] = 1.0
        
        # Create module
        embedding = CellTypeEmbedding(num_cell_types=num_cell_types, dim=dim)
        
        # Check forward pass
        output = embedding(x)
        
        assert output.shape == (batch_size, dim)


class TestNascentEncoder:
    """Tests for the NascentEncoder model."""
    
    @pytest.fixture
    def model_and_batch(self):
        """Create a model and a batch of data."""
        # Model parameters
        dim = 24
        num_layers = 2
        num_heads = 3
        hidden_dim = 96
        num_cell_types = 2
        max_seq_len = 16
        
        # Create model
        model = NascentEncoder(
            dim=dim,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            num_cell_types=num_cell_types,
            max_seq_len=max_seq_len
        )
        
        # Create a batch of data
        batch_size = 2
        batch_data = {
            'masked_sequence': np.random.rand(batch_size, max_seq_len, 4),
            'masked_pos_coverage': np.random.rand(batch_size, max_seq_len),
            'masked_neg_coverage': np.random.rand(batch_size, max_seq_len),
            'cell_type': np.eye(num_cell_types)[:batch_size],
            'sequence': np.random.rand(batch_size, max_seq_len, 4),
            'pos_coverage': np.random.rand(batch_size, max_seq_len),
            'neg_coverage': np.random.rand(batch_size, max_seq_len),
            'mask_indices': [list(range(3, 8)) for _ in range(batch_size)]
        }
        
        return model, batch_data
    
    def test_forward(self, model_and_batch):
        """Test the forward pass of the model."""
        model, batch_data = model_and_batch
        
        # Forward pass
        outputs = model(batch_data)
        
        # Check outputs
        assert 'pred_sequence' in outputs
        assert 'pred_pos_coverage' in outputs
        assert 'pred_neg_coverage' in outputs
        assert 'pred_cell_type' in outputs
        
        # Check shapes
        batch_size = len(batch_data['masked_sequence'])
        seq_len = batch_data['masked_sequence'][0].shape[0]
        num_cell_types = batch_data['cell_type'].shape[1]
        
        assert outputs['pred_sequence'].shape == (batch_size, seq_len, 4)
        assert outputs['pred_pos_coverage'].shape == (batch_size, seq_len)
        assert outputs['pred_neg_coverage'].shape == (batch_size, seq_len)
        assert outputs['pred_cell_type'].shape == (batch_size, num_cell_types)
        
        # Check that outputs sum to 1 where appropriate
        assert torch.allclose(outputs['pred_sequence'].sum(dim=2), torch.ones(batch_size, seq_len))
        assert torch.allclose(outputs['pred_cell_type'].sum(dim=1), torch.ones(batch_size))
        
        # Check coverage normalization
        total_coverage = outputs['pred_pos_coverage'].sum(dim=1) + outputs['pred_neg_coverage'].sum(dim=1)
        assert torch.allclose(total_coverage, torch.ones(batch_size))


class TestMixedLoss:
    """Tests for the MixedLoss function."""
    
    def test_loss_calculation(self):
        """Test loss calculation."""
        # Create outputs and targets
        batch_size = 2
        seq_len = 16
        num_cell_types = 2
        
        outputs = {
            'pred_sequence': torch.softmax(torch.randn(batch_size, seq_len, 4), dim=2),
            'pred_pos_coverage': torch.relu(torch.randn(batch_size, seq_len)),
            'pred_neg_coverage': torch.relu(torch.randn(batch_size, seq_len)),
            'pred_cell_type': torch.softmax(torch.randn(batch_size, num_cell_types), dim=1)
        }
        
        # Normalize coverage
        coverage_sum = outputs['pred_pos_coverage'].sum(dim=1, keepdim=True) + outputs['pred_neg_coverage'].sum(dim=1, keepdim=True)
        outputs['pred_pos_coverage'] = outputs['pred_pos_coverage'] / coverage_sum
        outputs['pred_neg_coverage'] = outputs['pred_neg_coverage'] / coverage_sum
        
        targets = {
            'sequence': torch.softmax(torch.randn(batch_size, seq_len, 4), dim=2),
            'pos_coverage': torch.softmax(torch.randn(batch_size, seq_len), dim=1),
            'neg_coverage': torch.softmax(torch.randn(batch_size, seq_len), dim=1),
            'cell_type': torch.softmax(torch.randn(batch_size, num_cell_types), dim=1)
        }
        
        mask_indices = [list(range(3, 8)) for _ in range(batch_size)]
        
        # Create loss function
        loss_fn = MixedLoss()
        
        # Calculate loss
        loss, loss_components = loss_fn(outputs, targets, mask_indices)
        
        # Check loss
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # scalar
        assert loss.item() > 0
        
        # Check loss components
        assert 'sequence_loss' in loss_components
        assert 'coverage_loss' in loss_components
        assert 'cell_type_loss' in loss_components
        assert 'total_loss' in loss_components
        
        # Check that loss components are positive
        assert loss_components['sequence_loss'] >= 0
        assert loss_components['coverage_loss'] >= 0
        assert loss_components['cell_type_loss'] >= 0
        assert loss_components['total_loss'] >= 0


class TestSparseAutoencoder:
    """Tests for the SparseAutoencoder model."""
    
    def test_forward(self):
        """Test the forward pass of the model."""
        batch_size = 2
        input_dim = 100
        hidden_dim = 50
        
        # Create input tensor
        x = torch.randn(batch_size, input_dim)
        
        # Create model
        model = SparseAutoencoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            sparsity_k=10
        )
        
        # Forward pass
        outputs = model(x)
        
        # Check outputs
        assert 'hidden' in outputs
        assert 'hidden_sparse' in outputs
        assert 'reconstructed' in outputs
        
        # Check shapes
        assert outputs['hidden'].shape == (batch_size, hidden_dim)
        assert outputs['hidden_sparse'].shape == (batch_size, hidden_dim)
        assert outputs['reconstructed'].shape == (batch_size, input_dim)
        
        # Check sparsity
        non_zero = torch.count_nonzero(outputs['hidden_sparse'], dim=1)
        assert torch.all(non_zero == 10)  # sparsity_k = 10
    
    def test_extract_features(self):
        """Test feature extraction."""
        batch_size = 10
        input_dim = 100
        hidden_dim = 50
        
        # Create input tensor
        x = torch.randn(batch_size, input_dim)
        
        # Create model
        model = SparseAutoencoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            sparsity_k=10
        )
        
        # Extract features
        features = model.extract_features(x, threshold=2.0)
        
        # Check outputs
        assert 'active_indices' in features
        assert 'active_values' in features
        assert 'hidden' in features
        
        # Check shapes
        assert len(features['active_indices']) == batch_size
        assert len(features['active_values']) == batch_size
        assert features['hidden'].shape == (batch_size, hidden_dim)
        
        # Check that active values are above threshold
        for i in range(batch_size):
            for j, idx in enumerate(features['active_indices'][i]):
                value = features['active_values'][i][j]
                assert value > 0  # Should be positive
    
    def test_get_attributions(self):
        """Test attribution calculation."""
        batch_size = 2
        input_dim = 100
        hidden_dim = 50
        
        # Create input tensor
        x = torch.randn(batch_size, input_dim)
        
        # Create model
        model = SparseAutoencoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            sparsity_k=10
        )
        
        # Get attributions for a feature
        attributions = model.get_attributions(x, feature_idx=0)
        
        # Check shape
        assert attributions.shape == x.shape


class TestSparseAutoencoderLoss:
    """Tests for the SparseAutoencoderLoss function."""
    
    def test_loss_calculation(self):
        """Test loss calculation."""
        # Create outputs and inputs
        batch_size = 2
        input_dim = 100
        hidden_dim = 50
        
        inputs = torch.randn(batch_size, input_dim)
        
        outputs = {
            'hidden': torch.randn(batch_size, hidden_dim),
            'hidden_sparse': torch.zeros(batch_size, hidden_dim),
            'reconstructed': torch.randn(batch_size, input_dim)
        }
        
        # Set some sparse activations
        for i in range(batch_size):
            indices = torch.randperm(hidden_dim)[:10]
            outputs['hidden_sparse'][i, indices] = outputs['hidden'][i, indices]
        
        # Create loss function
        loss_fn = SparseAutoencoderLoss()
        
        # Calculate loss
        loss, loss_components = loss_fn(outputs, inputs)
        
        # Check loss
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # scalar
        assert loss.item() > 0
        
        # Check loss components
        assert 'reconstruction_loss' in loss_components
        assert 'sparsity_loss' in loss_components
        assert 'total_loss' in loss_components
        
        # Check that loss components are positive
        assert loss_components['reconstruction_loss'] >= 0
        assert loss_components['sparsity_loss'] >= 0
        assert loss_components['total_loss'] >= 0