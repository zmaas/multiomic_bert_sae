"""
Simple training example for the nascent transcription encoder model.

This script demonstrates how to use the model with mock data.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import tempfile
from pathlib import Path

# Add parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from nasencoder.models.encoder import NascentEncoder, SparseAutoencoder
from nasencoder.models.loss import MixedLoss, SparseAutoencoderLoss
from nasencoder.training.trainer import ModelTrainer, SparseAutoencoderTrainer
from nasencoder.utils.visualization import (
    visualize_batch, plot_loss_curves, visualize_model_outputs, visualize_sae_features
)


class MockNascentDataset:
    """Mock dataset for testing."""
    
    def __init__(self, num_cell_types=3, region_size=256, batch_size=8):
        """Initialize the mock dataset."""
        self.num_cell_types = num_cell_types
        self.region_size = region_size
        self.batch_size = batch_size
        self.cell_types = [f'cell_type{i+1}' for i in range(num_cell_types)]
    
    def get_random_batch(self):
        """Get a random batch of data."""
        batch_size = self.batch_size
        region_size = self.region_size
        
        # Create random data
        batch_data = {
            'sequence': np.zeros((batch_size, region_size, 4)),
            'masked_sequence': np.zeros((batch_size, region_size, 4)),
            'pos_coverage': np.zeros((batch_size, region_size)),
            'masked_pos_coverage': np.zeros((batch_size, region_size)),
            'neg_coverage': np.zeros((batch_size, region_size)),
            'masked_neg_coverage': np.zeros((batch_size, region_size)),
            'cell_type': np.zeros((batch_size, self.num_cell_types)),
            'mask_indices': [],
            'metadata': []
        }
        
        # Fill with random data
        for i in range(batch_size):
            # Create random sequence
            seq = np.zeros((region_size, 4))
            for j in range(region_size):
                idx = np.random.randint(0, 4)
                seq[j, idx] = 1.0
            
            # Create masked sequence with ~30% masking
            mask_indices = np.random.choice(region_size, size=int(region_size * 0.3), replace=False)
            masked_seq = seq.copy()
            masked_seq[mask_indices] = 0.0
            
            # Create random coverage
            pos_coverage = np.random.rand(region_size)
            neg_coverage = np.random.rand(region_size)
            
            # Normalize coverage
            total = pos_coverage.sum() + neg_coverage.sum()
            pos_coverage /= total
            neg_coverage /= total
            
            # Create masked coverage
            masked_pos_coverage = pos_coverage.copy()
            masked_neg_coverage = neg_coverage.copy()
            masked_pos_coverage[mask_indices] = 0.0
            masked_neg_coverage[mask_indices] = 0.0
            
            # Create random cell type
            cell_type_idx = np.random.randint(0, self.num_cell_types)
            cell_type = np.zeros(self.num_cell_types)
            cell_type[cell_type_idx] = 1.0
            
            # Add to batch
            batch_data['sequence'][i] = seq
            batch_data['masked_sequence'][i] = masked_seq
            batch_data['pos_coverage'][i] = pos_coverage
            batch_data['masked_pos_coverage'][i] = masked_pos_coverage
            batch_data['neg_coverage'][i] = neg_coverage
            batch_data['masked_neg_coverage'][i] = masked_neg_coverage
            batch_data['cell_type'][i] = cell_type
            batch_data['mask_indices'].append(mask_indices.tolist())
            
            # Add metadata
            batch_data['metadata'].append({
                'chrom': f'chr{np.random.randint(1, 23)}',
                'start': np.random.randint(0, 1000000),
                'end': np.random.randint(1000000, 2000000),
                'center': np.random.randint(500000, 1500000),
                'cell_type': self.cell_types[cell_type_idx]
            })
        
        return batch_data


def train_base_model(args):
    """Train the base model."""
    print("Creating mock dataset...")
    dataset = MockNascentDataset(
        num_cell_types=args.num_cell_types,
        region_size=args.region_size,
        batch_size=args.batch_size
    )
    
    print("Creating model...")
    model = NascentEncoder(
        dim=args.dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        hidden_dim=args.hidden_dim,
        num_cell_types=args.num_cell_types,
        max_seq_len=args.region_size,
        dropout=args.dropout
    )
    
    print("Creating loss function...")
    loss_fn = MixedLoss(
        sequence_weight=args.sequence_weight,
        coverage_weight=args.coverage_weight,
        cell_type_weight=args.cell_type_weight
    )
    
    print("Creating trainer...")
    trainer = ModelTrainer(
        model=model,
        loss_fn=loss_fn,
        dataset=dataset,
        device=args.device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        log_dir=args.log_dir
    )
    
    print("Starting training...")
    metrics = trainer.train(
        num_epochs=args.num_epochs,
        steps_per_epoch=args.steps_per_epoch,
        validate_every=args.validate_every,
        save_path=args.save_dir
    )
    
    print("Training completed.")
    
    # Plot loss curves
    print("Plotting loss curves...")
    fig = plot_loss_curves(metrics)
    plt.savefig(os.path.join(args.save_dir, 'loss_curves.png'))
    
    # Visualize a batch
    print("Visualizing a batch...")
    batch_data = dataset.get_random_batch()
    figures = visualize_batch(batch_data, indices=range(2))
    for i, fig in enumerate(figures):
        plt.figure(fig.number)
        plt.savefig(os.path.join(args.save_dir, f'batch_example_{i}.png'))
    
    # Visualize model outputs
    print("Visualizing model outputs...")
    model.eval()
    with torch.no_grad():
        outputs = model(batch_data)
    
    figures = visualize_model_outputs(
        outputs,
        {
            'sequence': batch_data['sequence'],
            'pos_coverage': batch_data['pos_coverage'],
            'neg_coverage': batch_data['neg_coverage'],
            'cell_type': batch_data['cell_type']
        },
        batch_data['mask_indices'],
        indices=range(2)
    )
    
    for i, fig in enumerate(figures):
        plt.figure(fig.number)
        plt.savefig(os.path.join(args.save_dir, f'model_output_{i}.png'))
    
    return model


def train_sparse_autoencoder(args, base_model):
    """Train the sparse autoencoder."""
    print("Creating mock dataset...")
    dataset = MockNascentDataset(
        num_cell_types=args.num_cell_types,
        region_size=args.region_size,
        batch_size=args.batch_size
    )
    
    print("Creating sparse autoencoder...")
    # Get activation dimensions from the base model
    if args.sae_input_dim is None:
        # Assuming activations are from the feed forward layer of the last transformer layer
        args.sae_input_dim = base_model.transformer_layers[-1].feed_forward.net[0].out_features
    
    sae_model = SparseAutoencoder(
        input_dim=args.sae_input_dim,
        hidden_dim=args.sae_hidden_dim,
        sparsity_k=args.sae_sparsity_k
    )
    
    print("Creating loss function...")
    loss_fn = SparseAutoencoderLoss(
        sparsity_weight=args.sae_sparsity_weight
    )
    
    print("Creating trainer...")
    trainer = SparseAutoencoderTrainer(
        model=sae_model,
        loss_fn=loss_fn,
        base_model=base_model,
        dataset=dataset,
        device=args.device,
        learning_rate=args.sae_learning_rate,
        weight_decay=args.sae_weight_decay,
        log_dir=args.log_dir
    )
    
    print("Starting training...")
    metrics = trainer.train(
        num_epochs=args.sae_num_epochs,
        steps_per_epoch=args.sae_steps_per_epoch,
        validate_every=args.sae_validate_every,
        save_path=args.save_dir
    )
    
    print("Training completed.")
    
    # Plot loss curves
    print("Plotting SAE loss curves...")
    fig = plot_loss_curves(metrics, title='Sparse Autoencoder Loss')
    plt.savefig(os.path.join(args.save_dir, 'sae_loss_curves.png'))
    
    # Visualize SAE features
    print("Visualizing SAE features...")
    batch_data = dataset.get_random_batch()
    
    # Get activations from the base model
    base_model.eval()
    with torch.no_grad():
        base_model(batch_data)
        activations = base_model.transformer_layers[-1].feed_forward.net[0].weight
        activations = activations.flatten(1)
    
    # Visualize features
    figures = visualize_sae_features(
        sae_model,
        feature_indices=range(min(5, args.sae_hidden_dim)),
        input_data=activations,
        save_dir=args.save_dir
    )
    
    for i, fig in enumerate(figures):
        plt.figure(fig.number)
        plt.savefig(os.path.join(args.save_dir, f'sae_feature_{i}.png'))
    
    return sae_model


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Train the nascent transcription encoder model')
    
    # General parameters
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for training')
    parser.add_argument('--save_dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--log_dir', type=str, default=None, help='Directory for tensorboard logs')
    
    # Dataset parameters
    parser.add_argument('--num_cell_types', type=int, default=3, help='Number of cell types')
    parser.add_argument('--region_size', type=int, default=256, help='Size of genomic regions')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    
    # Model parameters
    parser.add_argument('--dim', type=int, default=128, help='Model dimension')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension in feed forward layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Loss parameters
    parser.add_argument('--sequence_weight', type=float, default=1.0, help='Weight for sequence loss')
    parser.add_argument('--coverage_weight', type=float, default=1.0, help='Weight for coverage loss')
    parser.add_argument('--cell_type_weight', type=float, default=1.0, help='Weight for cell type loss')
    
    # Training parameters
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='Weight decay')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--steps_per_epoch', type=int, default=20, help='Steps per epoch')
    parser.add_argument('--validate_every', type=int, default=5, help='Validate every N steps')
    
    # Sparse autoencoder parameters
    parser.add_argument('--sae_input_dim', type=int, default=None, help='Input dimension for SAE')
    parser.add_argument('--sae_hidden_dim', type=int, default=64, help='Hidden dimension for SAE')
    parser.add_argument('--sae_sparsity_k', type=int, default=5, help='Top-k sparsity for SAE')
    parser.add_argument('--sae_sparsity_weight', type=float, default=0.1, help='Weight for sparsity loss')
    parser.add_argument('--sae_learning_rate', type=float, default=1e-4, help='Learning rate for SAE')
    parser.add_argument('--sae_weight_decay', type=float, default=1e-6, help='Weight decay for SAE')
    parser.add_argument('--sae_num_epochs', type=int, default=3, help='Number of epochs for SAE')
    parser.add_argument('--sae_steps_per_epoch', type=int, default=10, help='Steps per epoch for SAE')
    parser.add_argument('--sae_validate_every', type=int, default=5, help='Validate every N steps for SAE')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU instead.")
        args.device = 'cpu'
    
    args.device = torch.device(args.device)
    
    # Train base model
    base_model = train_base_model(args)
    
    # Train sparse autoencoder
    sae_model = train_sparse_autoencoder(args, base_model)
    
    print("Done!")


if __name__ == '__main__':
    main()