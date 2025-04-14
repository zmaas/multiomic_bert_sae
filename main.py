#!/usr/bin/env python3
"""
Main script for running the nascent transcription encoder model.

This script provides a command-line interface for training and using the model.
"""

import argparse
import os
import sys
import torch
import numpy as np
import logging
from pathlib import Path
from datetime import datetime

from nasencoder.data.loader import GenomicDataLoader, NascentDataset
from nasencoder.models.encoder import NascentEncoder, SparseAutoencoder
from nasencoder.models.loss import MixedLoss, SparseAutoencoderLoss
from nasencoder.training.trainer import ModelTrainer, SparseAutoencoderTrainer
from nasencoder.utils.analysis import FeatureAnalyzer
from nasencoder.utils.visualization import visualize_batch, plot_loss_curves


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('nasencoder.log')
    ]
)

logger = logging.getLogger('main')


def setup_parser():
    """Set up the argument parser."""
    parser = argparse.ArgumentParser(description='Nascent Transcription Encoder')
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--config', type=str, help='Config file for training')
    train_parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    train_parser.add_argument('--data_dir', type=str, help='Directory containing data files')
    train_parser.add_argument('--fasta_file', type=str, help='FASTA file with reference genome')
    train_parser.add_argument('--cell_types', type=str, nargs='+', help='Cell types to use')
    train_parser.add_argument('--bam_files', type=str, nargs='+', help='BAM files for each cell type')
    train_parser.add_argument('--region_size', type=int, default=256, help='Size of genomic regions')
    train_parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    train_parser.add_argument('--mask_ratio', type=float, nargs=2, default=[0.15, 0.4], help='Range of masking ratio')
    train_parser.add_argument('--dim', type=int, default=384, help='Model dimension')
    train_parser.add_argument('--num_layers', type=int, default=6, help='Number of transformer layers')
    train_parser.add_argument('--num_heads', type=int, default=6, help='Number of attention heads')
    train_parser.add_argument('--hidden_dim', type=int, default=1536, help='Hidden dimension')
    train_parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    train_parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    train_parser.add_argument('--weight_decay', type=float, default=1e-6, help='Weight decay')
    train_parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    train_parser.add_argument('--steps_per_epoch', type=int, default=100, help='Steps per epoch')
    train_parser.add_argument('--validate_every', type=int, default=10, help='Validate every N steps')
    train_parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    train_parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Train SAE command
    sae_parser = subparsers.add_parser('train_sae', help='Train the sparse autoencoder')
    sae_parser.add_argument('--config', type=str, help='Config file for training')
    sae_parser.add_argument('--model_path', type=str, required=True, help='Path to trained base model')
    sae_parser.add_argument('--output_dir', type=str, default='results_sae', help='Output directory')
    sae_parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension for SAE')
    sae_parser.add_argument('--sparsity_k', type=int, default=20, help='Top-k sparsity')
    sae_parser.add_argument('--sparsity_weight', type=float, default=0.1, help='Sparsity weight')
    sae_parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    sae_parser.add_argument('--weight_decay', type=float, default=1e-6, help='Weight decay')
    sae_parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    sae_parser.add_argument('--steps_per_epoch', type=int, default=50, help='Steps per epoch')
    sae_parser.add_argument('--validate_every', type=int, default=10, help='Validate every N steps')
    sae_parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    sae_parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze features learned by the sparse autoencoder')
    analyze_parser.add_argument('--model_path', type=str, required=True, help='Path to trained base model')
    analyze_parser.add_argument('--sae_path', type=str, required=True, help='Path to trained SAE model')
    analyze_parser.add_argument('--output_dir', type=str, default='analysis', help='Output directory')
    analyze_parser.add_argument('--num_batches', type=int, default=10, help='Number of batches to analyze')
    analyze_parser.add_argument('--threshold', type=float, default=3.0, help='Activation threshold (in sigmas)')
    analyze_parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    return parser


def load_checkpoint(path, device):
    """Load a checkpoint from a file."""
    logger.info(f"Loading checkpoint from {path}")
    checkpoint = torch.load(path, map_location=device)
    return checkpoint


def train(args):
    """Train the base model."""
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU instead.")
        args.device = 'cpu'
    device = torch.device(args.device)
    
    # Create dataset
    logger.info("Creating dataset...")
    if args.cell_types and args.bam_files and args.fasta_file:
        # Check if the number of cell types and BAM files match
        if len(args.cell_types) != len(args.bam_files):
            logger.error("Number of cell types and BAM files must match.")
            return
        
        # Create dataset with real data
        cell_type_data = {
            cell_type: (args.fasta_file, bam_file)
            for cell_type, bam_file in zip(args.cell_types, args.bam_files)
        }
        
        dataset = NascentDataset(
            cell_type_data=cell_type_data,
            region_size=args.region_size,
            batch_size=args.batch_size,
            mask_ratio=tuple(args.mask_ratio)
        )
    else:
        # Create mock dataset for testing
        from nasencoder.examples.simple_training import MockNascentDataset
        logger.warning("Using mock dataset for training. For real training, provide cell_types, bam_files, and fasta_file.")
        dataset = MockNascentDataset(
            num_cell_types=len(args.cell_types) if args.cell_types else 3,
            region_size=args.region_size,
            batch_size=args.batch_size
        )
    
    # Create model
    logger.info("Creating model...")
    model = NascentEncoder(
        dim=args.dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        hidden_dim=args.hidden_dim,
        num_cell_types=len(dataset.cell_types),
        max_seq_len=args.region_size,
        dropout=args.dropout
    )
    
    # Create loss function
    logger.info("Creating loss function...")
    loss_fn = MixedLoss()
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = ModelTrainer(
        model=model,
        loss_fn=loss_fn,
        dataset=dataset,
        device=device,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        log_dir=os.path.join(args.output_dir, 'logs')
    )
    
    # Train model
    logger.info("Starting training...")
    metrics = trainer.train(
        num_epochs=args.num_epochs,
        steps_per_epoch=args.steps_per_epoch,
        validate_every=args.validate_every,
        save_path=args.output_dir
    )
    
    # Plot loss curves
    logger.info("Plotting loss curves...")
    fig = plot_loss_curves(metrics)
    fig.savefig(os.path.join(args.output_dir, 'loss_curves.png'))
    
    # Visualize batch
    logger.info("Visualizing batch...")
    batch_data = dataset.get_random_batch()
    figs = visualize_batch(batch_data, indices=range(min(4, args.batch_size)))
    for i, fig in enumerate(figs):
        fig.savefig(os.path.join(args.output_dir, f'batch_{i}.png'))
    
    logger.info(f"Training completed. Results saved to {args.output_dir}")


def train_sae(args):
    """Train the sparse autoencoder."""
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU instead.")
        args.device = 'cpu'
    device = torch.device(args.device)
    
    # Load base model
    logger.info(f"Loading base model from {args.model_path}")
    checkpoint = load_checkpoint(args.model_path, device)
    
    # Recreate model architecture
    base_model = NascentEncoder(
        dim=checkpoint['model_state_dict']['sequence_embedding.conv3.weight'].shape[0],
        num_layers=len([k for k in checkpoint['model_state_dict'].keys() if 'transformer_layers' in k and 'norm1.weight' in k]),
        num_heads=checkpoint['model_state_dict']['transformer_layers.0.attention.q_proj.weight'].shape[0] // checkpoint['model_state_dict']['transformer_layers.0.attention.q_proj.weight'].shape[1],
        hidden_dim=checkpoint['model_state_dict']['transformer_layers.0.feed_forward.net.0.weight'].shape[0],
        num_cell_types=checkpoint['model_state_dict']['cell_type_head.weight'].shape[0],
        max_seq_len=checkpoint['model_state_dict']['sequence_head.weight'].shape[0] // 4
    )
    
    # Load model weights
    base_model.load_state_dict(checkpoint['model_state_dict'])
    base_model = base_model.to(device)
    base_model.eval()  # Set to evaluation mode
    
    # Create mock dataset for testing
    from nasencoder.examples.simple_training import MockNascentDataset
    logger.info("Creating mock dataset...")
    dataset = MockNascentDataset(
        num_cell_types=checkpoint['model_state_dict']['cell_type_head.weight'].shape[0],
        region_size=checkpoint['model_state_dict']['sequence_head.weight'].shape[0] // 4,
        batch_size=args.steps_per_epoch
    )
    
    # Create sparse autoencoder
    logger.info("Creating sparse autoencoder...")
    input_dim = base_model.transformer_layers[-1].feed_forward.net[0].out_features
    sae_model = SparseAutoencoder(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        sparsity_k=args.sparsity_k
    )
    sae_model = sae_model.to(device)
    
    # Create loss function
    logger.info("Creating loss function...")
    loss_fn = SparseAutoencoderLoss(sparsity_weight=args.sparsity_weight)
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = SparseAutoencoderTrainer(
        model=sae_model,
        loss_fn=loss_fn,
        base_model=base_model,
        dataset=dataset,
        device=device,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        log_dir=os.path.join(args.output_dir, 'logs')
    )
    
    # Train model
    logger.info("Starting training...")
    metrics = trainer.train(
        num_epochs=args.num_epochs,
        steps_per_epoch=args.steps_per_epoch,
        validate_every=args.validate_every,
        save_path=args.output_dir
    )
    
    # Plot loss curves
    logger.info("Plotting loss curves...")
    fig = plot_loss_curves(metrics, title='Sparse Autoencoder Loss')
    fig.savefig(os.path.join(args.output_dir, 'sae_loss_curves.png'))
    
    logger.info(f"Training completed. Results saved to {args.output_dir}")


def analyze(args):
    """Analyze features learned by the sparse autoencoder."""
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU instead.")
        args.device = 'cpu'
    device = torch.device(args.device)
    
    # Load base model
    logger.info(f"Loading base model from {args.model_path}")
    base_checkpoint = load_checkpoint(args.model_path, device)
    
    # Recreate model architecture
    base_model = NascentEncoder(
        dim=base_checkpoint['model_state_dict']['sequence_embedding.conv3.weight'].shape[0],
        num_layers=len([k for k in base_checkpoint['model_state_dict'].keys() if 'transformer_layers' in k and 'norm1.weight' in k]),
        num_heads=base_checkpoint['model_state_dict']['transformer_layers.0.attention.q_proj.weight'].shape[0] // base_checkpoint['model_state_dict']['transformer_layers.0.attention.q_proj.weight'].shape[1],
        hidden_dim=base_checkpoint['model_state_dict']['transformer_layers.0.feed_forward.net.0.weight'].shape[0],
        num_cell_types=base_checkpoint['model_state_dict']['cell_type_head.weight'].shape[0],
        max_seq_len=base_checkpoint['model_state_dict']['sequence_head.weight'].shape[0] // 4
    )
    
    # Load model weights
    base_model.load_state_dict(base_checkpoint['model_state_dict'])
    base_model = base_model.to(device)
    base_model.eval()  # Set to evaluation mode
    
    # Load SAE model
    logger.info(f"Loading SAE model from {args.sae_path}")
    sae_checkpoint = load_checkpoint(args.sae_path, device)
    
    # Recreate model architecture
    input_dim = base_model.transformer_layers[-1].feed_forward.net[0].out_features
    hidden_dim = sae_checkpoint['model_state_dict']['encoder.weight'].shape[0]
    
    sae_model = SparseAutoencoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        sparsity_k=20  # Default value, not actually used during analysis
    )
    
    # Load model weights
    sae_model.load_state_dict(sae_checkpoint['model_state_dict'])
    sae_model = sae_model.to(device)
    sae_model.eval()  # Set to evaluation mode
    
    # Create mock dataset for testing
    from nasencoder.examples.simple_training import MockNascentDataset
    logger.info("Creating mock dataset...")
    dataset = MockNascentDataset(
        num_cell_types=base_checkpoint['model_state_dict']['cell_type_head.weight'].shape[0],
        region_size=base_checkpoint['model_state_dict']['sequence_head.weight'].shape[0] // 4,
        batch_size=32
    )
    
    # Create feature analyzer
    logger.info("Creating feature analyzer...")
    analyzer = FeatureAnalyzer(
        sparse_autoencoder=sae_model,
        base_model=base_model,
        dataset=dataset
    )
    
    # Analyze activations
    logger.info("Analyzing activations...")
    activation_stats = analyzer.analyze_activations(num_batches=args.num_batches)
    
    # Log statistics
    logger.info(f"Found {len(activation_stats['feature_indices'])} features above {args.threshold} sigma threshold")
    
    # Visualize features
    logger.info("Visualizing features...")
    for feature_idx in activation_stats['feature_indices']:
        # Get attributions
        attributions = analyzer.get_feature_attributions(feature_idx, num_samples=10)
        
        # Visualize feature
        fig = analyzer.visualize_feature(feature_idx, attributions, save_path=args.output_dir)
    
    # Export feature attributions
    logger.info("Exporting feature attributions...")
    df = analyzer.export_feature_attributions(
        feature_indices=activation_stats['feature_indices'],
        output_path=os.path.join(args.output_dir, 'feature_attributions.csv')
    )
    
    logger.info(f"Analysis completed. Results saved to {args.output_dir}")


def main():
    """Main function."""
    # Parse arguments
    parser = setup_parser()
    args = parser.parse_args()
    
    # Run the appropriate command
    if args.command == 'train':
        train(args)
    elif args.command == 'train_sae':
        train_sae(args)
    elif args.command == 'analyze':
        analyze(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
