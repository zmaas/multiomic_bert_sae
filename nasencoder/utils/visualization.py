import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import torch


def plot_coverage(pos_coverage, neg_coverage, ax=None, title=None):
    """
    Plot positive and negative strand coverage.
    
    Args:
        pos_coverage (np.ndarray): Positive strand coverage
        neg_coverage (np.ndarray): Negative strand coverage
        ax (matplotlib.axes.Axes, optional): Axes to plot on
        title (str, optional): Plot title
        
    Returns:
        matplotlib.axes.Axes: The plot axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    
    # Plot coverage
    ax.plot(pos_coverage, label='Positive Strand', color='blue')
    ax.plot(neg_coverage, label='Negative Strand', color='red')
    
    # Add labels and legend
    ax.set_xlabel('Position')
    ax.set_ylabel('Coverage')
    if title:
        ax.set_title(title)
    ax.legend()
    
    return ax


def plot_sequence(sequence, ax=None, title=None):
    """
    Plot one-hot encoded sequence as a heatmap.
    
    Args:
        sequence (np.ndarray): One-hot encoded sequence of shape (seq_len, 4)
        ax (matplotlib.axes.Axes, optional): Axes to plot on
        title (str, optional): Plot title
        
    Returns:
        matplotlib.axes.Axes: The plot axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 2))
    
    # Define nucleotide colormap
    colors = ['green', 'blue', 'orange', 'red']
    cmap = LinearSegmentedColormap.from_list('ACGT', colors, N=4)
    
    # Convert one-hot encoding to indices
    sequence_indices = np.argmax(sequence, axis=1)
    
    # Create a colormap for visualizing sequence
    seq_colors = np.array([colors[i] for i in sequence_indices])
    
    # Plot sequence as colored blocks
    for i, (idx, color) in enumerate(zip(sequence_indices, seq_colors)):
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, facecolor=color, edgecolor='none'))
    
    # Add labels for nucleotides
    ax.text(-0.05, 0.5, 'A', color=colors[0], fontsize=12, ha='right', va='center', transform=ax.transAxes)
    ax.text(-0.05, 1.5, 'C', color=colors[1], fontsize=12, ha='right', va='center', transform=ax.transAxes)
    ax.text(-0.05, 2.5, 'G', color=colors[2], fontsize=12, ha='right', va='center', transform=ax.transAxes)
    ax.text(-0.05, 3.5, 'T', color=colors[3], fontsize=12, ha='right', va='center', transform=ax.transAxes)
    
    # Set limits and title
    ax.set_xlim(0, len(sequence))
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel('Position')
    if title:
        ax.set_title(title)
    
    return ax


def visualize_region(region_data, title=None, save_path=None):
    """
    Visualize a genomic region with sequence and coverage.
    
    Args:
        region_data (dict): Region data dictionary
        title (str, optional): Plot title
        save_path (str, optional): Path to save the plot
        
    Returns:
        matplotlib.figure.Figure: The figure
    """
    # Create figure
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 3])
    
    # Extract data
    sequence = region_data['sequence']
    pos_coverage = region_data['pos_coverage']
    neg_coverage = region_data['neg_coverage']
    
    # Plot sequence
    ax1 = plt.subplot(gs[0])
    plot_sequence(sequence, ax=ax1, title='DNA Sequence')
    
    # Plot coverage
    ax2 = plt.subplot(gs[1])
    plot_coverage(pos_coverage, neg_coverage, ax=ax2, title='Strand Coverage')
    
    # Add metadata title
    if title is None and 'metadata' in region_data:
        metadata = region_data['metadata']
        title = f"{metadata['chrom']}:{metadata['start']}-{metadata['end']} ({metadata['cell_type']})"
    
    if title:
        plt.suptitle(title, fontsize=14)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def visualize_batch(batch_data, indices=None, save_dir=None):
    """
    Visualize multiple regions from a batch.
    
    Args:
        batch_data (dict): Batch data dictionary
        indices (list, optional): Indices of regions to visualize
        save_dir (str, optional): Directory to save plots
        
    Returns:
        list: List of figures
    """
    # Get the batch size
    batch_size = batch_data['sequence'].shape[0]
    
    # Default to first few regions if indices not provided
    if indices is None:
        indices = range(min(4, batch_size))
    
    # Visualize each region
    figures = []
    for i in indices:
        # Extract region data
        region_data = {
            'sequence': batch_data['sequence'][i],
            'pos_coverage': batch_data['pos_coverage'][i],
            'neg_coverage': batch_data['neg_coverage'][i]
        }
        
        # Add metadata if available
        if 'metadata' in batch_data:
            region_data['metadata'] = batch_data['metadata'][i]
        
        # Create figure title
        if 'metadata' in batch_data:
            metadata = batch_data['metadata'][i]
            title = f"Region {i}: {metadata['chrom']}:{metadata['start']}-{metadata['end']} ({metadata['cell_type']})"
        else:
            title = f"Region {i}"
        
        # Create save path if requested
        save_path = None
        if save_dir:
            save_path = f"{save_dir}/region_{i}.png"
        
        # Visualize region
        fig = visualize_region(region_data, title=title, save_path=save_path)
        figures.append(fig)
    
    return figures


def plot_loss_curves(metrics, title='Training and Validation Loss', save_path=None):
    """
    Plot training and validation loss curves.
    
    Args:
        metrics (dict): Dictionary of metrics from the trainer
        title (str, optional): Plot title
        save_path (str, optional): Path to save the plot
        
    Returns:
        matplotlib.figure.Figure: The figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot loss curves
    train_loss = metrics.get('train_loss', [])
    val_loss = metrics.get('val_loss', [])
    
    epochs = range(1, len(train_loss) + 1)
    
    ax.plot(epochs, train_loss, 'b-', label='Training Loss')
    
    if val_loss:
        ax.plot(epochs, val_loss, 'r-', label='Validation Loss')
    
    # Add labels and legend
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.legend()
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Save plot if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def visualize_model_outputs(model_outputs, targets, mask_indices, indices=None, save_dir=None):
    """
    Visualize model outputs against targets.
    
    Args:
        model_outputs (dict): Model outputs
        targets (dict): Target values
        mask_indices (list): List of masked indices for each example
        indices (list, optional): Indices of examples to visualize
        save_dir (str, optional): Directory to save plots
        
    Returns:
        list: List of figures
    """
    # Convert outputs and targets to numpy arrays
    outputs = {k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in model_outputs.items()}
    targets = {k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in targets.items()}
    
    # Get the batch size
    batch_size = outputs['pred_sequence'].shape[0]
    
    # Default to first few examples if indices not provided
    if indices is None:
        indices = range(min(4, batch_size))
    
    # Visualize each example
    figures = []
    for i in indices:
        # Create figure
        fig = plt.figure(figsize=(12, 10))
        gs = gridspec.GridSpec(4, 2, height_ratios=[1, 1, 3, 3])
        
        # Extract mask indices for this example
        example_mask = np.zeros(outputs['pred_sequence'].shape[1], dtype=bool)
        example_mask[mask_indices[i]] = True
        
        # Plot target sequence
        ax1 = plt.subplot(gs[0, 0])
        plot_sequence(targets['sequence'][i], ax=ax1, title='Target Sequence')
        
        # Plot predicted sequence
        ax2 = plt.subplot(gs[0, 1])
        plot_sequence(outputs['pred_sequence'][i], ax=ax2, title='Predicted Sequence')
        
        # Highlight masked positions
        for idx in mask_indices[i]:
            ax1.add_patch(plt.Rectangle((idx, 0), 1, 1, facecolor='none', edgecolor='black', linewidth=2))
            ax2.add_patch(plt.Rectangle((idx, 0), 1, 1, facecolor='none', edgecolor='black', linewidth=2))
        
        # Plot cell type
        ax3 = plt.subplot(gs[1, 0])
        ax3.bar(range(len(targets['cell_type'][i])), targets['cell_type'][i])
        ax3.set_title('Target Cell Type')
        ax3.set_xticks(range(len(targets['cell_type'][i])))
        
        ax4 = plt.subplot(gs[1, 1])
        ax4.bar(range(len(outputs['pred_cell_type'][i])), outputs['pred_cell_type'][i])
        ax4.set_title('Predicted Cell Type')
        ax4.set_xticks(range(len(outputs['pred_cell_type'][i])))
        
        # Plot target coverage
        ax5 = plt.subplot(gs[2, 0])
        plot_coverage(targets['pos_coverage'][i], targets['neg_coverage'][i], ax=ax5, title='Target Coverage')
        
        # Highlight masked positions
        for idx in mask_indices[i]:
            ax5.axvspan(idx - 0.5, idx + 0.5, color='gray', alpha=0.3)
        
        # Plot predicted coverage
        ax6 = plt.subplot(gs[2, 1])
        plot_coverage(outputs['pred_pos_coverage'][i], outputs['pred_neg_coverage'][i], ax=ax6, title='Predicted Coverage')
        
        # Highlight masked positions
        for idx in mask_indices[i]:
            ax6.axvspan(idx - 0.5, idx + 0.5, color='gray', alpha=0.3)
        
        # Plot masked positions
        ax7 = plt.subplot(gs[3, 0])
        ax7.plot(targets['pos_coverage'][i][example_mask], label='Target Pos', marker='o', linestyle='-', color='blue')
        ax7.plot(targets['neg_coverage'][i][example_mask], label='Target Neg', marker='o', linestyle='-', color='red')
        ax7.plot(outputs['pred_pos_coverage'][i][example_mask], label='Pred Pos', marker='x', linestyle='--', color='blue')
        ax7.plot(outputs['pred_neg_coverage'][i][example_mask], label='Pred Neg', marker='x', linestyle='--', color='red')
        ax7.set_title('Coverage at Masked Positions')
        ax7.set_xlabel('Masked Position Index')
        ax7.set_ylabel('Coverage')
        ax7.legend()
        
        # Plot sequence at masked positions
        ax8 = plt.subplot(gs[3, 1])
        # Visualize nucleotide predictions at masked positions
        masked_target = targets['sequence'][i][example_mask]
        masked_pred = outputs['pred_sequence'][i][example_mask]
        
        # Get the most likely nucleotide for each position
        target_bases = np.argmax(masked_target, axis=1)
        pred_bases = np.argmax(masked_pred, axis=1)
        
        # Convert to nucleotide names
        nucleotides = ['A', 'C', 'G', 'T']
        target_nucs = [nucleotides[b] for b in target_bases]
        pred_nucs = [nucleotides[b] for b in pred_bases]
        
        # Plot as a table
        table_data = []
        for t, p in zip(target_nucs, pred_nucs):
            table_data.append([t, p])
        
        ax8.table(
            cellText=table_data,
            colLabels=['Target', 'Predicted'],
            loc='center',
            cellLoc='center'
        )
        ax8.set_title('Sequence at Masked Positions')
        ax8.axis('off')
        
        # Set title for the whole figure
        plt.suptitle(f'Example {i}', fontsize=16)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save figure if requested
        if save_dir:
            plt.savefig(f'{save_dir}/example_{i}_prediction.png', dpi=300, bbox_inches='tight')
        
        figures.append(fig)
    
    return figures


def visualize_sae_features(sae_model, feature_indices, input_data=None, save_dir=None):
    """
    Visualize sparse autoencoder features.
    
    Args:
        sae_model: The sparse autoencoder model
        feature_indices (list): Indices of features to visualize
        input_data (torch.Tensor, optional): Input data to the sparse autoencoder
        save_dir (str, optional): Directory to save plots
        
    Returns:
        list: List of figures
    """
    # Convert model weights to numpy
    if isinstance(sae_model, torch.nn.Module):
        decoder_weights = sae_model.decoder.weight.detach().cpu().numpy()
    else:
        decoder_weights = sae_model.decoder.weight

    # Create a figure for each feature
    figures = []
    for feature_idx in feature_indices:
        # Get the decoder weights for this feature
        feature_weights = decoder_weights[feature_idx]
        
        # Reshape weights if needed
        if feature_weights.ndim == 1:
            # Assume the weights are for a flattened 2D input
            # Try to infer the original shape (square-ish)
            side_length = int(np.sqrt(len(feature_weights)))
            feature_weights = feature_weights[:side_length**2].reshape(side_length, side_length)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot the feature weights
        im = ax.imshow(feature_weights, cmap='viridis', aspect='auto')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Weight')
        
        # Set title and labels
        ax.set_title(f'Feature {feature_idx} Decoder Weights')
        
        # If input data is provided, we can show feature activations
        if input_data is not None:
            # Forward pass through the model
            with torch.no_grad():
                outputs = sae_model(input_data)
            
            # Get activations for this feature
            activations = outputs['hidden_sparse'][:, feature_idx].detach().cpu().numpy()
            
            # Add a histogram of activations in a small subplot
            inset_ax = fig.add_axes([0.65, 0.65, 0.3, 0.2])
            inset_ax.hist(activations, bins=20, color='salmon')
            inset_ax.set_title('Activations')
            inset_ax.set_yscale('log')
        
        # Save figure if requested
        if save_dir:
            plt.savefig(f'{save_dir}/feature_{feature_idx}.png', dpi=300, bbox_inches='tight')
        
        figures.append(fig)
    
    return figures