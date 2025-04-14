import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.stats import norm
import pandas as pd
import logging


class FeatureAnalyzer:
    """
    Feature analysis and visualization utility.
    
    Used to analyze the features learned by the sparse autoencoder.
    """
    def __init__(self, sparse_autoencoder, base_model, dataset):
        """
        Initialize the feature analyzer.
        
        Args:
            sparse_autoencoder: The sparse autoencoder model
            base_model: The base model
            dataset: The dataset
        """
        self.sparse_autoencoder = sparse_autoencoder
        self.base_model = base_model
        self.dataset = dataset
        
        # Initialize logger
        self.logger = logging.getLogger('FeatureAnalyzer')
        self.logger.setLevel(logging.INFO)
        
        # Add console handler if not already present
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
    
    def get_activations(self, batch_data):
        """
        Get activations from the base model.
        
        Args:
            batch_data: Batch of data from the dataset
            
        Returns:
            torch.Tensor: Activations
        """
        # Set models to evaluation mode
        self.base_model.eval()
        self.sparse_autoencoder.eval()
        
        # Get device
        device = next(self.base_model.parameters()).device
        
        # Prepare inputs
        inputs = {}
        for key, value in batch_data.items():
            if key not in ['metadata', 'mask_indices']:
                inputs[key] = torch.tensor(value, dtype=torch.float32, device=device)
        
        # Forward pass through the base model
        with torch.no_grad():
            # Get activations from the base model
            self.base_model(inputs)
            
            # Get activations from the final transformer layer
            activations = self.base_model.transformer_layers[-1].feed_forward.net[0].weight
            
            # Flatten activations
            activations = activations.flatten(1)
            
            # Forward pass through the sparse autoencoder
            sae_outputs = self.sparse_autoencoder(activations)
            
            # Get sparse activations
            sparse_activations = sae_outputs['hidden_sparse']
            
        return sparse_activations.cpu().numpy()
    
    def analyze_activations(self, num_batches=10):
        """
        Analyze activations from the sparse autoencoder.
        
        Args:
            num_batches: Number of batches to analyze
            
        Returns:
            dict: Dictionary of activation statistics
        """
        # Initialize lists to store statistics
        all_activations = []
        
        # Analyze activations for multiple batches
        for _ in range(num_batches):
            # Get a batch of data
            batch_data = self.dataset.get_random_batch()
            
            # Get activations
            activations = self.get_activations(batch_data)
            
            # Add to list
            all_activations.append(activations)
        
        # Concatenate activations
        all_activations = np.vstack(all_activations)
        
        # Calculate statistics
        mean = np.mean(all_activations, axis=0)
        std = np.std(all_activations, axis=0)
        max_val = np.max(all_activations, axis=0)
        sparsity = np.mean(all_activations == 0, axis=0)
        
        # Identify features that activate above threshold
        threshold = 3.0  # 3 standard deviations
        above_threshold = max_val > (mean + threshold * std)
        
        # Get feature indices above threshold
        feature_indices = np.where(above_threshold)[0]
        
        # Log results
        self.logger.info(f'Found {len(feature_indices)} features above {threshold} sigma threshold')
        
        return {
            'activations': all_activations,
            'mean': mean,
            'std': std,
            'max': max_val,
            'sparsity': sparsity,
            'feature_indices': feature_indices
        }
    
    def get_feature_attributions(self, feature_idx, num_samples=10):
        """
        Get attributions for a specific feature.
        
        Args:
            feature_idx: Index of the feature to analyze
            num_samples: Number of samples to analyze
            
        Returns:
            dict: Dictionary of attributions
        """
        # Initialize lists to store attributions
        sequence_attributions = []
        pos_coverage_attributions = []
        neg_coverage_attributions = []
        cell_type_attributions = []
        metadata_list = []
        
        # Set models to evaluation mode
        self.base_model.eval()
        self.sparse_autoencoder.eval()
        
        # Get device
        device = next(self.base_model.parameters()).device
        
        # Get attributions for multiple samples
        for _ in range(num_samples):
            # Get a batch of data
            batch_data = self.dataset.get_random_batch()
            
            # Prepare inputs
            inputs = {}
            for key, value in batch_data.items():
                if key not in ['metadata', 'mask_indices']:
                    inputs[key] = torch.tensor(value, dtype=torch.float32, device=device)
            
            # Forward pass through the base model
            with torch.no_grad():
                # Get base model outputs
                self.base_model(inputs)
                
                # Get activations from the final transformer layer
                activations = self.base_model.transformer_layers[-1].feed_forward.net[0].weight
                
                # Flatten activations
                activations = activations.flatten(1)
                
                # Get decoder weights for the feature
                decoder_weights = self.sparse_autoencoder.decoder.weight[feature_idx]
                
                # Calculate attributions as the product of activations and decoder weights
                attributions = activations * decoder_weights.unsqueeze(0)
                
                # Project attributions back to input space
                sequence_attr = self.base_model.sequence_embedding.conv3.weight @ attributions[:, :self.base_model.sequence_embedding.conv3.weight.shape[1]]
                coverage_attr = self.base_model.transcription_embedding.conv3.weight @ attributions[:, self.base_model.sequence_embedding.conv3.weight.shape[1]:self.base_model.sequence_embedding.conv3.weight.shape[1] + self.base_model.transcription_embedding.conv3.weight.shape[1]]
                cell_type_attr = self.base_model.cell_type_embedding.embedding.weight @ attributions[:, -self.base_model.cell_type_embedding.embedding.weight.shape[0]:]
                
                # Add to lists
                sequence_attributions.append(sequence_attr.cpu().numpy())
                pos_coverage_attributions.append(coverage_attr[:, 0].cpu().numpy())
                neg_coverage_attributions.append(coverage_attr[:, 1].cpu().numpy())
                cell_type_attributions.append(cell_type_attr.cpu().numpy())
                metadata_list.extend(batch_data['metadata'])
        
        # Concatenate attributions
        sequence_attributions = np.vstack(sequence_attributions)
        pos_coverage_attributions = np.vstack(pos_coverage_attributions)
        neg_coverage_attributions = np.vstack(neg_coverage_attributions)
        cell_type_attributions = np.vstack(cell_type_attributions)
        
        return {
            'sequence_attributions': sequence_attributions,
            'pos_coverage_attributions': pos_coverage_attributions,
            'neg_coverage_attributions': neg_coverage_attributions,
            'cell_type_attributions': cell_type_attributions,
            'metadata': metadata_list
        }
    
    def visualize_feature(self, feature_idx, attributions=None, save_path=None):
        """
        Visualize attributions for a specific feature.
        
        Args:
            feature_idx: Index of the feature to visualize
            attributions: Precomputed attributions (if None, they will be computed)
            save_path: Path to save the visualization
            
        Returns:
            matplotlib.figure.Figure: The figure
        """
        # Get attributions if not provided
        if attributions is None:
            attributions = self.get_feature_attributions(feature_idx)
        
        # Create figure
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        
        # Plot sequence attributions
        sequence_attr = attributions['sequence_attributions'].mean(axis=0)
        axs[0].bar(['A', 'C', 'G', 'T'], sequence_attr)
        axs[0].set_title(f'Feature {feature_idx} - Sequence Attributions')
        axs[0].set_ylabel('Attribution')
        
        # Plot coverage attributions
        pos_attr = attributions['pos_coverage_attributions'].mean(axis=0)
        neg_attr = attributions['neg_coverage_attributions'].mean(axis=0)
        axs[1].plot(pos_attr, label='Positive Strand')
        axs[1].plot(neg_attr, label='Negative Strand')
        axs[1].set_title(f'Feature {feature_idx} - Coverage Attributions')
        axs[1].set_xlabel('Position')
        axs[1].set_ylabel('Attribution')
        axs[1].legend()
        
        # Plot cell type attributions
        cell_type_attr = attributions['cell_type_attributions'].mean(axis=0)
        axs[2].bar(range(len(cell_type_attr)), cell_type_attr)
        axs[2].set_title(f'Feature {feature_idx} - Cell Type Attributions')
        axs[2].set_xlabel('Cell Type')
        axs[2].set_ylabel('Attribution')
        axs[2].set_xticks(range(len(cell_type_attr)))
        axs[2].set_xticklabels(self.dataset.cell_types, rotation=45, ha='right')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure if requested
        if save_path:
            plt.savefig(f'{save_path}/feature_{feature_idx}.png', dpi=300, bbox_inches='tight')
        
        return fig
    
    def export_feature_attributions(self, feature_indices, output_path):
        """
        Export feature attributions to a CSV file.
        
        Args:
            feature_indices: List of feature indices to export
            output_path: Path to save the CSV file
            
        Returns:
            pd.DataFrame: Dataframe of attributions
        """
        # Initialize list to store data
        data = []
        
        # Get attributions for each feature
        for feature_idx in feature_indices:
            # Get attributions
            attributions = self.get_feature_attributions(feature_idx)
            
            # Calculate mean attributions
            mean_seq_attr = attributions['sequence_attributions'].mean(axis=0)
            mean_pos_attr = attributions['pos_coverage_attributions'].mean()
            mean_neg_attr = attributions['neg_coverage_attributions'].mean()
            mean_cell_attr = attributions['cell_type_attributions'].mean(axis=0)
            
            # Identify most attributed nucleotide and cell type
            nucleotide = ['A', 'C', 'G', 'T'][np.argmax(mean_seq_attr)]
            cell_type = self.dataset.cell_types[np.argmax(mean_cell_attr)]
            
            # Add to data
            data.append({
                'feature_idx': feature_idx,
                'top_nucleotide': nucleotide,
                'nucleotide_attribution': np.max(mean_seq_attr),
                'pos_coverage_attribution': mean_pos_attr,
                'neg_coverage_attribution': mean_neg_attr,
                'top_cell_type': cell_type,
                'cell_type_attribution': np.max(mean_cell_attr)
            })
        
        # Create dataframe
        df = pd.DataFrame(data)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        
        return df