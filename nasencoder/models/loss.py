import torch
import torch.nn as nn
import torch.nn.functional as F


class MixedLoss(nn.Module):
    """
    Mixed loss function for the nascent transcription encoder model.
    
    Combines:
    1. KL divergence for sequence prediction
    2. MSE loss for read coverage prediction
    3. KL divergence for cell type prediction
    
    Uses VQGAN-style gradient normalization.
    """
    def __init__(self, sequence_weight=1.0, coverage_weight=1.0, cell_type_weight=1.0):
        super().__init__()
        self.sequence_weight = sequence_weight
        self.coverage_weight = coverage_weight
        self.cell_type_weight = cell_type_weight
        
        # Track running averages for gradient normalization
        self.register_buffer('sequence_grad_mean', torch.tensor(0.0))
        self.register_buffer('coverage_grad_mean', torch.tensor(0.0))
        self.register_buffer('cell_type_grad_mean', torch.tensor(0.0))
        
        # Momentum for running averages
        self.momentum = 0.99
    
    def forward(self, outputs, targets, mask_indices):
        """
        Forward pass to compute the loss.
        
        Args:
            outputs (dict): Model outputs
            targets (dict): Target values
            mask_indices (list): List of indices that were masked for each example
            
        Returns:
            tuple: (total_loss, loss_components)
        """
        batch_size = outputs['pred_sequence'].shape[0]
        
        # Extract model outputs
        pred_sequence = outputs['pred_sequence']
        pred_pos_coverage = outputs['pred_pos_coverage']
        pred_neg_coverage = outputs['pred_neg_coverage']
        pred_cell_type = outputs['pred_cell_type']
        
        # Extract target values
        target_sequence = targets['sequence']
        target_pos_coverage = targets['pos_coverage']
        target_neg_coverage = targets['neg_coverage']
        target_cell_type = targets['cell_type']
        
        # Move targets to the same device as outputs
        device = pred_sequence.device
        target_sequence = torch.tensor(target_sequence, dtype=torch.float32, device=device)
        target_pos_coverage = torch.tensor(target_pos_coverage, dtype=torch.float32, device=device)
        target_neg_coverage = torch.tensor(target_neg_coverage, dtype=torch.float32, device=device)
        target_cell_type = torch.tensor(target_cell_type, dtype=torch.float32, device=device)
        
        # Create mask tensor for masked regions
        masks = []
        for i, indices in enumerate(mask_indices):
            mask = torch.zeros(pred_sequence.shape[1], dtype=torch.bool, device=device)
            mask[indices] = True
            masks.append(mask)
        
        mask_tensor = torch.stack(masks)
        
        # Compute sequence loss (KL divergence) only on masked positions
        sequence_loss = 0
        for i in range(batch_size):
            # Get masked positions for this example
            masked_positions = mask_tensor[i]
            
            if masked_positions.sum() > 0:
                # Calculate KL divergence for masked positions
                kl_div = F.kl_div(
                    torch.log(pred_sequence[i, masked_positions] + 1e-10),
                    target_sequence[i, masked_positions],
                    reduction='batchmean'
                )
                sequence_loss += kl_div
        
        # Average over batch
        sequence_loss = sequence_loss / batch_size if batch_size > 0 else 0
        
        # Compute coverage loss (MSE) only on masked positions
        coverage_loss = 0
        for i in range(batch_size):
            # Get masked positions for this example
            masked_positions = mask_tensor[i]
            
            if masked_positions.sum() > 0:
                # Calculate MSE for masked positions
                pos_mse = F.mse_loss(
                    pred_pos_coverage[i, masked_positions],
                    target_pos_coverage[i, masked_positions]
                )
                
                neg_mse = F.mse_loss(
                    pred_neg_coverage[i, masked_positions],
                    target_neg_coverage[i, masked_positions]
                )
                
                coverage_loss += (pos_mse + neg_mse) / 2
        
        # Average over batch
        coverage_loss = coverage_loss / batch_size if batch_size > 0 else 0
        
        # Compute cell type loss (KL divergence)
        cell_type_loss = F.kl_div(
            torch.log(pred_cell_type + 1e-10),
            target_cell_type,
            reduction='batchmean'
        )
        
        # Apply weights and compute total loss
        weighted_sequence_loss = self.sequence_weight * sequence_loss
        weighted_coverage_loss = self.coverage_weight * coverage_loss
        weighted_cell_type_loss = self.cell_type_weight * cell_type_loss
        
        # VQGAN-style gradient normalization
        with torch.no_grad():
            # Update running averages of gradients
            if self.training:
                # Compute gradients (hacky way to get gradient norms)
                dummy_sequence_loss = sequence_loss.clone()
                dummy_sequence_loss.backward(retain_graph=True)
                sequence_grad_norm = torch.norm(pred_sequence.grad)
                pred_sequence.grad = None
                
                dummy_coverage_loss = coverage_loss.clone()
                dummy_coverage_loss.backward(retain_graph=True)
                coverage_grad_norm = torch.norm(pred_pos_coverage.grad) + torch.norm(pred_neg_coverage.grad)
                pred_pos_coverage.grad = None
                pred_neg_coverage.grad = None
                
                dummy_cell_type_loss = cell_type_loss.clone()
                dummy_cell_type_loss.backward(retain_graph=True)
                cell_type_grad_norm = torch.norm(pred_cell_type.grad)
                pred_cell_type.grad = None
                
                # Update running averages
                self.sequence_grad_mean = self.momentum * self.sequence_grad_mean + (1 - self.momentum) * sequence_grad_norm
                self.coverage_grad_mean = self.momentum * self.coverage_grad_mean + (1 - self.momentum) * coverage_grad_norm
                self.cell_type_grad_mean = self.momentum * self.cell_type_grad_mean + (1 - self.momentum) * cell_type_grad_norm
            
            # Reweight losses based on gradient norms
            if self.sequence_grad_mean > 0 and self.coverage_grad_mean > 0 and self.cell_type_grad_mean > 0:
                weighted_sequence_loss = self.sequence_weight * sequence_loss * (self.coverage_grad_mean / self.sequence_grad_mean)
                weighted_coverage_loss = self.coverage_weight * coverage_loss
                weighted_cell_type_loss = self.cell_type_weight * cell_type_loss * (self.coverage_grad_mean / self.cell_type_grad_mean)
        
        # Compute total loss
        total_loss = weighted_sequence_loss + weighted_coverage_loss + weighted_cell_type_loss
        
        # Return total loss and individual components
        return total_loss, {
            'sequence_loss': sequence_loss.item(),
            'coverage_loss': coverage_loss.item(),
            'cell_type_loss': cell_type_loss.item(),
            'total_loss': total_loss.item()
        }


class SparseAutoencoderLoss(nn.Module):
    """
    Loss function for the sparse autoencoder.
    
    Combines:
    1. Reconstruction loss
    2. Sparsity loss
    """
    def __init__(self, sparsity_weight=0.1):
        super().__init__()
        self.sparsity_weight = sparsity_weight
        
    def forward(self, outputs, inputs):
        """
        Forward pass to compute the loss.
        
        Args:
            outputs (dict): Dictionary containing:
                - hidden: Hidden activations
                - hidden_sparse: Sparsified hidden activations
                - reconstructed: Reconstructed input
            inputs (tensor): Original input tensor
            
        Returns:
            tuple: (total_loss, loss_components)
        """
        # Extract outputs
        hidden = outputs['hidden']
        hidden_sparse = outputs['hidden_sparse']
        reconstructed = outputs['reconstructed']
        
        # Compute reconstruction loss (MSE)
        recon_loss = F.mse_loss(reconstructed, inputs)
        
        # Compute sparsity loss (L1 norm)
        sparsity_loss = torch.mean(torch.abs(hidden))
        
        # Compute total loss
        total_loss = recon_loss + self.sparsity_weight * sparsity_loss
        
        # Return total loss and individual components
        return total_loss, {
            'reconstruction_loss': recon_loss.item(),
            'sparsity_loss': sparsity_loss.item(),
            'total_loss': total_loss.item()
        }