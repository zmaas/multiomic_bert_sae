import torch
import torch.optim as optim
import numpy as np
import time
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


class ModelTrainer:
    """
    Trainer class for the nascent transcription encoder model.
    """
    def __init__(
        self,
        model,
        loss_fn,
        dataset,
        device=None,
        learning_rate=1e-4,
        weight_decay=1e-6,
        log_dir=None
    ):
        """
        Initialize the trainer.
        
        Args:
            model: The model to train
            loss_fn: The loss function
            dataset: The dataset
            device: The device to use for training
            learning_rate: The learning rate
            weight_decay: The weight decay
            log_dir: Directory for tensorboard logs
        """
        self.model = model
        self.loss_fn = loss_fn
        self.dataset = dataset
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Initialize learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Initialize tensorboard writer
        self.writer = SummaryWriter(log_dir) if log_dir else None
        
        # Initialize logger
        self.logger = logging.getLogger('ModelTrainer')
        self.logger.setLevel(logging.INFO)
        
        # Add console handler if not already present
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
    
    def train_step(self, batch_data):
        """
        Perform a single training step.
        
        Args:
            batch_data: Batch of data from the dataset
            
        Returns:
            dict: Dictionary of metrics
        """
        # Set model to training mode
        self.model.train()
        
        # Reset gradients
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self.model(batch_data)
        
        # Compute loss
        loss, loss_components = self.loss_fn(
            outputs,
            {
                'sequence': batch_data['sequence'],
                'pos_coverage': batch_data['pos_coverage'],
                'neg_coverage': batch_data['neg_coverage'],
                'cell_type': batch_data['cell_type']
            },
            batch_data['mask_indices']
        )
        
        # Backward pass
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        # Update weights
        self.optimizer.step()
        
        return loss_components
    
    def validate(self, num_batches=10):
        """
        Validate the model.
        
        Args:
            num_batches: Number of batches to validate on
            
        Returns:
            dict: Dictionary of metrics
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize metrics
        metrics = {
            'sequence_loss': 0.0,
            'coverage_loss': 0.0,
            'cell_type_loss': 0.0,
            'total_loss': 0.0
        }
        
        # Validate on multiple batches
        with torch.no_grad():
            for _ in range(num_batches):
                # Get a batch of data
                batch_data = self.dataset.get_random_batch()
                
                # Forward pass
                outputs = self.model(batch_data)
                
                # Compute loss
                _, loss_components = self.loss_fn(
                    outputs,
                    {
                        'sequence': batch_data['sequence'],
                        'pos_coverage': batch_data['pos_coverage'],
                        'neg_coverage': batch_data['neg_coverage'],
                        'cell_type': batch_data['cell_type']
                    },
                    batch_data['mask_indices']
                )
                
                # Update metrics
                for key, value in loss_components.items():
                    metrics[key] += value / num_batches
        
        return metrics
    
    def train(self, num_epochs, steps_per_epoch=100, validate_every=10, save_path=None):
        """
        Train the model.
        
        Args:
            num_epochs: Number of epochs to train for
            steps_per_epoch: Number of steps per epoch
            validate_every: Validate every N steps
            save_path: Path to save the model
            
        Returns:
            dict: Dictionary of metrics
        """
        # Initialize metrics
        metrics = {
            'train_loss': [],
            'val_loss': [],
            'lr': []
        }
        
        # Start timer
        start_time = time.time()
        
        # Training loop
        for epoch in range(num_epochs):
            self.logger.info(f'Epoch {epoch + 1}/{num_epochs}')
            
            # Initialize epoch metrics
            epoch_metrics = {
                'sequence_loss': 0.0,
                'coverage_loss': 0.0,
                'cell_type_loss': 0.0,
                'total_loss': 0.0
            }
            
            # Training loop for this epoch
            for step in tqdm(range(steps_per_epoch), desc='Training'):
                # Get a batch of data
                batch_data = self.dataset.get_random_batch()
                
                # Perform training step
                step_metrics = self.train_step(batch_data)
                
                # Update epoch metrics
                for key, value in step_metrics.items():
                    epoch_metrics[key] += value / steps_per_epoch
                
                # Validate periodically
                if (step + 1) % validate_every == 0:
                    val_metrics = self.validate(num_batches=5)
                    
                    # Log validation metrics
                    self.logger.info(
                        f'Step {step + 1}/{steps_per_epoch}, '
                        f'Val Loss: {val_metrics["total_loss"]:.4f}, '
                        f'Seq Loss: {val_metrics["sequence_loss"]:.4f}, '
                        f'Cov Loss: {val_metrics["coverage_loss"]:.4f}, '
                        f'Cell Loss: {val_metrics["cell_type_loss"]:.4f}'
                    )
                    
                    # Log to tensorboard
                    if self.writer:
                        global_step = epoch * steps_per_epoch + step
                        self.writer.add_scalar('val/total_loss', val_metrics['total_loss'], global_step)
                        self.writer.add_scalar('val/sequence_loss', val_metrics['sequence_loss'], global_step)
                        self.writer.add_scalar('val/coverage_loss', val_metrics['coverage_loss'], global_step)
                        self.writer.add_scalar('val/cell_type_loss', val_metrics['cell_type_loss'], global_step)
            
            # Log epoch metrics
            self.logger.info(
                f'Epoch {epoch + 1}/{num_epochs}, '
                f'Train Loss: {epoch_metrics["total_loss"]:.4f}, '
                f'Seq Loss: {epoch_metrics["sequence_loss"]:.4f}, '
                f'Cov Loss: {epoch_metrics["coverage_loss"]:.4f}, '
                f'Cell Loss: {epoch_metrics["cell_type_loss"]:.4f}'
            )
            
            # Update learning rate scheduler
            self.scheduler.step(epoch_metrics['total_loss'])
            
            # Log learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            metrics['lr'].append(current_lr)
            
            # Log to tensorboard
            if self.writer:
                self.writer.add_scalar('train/total_loss', epoch_metrics['total_loss'], epoch)
                self.writer.add_scalar('train/sequence_loss', epoch_metrics['sequence_loss'], epoch)
                self.writer.add_scalar('train/coverage_loss', epoch_metrics['coverage_loss'], epoch)
                self.writer.add_scalar('train/cell_type_loss', epoch_metrics['cell_type_loss'], epoch)
                self.writer.add_scalar('lr', current_lr, epoch)
            
            # Update metrics
            metrics['train_loss'].append(epoch_metrics['total_loss'])
            
            # Validate at the end of the epoch
            val_metrics = self.validate(num_batches=10)
            metrics['val_loss'].append(val_metrics['total_loss'])
            
            # Save model if requested
            if save_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'metrics': metrics
                }, f'{save_path}/epoch_{epoch + 1}.pt')
        
        # Log training time
        training_time = time.time() - start_time
        self.logger.info(f'Training completed in {training_time / 60:.2f} minutes')
        
        return metrics


class SparseAutoencoderTrainer:
    """
    Trainer class for the sparse autoencoder.
    """
    def __init__(
        self,
        model,
        loss_fn,
        base_model,
        dataset,
        device=None,
        learning_rate=1e-4,
        weight_decay=1e-6,
        log_dir=None
    ):
        """
        Initialize the trainer.
        
        Args:
            model: The sparse autoencoder model
            loss_fn: The loss function
            base_model: The base model that produces activations
            dataset: The dataset
            device: The device to use for training
            learning_rate: The learning rate
            weight_decay: The weight decay
            log_dir: Directory for tensorboard logs
        """
        self.model = model
        self.loss_fn = loss_fn
        self.base_model = base_model
        self.dataset = dataset
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Move models to device
        self.model = self.model.to(self.device)
        self.base_model = self.base_model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Initialize tensorboard writer
        self.writer = SummaryWriter(log_dir) if log_dir else None
        
        # Initialize logger
        self.logger = logging.getLogger('SparseAutoencoderTrainer')
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
        # Set base model to evaluation mode
        self.base_model.eval()
        
        # Forward pass through the base model
        with torch.no_grad():
            outputs = self.base_model(batch_data)
        
        # Get the activations from the transformer layers
        # We use the final layer's output (you might want to customize this)
        activations = self.base_model.transformer_layers[-1].feed_forward.net[0].weight
        
        return activations.flatten(1)
    
    def train_step(self, batch_data):
        """
        Perform a single training step.
        
        Args:
            batch_data: Batch of data from the dataset
            
        Returns:
            dict: Dictionary of metrics
        """
        # Set model to training mode
        self.model.train()
        
        # Get activations from the base model
        activations = self.get_activations(batch_data)
        
        # Reset gradients
        self.optimizer.zero_grad()
        
        # Forward pass through the sparse autoencoder
        outputs = self.model(activations)
        
        # Compute loss
        loss, loss_components = self.loss_fn(outputs, activations)
        
        # Backward pass
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        # Update weights
        self.optimizer.step()
        
        return loss_components
    
    def validate(self, num_batches=10):
        """
        Validate the model.
        
        Args:
            num_batches: Number of batches to validate on
            
        Returns:
            dict: Dictionary of metrics
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize metrics
        metrics = {
            'reconstruction_loss': 0.0,
            'sparsity_loss': 0.0,
            'total_loss': 0.0
        }
        
        # Validate on multiple batches
        with torch.no_grad():
            for _ in range(num_batches):
                # Get a batch of data
                batch_data = self.dataset.get_random_batch()
                
                # Get activations from the base model
                activations = self.get_activations(batch_data)
                
                # Forward pass through the sparse autoencoder
                outputs = self.model(activations)
                
                # Compute loss
                _, loss_components = self.loss_fn(outputs, activations)
                
                # Update metrics
                for key, value in loss_components.items():
                    metrics[key] += value / num_batches
        
        return metrics
    
    def train(self, num_epochs, steps_per_epoch=100, validate_every=10, save_path=None):
        """
        Train the model.
        
        Args:
            num_epochs: Number of epochs to train for
            steps_per_epoch: Number of steps per epoch
            validate_every: Validate every N steps
            save_path: Path to save the model
            
        Returns:
            dict: Dictionary of metrics
        """
        # Initialize metrics
        metrics = {
            'train_loss': [],
            'val_loss': []
        }
        
        # Start timer
        start_time = time.time()
        
        # Training loop
        for epoch in range(num_epochs):
            self.logger.info(f'Epoch {epoch + 1}/{num_epochs}')
            
            # Initialize epoch metrics
            epoch_metrics = {
                'reconstruction_loss': 0.0,
                'sparsity_loss': 0.0,
                'total_loss': 0.0
            }
            
            # Training loop for this epoch
            for step in tqdm(range(steps_per_epoch), desc='Training'):
                # Get a batch of data
                batch_data = self.dataset.get_random_batch()
                
                # Perform training step
                step_metrics = self.train_step(batch_data)
                
                # Update epoch metrics
                for key, value in step_metrics.items():
                    epoch_metrics[key] += value / steps_per_epoch
                
                # Validate periodically
                if (step + 1) % validate_every == 0:
                    val_metrics = self.validate(num_batches=5)
                    
                    # Log validation metrics
                    self.logger.info(
                        f'Step {step + 1}/{steps_per_epoch}, '
                        f'Val Loss: {val_metrics["total_loss"]:.4f}, '
                        f'Recon Loss: {val_metrics["reconstruction_loss"]:.4f}, '
                        f'Sparsity Loss: {val_metrics["sparsity_loss"]:.4f}'
                    )
                    
                    # Log to tensorboard
                    if self.writer:
                        global_step = epoch * steps_per_epoch + step
                        self.writer.add_scalar('val/total_loss', val_metrics['total_loss'], global_step)
                        self.writer.add_scalar('val/reconstruction_loss', val_metrics['reconstruction_loss'], global_step)
                        self.writer.add_scalar('val/sparsity_loss', val_metrics['sparsity_loss'], global_step)
            
            # Log epoch metrics
            self.logger.info(
                f'Epoch {epoch + 1}/{num_epochs}, '
                f'Train Loss: {epoch_metrics["total_loss"]:.4f}, '
                f'Recon Loss: {epoch_metrics["reconstruction_loss"]:.4f}, '
                f'Sparsity Loss: {epoch_metrics["sparsity_loss"]:.4f}'
            )
            
            # Log to tensorboard
            if self.writer:
                self.writer.add_scalar('train/total_loss', epoch_metrics['total_loss'], epoch)
                self.writer.add_scalar('train/reconstruction_loss', epoch_metrics['reconstruction_loss'], epoch)
                self.writer.add_scalar('train/sparsity_loss', epoch_metrics['sparsity_loss'], epoch)
            
            # Update metrics
            metrics['train_loss'].append(epoch_metrics['total_loss'])
            
            # Validate at the end of the epoch
            val_metrics = self.validate(num_batches=10)
            metrics['val_loss'].append(val_metrics['total_loss'])
            
            # Save model if requested
            if save_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'metrics': metrics
                }, f'{save_path}/sae_epoch_{epoch + 1}.pt')
        
        # Log training time
        training_time = time.time() - start_time
        self.logger.info(f'Training completed in {training_time / 60:.2f} minutes')
        
        return metrics