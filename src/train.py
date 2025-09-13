import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
import os
import json
from datetime import datetime
import logging
from tqdm import tqdm

from vae_model import VariationalAutoencoder
from data_utils import prepare_experiment_data
from evaluation_metrics import VAEEvaluator
from visualization import VAEVisualizer

class VAETrainer:
    """Trainer class for Variational Autoencoder."""
    
    def __init__(self, model: VariationalAutoencoder, device: torch.device,
                 learning_rate: float = 1e-3, weight_decay: float = 1e-5):
        """
        Args:
            model: VAE model to train
            device: Device to train on
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=10, factor=0.5)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.recon_losses = []
        self.kl_losses = []
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dictionary containing epoch metrics
        """
        self.model.train()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch_idx, data in enumerate(progress_bar):
            if isinstance(data, (list, tuple)):
                data = data[0]  # Handle datasets that return (data, labels)
            
            data = data.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            reconstruction, mu, logvar = self.model(data)
            
            # Compute loss
            loss, recon_loss, kl_loss = self.model.compute_loss(data, reconstruction, mu, logvar)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate losses
            batch_size = data.size(0)
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item()/batch_size:.4f}',
                'Recon': f'{recon_loss.item()/batch_size:.4f}',
                'KL': f'{kl_loss.item()/batch_size:.4f}'
            })
        
        # Calculate average losses
        avg_loss = total_loss / len(train_loader.dataset)
        avg_recon_loss = total_recon_loss / len(train_loader.dataset)
        avg_kl_loss = total_kl_loss / len(train_loader.dataset)
        
        return {
            'loss': avg_loss,
            'recon_loss': avg_recon_loss,
            'kl_loss': avg_kl_loss
        }
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary containing validation metrics
        """
        self.model.eval()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        
        with torch.no_grad():
            for data in val_loader:
                if isinstance(data, (list, tuple)):
                    data = data[0]
                
                data = data.to(self.device)
                
                # Forward pass
                reconstruction, mu, logvar = self.model(data)
                
                # Compute loss
                loss, recon_loss, kl_loss = self.model.compute_loss(data, reconstruction, mu, logvar)
                
                # Accumulate losses
                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()
        
        # Calculate average losses
        avg_loss = total_loss / len(val_loader.dataset)
        avg_recon_loss = total_recon_loss / len(val_loader.dataset)
        avg_kl_loss = total_kl_loss / len(val_loader.dataset)
        
        return {
            'loss': avg_loss,
            'recon_loss': avg_recon_loss,
            'kl_loss': avg_kl_loss
        }
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int, save_dir: str = './models', 
              early_stopping_patience: int = 20) -> Dict[str, List[float]]:
        """Train the VAE model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            save_dir: Directory to save models
            early_stopping_patience: Patience for early stopping
            
        Returns:
            Training history
        """
        os.makedirs(save_dir, exist_ok=True)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        self.logger.info(f"Starting training for {epochs} epochs")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")
        
        for epoch in range(epochs):
            self.logger.info(f"Epoch {epoch+1}/{epochs}")
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            
            # Validation
            val_metrics = self.validate_epoch(val_loader)
            
            # Update learning rate scheduler
            self.scheduler.step(val_metrics['loss'])
            
            # Store metrics
            self.train_losses.append(train_metrics['loss'])
            self.val_losses.append(val_metrics['loss'])
            self.recon_losses.append(train_metrics['recon_loss'])
            self.kl_losses.append(train_metrics['kl_loss'])
            
            # Log metrics
            self.logger.info(
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Recon Loss: {train_metrics['recon_loss']:.4f}, "
                f"KL Loss: {train_metrics['kl_loss']:.4f}, "
                f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
            )
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                self.save_model(os.path.join(save_dir, 'best_model.pth'), epoch, val_metrics['loss'])
                self.logger.info(f"New best model saved with validation loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_model(os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'), 
                              epoch, val_metrics['loss'])
        
        # Save final model
        self.save_model(os.path.join(save_dir, 'final_model.pth'), epochs-1, val_metrics['loss'])
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'recon_losses': self.recon_losses,
            'kl_losses': self.kl_losses
        }
    
    def save_model(self, path: str, epoch: int, val_loss: float):
        """Save model checkpoint.
        
        Args:
            path: Path to save the model
            epoch: Current epoch
            val_loss: Validation loss
        """
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'recon_losses': self.recon_losses,
            'kl_losses': self.kl_losses
        }, path)
    
    def load_model(self, path: str):
        """Load model checkpoint.
        
        Args:
            path: Path to the saved model
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.recon_losses = checkpoint.get('recon_losses', [])
        self.kl_losses = checkpoint.get('kl_losses', [])
        
        self.logger.info(f"Model loaded from {path}")
        return checkpoint['epoch'], checkpoint['val_loss']
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history.
        
        Args:
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Total loss
        axes[0, 0].plot(self.train_losses, label='Train Loss')
        axes[0, 0].plot(self.val_losses, label='Val Loss')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Reconstruction loss
        axes[0, 1].plot(self.recon_losses, label='Reconstruction Loss', color='orange')
        axes[0, 1].set_title('Reconstruction Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # KL divergence loss
        axes[1, 0].plot(self.kl_losses, label='KL Divergence', color='green')
        axes[1, 0].set_title('KL Divergence Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate
        if hasattr(self.scheduler, '_last_lr'):
            lrs = [group['lr'] for group in self.optimizer.param_groups]
            axes[1, 1].plot(lrs, label='Learning Rate', color='red')
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('LR')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

def main():
    """Main training function."""
    # Configuration
    config = {
        'dataset': 'mnist',  # 'mnist', 'cifar10', 'synthetic'
        'input_dim': 784,
        'hidden_dims': [512, 256],
        'latent_dim': 20,
        'beta': 1.0,
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        'batch_size': 128,
        'epochs': 100,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # Setup device
    device = torch.device(config['device'])
    print(f"Using device: {device}")
    
    # Load data
    if config['dataset'] == 'synthetic':
        train_loader, val_loader, input_dim, preprocessor = prepare_experiment_data(
            config['dataset'], 
            dataset_type='blobs',
            n_samples=2000,
            n_features=config['input_dim']
        )
        config['input_dim'] = input_dim
    else:
        train_loader, val_loader, input_dim = prepare_experiment_data(config['dataset'])
        config['input_dim'] = input_dim
    
    # Create model
    model = VariationalAutoencoder(
        input_dim=config['input_dim'],
        hidden_dims=config['hidden_dims'],
        latent_dim=config['latent_dim'],
        beta=config['beta']
    )
    
    # Create trainer
    trainer = VAETrainer(
        model=model,
        device=device,
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Train model
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['epochs']
    )
    
    # Plot training history
    trainer.plot_training_history(save_path='./results/training_history.png')
    
    # Save configuration
    os.makedirs('./results', exist_ok=True)
    with open('./results/config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()