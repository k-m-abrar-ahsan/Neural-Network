import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from typing import Optional, Tuple, List, Union
import os
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class VAEVisualizer:
    """Visualization utilities for VAE models and results."""
    
    def __init__(self, model, device: torch.device):
        """
        Args:
            model: Trained VAE model
            device: Device to run model on
        """
        self.model = model
        self.device = device
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_reconstructions(self, data_loader: DataLoader, n_samples: int = 8, 
                           save_path: Optional[str] = None, figsize: Tuple[int, int] = (12, 6)):
        """Plot original vs reconstructed samples.
        
        Args:
            data_loader: DataLoader containing test data
            n_samples: Number of samples to visualize
            save_path: Path to save the plot
            figsize: Figure size
        """
        self.model.eval()
        
        # Get a batch of data
        data_iter = iter(data_loader)
        batch = next(data_iter)
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        
        batch = batch[:n_samples].to(self.device)
        
        with torch.no_grad():
            reconstruction, _, _ = self.model(batch)
        
        # Convert to numpy
        original = batch.cpu().numpy()
        reconstructed = reconstruction.cpu().numpy()
        
        # Determine if data is image-like
        is_image = self._is_image_data(original)
        
        if is_image:
            self._plot_image_reconstructions(original, reconstructed, save_path, figsize)
        else:
            self._plot_vector_reconstructions(original, reconstructed, save_path, figsize)
    
    def _is_image_data(self, data: np.ndarray) -> bool:
        """Check if data appears to be image data."""
        if len(data.shape) == 4:  # (batch, channels, height, width)
            return True
        elif len(data.shape) == 2:  # (batch, features)
            # Check if features could be flattened square images
            n_features = data.shape[1]
            sqrt_features = int(np.sqrt(n_features))
            return sqrt_features * sqrt_features == n_features
        return False
    
    def _plot_image_reconstructions(self, original: np.ndarray, reconstructed: np.ndarray,
                                  save_path: Optional[str], figsize: Tuple[int, int]):
        """Plot image reconstructions."""
        n_samples = original.shape[0]
        
        # Reshape if flattened
        if len(original.shape) == 2:
            img_size = int(np.sqrt(original.shape[1]))
            original = original.reshape(n_samples, img_size, img_size)
            reconstructed = reconstructed.reshape(n_samples, img_size, img_size)
        elif len(original.shape) == 4:
            # Remove channel dimension if grayscale
            if original.shape[1] == 1:
                original = original.squeeze(1)
                reconstructed = reconstructed.squeeze(1)
        
        fig, axes = plt.subplots(2, n_samples, figsize=figsize)
        
        for i in range(n_samples):
            # Original
            axes[0, i].imshow(original[i], cmap='gray')
            axes[0, i].set_title(f'Original {i+1}')
            axes[0, i].axis('off')
            
            # Reconstructed
            axes[1, i].imshow(reconstructed[i], cmap='gray')
            axes[1, i].set_title(f'Reconstructed {i+1}')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _plot_vector_reconstructions(self, original: np.ndarray, reconstructed: np.ndarray,
                                   save_path: Optional[str], figsize: Tuple[int, int]):
        """Plot vector data reconstructions."""
        n_samples = original.shape[0]
        n_features = min(original.shape[1], 10)  # Show at most 10 features
        
        fig, axes = plt.subplots(1, n_samples, figsize=figsize)
        if n_samples == 1:
            axes = [axes]
        
        for i in range(n_samples):
            x = np.arange(n_features)
            axes[i].plot(x, original[i, :n_features], 'o-', label='Original', alpha=0.7)
            axes[i].plot(x, reconstructed[i, :n_features], 's-', label='Reconstructed', alpha=0.7)
            axes[i].set_title(f'Sample {i+1}')
            axes[i].set_xlabel('Feature Index')
            axes[i].set_ylabel('Value')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_generated_samples(self, n_samples: int = 16, save_path: Optional[str] = None,
                             figsize: Tuple[int, int] = (12, 8)):
        """Plot generated samples from the model.
        
        Args:
            n_samples: Number of samples to generate
            save_path: Path to save the plot
            figsize: Figure size
        """
        self.model.eval()
        
        # Generate samples
        with torch.no_grad():
            samples = self.model.sample(n_samples, self.device)
        
        samples = samples.cpu().numpy()
        
        # Determine if data is image-like
        is_image = self._is_image_data(samples)
        
        if is_image:
            self._plot_generated_images(samples, save_path, figsize)
        else:
            self._plot_generated_vectors(samples, save_path, figsize)
    
    def _plot_generated_images(self, samples: np.ndarray, save_path: Optional[str],
                             figsize: Tuple[int, int]):
        """Plot generated image samples."""
        n_samples = samples.shape[0]
        
        # Reshape if flattened
        if len(samples.shape) == 2:
            img_size = int(np.sqrt(samples.shape[1]))
            samples = samples.reshape(n_samples, img_size, img_size)
        
        # Determine grid size
        grid_size = int(np.ceil(np.sqrt(n_samples)))
        
        fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
        axes = axes.flatten() if grid_size > 1 else [axes]
        
        for i in range(n_samples):
            axes[i].imshow(samples[i], cmap='gray')
            axes[i].set_title(f'Generated {i+1}')
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(n_samples, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _plot_generated_vectors(self, samples: np.ndarray, save_path: Optional[str],
                              figsize: Tuple[int, int]):
        """Plot generated vector samples."""
        n_samples = samples.shape[0]
        n_features = min(samples.shape[1], 10)  # Show at most 10 features
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Feature distributions
        for i in range(min(4, n_features)):
            row, col = i // 2, i % 2
            axes[row, col].hist(samples[:, i], bins=20, alpha=0.7, density=True)
            axes[row, col].set_title(f'Feature {i+1} Distribution')
            axes[row, col].set_xlabel('Value')
            axes[row, col].set_ylabel('Density')
            axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_latent_space(self, data_loader: DataLoader, labels: Optional[np.ndarray] = None,
                         method: str = 'tsne', save_path: Optional[str] = None,
                         figsize: Tuple[int, int] = (10, 8)):
        """Plot latent space representation.
        
        Args:
            data_loader: DataLoader containing data
            labels: Optional labels for coloring points
            method: Dimensionality reduction method ('tsne', 'pca')
            save_path: Path to save the plot
            figsize: Figure size
        """
        self.model.eval()
        
        # Extract latent representations
        latent_representations = []
        data_labels = []
        
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                if isinstance(batch, (list, tuple)):
                    if labels is not None:
                        data_labels.extend(batch[1].numpy())
                    batch = batch[0]
                
                batch = batch.to(self.device)
                mu, _ = self.model.encode(batch)
                latent_representations.append(mu.cpu().numpy())
        
        latent_data = np.concatenate(latent_representations, axis=0)
        
        if labels is not None and len(data_labels) == 0:
            data_labels = labels
        
        # Reduce dimensionality for visualization
        if latent_data.shape[1] > 2:
            if method == 'tsne':
                reducer = TSNE(n_components=2, random_state=42, perplexity=30)
            elif method == 'pca':
                reducer = PCA(n_components=2)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            latent_2d = reducer.fit_transform(latent_data)
        else:
            latent_2d = latent_data
        
        # Plot
        plt.figure(figsize=figsize)
        
        if data_labels is not None and len(data_labels) > 0:
            scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], 
                                c=data_labels, cmap='tab10', alpha=0.6, s=20)
            plt.colorbar(scatter)
        else:
            plt.scatter(latent_2d[:, 0], latent_2d[:, 1], alpha=0.6, s=20)
        
        plt.title(f'Latent Space Visualization ({method.upper()})')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_interpolation(self, data_loader: DataLoader, n_pairs: int = 3, 
                         n_steps: int = 10, save_path: Optional[str] = None,
                         figsize: Tuple[int, int] = (15, 8)):
        """Plot interpolations between pairs of samples.
        
        Args:
            data_loader: DataLoader containing data
            n_pairs: Number of interpolation pairs
            n_steps: Number of interpolation steps
            save_path: Path to save the plot
            figsize: Figure size
        """
        self.model.eval()
        
        # Get data
        data_iter = iter(data_loader)
        batch = next(data_iter)
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        
        batch = batch.to(self.device)
        
        fig, axes = plt.subplots(n_pairs, n_steps, figsize=figsize)
        if n_pairs == 1:
            axes = axes.reshape(1, -1)
        
        for pair_idx in range(n_pairs):
            x1 = batch[2*pair_idx:2*pair_idx+1]
            x2 = batch[2*pair_idx+1:2*pair_idx+2]
            
            # Generate interpolation
            with torch.no_grad():
                interpolations = self.model.interpolate(x1, x2, steps=n_steps)
            
            # Plot interpolation sequence
            for step_idx in range(n_steps):
                img = interpolations[step_idx].cpu().numpy().squeeze()
                
                # Handle different data types
                if len(img.shape) == 1:
                    # Vector data - reshape if possible
                    img_size = int(np.sqrt(len(img)))
                    if img_size * img_size == len(img):
                        img = img.reshape(img_size, img_size)
                        axes[pair_idx, step_idx].imshow(img, cmap='gray')
                    else:
                        # Plot as line graph
                        axes[pair_idx, step_idx].plot(img[:20])  # Show first 20 features
                        axes[pair_idx, step_idx].set_ylim([0, 1])
                else:
                    axes[pair_idx, step_idx].imshow(img, cmap='gray')
                
                axes[pair_idx, step_idx].axis('off')
                if pair_idx == 0:
                    axes[pair_idx, step_idx].set_title(f'Step {step_idx+1}')
        
        plt.suptitle('Latent Space Interpolations')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_loss_curves(self, train_losses: List[float], val_losses: List[float],
                        recon_losses: List[float], kl_losses: List[float],
                        save_path: Optional[str] = None, figsize: Tuple[int, int] = (12, 8)):
        """Plot training loss curves.
        
        Args:
            train_losses: Training losses
            val_losses: Validation losses
            recon_losses: Reconstruction losses
            kl_losses: KL divergence losses
            save_path: Path to save the plot
            figsize: Figure size
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        epochs = range(1, len(train_losses) + 1)
        
        # Total loss
        axes[0, 0].plot(epochs, train_losses, label='Train Loss', color='blue')
        axes[0, 0].plot(epochs, val_losses, label='Val Loss', color='red')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Reconstruction loss
        axes[0, 1].plot(epochs, recon_losses, label='Reconstruction Loss', color='orange')
        axes[0, 1].set_title('Reconstruction Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # KL divergence loss
        axes[1, 0].plot(epochs, kl_losses, label='KL Divergence', color='green')
        axes[1, 0].set_title('KL Divergence Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Loss components comparison
        axes[1, 1].plot(epochs, recon_losses, label='Reconstruction', color='orange')
        axes[1, 1].plot(epochs, kl_losses, label='KL Divergence', color='green')
        axes[1, 1].set_title('Loss Components')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_latent_dimensions(self, data_loader: DataLoader, save_path: Optional[str] = None,
                             figsize: Tuple[int, int] = (12, 8)):
        """Plot statistics of latent dimensions.
        
        Args:
            data_loader: DataLoader containing data
            save_path: Path to save the plot
            figsize: Figure size
        """
        self.model.eval()
        
        # Extract latent representations
        latent_representations = []
        
        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                
                batch = batch.to(self.device)
                mu, logvar = self.model.encode(batch)
                latent_representations.append(mu.cpu().numpy())
        
        latent_data = np.concatenate(latent_representations, axis=0)
        
        # Compute statistics
        means = np.mean(latent_data, axis=0)
        stds = np.std(latent_data, axis=0)
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Mean values per dimension
        axes[0, 0].bar(range(len(means)), means)
        axes[0, 0].set_title('Mean Values per Latent Dimension')
        axes[0, 0].set_xlabel('Latent Dimension')
        axes[0, 0].set_ylabel('Mean Value')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Standard deviation per dimension
        axes[0, 1].bar(range(len(stds)), stds)
        axes[0, 1].set_title('Standard Deviation per Latent Dimension')
        axes[0, 1].set_xlabel('Latent Dimension')
        axes[0, 1].set_ylabel('Standard Deviation')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Distribution of first few dimensions
        n_dims_to_show = min(4, latent_data.shape[1])
        for i in range(n_dims_to_show):
            row, col = (i // 2) + (1 if i >= 2 else 0), i % 2
            if i < 2:
                row, col = 1, i
            
            axes[row, col].hist(latent_data[:, i], bins=30, alpha=0.7, density=True)
            axes[row, col].set_title(f'Latent Dimension {i+1} Distribution')
            axes[row, col].set_xlabel('Value')
            axes[row, col].set_ylabel('Density')
            axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_comprehensive_report(self, data_loader: DataLoader, 
                                  train_history: dict, evaluation_results: dict,
                                  save_dir: str = './results'):
        """Create a comprehensive visualization report.
        
        Args:
            data_loader: Test data loader
            train_history: Training history dictionary
            evaluation_results: Evaluation results dictionary
            save_dir: Directory to save visualizations
        """
        os.makedirs(save_dir, exist_ok=True)
        
        print("Generating comprehensive visualization report...")
        
        # 1. Training curves
        self.plot_loss_curves(
            train_history['train_losses'],
            train_history['val_losses'],
            train_history['recon_losses'],
            train_history['kl_losses'],
            save_path=os.path.join(save_dir, 'training_curves.png')
        )
        
        # 2. Reconstructions
        self.plot_reconstructions(
            data_loader,
            n_samples=8,
            save_path=os.path.join(save_dir, 'reconstructions.png')
        )
        
        # 3. Generated samples
        self.plot_generated_samples(
            n_samples=16,
            save_path=os.path.join(save_dir, 'generated_samples.png')
        )
        
        # 4. Latent space
        self.plot_latent_space(
            data_loader,
            method='tsne',
            save_path=os.path.join(save_dir, 'latent_space_tsne.png')
        )
        
        self.plot_latent_space(
            data_loader,
            method='pca',
            save_path=os.path.join(save_dir, 'latent_space_pca.png')
        )
        
        # 5. Interpolations
        self.plot_interpolation(
            data_loader,
            n_pairs=3,
            n_steps=10,
            save_path=os.path.join(save_dir, 'interpolations.png')
        )
        
        # 6. Latent dimensions analysis
        self.plot_latent_dimensions(
            data_loader,
            save_path=os.path.join(save_dir, 'latent_dimensions.png')
        )
        
        print(f"Visualization report saved to {save_dir}")