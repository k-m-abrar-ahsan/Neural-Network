import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import numpy as np
from typing import Tuple, Optional

class Encoder(nn.Module):
    """Encoder network for VAE that maps input to latent distribution parameters."""
    
    def __init__(self, input_dim: int, hidden_dims: list, latent_dim: int):
        super(Encoder, self).__init__()
        
        # Build encoder layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Latent distribution parameters
        self.mu_layer = nn.Linear(prev_dim, latent_dim)
        self.logvar_layer = nn.Linear(prev_dim, latent_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through encoder.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            mu: Mean of latent distribution (batch_size, latent_dim)
            logvar: Log variance of latent distribution (batch_size, latent_dim)
        """
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar

class Decoder(nn.Module):
    """Decoder network for VAE that reconstructs input from latent representation."""
    
    def __init__(self, latent_dim: int, hidden_dims: list, output_dim: int):
        super(Decoder, self).__init__()
        
        # Build decoder layers (reverse of encoder)
        layers = []
        prev_dim = latent_dim
        
        for hidden_dim in reversed(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Sigmoid())  # For normalized data [0,1]
        
        self.decoder = nn.Sequential(*layers)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass through decoder.
        
        Args:
            z: Latent representation (batch_size, latent_dim)
            
        Returns:
            reconstruction: Reconstructed input (batch_size, output_dim)
        """
        return self.decoder(z)

class VariationalAutoencoder(nn.Module):
    """Variational Autoencoder with stochastic sampling and reparameterization trick."""
    
    def __init__(self, input_dim: int, hidden_dims: list, latent_dim: int, beta: float = 1.0):
        super(VariationalAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta
        
        # Initialize encoder and decoder
        self.encoder = Encoder(input_dim, hidden_dims, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dims, input_dim)
        
        # Prior distribution (standard normal)
        self.prior = dist.Normal(0, 1)
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for stochastic sampling.
        
        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            
        Returns:
            z: Sampled latent representation
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters."""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to reconstruction."""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through VAE.
        
        Args:
            x: Input tensor
            
        Returns:
            reconstruction: Reconstructed input
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar
    
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """Generate samples from the model.
        
        Args:
            num_samples: Number of samples to generate
            device: Device to generate samples on
            
        Returns:
            Generated samples
        """
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim, device=device)
            samples = self.decode(z)
        return samples
    
    def compute_loss(self, x: torch.Tensor, reconstruction: torch.Tensor, 
                    mu: torch.Tensor, logvar: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute VAE loss (ELBO).
        
        Args:
            x: Original input
            reconstruction: Reconstructed input
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            
        Returns:
            total_loss: Total VAE loss
            recon_loss: Reconstruction loss
            kl_loss: KL divergence loss
        """
        # Reconstruction loss (Binary Cross Entropy)
        recon_loss = F.binary_cross_entropy(reconstruction, x, reduction='sum')
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss (ELBO)
        total_loss = recon_loss + self.beta * kl_loss
        
        return total_loss, recon_loss, kl_loss
    
    def get_latent_representation(self, x: torch.Tensor) -> torch.Tensor:
        """Get latent representation of input (using mean, no sampling)."""
        with torch.no_grad():
            mu, _ = self.encode(x)
        return mu
    
    def interpolate(self, x1: torch.Tensor, x2: torch.Tensor, steps: int = 10) -> torch.Tensor:
        """Interpolate between two inputs in latent space.
        
        Args:
            x1, x2: Input tensors to interpolate between
            steps: Number of interpolation steps
            
        Returns:
            Interpolated samples
        """
        with torch.no_grad():
            mu1, _ = self.encode(x1)
            mu2, _ = self.encode(x2)
            
            # Linear interpolation in latent space
            alphas = torch.linspace(0, 1, steps, device=x1.device)
            interpolations = []
            
            for alpha in alphas:
                z_interp = (1 - alpha) * mu1 + alpha * mu2
                x_interp = self.decode(z_interp)
                interpolations.append(x_interp)
            
            return torch.stack(interpolations)