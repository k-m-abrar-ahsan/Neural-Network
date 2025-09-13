#!/usr/bin/env python3
"""
Simplified VAE implementation without batch normalization for testing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

class SimpleEncoder(nn.Module):
    """Simple encoder without batch normalization."""
    
    def __init__(self, input_dim: int, hidden_dims: list, latent_dim: int):
        super(SimpleEncoder, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        self.mu_layer = nn.Linear(prev_dim, latent_dim)
        self.logvar_layer = nn.Linear(prev_dim, latent_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar

class SimpleDecoder(nn.Module):
    """Simple decoder without batch normalization."""
    
    def __init__(self, latent_dim: int, hidden_dims: list, output_dim: int):
        super(SimpleDecoder, self).__init__()
        
        layers = []
        prev_dim = latent_dim
        
        for hidden_dim in reversed(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        layers.extend([
            nn.Linear(prev_dim, output_dim),
            nn.Sigmoid()
        ])
        
        self.decoder = nn.Sequential(*layers)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

class SimpleVAE(nn.Module):
    """Simple VAE without batch normalization."""
    
    def __init__(self, input_dim: int, hidden_dims: list, latent_dim: int, beta: float = 1.0):
        super(SimpleVAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta
        
        self.encoder = SimpleEncoder(input_dim, hidden_dims, latent_dim)
        self.decoder = SimpleDecoder(latent_dim, hidden_dims, input_dim)
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decoder(z)
        return reconstruction, mu, logvar
    
    def compute_loss(self, x: torch.Tensor, reconstruction: torch.Tensor, 
                    mu: torch.Tensor, logvar: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Reconstruction loss
        recon_loss = F.binary_cross_entropy(reconstruction, x, reduction='sum')
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        total_loss = recon_loss + self.beta * kl_loss
        
        return total_loss, recon_loss, kl_loss
    
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim, device=device)
            samples = self.decoder(z)
        return samples
    
    def interpolate(self, x1: torch.Tensor, x2: torch.Tensor, steps: int = 10) -> torch.Tensor:
        with torch.no_grad():
            mu1, _ = self.encoder(x1.unsqueeze(0))
            mu2, _ = self.encoder(x2.unsqueeze(0))
            
            alphas = torch.linspace(0, 1, steps, device=x1.device)
            interpolations = []
            
            for alpha in alphas:
                z_interp = (1 - alpha) * mu1 + alpha * mu2
                x_interp = self.decoder(z_interp)
                interpolations.append(x_interp.squeeze(0))
            
            return torch.stack(interpolations)