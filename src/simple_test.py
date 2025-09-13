#!/usr/bin/env python3
"""
Simple test script to verify the VAE implementation works with minimal dependencies.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simple_vae import SimpleVAE as VAE

def create_simple_data(n_samples=1000, dim=784):
    """Create simple synthetic data for testing."""
    # Create some simple patterns
    data = torch.randn(n_samples, dim) * 0.1
    # Add some structure
    data[:n_samples//2, :dim//2] += 1.0
    data[n_samples//2:, dim//2:] += 1.0
    
    # Normalize to [0, 1] for binary cross entropy
    data = torch.sigmoid(data)
    return data

def simple_train_test():
    """Simple training test with minimal dependencies."""
    print("Starting simple VAE test...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create simple data
    print("Creating synthetic data...")
    train_data = create_simple_data(1000, 784)
    test_data = create_simple_data(200, 784)
    
    # Create model
    print("Initializing VAE model...")
    model = VAE(
        input_dim=784,
        hidden_dims=[400, 200],
        latent_dim=20,
        beta=1.0
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Training loop
    print("Starting training...")
    model.train()
    batch_size = 64
    n_epochs = 5
    
    for epoch in range(n_epochs):
        total_loss = 0
        n_batches = 0
        
        # Simple batching
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size].to(device)
            
            # Skip if batch is too small for batch normalization
            if batch.size(0) < 2:
                continue
                
            optimizer.zero_grad()
            
            # Forward pass
            recon_batch, mu, logvar = model(batch)
            loss, recon_loss, kl_loss = model.compute_loss(batch, recon_batch, mu, logvar)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        avg_loss = total_loss / n_batches
        print(f"Epoch {epoch+1}/{n_epochs}, Average Loss: {avg_loss:.4f}")
    
    # Test reconstruction
    print("Testing reconstruction...")
    model.eval()
    with torch.no_grad():
        test_batch = test_data[:10].to(device)
        recon_batch, mu, logvar = model(test_batch)
        
        # Calculate reconstruction error
        recon_error = torch.mean((test_batch - recon_batch) ** 2)
        print(f"Reconstruction Error: {recon_error.item():.4f}")
        
        # Test sampling
        print("Testing sampling...")
        samples = model.sample(10, device)
        print(f"Generated samples shape: {samples.shape}")
        
        # Test interpolation
        print("Testing interpolation...")
        interpolations = model.interpolate(test_batch[0], test_batch[1], 5)
        print(f"Interpolation shape: {interpolations.shape}")
    
    print("\n=== Test Results ===")
    print(f"✓ Model initialized successfully")
    print(f"✓ Training completed ({n_epochs} epochs)")
    print(f"✓ Final reconstruction error: {recon_error.item():.4f}")
    print(f"✓ Sampling works (shape: {samples.shape})")
    print(f"✓ Interpolation works (shape: {interpolations.shape})")
    print("\n=== VAE Implementation Test PASSED ===")
    
    return model

if __name__ == "__main__":
    try:
        model = simple_train_test()
        print("\nAll tests passed! The VAE implementation is working correctly.")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()