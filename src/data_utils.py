import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.datasets import make_blobs, make_moons, make_circles
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Union
import os

class SyntheticDataset(Dataset):
    """Custom dataset for synthetic data generation."""
    
    def __init__(self, data: np.ndarray, transform: Optional[callable] = None):
        self.data = torch.FloatTensor(data)
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample

class DataPreprocessor:
    """Data preprocessing utilities for various datasets."""
    
    def __init__(self, normalization: str = 'minmax'):
        """
        Args:
            normalization: Type of normalization ('minmax', 'standard', 'none')
        """
        self.normalization = normalization
        self.scaler = None
        
        if normalization == 'minmax':
            self.scaler = MinMaxScaler()
        elif normalization == 'standard':
            self.scaler = StandardScaler()
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit scaler and transform data."""
        if self.scaler is not None:
            return self.scaler.fit_transform(data)
        return data
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data using fitted scaler."""
        if self.scaler is not None:
            return self.scaler.transform(data)
        return data
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform data."""
        if self.scaler is not None:
            return self.scaler.inverse_transform(data)
        return data

def create_synthetic_data(dataset_type: str = 'blobs', n_samples: int = 1000, 
                         n_features: int = 2, **kwargs) -> np.ndarray:
    """Create synthetic datasets for testing.
    
    Args:
        dataset_type: Type of synthetic data ('blobs', 'moons', 'circles')
        n_samples: Number of samples
        n_features: Number of features
        **kwargs: Additional arguments for dataset generation
        
    Returns:
        Generated synthetic data
    """
    if dataset_type == 'blobs':
        data, _ = make_blobs(n_samples=n_samples, centers=kwargs.get('centers', 3),
                           n_features=n_features, random_state=42,
                           cluster_std=kwargs.get('cluster_std', 1.0))
    elif dataset_type == 'moons':
        data, _ = make_moons(n_samples=n_samples, noise=kwargs.get('noise', 0.1),
                           random_state=42)
    elif dataset_type == 'circles':
        data, _ = make_circles(n_samples=n_samples, noise=kwargs.get('noise', 0.1),
                             factor=kwargs.get('factor', 0.5), random_state=42)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    return data

def load_mnist_data(data_dir: str = './data', flatten: bool = True, 
                   normalize: bool = True) -> Tuple[DataLoader, DataLoader]:
    """Load MNIST dataset.
    
    Args:
        data_dir: Directory to store/load data
        flatten: Whether to flatten images
        normalize: Whether to normalize pixel values
        
    Returns:
        train_loader, test_loader
    """
    transform_list = []
    
    if normalize:
        transform_list.append(transforms.ToTensor())
    else:
        transform_list.append(transforms.ToTensor())
    
    if flatten:
        transform_list.append(transforms.Lambda(lambda x: x.view(-1)))
    
    transform = transforms.Compose(transform_list)
    
    # Download and load datasets
    train_dataset = datasets.MNIST(root=data_dir, train=True, 
                                 download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, 
                                download=True, transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    return train_loader, test_loader

def load_cifar10_data(data_dir: str = './data', flatten: bool = True,
                     normalize: bool = True) -> Tuple[DataLoader, DataLoader]:
    """Load CIFAR-10 dataset.
    
    Args:
        data_dir: Directory to store/load data
        flatten: Whether to flatten images
        normalize: Whether to normalize pixel values
        
    Returns:
        train_loader, test_loader
    """
    transform_list = []
    
    if normalize:
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        transform_list.append(transforms.ToTensor())
    
    if flatten:
        transform_list.append(transforms.Lambda(lambda x: x.view(-1)))
    
    transform = transforms.Compose(transform_list)
    
    # Download and load datasets
    train_dataset = datasets.CIFAR10(root=data_dir, train=True,
                                   download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False,
                                  download=True, transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    return train_loader, test_loader

def create_dataloader(data: np.ndarray, batch_size: int = 128, 
                     shuffle: bool = True, preprocessor: Optional[DataPreprocessor] = None) -> DataLoader:
    """Create DataLoader from numpy array.
    
    Args:
        data: Input data as numpy array
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle data
        preprocessor: Optional data preprocessor
        
    Returns:
        DataLoader object
    """
    if preprocessor is not None:
        data = preprocessor.fit_transform(data)
    
    dataset = SyntheticDataset(data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def visualize_data(data: np.ndarray, title: str = "Data Visualization", 
                  save_path: Optional[str] = None, figsize: Tuple[int, int] = (8, 6)):
    """Visualize 2D data.
    
    Args:
        data: 2D numpy array to visualize
        title: Plot title
        save_path: Path to save the plot
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    if data.shape[1] == 2:
        plt.scatter(data[:, 0], data[:, 1], alpha=0.6, s=20)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
    else:
        # For higher dimensional data, show first two dimensions
        plt.scatter(data[:, 0], data[:, 1], alpha=0.6, s=20)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(f"{title} (showing first 2 dimensions)")
    
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def get_data_statistics(data: np.ndarray) -> dict:
    """Compute basic statistics of the data.
    
    Args:
        data: Input data
        
    Returns:
        Dictionary containing statistics
    """
    stats = {
        'shape': data.shape,
        'mean': np.mean(data, axis=0),
        'std': np.std(data, axis=0),
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0),
        'median': np.median(data, axis=0)
    }
    return stats

def split_data(data: np.ndarray, train_ratio: float = 0.8, 
              random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Split data into train and test sets.
    
    Args:
        data: Input data
        train_ratio: Ratio of training data
        random_state: Random seed
        
    Returns:
        train_data, test_data
    """
    np.random.seed(random_state)
    n_samples = len(data)
    n_train = int(n_samples * train_ratio)
    
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    return data[train_indices], data[test_indices]

# Example usage and data preparation functions
def prepare_experiment_data(dataset_name: str = 'mnist', **kwargs):
    """Prepare data for experiments.
    
    Args:
        dataset_name: Name of dataset ('mnist', 'cifar10', 'synthetic')
        **kwargs: Additional arguments
        
    Returns:
        Prepared data loaders and metadata
    """
    if dataset_name == 'mnist':
        train_loader, test_loader = load_mnist_data(**kwargs)
        input_dim = 784  # 28*28
        return train_loader, test_loader, input_dim
    
    elif dataset_name == 'cifar10':
        train_loader, test_loader = load_cifar10_data(**kwargs)
        input_dim = 3072  # 32*32*3
        return train_loader, test_loader, input_dim
    
    elif dataset_name == 'synthetic':
        data = create_synthetic_data(**kwargs)
        preprocessor = DataPreprocessor(normalization='minmax')
        train_data, test_data = split_data(data)
        
        train_loader = create_dataloader(train_data, preprocessor=preprocessor)
        test_loader = create_dataloader(test_data, preprocessor=preprocessor)
        
        input_dim = data.shape[1]
        return train_loader, test_loader, input_dim, preprocessor
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")