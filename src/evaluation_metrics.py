import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import linalg
from sklearn.metrics import silhouette_score, adjusted_rand_index, normalized_mutual_info_score
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional, Union
from torchvision.models import inception_v3
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')

class InceptionFeatureExtractor(nn.Module):
    """Feature extractor using pre-trained Inception v3 for FID and IS computation."""
    
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        
        # Load pre-trained Inception v3
        self.inception = inception_v3(pretrained=True, transform_input=False)
        self.inception.fc = nn.Identity()  # Remove final classification layer
        self.inception.eval()
        self.inception.to(device)
        
        # Freeze parameters
        for param in self.inception.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input images.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Features of shape (batch_size, 2048)
        """
        # Resize to 299x299 for Inception v3
        if x.size(-1) != 299:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        
        # Convert grayscale to RGB if needed
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        
        with torch.no_grad():
            features = self.inception(x)
        
        return features

class VAEEvaluator:
    """Comprehensive evaluation metrics for VAE models."""
    
    def __init__(self, model, device: torch.device, use_inception: bool = True):
        """
        Args:
            model: Trained VAE model
            device: Device to run evaluations on
            use_inception: Whether to use Inception v3 for FID/IS (requires image data)
        """
        self.model = model
        self.device = device
        self.use_inception = use_inception
        
        if use_inception:
            try:
                self.inception_extractor = InceptionFeatureExtractor(device)
            except Exception as e:
                print(f"Warning: Could not load Inception v3: {e}")
                self.use_inception = False
    
    def compute_reconstruction_error(self, data_loader: DataLoader) -> Dict[str, float]:
        """Compute reconstruction error metrics.
        
        Args:
            data_loader: DataLoader containing test data
            
        Returns:
            Dictionary containing reconstruction metrics
        """
        self.model.eval()
        total_mse = 0
        total_mae = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                
                batch = batch.to(self.device)
                reconstruction, _, _ = self.model(batch)
                
                # Compute errors
                mse = F.mse_loss(reconstruction, batch, reduction='sum')
                mae = F.l1_loss(reconstruction, batch, reduction='sum')
                
                total_mse += mse.item()
                total_mae += mae.item()
                total_samples += batch.size(0) * batch.numel() // batch.size(0)
        
        return {
            'mse': total_mse / total_samples,
            'mae': total_mae / total_samples,
            'rmse': np.sqrt(total_mse / total_samples)
        }
    
    def compute_fid_score(self, real_data: torch.Tensor, generated_data: torch.Tensor) -> float:
        """Compute Fréchet Inception Distance (FID).
        
        Args:
            real_data: Real data tensor
            generated_data: Generated data tensor
            
        Returns:
            FID score (lower is better)
        """
        if not self.use_inception:
            raise ValueError("Inception v3 not available for FID computation")
        
        # Extract features
        real_features = self.inception_extractor(real_data).cpu().numpy()
        gen_features = self.inception_extractor(generated_data).cpu().numpy()
        
        # Compute statistics
        mu_real = np.mean(real_features, axis=0)
        sigma_real = np.cov(real_features, rowvar=False)
        
        mu_gen = np.mean(gen_features, axis=0)
        sigma_gen = np.cov(gen_features, rowvar=False)
        
        # Compute FID
        diff = mu_real - mu_gen
        covmean, _ = linalg.sqrtm(sigma_real.dot(sigma_gen), disp=False)
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = diff.dot(diff) + np.trace(sigma_real + sigma_gen - 2 * covmean)
        return float(fid)
    
    def compute_inception_score(self, generated_data: torch.Tensor, splits: int = 10) -> Tuple[float, float]:
        """Compute Inception Score (IS).
        
        Args:
            generated_data: Generated data tensor
            splits: Number of splits for computing IS
            
        Returns:
            Mean and standard deviation of IS (higher is better)
        """
        if not self.use_inception:
            raise ValueError("Inception v3 not available for IS computation")
        
        # Get predictions from Inception v3
        with torch.no_grad():
            # Use the full Inception model for classification
            inception_full = inception_v3(pretrained=True, transform_input=False)
            inception_full.eval().to(self.device)
            
            # Resize and preprocess
            if generated_data.size(-1) != 299:
                generated_data = F.interpolate(generated_data, size=(299, 299), mode='bilinear', align_corners=False)
            
            if generated_data.size(1) == 1:
                generated_data = generated_data.repeat(1, 3, 1, 1)
            
            preds = F.softmax(inception_full(generated_data), dim=1).cpu().numpy()
        
        # Compute IS
        scores = []
        for i in range(splits):
            part = preds[i * len(preds) // splits: (i + 1) * len(preds) // splits]
            kl_div = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, axis=0), 0)))
            kl_div = np.mean(np.sum(kl_div, axis=1))
            scores.append(np.exp(kl_div))
        
        return np.mean(scores), np.std(scores)
    
    def compute_latent_space_metrics(self, data_loader: DataLoader, 
                                   true_labels: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Compute latent space quality metrics.
        
        Args:
            data_loader: DataLoader containing test data
            true_labels: True cluster labels (if available)
            
        Returns:
            Dictionary containing latent space metrics
        """
        self.model.eval()
        latent_representations = []
        
        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                
                batch = batch.to(self.device)
                mu, _ = self.model.encode(batch)
                latent_representations.append(mu.cpu().numpy())
        
        latent_data = np.concatenate(latent_representations, axis=0)
        
        metrics = {}
        
        # Compute clustering metrics if labels are provided
        if true_labels is not None:
            # K-means clustering
            n_clusters = len(np.unique(true_labels))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            pred_labels = kmeans.fit_predict(latent_data)
            
            metrics['silhouette_score'] = silhouette_score(latent_data, pred_labels)
            metrics['ari'] = adjusted_rand_index(true_labels, pred_labels)
            metrics['nmi'] = normalized_mutual_info_score(true_labels, pred_labels)
        
        # Compute latent space statistics
        metrics['latent_mean'] = np.mean(latent_data, axis=0)
        metrics['latent_std'] = np.std(latent_data, axis=0)
        metrics['latent_variance_explained'] = np.var(latent_data, axis=0)
        
        # Compute effective dimensionality
        pca = PCA()
        pca.fit(latent_data)
        cumsum_var = np.cumsum(pca.explained_variance_ratio_)
        effective_dim = np.argmax(cumsum_var >= 0.95) + 1
        metrics['effective_dimensionality'] = effective_dim
        
        return metrics
    
    def compute_interpolation_quality(self, data_loader: DataLoader, 
                                    n_pairs: int = 10, n_steps: int = 10) -> float:
        """Compute interpolation quality in latent space.
        
        Args:
            data_loader: DataLoader containing test data
            n_pairs: Number of pairs to interpolate between
            n_steps: Number of interpolation steps
            
        Returns:
            Average interpolation smoothness score
        """
        self.model.eval()
        
        # Get random pairs of data points
        data_iter = iter(data_loader)
        batch = next(data_iter)
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        
        batch = batch.to(self.device)
        
        smoothness_scores = []
        
        with torch.no_grad():
            for i in range(min(n_pairs, batch.size(0) // 2)):
                x1 = batch[2*i:2*i+1]
                x2 = batch[2*i+1:2*i+2]
                
                # Interpolate in latent space
                interpolations = self.model.interpolate(x1, x2, steps=n_steps)
                
                # Compute smoothness (variance of consecutive differences)
                diffs = []
                for j in range(len(interpolations) - 1):
                    diff = torch.mean((interpolations[j+1] - interpolations[j])**2)
                    diffs.append(diff.item())
                
                smoothness = 1.0 / (1.0 + np.var(diffs))  # Higher is smoother
                smoothness_scores.append(smoothness)
        
        return np.mean(smoothness_scores)
    
    def compute_generation_diversity(self, n_samples: int = 1000) -> Dict[str, float]:
        """Compute diversity metrics for generated samples.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Dictionary containing diversity metrics
        """
        self.model.eval()
        
        # Generate samples
        generated_samples = self.model.sample(n_samples, self.device)
        generated_samples = generated_samples.cpu().numpy()
        
        # Compute pairwise distances
        from scipy.spatial.distance import pdist
        
        distances = pdist(generated_samples.reshape(n_samples, -1))
        
        metrics = {
            'mean_pairwise_distance': np.mean(distances),
            'std_pairwise_distance': np.std(distances),
            'min_pairwise_distance': np.min(distances),
            'max_pairwise_distance': np.max(distances)
        }
        
        return metrics
    
    def compute_uncertainty_metrics(self, data_loader: DataLoader, 
                                  n_samples: int = 10) -> Dict[str, float]:
        """Compute uncertainty quantification metrics.
        
        Args:
            data_loader: DataLoader containing test data
            n_samples: Number of stochastic forward passes
            
        Returns:
            Dictionary containing uncertainty metrics
        """
        self.model.eval()
        
        total_epistemic_uncertainty = 0
        total_aleatoric_uncertainty = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                
                batch = batch.to(self.device)
                
                # Multiple stochastic forward passes
                reconstructions = []
                mus = []
                logvars = []
                
                for _ in range(n_samples):
                    recon, mu, logvar = self.model(batch)
                    reconstructions.append(recon)
                    mus.append(mu)
                    logvars.append(logvar)
                
                # Stack results
                reconstructions = torch.stack(reconstructions)  # (n_samples, batch_size, ...)
                mus = torch.stack(mus)
                logvars = torch.stack(logvars)
                
                # Epistemic uncertainty (variance across samples)
                epistemic = torch.var(reconstructions, dim=0).mean()
                
                # Aleatoric uncertainty (average of individual uncertainties)
                aleatoric = torch.mean(torch.exp(logvars), dim=0).mean()
                
                total_epistemic_uncertainty += epistemic.item() * batch.size(0)
                total_aleatoric_uncertainty += aleatoric.item() * batch.size(0)
                total_samples += batch.size(0)
        
        return {
            'epistemic_uncertainty': total_epistemic_uncertainty / total_samples,
            'aleatoric_uncertainty': total_aleatoric_uncertainty / total_samples,
            'total_uncertainty': (total_epistemic_uncertainty + total_aleatoric_uncertainty) / total_samples
        }
    
    def comprehensive_evaluation(self, test_loader: DataLoader, 
                               true_labels: Optional[np.ndarray] = None,
                               n_generated_samples: int = 1000) -> Dict[str, Union[float, Dict]]:
        """Perform comprehensive evaluation of the VAE model.
        
        Args:
            test_loader: Test data loader
            true_labels: True labels for clustering evaluation
            n_generated_samples: Number of samples to generate for evaluation
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        print("Computing comprehensive evaluation metrics...")
        
        results = {}
        
        # Reconstruction metrics
        print("Computing reconstruction metrics...")
        results['reconstruction'] = self.compute_reconstruction_error(test_loader)
        
        # Latent space metrics
        print("Computing latent space metrics...")
        results['latent_space'] = self.compute_latent_space_metrics(test_loader, true_labels)
        
        # Generation diversity
        print("Computing generation diversity...")
        results['diversity'] = self.compute_generation_diversity(n_generated_samples)
        
        # Interpolation quality
        print("Computing interpolation quality...")
        results['interpolation'] = self.compute_interpolation_quality(test_loader)
        
        # Uncertainty metrics
        print("Computing uncertainty metrics...")
        results['uncertainty'] = self.compute_uncertainty_metrics(test_loader)
        
        # FID and IS (if inception is available and data is image-like)
        if self.use_inception:
            try:
                print("Computing FID and IS scores...")
                # Generate samples for comparison
                generated_samples = self.model.sample(n_generated_samples, self.device)
                
                # Get real samples
                real_samples = []
                for batch in test_loader:
                    if isinstance(batch, (list, tuple)):
                        batch = batch[0]
                    real_samples.append(batch)
                    if len(real_samples) * batch.size(0) >= n_generated_samples:
                        break
                
                real_samples = torch.cat(real_samples)[:n_generated_samples]
                
                # Reshape for image evaluation (assuming square images)
                if len(generated_samples.shape) == 2:
                    # Assume flattened images, try to reshape to square
                    img_size = int(np.sqrt(generated_samples.shape[1]))
                    if img_size * img_size == generated_samples.shape[1]:
                        generated_samples = generated_samples.view(-1, 1, img_size, img_size)
                        real_samples = real_samples.view(-1, 1, img_size, img_size)
                    else:
                        print("Cannot reshape data for FID/IS computation")
                        self.use_inception = False
                
                if self.use_inception:
                    results['fid'] = self.compute_fid_score(real_samples.to(self.device), generated_samples)
                    is_mean, is_std = self.compute_inception_score(generated_samples)
                    results['inception_score'] = {'mean': is_mean, 'std': is_std}
                    
            except Exception as e:
                print(f"Could not compute FID/IS: {e}")
        
        return results
    
    def print_evaluation_summary(self, results: Dict):
        """Print a formatted summary of evaluation results.
        
        Args:
            results: Results dictionary from comprehensive_evaluation
        """
        print("\n" + "="*60)
        print("VAE MODEL EVALUATION SUMMARY")
        print("="*60)
        
        # Reconstruction metrics
        if 'reconstruction' in results:
            print("\nReconstruction Metrics:")
            print(f"  MSE: {results['reconstruction']['mse']:.6f}")
            print(f"  MAE: {results['reconstruction']['mae']:.6f}")
            print(f"  RMSE: {results['reconstruction']['rmse']:.6f}")
        
        # Latent space metrics
        if 'latent_space' in results:
            print("\nLatent Space Metrics:")
            if 'silhouette_score' in results['latent_space']:
                print(f"  Silhouette Score: {results['latent_space']['silhouette_score']:.4f}")
                print(f"  Adjusted Rand Index: {results['latent_space']['ari']:.4f}")
                print(f"  Normalized Mutual Info: {results['latent_space']['nmi']:.4f}")
            print(f"  Effective Dimensionality: {results['latent_space']['effective_dimensionality']}")
        
        # Generation metrics
        if 'diversity' in results:
            print("\nGeneration Diversity:")
            print(f"  Mean Pairwise Distance: {results['diversity']['mean_pairwise_distance']:.4f}")
            print(f"  Std Pairwise Distance: {results['diversity']['std_pairwise_distance']:.4f}")
        
        # Uncertainty metrics
        if 'uncertainty' in results:
            print("\nUncertainty Metrics:")
            print(f"  Epistemic Uncertainty: {results['uncertainty']['epistemic_uncertainty']:.6f}")
            print(f"  Aleatoric Uncertainty: {results['uncertainty']['aleatoric_uncertainty']:.6f}")
            print(f"  Total Uncertainty: {results['uncertainty']['total_uncertainty']:.6f}")
        
        # FID and IS
        if 'fid' in results:
            print("\nImage Quality Metrics:")
            print(f"  FID Score: {results['fid']:.4f}")
        
        if 'inception_score' in results:
            print(f"  Inception Score: {results['inception_score']['mean']:.4f} ± {results['inception_score']['std']:.4f}")
        
        if 'interpolation' in results:
            print(f"\nInterpolation Quality: {results['interpolation']:.4f}")
        
        print("\n" + "="*60)