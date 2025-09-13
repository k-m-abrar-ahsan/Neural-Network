# Non-Deterministic Unsupervised Neural Network Model

## Neural Networks Course Assignment
**Due: September 14th, 2025**

This project implements a **Variational Autoencoder (VAE)** as a non-deterministic unsupervised neural network model for data generation, with comprehensive evaluation metrics and analysis.

##  Objective

Design and implement a non-deterministic unsupervised neural network model that:
- Uses stochastic sampling for better data space exploration
- Provides uncertainty quantification
- Generates high-quality samples
- Includes comprehensive evaluation and analysis

##  Project Structure

```
425/
├── src/                          # Source code
│   ├── vae_model.py             # VAE architecture implementation
│   ├── data_utils.py            # Data loading and preprocessing
│   ├── train.py                 # Training utilities and trainer class
│   ├── evaluation_metrics.py    # Comprehensive evaluation metrics
│   ├── visualization.py         # Visualization utilities
│   └── main_experiment.py       # Main experiment runner
├── data/                        # Dataset storage
├── models/                      # Trained model checkpoints
├── results/                     # Experiment results and visualizations
├── docs/                        # Documentation and reports
├── notebooks/                   # Jupyter notebooks for analysis
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

##  Quick Start

### 1. Environment Setup

```bash
# Clone or navigate to the project directory
cd 425

# Create virtual environment (recommended)
python -m venv vae_env

# Activate virtual environment
# On Windows:
vae_env\Scripts\activate
# On macOS/Linux:
source vae_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Basic Experiment

```bash
# Run with default settings (MNIST dataset)
python src/main_experiment.py

# Run with custom parameters
python src/main_experiment.py --dataset mnist --epochs 50 --latent-dim 10 --beta 1.0

# Run with synthetic data
python src/main_experiment.py --dataset synthetic --epochs 100 --latent-dim 5
```

### 3. Available Datasets

- **MNIST**: Handwritten digits (28x28 grayscale images)
- **CIFAR-10**: Natural images (32x32 RGB images)
- **Synthetic**: Generated 2D datasets (blobs, moons, circles)

##  Model Architecture

### Variational Autoencoder (VAE)

The implemented VAE consists of:

1. **Encoder Network**: Maps input data to latent distribution parameters (μ, σ²)
2. **Stochastic Sampling**: Uses reparameterization trick for differentiable sampling
3. **Decoder Network**: Reconstructs data from latent representations
4. **Loss Function**: ELBO = Reconstruction Loss + β × KL Divergence

```
Loss = E_q(z|x)[log p(x|z)] - β × D_KL(q(z|x) || p(z))
```

### Key Features

- **Non-deterministic**: Stochastic sampling in latent space
- **Uncertainty Quantification**: Epistemic and aleatoric uncertainty
- **Flexible Architecture**: Configurable hidden layers and latent dimensions
- **β-VAE Support**: Controllable disentanglement via β parameter

##  Evaluation Metrics

### For Generative Models
- **Fréchet Inception Distance (FID)**: Lower is better
- **Inception Score (IS)**: Higher is better
- **Reconstruction Error**: MSE, MAE, RMSE
- **Visual Quality Assessment**

### For Clustering (when labels available)
- **Silhouette Score**: Cluster separation (-1 to 1)
- **Adjusted Rand Index (ARI)**: Similarity to true clusters (0 to 1)
- **Normalized Mutual Information (NMI)**: Information-theoretic measure (0 to 1)

### For Dimensionality Reduction
- **Reconstruction Error**: Information preservation
- **Effective Dimensionality**: PCA-based analysis
- **Interpolation Quality**: Latent space smoothness

### Uncertainty Metrics
- **Epistemic Uncertainty**: Model uncertainty
- **Aleatoric Uncertainty**: Data uncertainty
- **Generation Diversity**: Sample variety measures

##  Visualizations

The project generates comprehensive visualizations:

1. **Training Curves**: Loss progression over epochs
2. **Reconstructions**: Original vs reconstructed samples
3. **Generated Samples**: Model-generated data
4. **Latent Space**: t-SNE and PCA projections
5. **Interpolations**: Smooth transitions in latent space
6. **Latent Dimensions**: Statistical analysis of learned representations

##  Configuration

### Command Line Arguments

```bash
python src/main_experiment.py --help

optional arguments:
  --config CONFIG       Path to configuration file
  --dataset {mnist,cifar10,synthetic}
                        Dataset to use
  --epochs EPOCHS       Number of training epochs
  --latent-dim LATENT_DIM
                        Latent dimension size
  --beta BETA           Beta parameter for VAE loss
  --batch-size BATCH_SIZE
                        Batch size for training
  --learning-rate LEARNING_RATE
                        Learning rate
  --device {auto,cpu,cuda}
                        Device to use for training
  --seed SEED           Random seed for reproducibility
```

### Configuration File

Create a JSON configuration file for advanced settings:

```json
{
  "dataset": "mnist",
  "model": {
    "hidden_dims": [512, 256],
    "latent_dim": 20,
    "beta": 1.0
  },
  "training": {
    "learning_rate": 0.001,
    "epochs": 100,
    "batch_size": 128,
    "early_stopping_patience": 20
  },
  "evaluation": {
    "n_generated_samples": 1000,
    "use_inception": true
  }
}
```

##  Results and Analysis

After running an experiment, results are saved to:

- `results/experiment_results.json`: Complete numerical results
- `results/training_curves.png`: Training progress visualization
- `results/reconstructions.png`: Sample reconstructions
- `results/generated_samples.png`: Generated samples
- `results/latent_space_*.png`: Latent space visualizations
- `results/interpolations.png`: Latent space interpolations
- `models/best_model.pth`: Best trained model checkpoint

##  Advanced Usage

### Custom Datasets

To use custom datasets, modify `data_utils.py`:

```python
def load_custom_data():
    # Load your data here
    return train_loader, test_loader, input_dim
```

### Model Modifications

Customize the VAE architecture in `vae_model.py`:

```python
# Modify encoder/decoder architectures
# Add custom loss functions
# Implement different sampling strategies
```

### Evaluation Extensions

Add custom metrics in `evaluation_metrics.py`:

```python
def custom_metric(self, data_loader):
    # Implement your evaluation metric
    return metric_value
```

##  Experiments and Comparisons

### Baseline Comparisons

The project includes comparisons with:
- Standard Autoencoder (deterministic baseline)
- Different β values for β-VAE analysis
- Various latent dimensionalities
- Different architectural choices

### Ablation Studies

Run ablation studies by varying:
```bash
# Different β values
python src/main_experiment.py --beta 0.5
python src/main_experiment.py --beta 2.0
python src/main_experiment.py --beta 4.0

# Different latent dimensions
python src/main_experiment.py --latent-dim 5
python src/main_experiment.py --latent-dim 10
python src/main_experiment.py --latent-dim 50
```

##  Report Generation

The project automatically generates:

1. **Quantitative Results**: Numerical evaluation metrics
2. **Qualitative Analysis**: Visual assessments and interpretations
3. **Statistical Significance**: Multiple runs with different seeds
4. **Uncertainty Analysis**: Epistemic vs aleatoric uncertainty
5. **Failure Case Analysis**: Limitations and edge cases

##  Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   ```bash
   python src/main_experiment.py --batch-size 64 --device cpu
   ```

2. **Inception Model Issues**:
   ```bash
   # Disable FID/IS computation
   # Modify config: "use_inception": false
   ```

3. **Slow Training**:
   ```bash
   # Reduce model size or use GPU
   python src/main_experiment.py --device cuda
   ```

### Performance Tips

- Use GPU when available for faster training
- Adjust batch size based on available memory
- Use early stopping to prevent overfitting
- Monitor training curves for convergence

##  Theoretical Background

### Variational Autoencoders

VAEs are generative models that learn a probabilistic mapping between data and latent space:

1. **Encoder**: q_φ(z|x) ≈ p(z|x)
2. **Decoder**: p_θ(x|z)
3. **Prior**: p(z) = N(0, I)

### Reparameterization Trick

Enables backpropagation through stochastic nodes:
```
z = μ + σ ⊙ ε, where ε ~ N(0, I)
```

### β-VAE

Controls the trade-off between reconstruction and regularization:
- β < 1: Emphasizes reconstruction
- β > 1: Emphasizes disentanglement

##  Contributing

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and documentation
5. Submit a pull request

## 📄 License

This project is for educational purposes as part of the Neural Networks course assignment.

##  Support

For questions or issues:
1. Check the troubleshooting section
2. Review the code documentation
3. Create an issue with detailed description

##  Academic References

1. Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes. arXiv preprint arXiv:1312.6114.
2. Higgins, I., et al. (2016). Beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework.
3. Rezende, D. J., Mohamed, S., & Wierstra, D. (2014). Stochastic Backpropagation and Approximate Inference in Deep Generative Models.

---

**Note**: This implementation follows the assignment requirements for a non-deterministic unsupervised neural network model with comprehensive evaluation and analysis capabilities.
