import torch
import numpy as np
import os
import json
import argparse
from datetime import datetime
import logging
from typing import Dict, Any

from vae_model import VariationalAutoencoder
from data_utils import prepare_experiment_data
from train import VAETrainer
from evaluation_metrics import VAEEvaluator
from visualization import VAEVisualizer

def setup_logging(log_dir: str = './logs'):
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'experiment_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load experiment configuration.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    default_config = {
        # Data configuration
        'dataset': 'mnist',  # 'mnist', 'cifar10', 'synthetic'
        'data_dir': './data',
        'synthetic_params': {
            'dataset_type': 'blobs',
            'n_samples': 2000,
            'n_features': 2,
            'centers': 3,
            'cluster_std': 1.0
        },
        
        # Model configuration
        'model': {
            'hidden_dims': [512, 256],
            'latent_dim': 20,
            'beta': 1.0
        },
        
        # Training configuration
        'training': {
            'learning_rate': 1e-3,
            'weight_decay': 1e-5,
            'batch_size': 128,
            'epochs': 100,
            'early_stopping_patience': 20
        },
        
        # Evaluation configuration
        'evaluation': {
            'n_generated_samples': 1000,
            'use_inception': True,
            'compute_fid': True,
            'compute_is': True
        },
        
        # Visualization configuration
        'visualization': {
            'n_reconstruction_samples': 8,
            'n_generated_samples': 16,
            'n_interpolation_pairs': 3,
            'interpolation_steps': 10
        },
        
        # Output configuration
        'output': {
            'results_dir': './results',
            'models_dir': './models',
            'logs_dir': './logs'
        },
        
        # Hardware configuration
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'random_seed': 42
    }
    
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            user_config = json.load(f)
        
        # Merge configurations (user config overrides default)
        def merge_dicts(default, user):
            result = default.copy()
            for key, value in user.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = merge_dicts(result[key], value)
                else:
                    result[key] = value
            return result
        
        return merge_dicts(default_config, user_config)
    
    return default_config

def set_random_seeds(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_directories(config: Dict[str, Any]):
    """Create necessary directories."""
    directories = [
        config['output']['results_dir'],
        config['output']['models_dir'],
        config['output']['logs_dir'],
        config['data_dir']
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def run_experiment(config: Dict[str, Any], logger: logging.Logger) -> Dict[str, Any]:
    """Run the complete VAE experiment.
    
    Args:
        config: Experiment configuration
        logger: Logger instance
        
    Returns:
        Experiment results
    """
    logger.info("Starting VAE experiment")
    logger.info(f"Configuration: {json.dumps(config, indent=2)}")
    
    # Set device
    device = torch.device(config['device'])
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info("Loading data...")
    if config['dataset'] == 'synthetic':
        train_loader, test_loader, input_dim, preprocessor = prepare_experiment_data(
            config['dataset'],
            **config['synthetic_params']
        )
        config['input_dim'] = input_dim
    else:
        train_loader, test_loader, input_dim = prepare_experiment_data(
            config['dataset'],
            data_dir=config['data_dir']
        )
        config['input_dim'] = input_dim
        preprocessor = None
    
    logger.info(f"Data loaded. Input dimension: {input_dim}")
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    logger.info("Creating VAE model...")
    model = VariationalAutoencoder(
        input_dim=input_dim,
        hidden_dims=config['model']['hidden_dims'],
        latent_dim=config['model']['latent_dim'],
        beta=config['model']['beta']
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model created. Total parameters: {total_params}, Trainable: {trainable_params}")
    
    # Create trainer
    logger.info("Setting up trainer...")
    trainer = VAETrainer(
        model=model,
        device=device,
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Train model
    logger.info("Starting training...")
    training_history = trainer.train(
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=config['training']['epochs'],
        save_dir=config['output']['models_dir'],
        early_stopping_patience=config['training']['early_stopping_patience']
    )
    
    logger.info("Training completed")
    
    # Load best model for evaluation
    best_model_path = os.path.join(config['output']['models_dir'], 'best_model.pth')
    if os.path.exists(best_model_path):
        trainer.load_model(best_model_path)
        logger.info("Best model loaded for evaluation")
    
    # Evaluation
    logger.info("Starting evaluation...")
    evaluator = VAEEvaluator(
        model=trainer.model,
        device=device,
        use_inception=config['evaluation']['use_inception']
    )
    
    # Get true labels if available (for clustering evaluation)
    true_labels = None
    if config['dataset'] == 'synthetic':
        # For synthetic data, we can create labels based on clusters
        from sklearn.cluster import KMeans
        
        # Extract some data for labeling
        sample_data = []
        for batch in test_loader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            sample_data.append(batch.numpy())
            if len(sample_data) >= 5:  # Limit samples for efficiency
                break
        
        sample_data = np.concatenate(sample_data, axis=0)
        if preprocessor:
            sample_data = preprocessor.inverse_transform(sample_data)
        
        # Create labels using K-means on original data
        n_clusters = config['synthetic_params'].get('centers', 3)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        true_labels = kmeans.fit_predict(sample_data)
    
    # Comprehensive evaluation
    evaluation_results = evaluator.comprehensive_evaluation(
        test_loader=test_loader,
        true_labels=true_labels,
        n_generated_samples=config['evaluation']['n_generated_samples']
    )
    
    # Print evaluation summary
    evaluator.print_evaluation_summary(evaluation_results)
    
    logger.info("Evaluation completed")
    
    # Visualization
    logger.info("Generating visualizations...")
    visualizer = VAEVisualizer(model=trainer.model, device=device)
    
    # Create comprehensive visualization report
    visualizer.create_comprehensive_report(
        data_loader=test_loader,
        train_history=training_history,
        evaluation_results=evaluation_results,
        save_dir=config['output']['results_dir']
    )
    
    logger.info("Visualizations completed")
    
    # Save results
    results = {
        'config': config,
        'training_history': training_history,
        'evaluation_results': evaluation_results,
        'model_info': {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_dim': input_dim,
            'latent_dim': config['model']['latent_dim']
        }
    }
    
    # Save results to JSON
    results_file = os.path.join(config['output']['results_dir'], 'experiment_results.json')
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    serializable_results = convert_numpy(results)
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")
    
    return results

def main():
    """Main function to run the experiment."""
    parser = argparse.ArgumentParser(description='Run VAE experiment')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file')
    parser.add_argument('--dataset', type=str, default='mnist',
                       choices=['mnist', 'cifar10', 'synthetic'],
                       help='Dataset to use')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--latent-dim', type=int, default=20,
                       help='Latent dimension size')
    parser.add_argument('--beta', type=float, default=1.0,
                       help='Beta parameter for VAE loss')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for training')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.dataset != 'mnist':
        config['dataset'] = args.dataset
    if args.epochs != 100:
        config['training']['epochs'] = args.epochs
    if args.latent_dim != 20:
        config['model']['latent_dim'] = args.latent_dim
    if args.beta != 1.0:
        config['model']['beta'] = args.beta
    if args.batch_size != 128:
        config['training']['batch_size'] = args.batch_size
    if args.learning_rate != 1e-3:
        config['training']['learning_rate'] = args.learning_rate
    if args.device != 'auto':
        config['device'] = args.device
    elif config['device'] == 'cuda' and not torch.cuda.is_available():
        config['device'] = 'cpu'
    if args.seed != 42:
        config['random_seed'] = args.seed
    
    # Set random seeds
    set_random_seeds(config['random_seed'])
    
    # Create directories
    create_directories(config)
    
    # Setup logging
    logger = setup_logging(config['output']['logs_dir'])
    
    try:
        # Run experiment
        results = run_experiment(config, logger)
        
        logger.info("Experiment completed successfully!")
        
        # Print final summary
        print("\n" + "="*80)
        print("EXPERIMENT COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"Dataset: {config['dataset']}")
        print(f"Model: VAE with {config['model']['latent_dim']} latent dimensions")
        print(f"Training epochs: {len(results['training_history']['train_losses'])}")
        print(f"Final training loss: {results['training_history']['train_losses'][-1]:.4f}")
        print(f"Final validation loss: {results['training_history']['val_losses'][-1]:.4f}")
        
        if 'reconstruction' in results['evaluation_results']:
            print(f"Reconstruction MSE: {results['evaluation_results']['reconstruction']['mse']:.6f}")
        
        if 'fid' in results['evaluation_results']:
            print(f"FID Score: {results['evaluation_results']['fid']:.4f}")
        
        if 'inception_score' in results['evaluation_results']:
            is_result = results['evaluation_results']['inception_score']
            print(f"Inception Score: {is_result['mean']:.4f} ± {is_result['std']:.4f}")
        
        print(f"\nResults saved to: {config['output']['results_dir']}")
        print(f"Models saved to: {config['output']['models_dir']}")
        print(f"Logs saved to: {config['output']['logs_dir']}")
        print("="*80)
        
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()