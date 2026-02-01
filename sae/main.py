#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Train a sparse autoencoder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset config file (required)
    parser.add_argument('--dataset_config', type=str, required=True,
                       help='Path to JSON config file containing dataset paths (train_path, val_path, test_path)')
    
    # Model selection
    parser.add_argument('--model', type=str, default='topk',
                       choices=['topk', 'batch_topk', 'vanilla', 'jumprelu'],
                       help='Model architecture: topk (TopKSAE), batch_topk (BatchTopKSAE), vanilla (VanillaSAE), or jumprelu (JumpReLUSAE)')
    
    # Model architecture
    parser.add_argument('--n_latents', type=int, default=8192, 
                       help='Number of latent dimensions')
    parser.add_argument('--activation', type=str, default='topk', 
                       choices=['relu', 'topk'],
                       help='Activation function type')
    parser.add_argument('--topk_k', type=int, default=64, 
                       help='K for TopK activation')
    parser.add_argument('--tied_weights', action='store_true', default=True,
                       help='Use tied encoder/decoder weights')
    parser.add_argument('--normalize', action='store_true', default=False,
                       help='Use layer normalization on inputs')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=256, 
                       help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=50, 
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, 
                       help='Learning rate')
    parser.add_argument('--eps', type=float, default=6.25e-10,
                       help='Epsilon for Adam optimizer')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                       help='Maximum gradient norm for clipping')
    
    # Loss coefficients
    parser.add_argument('--auxk_coef', type=float, default=1/32, 
                       help='Coefficient for auxiliary TopK loss')
    parser.add_argument('--auxk_k', type=int, default=None,
                       help='K for auxiliary TopK (if None, uses same as topk_k)')
    parser.add_argument('--l1_coef', type=float, default=0.0001,
                       help='Coefficient for L1 sparsity penalty (vanilla SAE)')
    parser.add_argument('--bandwidth', type=float, default=0.001,
                       help='Bandwidth for JumpReLU activation (smoothness of threshold)')
    parser.add_argument('--mono_coef', type=float, default=0.0,
                       help='Coefficient for monosemanticity loss (1 - mono_score)')
    parser.add_argument('--mono_period', type=int, default=1,
                       help='Compute mono loss every N steps (1=per batch, >1=full dataset every N steps)')
    
    # Dead neuron handling
    parser.add_argument('--dead_steps_threshold', type=int, default=10_000_000, 
                       help='Steps threshold for considering a neuron dead')
    parser.add_argument('--dead_check_interval', type=int, default=1000,
                       help='Check for dead neurons every N steps')
    
    # Logging
    parser.add_argument('--wandb_project', type=str, default=None, 
                       help='Wandb project name (if None, no wandb logging)')
    parser.add_argument('--wandb_group', type=str, default=None,
                       help='Wandb group name for organizing runs')
    
    # System
    parser.add_argument('--exp_name', type=str, default='default',
                       help='Experiment name (checkpoints saved to output_dir/exp_name)')
    parser.add_argument('--num_workers', type=int, default=4, 
                       help='Number of DataLoader workers')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed for reproducibility')
    parser.add_argument('--output_dir', type=str, default='./checkpoints', 
                       help='Base output directory for checkpoints')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use for training')
    parser.add_argument('--skip_existing', action='store_true', default=False,
                       help='Skip training if final checkpoint already exists')

    args = parser.parse_args()

    # Check if checkpoint already exists (early exit if --skip_existing)
    checkpoint_path = Path(args.output_dir) / args.exp_name / 'autoencoder_final.pt'
    if args.skip_existing and checkpoint_path.exists():
        print(f"Checkpoint already exists at {checkpoint_path}")
        print("Skipping training (--skip_existing is set)")
        return None

    # Load dataset paths from config
    print(f"Loading dataset paths from {args.dataset_config}")
    with open(args.dataset_config, 'r') as f:
        config_dict = json.load(f)
    
    # Load dataset paths from config
    args.train_path = config_dict.get('train_path', None)
    args.val_path = config_dict.get('val_path', None)
    args.test_path = config_dict.get('test_path', None)
    
    print(f"Dataset paths loaded: train={args.train_path is not None}, "
          f"val={args.val_path is not None}, test={args.test_path is not None}")
    
    # Validate that train_path is provided
    if args.train_path is None:
        parser.error("train_path must be specified in dataset config file")
    
    # Import appropriate training module based on model type
    if args.model == 'topk':
        from train_topk import TrainingConfig, train_autoencoder
        config = TrainingConfig(
            train_path=args.train_path,
            val_path=args.val_path,
            test_path=args.test_path,
            exp_name=args.exp_name,
            n_latents=args.n_latents,
            activation=args.activation,
            topk_k=args.topk_k,
            tied_weights=args.tied_weights,
            normalize=True, #args.normalize,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            lr=args.lr,
            eps=args.eps,
            auxk_coef=args.auxk_coef,
            auxk_k=args.auxk_k,
            mono_coef=args.mono_coef,
            mono_period=args.mono_period,
            dead_steps_threshold=args.dead_steps_threshold,
            dead_check_interval=args.dead_check_interval,
            wandb_project=args.wandb_project,
            wandb_name=args.exp_name,
            wandb_group=args.wandb_group,
            num_workers=args.num_workers,
            seed=args.seed,
            output_dir=args.output_dir,
            device=args.device
        )
    else:  # batch_topk, vanilla, jumprelu
        from train_other import TrainingConfig, train_autoencoder
        config = TrainingConfig(
            train_path=args.train_path,
            val_path=args.val_path,
            test_path=args.test_path,
            exp_name=args.exp_name,
            model_type=args.model,
            n_latents=args.n_latents,
            topk_k=args.topk_k,
            normalize=args.normalize,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm,
            auxk_coef=args.auxk_coef,
            auxk_k=args.auxk_k,
            l1_coef=args.l1_coef,
            bandwidth=args.bandwidth,
            mono_coef=args.mono_coef,
            mono_period=args.mono_period,
            dead_steps_threshold=args.dead_steps_threshold,
            wandb_project=args.wandb_project,
            wandb_name=args.exp_name,
            wandb_group=args.wandb_group,
            num_workers=args.num_workers,
            seed=args.seed,
            output_dir=args.output_dir,
            device=args.device
        )
    
    # Save config to output directory: output_dir/exp_name
    output_path = Path(args.output_dir) / args.exp_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    config_save_path = output_path / 'training_config.json'
    with open(config_save_path, 'w') as f:
        json.dump(vars(config), f, indent=2)
    print(f"Training configuration saved to {config_save_path}")
    
    # Print configuration
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    print(f"Experiment:       {config.exp_name}")
    print(f"Output directory: {output_path}")
    print(f"\nModel:")
    print(f"  Model type:     {getattr(config, 'model_type', 'topk')}")
    print(f"  Latents:        {config.n_latents}")
    if hasattr(config, 'activation'):
        print(f"  Activation:     {config.activation}")
        if config.activation == 'topk':
            print(f"  TopK K:         {config.topk_k}")
        print(f"  Tied weights:   {config.tied_weights}")
    else:
        print(f"  TopK K:         {config.topk_k}")
    print(f"  Normalize:      {config.normalize}")
    print(f"\nData:")
    print(f"  Train:          {config.train_path}")
    if config.val_path:
        print(f"  Val:            {config.val_path}")
    if config.test_path:
        print(f"  Test:           {config.test_path}")
    print(f"\nTraining:")
    print(f"  Batch size:     {config.batch_size}")
    print(f"  Num epochs:     {config.num_epochs}")
    print(f"  Learning rate:  {config.lr}")
    if hasattr(config, 'auxk_coef'):
        print(f"  AuxK coef:      {config.auxk_coef}")
    if hasattr(config, 'l1_coef') and config.l1_coef > 0:
        print(f"  L1 coef:        {config.l1_coef}")
    if hasattr(config, 'bandwidth'):
        print(f"  Bandwidth:      {config.bandwidth}")
    print(f"  Mono coef:      {config.mono_coef}")
    if config.mono_coef > 0:
        if config.mono_period == 1:
            print(f"  Mono period:    every step (batch-level)")
        else:
            print(f"  Mono period:    every {config.mono_period} steps (full dataset)")
    print(f"\nSystem:")
    print(f"  Device:         {config.device}")
    print(f"  Num workers:    {config.num_workers}")
    print(f"  Seed:           {config.seed}")
    if config.wandb_project:
        print(f"\nLogging:")
        print(f"  Wandb project:  {config.wandb_project}")
        print(f"  Wandb name:     {config.wandb_name}")
        if config.wandb_group:
            print(f"  Wandb group:    {config.wandb_group}")
    print("="*60 + "\n")
    
    # Train the model
    autoencoder = train_autoencoder(config)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"Final model saved to: {output_path / 'autoencoder_final.pt'}")
    print("="*60)
    
    return autoencoder


if __name__ == '__main__':
    main()
