import os
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import wandb

from dataset import LMDBFeatureDataset
# from dataset.samplers import ContiguousBatchSampler
from models import BatchTopKSAE, VanillaSAE, JumpReLUSAE
from loss import compute_monosemanticity_fast
from mono_loss import compute_monosemanticity_custom


@torch.no_grad()
def evaluate(autoencoder, dataloader, config, split_name='val'):
    """Evaluate autoencoder on a dataset."""
    autoencoder.eval()
    total_loss = 0
    total_l2 = 0
    total_active = 0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc=f"Evaluating {split_name}"):
        if not isinstance(batch, torch.Tensor):
            batch = torch.from_numpy(batch)
        
        x = batch.to(config.device)
        
        # All models return (acts/latents_pre_act, latents, recons)
        latents_pre_act, latents, recons = autoencoder(x)
        
        # Compute loss using model's compute_loss method
        loss, metrics = autoencoder.compute_loss(
            x.float(),
            latents_pre_act.float(),
            latents.float(),
            recons.float(),
            mono_coef=0.0
        )
        
        total_loss += loss.item()
        total_l2 += metrics.get('l2_loss', 0)
        total_active += metrics['num_active']
        num_batches += 1
    
    autoencoder.train()
    
    avg_loss = total_loss / num_batches
    avg_l2 = total_l2 / num_batches
    avg_active = total_active / num_batches
    
    return {
        f'{split_name}_loss': avg_loss,
        f'{split_name}_l2': avg_l2,
        f'{split_name}_active': avg_active
    }


@torch.no_grad()
def compute_r2(autoencoder, dataloader, device='cuda', eps: float = 1e-12, split_name='data'):
    """Compute R² (uniform average across features) over an entire dataloader.

    Uses a single streaming pass with sufficient statistics to avoid storing all data:
      - sum_y (per feature), sum_y2 (per feature), N (scalar), SS_res (per feature)
      - SS_tot = sum_y2 - (sum_y^2) / N (per feature)
      - R2_feature = 1 - SS_res / SS_tot (masked when SS_tot≈0)
      - R2 = mean over features with SS_tot > 0
    """
    sum_y = None
    sum_y2 = None
    ss_res = None
    total_n = 0

    autoencoder.eval()

    for batch in tqdm(dataloader, desc=f"Computing R2 {split_name}"):
        if not isinstance(batch, torch.Tensor):
            batch = torch.from_numpy(batch)
        x = batch.to(device)

        # All models return (acts/latents_pre_act, latents, recons)
        _, _, recons = autoencoder(x)

        # Cast to float64 for stable accumulation
        x64 = x.to(torch.float64)
        r64 = recons.to(torch.float64)

        if sum_y is None:
            D = x64.shape[1]
            sum_y = torch.zeros(D, dtype=torch.float64, device=device)
            sum_y2 = torch.zeros(D, dtype=torch.float64, device=device)
            ss_res = torch.zeros(D, dtype=torch.float64, device=device)

        sum_y += x64.sum(dim=0)
        sum_y2 += (x64 * x64).sum(dim=0)
        ss_res += ((x64 - r64) * (x64 - r64)).sum(dim=0)
        total_n += x64.shape[0]

    if total_n == 0:
        return 0.0

    ss_tot = sum_y2 - (sum_y * sum_y) / max(total_n, 1)
    valid = ss_tot > eps
    if valid.any():
        r2_per_feat = torch.zeros_like(ss_tot)
        r2_per_feat[valid] = 1.0 - (ss_res[valid] / (ss_tot[valid] + eps))
        r2 = r2_per_feat[valid].mean().item()
    else:
        r2 = 0.0

    autoencoder.train()
    return r2


@dataclass
class TrainingConfig:
    # Data (required field first)
    train_path: str
        
    # Data (optional)
    val_path: Optional[str] = None
    test_path: Optional[str] = None

    # Experiment
    exp_name: str = 'default'
    
    # Model architecture
    model_type: str = 'vanilla'  # 'batch_topk', 'vanilla', or 'jumprelu'
    n_latents: int = 8192
    d_model: Optional[int] = None  # Will be inferred from data
    topk_k: int = 64  # For batch_topk
    normalize: bool = False  # Layer normalization
    
    # Training
    batch_size: int = 4096
    num_epochs: int = 10
    lr: float = 1e-4
    eps: float = 6.25e-10
    max_grad_norm: float = 1.0
    
    # Loss coefficients
    auxk_coef: float = 1/32  # For batch_topk auxiliary loss
    auxk_k: Optional[int] = None  # For batch_topk (if None, uses 512)
    l1_coef: float = 0.0001  # Coefficient for L1/L0 sparsity penalty
    bandwidth: float = 0.001  # Bandwidth for JumpReLU activation
    mono_coef: float = 0.0  # Coefficient for monosemanticity loss
    mono_period: int = 1  # Compute mono loss every N steps (1=per batch)
    
    # Dead neuron handling (tracked by models)
    dead_steps_threshold: int = 10_000_000  # In tokens/samples
    
    # Logging
    wandb_project: Optional[str] = None
    wandb_name: Optional[str] = None
    wandb_group: Optional[str] = None
    
    # System
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers: int = 4
    seed: int = 42
    output_dir: str = './checkpoints'


def train_autoencoder(config: TrainingConfig):
    """
    Main training loop for sparse autoencoder (BatchTopK, Vanilla, JumpReLU).
    
    Args:
        config: Training configuration
        
    Returns:
        autoencoder: Trained autoencoder model
    """
    
    # Construct full output path: output_dir/exp_name
    config.output_dir = os.path.join(config.output_dir, config.exp_name)
    
    # Set seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Initialize wandb
    if config.wandb_project:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_name,
            group=config.wandb_group,
            config=vars(config)
        )
    
    # Load train dataset (LMDB format)
    print(f"Loading train dataset from {config.train_path}...")
    train_dataset = LMDBFeatureDataset(config.train_path, return_index=False, verbose=True)
    
    # Infer feature dimension
    d_model = train_dataset.get_feature_dim()
    config.d_model = d_model
    print(f"Feature dimension: {d_model}")
    
        
    # # Create train dataloader with contiguous batch sampling for faster LMDB reads
    # train_batch_sampler = ContiguousBatchSampler(
    #     n_samples=len(train_dataset),
    #     batch_size=config.batch_size,
    #     drop_last=True,
    #     shuffle=True
    # )
    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_sampler=train_batch_sampler,
    #     num_workers=config.num_workers,
    #     pin_memory=True,
    #     persistent_workers=config.num_workers > 0,
    #     prefetch_factor=4 if config.num_workers > 0 else None
    # )
    # Create train dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    # Create separate dataloader for mono loss computation (if needed)
    mono_loader = None
    if config.mono_coef > 0 and config.mono_period > 1:
        print(f"Creating separate DataLoader for periodic mono loss computation...")
        # mono_batch_sampler = ContiguousBatchSampler(
        #     n_samples=len(train_dataset),
        #     batch_size=config.batch_size,
        #     drop_last=False,  # Include all samples
        #     shuffle=True
        # )

        # mono_loader = DataLoader(
        #     train_dataset,
        #     batch_sampler=mono_batch_sampler,
        #     num_workers=config.num_workers,
        #     pin_memory=True,
        #     persistent_workers=config.num_workers > 0,
        #     prefetch_factor=4 if config.num_workers > 0 else None
        # )
    
        mono_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
            persistent_workers=config.num_workers > 0,
            prefetch_factor=4 if config.num_workers > 0 else None
        )
    
    # Load val dataset if provided
    val_loader = None
    if config.val_path:
        print(f"Loading val dataset from {config.val_path}...")
        val_dataset = LMDBFeatureDataset(config.val_path, return_index=False, verbose=False)
        
        # val_batch_sampler = ContiguousBatchSampler(
        #     n_samples=len(val_dataset),
        #     batch_size=config.batch_size,
        #     drop_last=False,
        #     shuffle=False  # Sequential for evaluation
        # )

        # val_loader = DataLoader(
        #     val_dataset,
        #     batch_sampler=val_batch_sampler,
        #     num_workers=config.num_workers,
        #     pin_memory=True,
        #     persistent_workers=config.num_workers > 0,
        #     prefetch_factor=4 if config.num_workers > 0 else None
        # )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=False
        )
        print(f"Val dataset: {len(val_dataset)} samples")
    
    # Load test dataset if provided  
    test_loader = None
    if config.test_path:
        print(f"Loading test dataset from {config.test_path}...")
        test_dataset = LMDBFeatureDataset(config.test_path, return_index=False, verbose=False)
        
        # test_batch_sampler = ContiguousBatchSampler(
        #     n_samples=len(test_dataset),
        #     batch_size=config.batch_size,
        #     drop_last=False,
        #     shuffle=False  # Sequential for evaluation
        # )

        # test_loader = DataLoader(
        #     test_dataset,
        #     batch_sampler=test_batch_sampler,
        #     num_workers=config.num_workers,
        #     pin_memory=True,
        #     persistent_workers=config.num_workers > 0,
        #     prefetch_factor=4 if config.num_workers > 0 else None
        # )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=False
        )
        print(f"Test dataset: {len(test_dataset)} samples")
    
    # Initialize model
    print(f"Initializing {config.model_type} autoencoder: {config.n_latents} latents, {d_model} features")
    
    if config.model_type == 'batch_topk':
        # BatchTopKSAE uses a config dictionary
        model_cfg = {
            'act_size': d_model,
            'dict_size': config.n_latents,
            'top_k': config.topk_k,
            'l1_coeff': 0.0,  # BatchTopK doesn't use L1
            'top_k_aux': config.auxk_k if config.auxk_k else 512,
            'aux_penalty': config.auxk_coef,
            'n_batches_to_dead': config.dead_steps_threshold // config.batch_size,
            'seed': config.seed,
            'device': config.device,
            'dtype': torch.float32,
            'input_unit_norm': config.normalize,
        }
        autoencoder = BatchTopKSAE(model_cfg)
    elif config.model_type == 'vanilla':
        # VanillaSAE uses a config dictionary
        model_cfg = {
            'act_size': d_model,
            'dict_size': config.n_latents,
            'l1_coeff': config.l1_coef,
            'n_batches_to_dead': config.dead_steps_threshold // config.batch_size,
            'seed': config.seed,
            'device': config.device,
            'dtype': torch.float32,
            'input_unit_norm': config.normalize,
        }
        autoencoder = VanillaSAE(model_cfg)
    elif config.model_type == 'jumprelu':
        # JumpReLUSAE uses a config dictionary
        model_cfg = {
            'act_size': d_model,
            'dict_size': config.n_latents,
            'l1_coeff': config.l1_coef,
            'bandwidth': config.bandwidth,
            'n_batches_to_dead': config.dead_steps_threshold // config.batch_size,
            'seed': config.seed,
            'device': config.device,
            'dtype': torch.float32,
            'input_unit_norm': config.normalize,
        }
        autoencoder = JumpReLUSAE(model_cfg)
    else:
        raise ValueError(f"Unknown model type: {config.model_type}. Use 'batch_topk', 'vanilla', or 'jumprelu'")
    
    # Optimizer
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=config.lr, eps=config.eps)
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Training loop
    print(f"Starting training for {config.num_epochs} epochs...")
    autoencoder.train()
    
    global_step = 0
    
    for epoch in range(config.num_epochs):
        # Track metrics for the epoch
        epoch_loss = 0.0
        epoch_l2 = 0.0
        epoch_active = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        
        for batch in pbar:
            # Convert to tensor if needed
            if not isinstance(batch, torch.Tensor):
                batch = torch.from_numpy(batch)
            
            x = batch.to(config.device)
            
            # Forward pass
            latents_pre_act, latents, recons = autoencoder(x)
            
            # Decide mono_coef for this step
            step_mono_coef = config.mono_coef
            if config.mono_coef > 0 and config.mono_period > 1:
                # Periodic mode: only compute mono loss every N steps
                step_mono_coef = 0.0  # Skip batch-level mono loss
            
            # Compute loss using model's compute_loss method
            loss, metrics = autoencoder.compute_loss(
                x.float(),
                latents_pre_act.float(),
                latents.float(),
                recons.float(),
                mono_coef=step_mono_coef
            )
            
            # Backward pass
            loss.backward()
            
            # Periodic full-dataset mono loss via custom autograd (if enabled and it's time)
            if config.mono_coef > 0 and config.mono_period > 1 and (global_step + 1) % config.mono_period == 0:
                print(f"\nComputing full dataset mono loss (custom autograd) at step {global_step + 1}...")
                m = compute_monosemanticity_custom(autoencoder, mono_loader, device=config.device, eps=1e-8, verbose=False)
                active_mask = (m != 0)
                active_count = int(active_mask.sum().item())
                denom = max(active_count, 1)
                # Backprop coef * (1 - mean_active(m))
                scalar = (-config.mono_coef / denom) * m[active_mask].sum()
                scalar.backward()

                if active_count > 0:
                    mono_mean = m[active_mask].mean().item()
                    mono_median = m[active_mask].median().item()
                else:
                    mono_mean = 0.0
                    mono_median = 0.0
                mono_metrics = {
                    'mono_score': mono_mean,
                    'mono_median': mono_median,
                    'mono_active_neurons': active_count,
                }
                metrics.update(mono_metrics)
                print(f"Full dataset mono score: {mono_mean:.4f}")
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), config.max_grad_norm)
            
            # Apply decoder constraints (all models use same method from BaseAutoencoder)
            autoencoder.make_decoder_weights_and_grad_unit_norm()
            
            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()
            
            # Accumulate metrics
            epoch_loss += loss.item()
            epoch_l2 += metrics.get('l2_loss', 0)
            epoch_active += metrics['num_active']
            num_batches += 1
            
            # Update progress bar
            num_dead = (autoencoder.num_batches_not_active > autoencoder.cfg["n_batches_to_dead"]).sum().item()
            l0 = metrics.get('num_active', 0)
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4e}",
                'active': f"{l0:.1f}/{config.n_latents}",
                'dead': num_dead
            })
            
            global_step += 1
        
        # Epoch-end logging
        avg_loss = epoch_loss / num_batches
        avg_l2 = epoch_l2 / num_batches
        avg_active = epoch_active / num_batches

        print(f"Epoch {epoch+1}/{config.num_epochs} - Train Loss: {avg_loss:.4e}, Train L2: {avg_l2:.4e}, Train Active: {avg_active:.1f}")

        # # Save checkpoint immediately after training (before evaluations)
        # checkpoint_path = Path(config.output_dir) / f'autoencoder_epoch{epoch+1}.pt'
        # torch.save({
        #     'epoch': epoch + 1,
        #     'step': global_step,
        #     'model_state_dict': autoencoder.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'config': vars(config),
        # }, checkpoint_path)
        # print(f"Checkpoint saved to {checkpoint_path}")

        # Validation at end of epoch
        if val_loader:
            val_metrics = evaluate(autoencoder, val_loader, config, 'val')
            print(f"Epoch {epoch+1}/{config.num_epochs} - Val Loss: {val_metrics['val_loss']:.4e}, "
                  f"Val L2: {val_metrics['val_l2']:.4e}, "
                  f"Val Active: {val_metrics['val_active']:.1f}")
        
        # R2 at end of epoch (train and val)
        train_r2 = compute_r2(autoencoder, train_loader, config.device, split_name='train')
        print(f"Epoch {epoch+1}/{config.num_epochs} - Train R2: {train_r2:.6f}")
        if val_loader:
            val_r2 = compute_r2(autoencoder, val_loader, config.device, split_name='val')
            print(f"Epoch {epoch+1}/{config.num_epochs} - Val   R2: {val_r2:.6f}")

        # Compute monosemanticity at end of epoch
        train_mono = compute_monosemanticity_fast(autoencoder, train_loader, config.device, 'train', verbose=False)
        train_mono_np = train_mono.cpu().numpy()
        train_mono_mean = np.mean(train_mono_np)
        train_mono_median = np.median(train_mono_np)
        print(f"Epoch {epoch+1}/{config.num_epochs} - Train Mono: mean={train_mono_mean:.4f}, median={train_mono_median:.4f}")
        
        epoch_log_data = {
            'epoch': epoch + 1,
            'train_loss': avg_loss,
            'train_l2': avg_l2,
            'train_active': avg_active,
            'train_r2': train_r2,
            'train_mono_mean': train_mono_mean,
            'train_mono_median': train_mono_median,
        }
        
        if val_loader:
            val_mono = compute_monosemanticity_fast(autoencoder, val_loader, config.device, 'val', verbose=False)
            val_mono_np = val_mono.cpu().numpy()
            val_mono_mean = np.mean(val_mono_np)
            val_mono_median = np.median(val_mono_np)
            print(f"Epoch {epoch+1}/{config.num_epochs} - Val Mono: mean={val_mono_mean:.4f}, median={val_mono_median:.4f}")
            
            epoch_log_data.update({
                **val_metrics,
                'val_r2': val_r2,
                'val_mono_mean': val_mono_mean,
                'val_mono_median': val_mono_median,
            })
        
        if config.wandb_project:
            wandb.log(epoch_log_data)
        
        print("")
    
    # Save final model
    final_path = Path(config.output_dir) / 'autoencoder_final.pt'
    torch.save({
        'epoch': config.num_epochs,
        'step': global_step,
        'model_state_dict': autoencoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': vars(config),
    }, final_path)
    print(f"\nTraining complete! Final model saved to {final_path}")
    
    # Final evaluation on all splits
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    # Dictionary to collect all final results
    final_results = {
        'config': vars(config),
        'final_epoch': config.num_epochs,
        'final_step': global_step,
    }
    
    # Train split
    print("Evaluating on train split...")
    train_metrics = evaluate(autoencoder, train_loader, config, 'train')
    print(f"Train - Loss: {train_metrics['train_loss']:.2e}, "
          f"L2: {train_metrics['train_l2']:.2e}, "
          f"Active: {train_metrics['train_active']:.1f}")
    final_results.update(train_metrics)
    
    # Val split
    if val_loader:
        print("Evaluating on val split...")
        val_metrics = evaluate(autoencoder, val_loader, config, 'val')
        print(f"Val   - Loss: {val_metrics['val_loss']:.2e}, "
              f"L2: {val_metrics['val_l2']:.2e}, "
              f"Active: {val_metrics['val_active']:.1f}")
        final_results.update(val_metrics)
        
        if config.wandb_project:
            wandb.log({**val_metrics})
    
    # Test split
    if test_loader:
        print("Evaluating on test split...")
        test_metrics = evaluate(autoencoder, test_loader, config, 'test')
        print(f"Test  - Loss: {test_metrics['test_loss']:.2e}, "
              f"L2: {test_metrics['test_l2']:.2e}, "
              f"Active: {test_metrics['test_active']:.1f}")
        final_results.update(test_metrics)
        
        if config.wandb_project:
            wandb.log({**test_metrics})
    
    # Compute R2 scores for all splits
    print("\n" + "="*60)
    print("R2 SCORES")
    print("="*60)
    
    print("Computing train R2 score...")
    train_r2_final = compute_r2(autoencoder, train_loader, config.device, split_name='train')
    print(f"Train - R2: {train_r2_final:.6f}")
    final_results['train_r2'] = train_r2_final

    if config.wandb_project:
        wandb.log({'final_train_r2': train_r2_final})

    if val_loader:
        print("Computing val R2 score...")
        val_r2_final = compute_r2(autoencoder, val_loader, config.device, split_name='val')
        print(f"Val   - R2: {val_r2_final:.6f}")
        final_results['val_r2'] = val_r2_final

        if config.wandb_project:
            wandb.log({'final_val_r2': val_r2_final})

    if test_loader:
        print("Computing test R2 score...")
        test_r2_final = compute_r2(autoencoder, test_loader, config.device, split_name='test')
        print(f"Test  - R2: {test_r2_final:.6f}")
        final_results['test_r2'] = test_r2_final
        
        if config.wandb_project:
            wandb.log({'final_test_r2': test_r2_final})
    
    # Compute monosemanticity scores
    print("\n" + "="*60)
    print("MONOSEMANTICITY EVALUATION")
    print("="*60)
    
    print("Computing train monosemanticity score...")
    train_mono = compute_monosemanticity_fast(autoencoder, train_loader, config.device, 'train', verbose=True)

    # Compute statistics: median, mean, 90th percentile
    train_mono_np = train_mono.cpu().numpy()
    train_mono_mean = np.mean(train_mono_np)
    train_mono_median = np.median(train_mono_np)
    train_mono_p90 = np.percentile(train_mono_np, 90)
    
    print(f"Train - Monosemanticity: mean={train_mono_mean:.4f}, median={train_mono_median:.4f}, 90th={train_mono_p90:.4f}")
    final_results['train_monosemanticity_mean'] = float(train_mono_mean)
    final_results['train_monosemanticity_median'] = float(train_mono_median)
    final_results['train_monosemanticity_p90'] = float(train_mono_p90)
    
    if config.wandb_project:
        wandb.log({
            'train_monosemanticity_mean': train_mono_mean,
            'train_monosemanticity_median': train_mono_median,
            'train_monosemanticity_p90': train_mono_p90
        })
    
    if val_loader:
        print("Computing val monosemanticity score (fast)...")
        val_mono = compute_monosemanticity_fast(autoencoder, val_loader, config.device, 'val', verbose=True)
        
        # Compute statistics: median, mean, 90th percentile
        val_mono_np = val_mono.cpu().numpy()
        val_mono_mean = np.mean(val_mono_np)
        val_mono_median = np.median(val_mono_np)
        val_mono_p90 = np.percentile(val_mono_np, 90)
        
        print(f"Val   - Monosemanticity: mean={val_mono_mean:.4f}, median={val_mono_median:.4f}, 90th={val_mono_p90:.4f}")
        final_results['val_monosemanticity_mean'] = float(val_mono_mean)
        final_results['val_monosemanticity_median'] = float(val_mono_median)
        final_results['val_monosemanticity_p90'] = float(val_mono_p90)
        
        if config.wandb_project:
            wandb.log({
                'val_monosemanticity_mean': val_mono_mean,
                'val_monosemanticity_median': val_mono_median,
                'val_monosemanticity_p90': val_mono_p90
            })
    
    if test_loader:
        print("Computing test monosemanticity score (fast)...")
        test_mono = compute_monosemanticity_fast(autoencoder, test_loader, config.device, 'test', verbose=True)
        
        # Compute statistics: median, mean, 90th percentile
        test_mono_np = test_mono.cpu().numpy()
        test_mono_mean = np.mean(test_mono_np)
        test_mono_median = np.median(test_mono_np)
        test_mono_p90 = np.percentile(test_mono_np, 90)
        
        print(f"Test  - Monosemanticity: mean={test_mono_mean:.4f}, median={test_mono_median:.4f}, 90th={test_mono_p90:.4f}")
        final_results['test_monosemanticity_mean'] = float(test_mono_mean)
        final_results['test_monosemanticity_median'] = float(test_mono_median)
        final_results['test_monosemanticity_p90'] = float(test_mono_p90)
        
        if config.wandb_project:
            wandb.log({
                'test_monosemanticity_mean': test_mono_mean,
                'test_monosemanticity_median': test_mono_median,
                'test_monosemanticity_p90': test_mono_p90
            })
    
    print("="*60)
    
    # Save all results to JSON file
    results_path = Path(config.output_dir) / 'results.json'
    print(f"\nSaving results to {results_path}")
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    print(f"Results saved successfully!")
    
    # Close datasets
    train_dataset.close()
    if val_loader:
        val_dataset.close()
    if test_loader:
        test_dataset.close()
    
    if config.wandb_project:
        wandb.finish()
    
    return autoencoder

