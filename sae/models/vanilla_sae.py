import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseAutoencoder
from loss import compute_monosemanticity_loss_batch


class VanillaSAE(BaseAutoencoder):
    """Vanilla Sparse Autoencoder with L1 sparsity penalty."""
    
    def __init__(self, cfg):
        super().__init__(cfg)

    def forward(self, x):
        """
        Forward pass through autoencoder.
        
        Args:
            x: input data (shape: [batch, n_inputs])
        
        Returns:
            acts: ReLU activations before sparsity (shape: [batch, n_latents])
            latents: Same as acts for vanilla SAE (shape: [batch, n_latents])
            recons: reconstructed data (shape: [batch, n_inputs])
        """
        x_preprocessed, x_mean, x_std = self.preprocess_input(x)
        
        x_cent = x_preprocessed - self.b_dec
        latents_pre_act = x_cent @ self.W_enc + self.b_enc
        acts = F.relu(latents_pre_act)
        x_reconstruct = acts @ self.W_dec + self.b_dec
        
        self.update_inactive_features(acts)
        recons = self.postprocess_output(x_reconstruct, x_mean, x_std)
        
        # Store preprocessed x for mono loss computation
        self._x_preprocessed = x_preprocessed
        
        # Return (acts, latents, recons) to match interface
        # For vanilla SAE, acts and latents are the same
        return latents_pre_act, acts, recons

    def compute_loss(self, x: torch.Tensor, acts: torch.Tensor, latents: torch.Tensor, 
                     recons: torch.Tensor, mono_coef: float = 0.0):
        """
        Compute loss for vanilla SAE.
        
        Args:
            x: Input features (B, D)
            acts: Activations (B, N)
            latents: Same as acts for vanilla SAE (B, N)
            recons: Reconstructions (B, D)
            mono_coef: Coefficient for monosemanticity loss
        
        Returns:
            loss: Total loss scalar
            output: Dictionary of metrics for logging
        """
        l2_loss = (recons.float() - x.float()).pow(2).mean()# / (x.float() ** 2).mean()
        l1_norm = latents.float().abs().sum(-1).mean()
        l1_loss = self.cfg["l1_coeff"] * l1_norm
        l0_norm = (latents > 0).float().sum(-1).mean()
        loss = l2_loss + l1_loss
        
        num_dead_features = (
            self.num_batches_not_active > self.cfg["n_batches_to_dead"]
        ).sum()

        output = {
            "num_dead_features": num_dead_features.item(),
            "l1_sparsity": l1_loss.item(),
            "l2_loss": l2_loss.item(),
            "num_active": l0_norm.item(),
            "l1_norm": l1_norm.item(),
        }

        # Monosemanticity loss (if coefficient > 0)
        if mono_coef > 0:
            # Use preprocessed x to match the input space where latents were computed
            x_for_mono = self._x_preprocessed if hasattr(self, '_x_preprocessed') else x
            mono_loss, mono_metrics = compute_monosemanticity_loss_batch(x_for_mono, latents)
            loss = loss + mono_coef * mono_loss
            output.update(mono_metrics)

        return loss, output