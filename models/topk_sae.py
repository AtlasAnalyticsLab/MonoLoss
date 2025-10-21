"""TopK Sparse Autoencoder implementation."""

from typing import Callable, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


"""
Loss functions for sparse autoencoders.

Implementation approach:
- NMSE (Normalized MSE): Per-sample normalization for scale-invariance
- AuxK Loss: Auxiliary loss for dead neuron revival using NMSE
- L1 Sparsity: Tracked as metric only (not in loss)
- Monosemanticity Loss: Custom addition for semantic coherence

References:
- OpenAI Sparse Autoencoder: https://github.com/openai/sparse_autoencoder
"""
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from tqdm import tqdm
import numpy as np
from loss import compute_monosemanticity_loss_batch


# ============================================================================
# RECONSTRUCTION LOSSES
# ============================================================================

def mse_loss(recons: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Compute mean squared error reconstruction loss.
    [DEPRECATED: Use normalized_mse_loss for scale-invariant training]
    
    Args:
        recons: Reconstructed features (B, D)
        x: Original features (B, D)
    
    Returns:
        MSE loss scalar
    """
    return F.mse_loss(recons, x)


def normalized_mse_loss(recons: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Compute NMSE (Normalized MSE) loss with per-sample normalization.
    
    Normalizes squared error per sample by dividing by the mean squared input value
    for that sample. This makes the loss scale-invariant and prevents samples with 
    larger magnitudes from dominating.
    
    Args:
        recons: Reconstructed features (B, D)
        x: Original features (B, D)
    
    Returns:
        NMSE loss scalar
    
    Formula:
        NMSE = mean_over_batch( mean_over_features((recons - x)^2) / mean_over_features(x^2) )
    """
    return (
        ((recons - x) ** 2).mean(dim=1) / (x ** 2).mean(dim=1)
    ).mean()


def auxk_loss(autoencoder, x: torch.Tensor, latents_pre_act: torch.Tensor, 
              recons: torch.Tensor, dead_steps_threshold: int, 
              k: int, eps: float = 1e-3) -> Tuple[torch.Tensor, int]:
    """
    Compute auxiliary TopK loss for dead neuron coverage.
    
    This loss reconstructs the residual using a second set of top-k activations,
    masking out neurons that have been recently active (non-dead neurons).
    
    Args:
        autoencoder: The autoencoder model
        x: Input features (B, D)
        latents_pre_act: Pre-activation latents (B, N)
        recons: Main reconstruction from primary top-k (B, D)
        dead_steps_threshold: Threshold for considering a neuron dead
        k: Number of auxiliary top-k neurons to select
        eps: Threshold for considering activation as active
    
    Returns:
        auxk_loss: Auxiliary reconstruction loss (NMSE on residual)
        num_auxk_active: Number of auxiliary neurons selected
    """
    # Get dead neuron mask (neurons that haven't fired recently)
    dead_mask = autoencoder.stats_last_nonzero > dead_steps_threshold
    
    # Mask out non-dead neurons from pre-activation
    masked_pre_act = latents_pre_act.clone()
    masked_pre_act[:, ~dead_mask] = float('-inf')
    
    # Get top-k auxiliary activations from dead neurons
    auxk_vals, auxk_indices = torch.topk(masked_pre_act, k=k, dim=-1)
    
    # Apply ReLU activation
    auxk_acts = torch.relu(auxk_vals)
    
    # Create sparse activation tensor
    auxk_latents = torch.zeros_like(latents_pre_act)
    auxk_latents.scatter_(-1, auxk_indices, auxk_acts)
    
    # Reconstruct from auxiliary latents
    _, info = autoencoder.preprocess(x)
    auxk_recons = autoencoder.decode(auxk_latents, info)
    
    # Auxiliary loss: reconstruct the residual using NMSE
    # Target is the residual between input and main reconstruction
    # Detach to avoid gradients flowing back through main reconstruction
    residual = x - recons.detach() + autoencoder.pre_bias.detach()
    loss = normalized_mse_loss(auxk_recons, residual)
    
    # Count how many auxiliary neurons were actually active
    num_active = (auxk_acts > eps).sum().item()
    
    return loss, num_active


def compute_autoencoder_loss(autoencoder, x: torch.Tensor, latents_pre_act: torch.Tensor,
                             latents: torch.Tensor, recons: torch.Tensor,
                             auxk_coef: float = 0.0, dead_steps_threshold: int = 1000,
                             auxk_k: Optional[int] = None, mono_coef: float = 0.0) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute total autoencoder loss with metrics.
    
    Loss = NMSE + auxk_coef * auxk_loss + mono_coef * mono_loss
    
    Args:
        autoencoder: The autoencoder model
        x: Input features (B, D)
        latents_pre_act: Pre-activation latents (B, N)
        latents: Activated latents (B, N)
        recons: Reconstructions (B, D)
        auxk_coef: Coefficient for auxiliary loss
        dead_steps_threshold: Threshold for dead neurons
        auxk_k: Number of auxiliary top-k neurons (if None, uses same k as model)
        mono_coef: Coefficient for monosemanticity loss (1 - mono_score)
    
    Returns:
        total_loss: Combined loss scalar
        metrics: Dictionary of metrics for logging
    """
    # ====================================================================
    # Main reconstruction loss - CHOOSE ONE:
    # ====================================================================
    recon_loss = normalized_mse_loss(recons, x)
    
    # ====================================================================
    
    # Compute sparsity metrics (L1 tracked as metric only, not added to loss)
    num_active = (latents > 0).float().sum(dim=-1).mean()
    l0_sparsity = num_active.item()
    l1_sparsity = latents.abs().sum(dim=-1).mean().item()
    
    metrics = {
        # Primary reconstruction loss (switch key based on which loss is active)
        'nmse_loss': recon_loss.item(),  
        
        # Sparsity metrics
        'num_active': l0_sparsity,
        'l1_sparsity': l1_sparsity,
    }
    
    total_loss = recon_loss
    
    # Auxiliary loss (if coefficient > 0 and model uses TopK)
    if auxk_coef > 0 and hasattr(autoencoder, 'activation'):
        from models import TopK
        if isinstance(autoencoder.activation, TopK):
            if auxk_k is None:
                auxk_k = autoencoder.activation.k
            
            aux_loss, num_auxk = auxk_loss(
                autoencoder, x, latents_pre_act, recons,
                dead_steps_threshold, auxk_k
            )
            
            total_loss = total_loss + auxk_coef * aux_loss
            metrics['auxk_loss'] = aux_loss.item()
            metrics['num_auxk_active'] = num_auxk
    
    # Monosemanticity loss (if coefficient > 0)
    if mono_coef > 0:
        mono_loss, mono_metrics = compute_monosemanticity_loss_batch(x, latents)
        total_loss = total_loss + mono_coef * mono_loss
        metrics.update(mono_metrics)
    
    return total_loss, metrics


def LN(x: torch.Tensor, eps: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Layer normalization helper function."""
    mu = x.mean(dim=-1, keepdim=True)
    x = x - mu
    std = x.std(dim=-1, keepdim=True)
    x = x / (std + eps)
    return x, mu, std


class TiedTranspose(nn.Module):
    """Tied decoder weights (transpose of encoder)."""
    
    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.linear = linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.linear.bias is None
        return F.linear(x, self.linear.weight.t(), None)

    @property
    def weight(self) -> torch.Tensor:
        return self.linear.weight.t()

    @property
    def bias(self) -> torch.Tensor:
        return self.linear.bias


class TopK(nn.Module):
    """TopK activation function that keeps only top-k values."""
    
    def __init__(self, k: int, postact_fn: Callable = nn.ReLU()) -> None:
        super().__init__()
        self.k = k
        self.postact_fn = postact_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        topk = torch.topk(x, k=self.k, dim=-1)
        values = self.postact_fn(topk.values)
        # make all other values 0
        result = torch.zeros_like(x)
        result.scatter_(-1, topk.indices, values)
        return result

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state_dict = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        state_dict.update({prefix + "k": self.k, prefix + "postact_fn": self.postact_fn.__class__.__name__})
        return state_dict

    @classmethod
    def from_state_dict(cls, state_dict: dict[str, torch.Tensor], strict: bool = True) -> "TopK":
        k = state_dict["k"]
        postact_fn = ACTIVATIONS_CLASSES[state_dict["postact_fn"]]()
        return cls(k=k, postact_fn=postact_fn)


class TopKSAE(nn.Module):
    """Sparse autoencoder with TopK activation.

    Implements:
        latents = activation(encoder(x - pre_bias) + latent_bias)
        recons = decoder(latents) + pre_bias
    """

    def __init__(
        self, n_latents: int, n_inputs: int, activation: Callable = nn.ReLU(), tied: bool = False,
        normalize: bool = False
    ) -> None:
        """
        Args:
            n_latents: dimension of the autoencoder latent
            n_inputs: dimensionality of the original data (e.g residual stream, number of MLP hidden units)
            activation: activation function
            tied: whether to tie the encoder and decoder weights
            normalize: whether to apply layer normalization
        """
        super().__init__()

        self.pre_bias = nn.Parameter(torch.zeros(n_inputs))
        self.encoder: nn.Module = nn.Linear(n_inputs, n_latents, bias=False)
        self.latent_bias = nn.Parameter(torch.zeros(n_latents))
        self.activation = activation
        if tied:
            self.decoder: nn.Linear | TiedTranspose = TiedTranspose(self.encoder)
        else:
            self.decoder = nn.Linear(n_latents, n_inputs, bias=False)
        self.normalize = normalize

        self.stats_last_nonzero: torch.Tensor
        self.latents_activation_frequency: torch.Tensor
        self.latents_mean_square: torch.Tensor
        self.register_buffer("stats_last_nonzero", torch.zeros(n_latents, dtype=torch.long))
        self.register_buffer(
            "latents_activation_frequency", torch.ones(n_latents, dtype=torch.float)
        )
        self.register_buffer("latents_mean_square", torch.zeros(n_latents, dtype=torch.float))

    def encode_pre_act(self, x: torch.Tensor, latent_slice: slice = slice(None)) -> torch.Tensor:
        """
        Encode input to pre-activation latents.
        
        Args:
            x: input data (shape: [batch, n_inputs])
            latent_slice: slice of latents to compute
                Example: latent_slice = slice(0, 10) to compute only the first 10 latents.
        
        Returns:
            autoencoder latents before activation (shape: [batch, n_latents])
        """
        x = x - self.pre_bias
        latents_pre_act = F.linear(
            x, self.encoder.weight[latent_slice], self.latent_bias[latent_slice]
        )
        return latents_pre_act

    def preprocess(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        """Apply optional preprocessing (layer norm)."""
        if not self.normalize:
            return x, dict()
        x, mu, std = LN(x)
        return x, dict(mu=mu, std=std)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Encode input to latents.
        
        Args:
            x: input data (shape: [batch, n_inputs])
        
        Returns:
            autoencoder latents (shape: [batch, n_latents])
            preprocessing info dict
        """
        x, info = self.preprocess(x)
        return self.activation(self.encode_pre_act(x)), info

    def decode(self, latents: torch.Tensor, info: dict[str, Any] | None = None) -> torch.Tensor:
        """
        Decode latents to reconstructed data.
        
        Args:
            latents: autoencoder latents (shape: [batch, n_latents])
            info: preprocessing info dict from encode
        
        Returns:
            reconstructed data (shape: [batch, n_inputs])
        """
        ret = self.decoder(latents) + self.pre_bias
        if self.normalize:
            assert info is not None
            ret = ret * info["std"] + info["mu"]
        return ret

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through autoencoder.
        
        Args:
            x: input data (shape: [batch, n_inputs])
        
        Returns:
            latents_pre_act: autoencoder latents pre activation (shape: [batch, n_latents])
            latents: autoencoder latents (shape: [batch, n_latents])
            recons: reconstructed data (shape: [batch, n_inputs])
        """
        x, info = self.preprocess(x)
        latents_pre_act = self.encode_pre_act(x)
        latents = self.activation(latents_pre_act)
        recons = self.decode(latents, info)

        # Update dead neuron statistics
        # set all indices of self.stats_last_nonzero where (latents != 0) to 0
        self.stats_last_nonzero *= (latents == 0).all(dim=0).long()
        self.stats_last_nonzero += 1

        return latents_pre_act, latents, recons

    def compute_loss(self, x: torch.Tensor, latents_pre_act: torch.Tensor, latents: torch.Tensor, recons: torch.Tensor,
                    auxk_coef: float = 0.0, dead_steps_threshold: int = 1000,
                    auxk_k: Optional[int] = None, mono_coef: float = 0.0) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total autoencoder loss with metrics.
        """
        return compute_autoencoder_loss(self, x, latents_pre_act, latents, recons, auxk_coef, dead_steps_threshold, auxk_k, mono_coef)


    @classmethod
    def from_state_dict(
        cls, state_dict: dict[str, torch.Tensor], strict: bool = True
    ) -> "TopKSAE":
        """Load autoencoder from state dict."""
        n_latents, d_model = state_dict["encoder.weight"].shape

        # Retrieve activation
        activation_class_name = state_dict.pop("activation", "ReLU")
        activation_class = ACTIVATIONS_CLASSES.get(activation_class_name, nn.ReLU)
        normalize = activation_class_name == "TopK"  # NOTE: hacky way to determine if normalization is enabled
        activation_state_dict = state_dict.pop("activation_state_dict", {})
        if hasattr(activation_class, "from_state_dict"):
            activation = activation_class.from_state_dict(
                activation_state_dict, strict=strict
            )
        else:
            activation = activation_class()
            if hasattr(activation, "load_state_dict"):
                activation.load_state_dict(activation_state_dict, strict=strict)

        autoencoder = cls(n_latents, d_model, activation=activation, normalize=normalize)
        # Load remaining state dict
        autoencoder.load_state_dict(state_dict, strict=strict)
        return autoencoder

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Save state dict with activation info."""
        sd = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        sd[prefix + "activation"] = self.activation.__class__.__name__
        if hasattr(self.activation, "state_dict"):
            sd[prefix + "activation_state_dict"] = self.activation.state_dict()
        return sd


# Activation classes registry
ACTIVATIONS_CLASSES = {
    "ReLU": nn.ReLU,
    "Identity": nn.Identity,
    "TopK": TopK,
}

