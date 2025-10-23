import torch
import torch.nn as nn
import torch.autograd as autograd

from .base import BaseAutoencoder
from loss import compute_monosemanticity_loss_batch


class RectangleFunction(autograd.Function):
    """Rectangle function with straight-through estimator for gradients."""
    
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return ((x > -0.5) & (x < 0.5)).float()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[(x <= -0.5) | (x >= 0.5)] = 0
        return grad_input


class JumpReLUFunction(autograd.Function):
    """JumpReLU activation with learnable thresholds."""
    
    @staticmethod
    def forward(ctx, x, log_threshold, bandwidth):
        ctx.save_for_backward(x, log_threshold, torch.tensor(bandwidth))
        threshold = torch.exp(log_threshold)
        return x * (x > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, log_threshold, bandwidth_tensor = ctx.saved_tensors
        bandwidth = bandwidth_tensor.item()
        threshold = torch.exp(log_threshold)
        x_grad = (x > threshold).float() * grad_output
        threshold_grad = (
            -(threshold / bandwidth)
            * RectangleFunction.apply((x - threshold) / bandwidth)
            * grad_output
        )
        return x_grad, threshold_grad, None  # None for bandwidth


class JumpReLU(nn.Module):
    """JumpReLU activation layer with per-feature learnable thresholds."""
    
    def __init__(self, feature_size, bandwidth, device='cpu'):
        super(JumpReLU, self).__init__()
        self.log_threshold = nn.Parameter(torch.zeros(feature_size, device=device))
        self.bandwidth = bandwidth

    def forward(self, x):
        return JumpReLUFunction.apply(x, self.log_threshold, self.bandwidth)


class StepFunction(autograd.Function):
    """Step function for computing L0 with smooth gradients."""
    
    @staticmethod
    def forward(ctx, x, log_threshold, bandwidth):
        ctx.save_for_backward(x, log_threshold, torch.tensor(bandwidth))
        threshold = torch.exp(log_threshold)
        return (x > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, log_threshold, bandwidth_tensor = ctx.saved_tensors
        bandwidth = bandwidth_tensor.item()
        threshold = torch.exp(log_threshold)
        x_grad = torch.zeros_like(x)
        threshold_grad = (
            -(1.0 / bandwidth)
            * RectangleFunction.apply((x - threshold) / bandwidth)
            * grad_output
        )
        return x_grad, threshold_grad, None  # None for bandwidth


class JumpReLUSAE(BaseAutoencoder):
    """Sparse Autoencoder with JumpReLU activation (learnable thresholds)."""
    
    def __init__(self, cfg):
        super().__init__(cfg)
        self.jumprelu = JumpReLU(
            feature_size=cfg["dict_size"], 
            bandwidth=cfg["bandwidth"], 
            device=cfg["device"]
        )

    def forward(self, x, use_pre_enc_bias=False):
        """
        Forward pass through autoencoder.
        
        Args:
            x: input data (shape: [batch, n_inputs])
            use_pre_enc_bias: whether to use pre-encoder bias
        
        Returns:
            pre_activations: ReLU activations before JumpReLU (shape: [batch, n_latents])
            feature_magnitudes: JumpReLU activations (shape: [batch, n_latents])
            recons: reconstructed data (shape: [batch, n_inputs])
        """
        x_preprocessed, x_mean, x_std = self.preprocess_input(x)

        if use_pre_enc_bias:
            x_preprocessed = x_preprocessed - self.b_dec

        pre_activations = torch.relu(x_preprocessed @ self.W_enc + self.b_enc)
        feature_magnitudes = self.jumprelu(pre_activations)

        x_reconstructed = feature_magnitudes @ self.W_dec + self.b_dec

        self.update_inactive_features(feature_magnitudes)
        recons = self.postprocess_output(x_reconstructed, x_mean, x_std)
        
        # Store preprocessed x for mono loss computation
        self._x_preprocessed = x_preprocessed
        
        # Return (pre_activations, feature_magnitudes, recons) to match interface
        return pre_activations, feature_magnitudes, recons

    def compute_loss(self, x: torch.Tensor, pre_activations: torch.Tensor, 
                     latents: torch.Tensor, recons: torch.Tensor, mono_coef: float = 0.0):
        """
        Compute loss for JumpReLU SAE.
        
        Args:
            x: Input features (B, D)
            pre_activations: ReLU activations before JumpReLU (B, N)
            latents: JumpReLU activations (B, N)
            recons: Reconstructions (B, D)
            mono_coef: Coefficient for monosemanticity loss
        
        Returns:
            loss: Total loss scalar
            output: Dictionary of metrics for logging
        """
        l2_loss = (recons.float() - x.float()).pow(2).mean()

        # L0 computed via StepFunction (differentiable approximation)
        l0 = StepFunction.apply(
            latents, 
            self.jumprelu.log_threshold, 
            self.cfg["bandwidth"]
        ).sum(dim=-1).mean()
        
        l0_loss = self.cfg["l1_coeff"] * l0
        l1_loss = l0_loss  # For JumpReLU, we penalize L0 directly

        loss = l2_loss + l1_loss
        
        num_dead_features = (
            self.num_batches_not_active > self.cfg["n_batches_to_dead"]
        ).sum()

        output = {
            "num_dead_features": num_dead_features.item(),
            "l1_sparsity": l1_loss.item(),
            "l2_loss": l2_loss.item(),
            "num_active": l0.item(),
            "l0_norm": l0.item(),
            "l1_norm": l0.item(),
        }

        # Monosemanticity loss (if coefficient > 0)
        if mono_coef > 0:
            # Use preprocessed x to match the input space where latents were computed
            x_for_mono = self._x_preprocessed if hasattr(self, '_x_preprocessed') else x
            mono_loss, mono_metrics = compute_monosemanticity_loss_batch(x_for_mono, latents)
            loss = loss + mono_coef * mono_loss
            output.update(mono_metrics)

        return loss, output

