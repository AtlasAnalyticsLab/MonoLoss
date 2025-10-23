import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseAutoencoder
from loss import compute_monosemanticity_loss_batch

class BatchTopKSAE(BaseAutoencoder):
    def __init__(self, cfg):
        super().__init__(cfg)

    def forward(self, x):
        x_preprocessed, x_mean, x_std = self.preprocess_input(x)

        x_cent = x_preprocessed - self.b_dec
        acts = F.relu(x_cent @ self.W_enc)
        acts_topk = torch.topk(acts.flatten(), self.cfg["top_k"] * x.shape[0], dim=-1)
        latents = (
            torch.zeros_like(acts.flatten())
            .scatter(-1, acts_topk.indices, acts_topk.values)
            .reshape(acts.shape)
        )
        x_reconstruct = latents @ self.W_dec + self.b_dec

        self.update_inactive_features(latents)
        recons = self.postprocess_output(x_reconstruct, x_mean, x_std)
        
        # Store preprocessed x for mono loss computation
        self._x_preprocessed = x_preprocessed
        
        # Return (acts, latents, recons) to match TopKSAE interface
        return acts, latents, recons

    def compute_loss(self, x: torch.Tensor, acts: torch.Tensor, latents: torch.Tensor, recons: torch.Tensor, mono_coef: float = 0.0):
        l2_loss = (recons.float() - x.float()).pow(2).mean()
        l1_norm = latents.float().abs().sum(-1).mean()
        l1_loss = self.cfg["l1_coeff"] * l1_norm
        l0_norm = (latents > 0).float().sum(-1).mean()
        aux_loss = self.get_auxiliary_loss(x, recons, acts)
        loss = l2_loss + l1_loss + aux_loss
        num_dead_features = (
            self.num_batches_not_active > self.cfg["n_batches_to_dead"]
        ).sum()
            
        output = {
            "num_dead_features": num_dead_features.item(),
            "l1_sparsity": l1_loss.item(),
            "l2_loss": l2_loss.item(),
            "num_active": l0_norm.item(),
            "auxk_loss": aux_loss.item(),
        }

        # Monosemanticity loss (if coefficient > 0)
        if mono_coef > 0:
            # Use preprocessed x to match the input space where latents were computed
            x_for_mono = self._x_preprocessed if hasattr(self, '_x_preprocessed') else x
            mono_loss, mono_metrics = compute_monosemanticity_loss_batch(x_for_mono, latents)
            loss = loss + mono_coef * mono_loss
            output.update(mono_metrics)
        

        return loss, output

    def get_auxiliary_loss(self, x, x_reconstruct, acts):
        dead_features = self.num_batches_not_active >= self.cfg["n_batches_to_dead"]
        if dead_features.sum() > 0:
            residual = x.float() - x_reconstruct.float()
            acts_topk_aux = torch.topk(
                acts[:, dead_features],
                min(self.cfg["top_k_aux"], dead_features.sum()),
                dim=-1,
            )
            acts_aux = torch.zeros_like(acts[:, dead_features]).scatter(
                -1, acts_topk_aux.indices, acts_topk_aux.values
            )
            x_reconstruct_aux = acts_aux @ self.W_dec[dead_features]
            l2_loss_aux = (
                self.cfg["aux_penalty"]
                * (x_reconstruct_aux.float() - residual.float()).pow(2).mean()
            )
            return l2_loss_aux
        else:
            return torch.tensor(0, dtype=x.dtype, device=x.device)
