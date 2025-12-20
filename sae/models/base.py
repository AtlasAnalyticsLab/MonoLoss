
import torch
from torch import nn 

class BaseAutoencoder(nn.Module):
    """Base class for autoencoder models."""

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        torch.manual_seed(self.cfg["seed"])

        self.b_dec = nn.Parameter(torch.zeros(self.cfg["act_size"]))
        self.b_enc = nn.Parameter(torch.zeros(self.cfg["dict_size"]))
        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.cfg["act_size"], self.cfg["dict_size"])
            )
        )
        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.cfg["dict_size"], self.cfg["act_size"])
            )
        )
        self.W_dec.data[:] = self.W_enc.t().data
        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        self.num_batches_not_active = torch.zeros((self.cfg["dict_size"],)).to(
            cfg["device"]
        )

        self.to(cfg["dtype"]).to(cfg["device"])

    def preprocess_input(self, x):
        if self.cfg.get("input_unit_norm", False):
            x_mean = x.mean(dim=-1, keepdim=True)
            x = x - x_mean
            x_std = x.std(dim=-1, keepdim=True)
            x = x / (x_std + 1e-5)
            return x, x_mean, x_std
        else:
            return x, None, None

    def postprocess_output(self, x_reconstruct, x_mean, x_std):
        if self.cfg.get("input_unit_norm", False):
            x_reconstruct = x_reconstruct * x_std + x_mean
        return x_reconstruct

    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(
            -1, keepdim=True
        ) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj
        self.W_dec.data = W_dec_normed

    def update_inactive_features(self, acts):
        self.num_batches_not_active += (acts.sum(0) == 0).float()
        self.num_batches_not_active[acts.sum(0) > 0] = 0

