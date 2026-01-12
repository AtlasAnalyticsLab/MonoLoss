import torch
import torch.nn.functional as F
from torch.autograd import Function
from tqdm import tqdm

class MonosemanticityFunction(Function):
    @staticmethod
    def forward(ctx, trigger, autoencoder, dataloader, device='cuda', eps=1e-8, verbose=False):
        # EXACT forward = your compute_monosemanticity_fast
        was_training = autoencoder.training
        autoencoder.eval()

        # Pass 1: min/max
        min_vals = None
        max_vals = None
        num_neurons = None
        feature_dim = None

        for batch in tqdm(dataloader, desc="Computing stats", disable=not verbose):
            if not isinstance(batch, torch.Tensor):
                batch = torch.from_numpy(batch)
            x = batch.to(device)
            with torch.amp.autocast(device_type=device.split(':')[0] if ':' in device else device):
                _, latents, _ = autoencoder(x)

            if min_vals is None:
                num_neurons = latents.shape[1]
                feature_dim = x.shape[1]
                min_vals = latents.min(dim=0, keepdim=True)[0]
                max_vals = latents.max(dim=0, keepdim=True)[0]
            else:
                min_vals = torch.min(min_vals, latents.min(dim=0, keepdim=True)[0])
                max_vals = torch.max(max_vals, latents.max(dim=0, keepdim=True)[0])

        # Pass 2: accumulators
        sum_a  = torch.zeros(num_neurons, device=device, dtype=torch.float32)
        sum_a2 = torch.zeros(num_neurons, device=device, dtype=torch.float32)
        V      = torch.zeros(feature_dim, num_neurons, device=device, dtype=torch.float32)

        for batch in tqdm(dataloader, desc="Accumulating", disable=not verbose):
            if not isinstance(batch, torch.Tensor):
                batch = torch.from_numpy(batch)
            x = batch.to(device)
            with torch.amp.autocast(device_type=device.split(':')[0] if ':' in device else device):
                _, latents, _ = autoencoder(x)

            a = (latents - min_vals) / (max_vals - min_vals + eps)  # (B,M)
            p = F.normalize(x, p=2, dim=1, eps=1e-8)

            a32 = a.float()
            p32 = p.float()
            sum_a  += a32.sum(dim=0)
            sum_a2 += (a32 * a32).sum(dim=0)
            V      += p32.T @ a32

        V_norm_sq = (V * V).sum(dim=0)
        W = 0.5 * (V_norm_sq - sum_a2)
        Z = 0.5 * (sum_a * sum_a - sum_a2)

        m = torch.where(
            Z > 1e-8,
            W / (Z + 1e-12),
            torch.zeros_like(Z)
        )

        # Save light state for backward replay
        ctx.autoencoder = autoencoder
        ctx.dataloader = dataloader
        ctx.device = device
        ctx.eps = eps
        ctx.num_neurons = num_neurons
        ctx.feature_dim = feature_dim
        ctx.save_for_backward(min_vals.detach(), max_vals.detach())

        autoencoder.train(was_training)
        return m + trigger.sum() * 0.0  # keep grad_fn alive

    @staticmethod
    def backward(ctx, grad_output):
        autoencoder = ctx.autoencoder
        dataloader = ctx.dataloader
        device = ctx.device
        eps = ctx.eps
        (min_vals, max_vals) = ctx.saved_tensors
        M = ctx.num_neurons
        D = ctx.feature_dim

        was_training = autoencoder.training
        autoencoder.eval()

        # ---- Rebuild q, sum_a, sum_a2, W, Z (no grad) ----
        with torch.no_grad():
            sum_a  = torch.zeros(M, device=device, dtype=torch.float32)
            sum_a2 = torch.zeros(M, device=device, dtype=torch.float32)
            V      = torch.zeros(D, M, device=device, dtype=torch.float32)

            for batch in dataloader:
                if not isinstance(batch, torch.Tensor):
                    batch = torch.from_numpy(batch)
                x = batch.to(device)
                with torch.amp.autocast(device_type=device.split(':')[0] if ':' in device else device):
                    _, latents, _ = autoencoder(x)

                a = (latents - min_vals) / (max_vals - min_vals + eps)
                p = F.normalize(x, p=2, dim=1, eps=1e-8)
                a32, p32 = a.float(), p.float()
                sum_a  += a32.sum(dim=0)
                sum_a2 += (a32 * a32).sum(dim=0)
                V      += p32.T @ a32

            q = V  # (D,M)
            V_norm_sq = (q * q).sum(dim=0)          # (M,)
            W = 0.5 * (V_norm_sq - sum_a2)
            Z = 0.5 * (sum_a * sum_a - sum_a2)

            active = (Z > 1e-8)
            Z_safe = Z.clone()
            Z_safe[~active] = 1.0

            go = grad_output.to(device).float()
            dLdW = torch.zeros_like(go)
            dLdZ = torch.zeros_like(go)
            dLdW[active] = go[active] / (Z_safe[active] + 1e-12)
            dLdZ[active] = -go[active] * W[active] / (Z_safe[active]**2 + 1e-12)

            G = q * dLdW  # (D,M), helper for intuition; not needed directly below.

        denom = (max_vals - min_vals + eps)  # (1,M)

        # ---- B1: global S1/S2 and tie counts (streaming, no grad) ----
        with torch.no_grad():
            S1_total = torch.zeros(M, device=device, dtype=torch.float32)
            S2_total = torch.zeros(M, device=device, dtype=torch.float32)
            min_count = torch.zeros(M, device=device, dtype=torch.float32)
            max_count = torch.zeros(M, device=device, dtype=torch.float32)

            for batch in dataloader:
                if not isinstance(batch, torch.Tensor):
                    batch = torch.from_numpy(batch)
                x = batch.to(device)

                with torch.amp.autocast(device_type=device.split(':')[0] if ':' in device else device):
                    _, latents, _ = autoencoder(x)      # (B,M)

                a = (latents - min_vals) / denom        # (B,M)
                p = F.normalize(x, p=2, dim=1, eps=1e-8).float()
                a32 = a.float()

                # dL/da = dLdW*(p@q - a) + dLdZ*(sum_a - a)
                Pq = p @ q                               # (B,M)
                dLa = dLdW.unsqueeze(0) * (Pq - a32) + dLdZ.unsqueeze(0) * (sum_a.unsqueeze(0) - a32)

                S1_total += dLa.sum(dim=0)
                S2_total += (dLa * (latents.float() - min_vals.float())).sum(dim=0)

                # count ties (match torch.min/torch.max semantics: all equal minima/maxima get gradient equally)
                lat_fp32 = latents.float()
                min_count += (lat_fp32 == min_vals.float()).sum(dim=0).float()
                max_count += (lat_fp32 == max_vals.float()).sum(dim=0).float()

            # avoid div-by-zero if (pathologically) no min/max found
            min_count = torch.clamp(min_count, min=1.0)
            max_count = torch.clamp(max_count, min=1.0)

            # Precompute extras (M,)
            denom32 = denom.float().squeeze(0)
            extra_min_all = (-S1_total / denom32) + (S2_total / (denom32 ** 2))
            extra_max_all = (-S2_total / (denom32 ** 2))

        # ---- B2: stream param grads via latents.backward(dlat) ----
        with torch.enable_grad():
            for batch in dataloader:
                if not isinstance(batch, torch.Tensor):
                    batch = torch.from_numpy(batch)
                x = batch.to(device)

                with torch.amp.autocast(device_type=device.split(':')[0] if ':' in device else device):
                    _, latents, _ = autoencoder(x)      # (B,M)

                a = (latents - min_vals) / denom        # (B,M), forward dtype
                p = F.normalize(x, p=2, dim=1, eps=1e-8).float()

                # Recompute dL/da for this batch (fp32 math)
                with torch.no_grad():
                    Pq = p @ q
                    a32 = a.float()
                    dLa = dLdW.unsqueeze(0) * (Pq - a32) + dLdZ.unsqueeze(0) * (sum_a.unsqueeze(0) - a32)

                    base = dLa / denom                   # (B,M)

                    lat_fp32 = latents.float()
                    is_min = (lat_fp32 == min_vals.float()).float()
                    is_max = (lat_fp32 == max_vals.float()).float()

                    add_min = is_min * (extra_min_all / min_count).unsqueeze(0)  # (B,M)
                    add_max = is_max * (extra_max_all / max_count).unsqueeze(0)  # (B,M)

                    dlat = base + add_min + add_max      # (B,M), fp32

                latents.backward(dlat.to(latents.dtype), retain_graph=False)

        autoencoder.train(was_training)

        # grads for inputs of .apply
        grad_trigger = torch.zeros((), device=device)
        return (grad_trigger, None, None, None, None, None)

def compute_monosemanticity_custom(autoencoder, dataloader, device='cuda', eps=1e-8, verbose=False):
    trigger = torch.zeros((), device=device, requires_grad=True)
    return MonosemanticityFunction.apply(trigger, autoencoder, dataloader, device, eps, verbose)
