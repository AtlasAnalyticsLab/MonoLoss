import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from tqdm import tqdm
import numpy as np



@torch.no_grad()
def compute_monosemanticity_ref(autoencoder, dataloader, device='cuda', split_name='val', 
                                pair_batch_size: int = 100, eps: float = 1e-8):
    """
    Reference implementation using explicit nested loops over all pairs.
    This is the ground truth implementation that matches the user's reference code.
    
    Args:
        autoencoder: The autoencoder model
        dataloader: DataLoader for the dataset
        device: Device to run on
        split_name: Name of the split for logging
        pair_batch_size: Batch size for j loop (memory control)
        eps: Small constant for numerical stability
    
    Returns:
        monosemanticity: (M,) tensor of scores per neuron
    """
    autoencoder.eval()
    dataset = dataloader.dataset
    num_images = len(dataset)
    
    # Pass 1: Collect all embeddings and activations with normalization
    print(f"First pass: collecting all embeddings and activations ({split_name})...")
    all_embeddings = []
    all_activations = []
    min_vals = None
    max_vals = None
    num_neurons = None
    
    for batch in tqdm(dataloader, desc=f"Collecting data for {split_name}"):
        if not isinstance(batch, torch.Tensor):
            batch = torch.from_numpy(batch)
        x = batch.to(device)
        
        with torch.amp.autocast(device_type=device.split(':')[0] if ':' in device else device):
            _, latents, _ = autoencoder(x)
        
        # Track min/max for normalization
        if min_vals is None:
            num_neurons = latents.shape[1]
            min_vals = latents.min(dim=0, keepdim=True)[0]
            max_vals = latents.max(dim=0, keepdim=True)[0]
        else:
            batch_min = latents.min(dim=0, keepdim=True)[0]
            batch_max = latents.max(dim=0, keepdim=True)[0]
            min_vals = torch.min(min_vals, batch_min)
            max_vals = torch.max(max_vals, batch_max)
        
        # Store embeddings and activations (keep on GPU)
        all_embeddings.append(x)
        all_activations.append(latents)
    
    # Concatenate all data
    embeddings = torch.cat(all_embeddings, dim=0)  # (N, D)
    activations = torch.cat(all_activations, dim=0)  # (N, M)
    
    # Normalize activations per neuron to [0, 1]
    activations = (activations - min_vals) / (max_vals - min_vals + eps)
    
    print(f"Dataset size: {num_images}")
    print(f"Number of neurons: {num_neurons}")
    
    # Pass 2: Nested loop over all pairs i < j
    print(f"Second pass: computing pairwise monosemanticity (ref {split_name})...")
    weighted_cosine_similarity_sum = torch.zeros(num_neurons, device=device)
    weight_sum = torch.zeros(num_neurons, device=device)
    
    for i in tqdm(range(num_images), desc="Processing image pairs"):
        for j_start in range(i + 1, num_images, pair_batch_size):  # Process in batches
            j_end = min(j_start + pair_batch_size, num_images)
            
            embeddings_i = embeddings[i]  # (D,)
            embeddings_j = embeddings[j_start:j_end]  # (batch, D)
            activations_i = activations[i]  # (M,)
            activations_j = activations[j_start:j_end]  # (batch, M)
            
            # Compute cosine similarity
            cosine_similarities = F.cosine_similarity(
                embeddings_i.unsqueeze(0).expand(j_end - j_start, -1),  # (batch, D)
                embeddings_j,
                dim=1
            )  # (batch,)
            
            # Compute weights and weighted similarities
            weights = activations_i.unsqueeze(0) * activations_j  # (batch, M)
            weighted_cosine_similarities = weights * cosine_similarities.unsqueeze(1)  # (batch, M)
            
            # Accumulate
            weighted_cosine_similarity_sum += weighted_cosine_similarities.sum(dim=0)  # (M,)
            weight_sum += weights.sum(dim=0)  # (M,)
    
    # Compute final monosemanticity scores
    monosemanticity = torch.where(
        weight_sum > eps,
        weighted_cosine_similarity_sum / weight_sum,
        torch.zeros_like(weight_sum)  # Use 0 instead of NaN for consistency
    )
    
    autoencoder.train()
    return monosemanticity


@torch.no_grad()
def compute_monosemanticity_fast(autoencoder, dataloader, device='cuda', split_name='val', eps: float = 1e-8):
    """
    Fast O(N·D·M) monosemanticity via streaming GEMMs (no pairwise loops).

    Uses identity for unit-norm embeddings p_i = e_i / ||e_i||:
      sum_{i<j} a_i a_j (p_i · p_j) = 0.5 (||sum_i a_i p_i||^2 - sum_i a_i^2)
      sum_{i<j} a_i a_j          = 0.5 ((sum_i a_i)^2 - sum_i a_i^2)

    Where a_i are per-neuron activations after per-neuron min-max normalization in [0, 1].

    Returns mean monosemanticity over neurons for logging parity with the slow path.
    """
    autoencoder.eval()

    dataset = dataloader.dataset

    # Pass 1: activation min/max per neuron for scaling
    print(f"First pass: computing activation statistics (fast {split_name})...")
    min_vals = None
    max_vals = None
    num_neurons = None
    feature_dim = None

    for batch in tqdm(dataloader, desc=f"Computing stats for {split_name}"):
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
            batch_min = latents.min(dim=0, keepdim=True)[0]
            batch_max = latents.max(dim=0, keepdim=True)[0]
            min_vals = torch.min(min_vals, batch_min)
            max_vals = torch.max(max_vals, batch_max)

    # Informational (parity with original outputs)
    dataset_size = len(dataset)
    print(f"Dataset size: {dataset_size}")
    print(f"Number of neurons: {num_neurons}")

    # Pass 2: stream accumulators
    print(f"Second pass: streaming accumulators...")
    # Accumulators in FP32 for numerical stability
    sum_a = torch.zeros(num_neurons, device=device, dtype=torch.float32)
    sum_a2 = torch.zeros(num_neurons, device=device, dtype=torch.float32)
    V = torch.zeros(feature_dim, num_neurons, device=device, dtype=torch.float32)

    for batch in tqdm(dataloader, desc=f"Accumulating {split_name}"):
        if not isinstance(batch, torch.Tensor):
            batch = torch.from_numpy(batch)
        x = batch.to(device)

        # Forward to get activations (AMP ok); then cast to fp32 for accumulation
        with torch.amp.autocast(device_type=device.split(':')[0] if ':' in device else device):
            _, latents, _ = autoencoder(x)

        # Normalize activations per neuron to [0, 1]
        latents = (latents - min_vals) / (max_vals - min_vals + eps)
        a = latents

        # Unit-normalize embeddings to match cosine similarity
        p = F.normalize(x, p=2, dim=1, eps=1e-8)

        # Cast to fp32 for accumulators
        a32 = a.float()
        p32 = p.float()

        # Update accumulators
        sum_a += a32.sum(dim=0)
        sum_a2 += (a32 * a32).sum(dim=0)
        V += p32.T @ a32  # (D, B) @ (B, M) -> (D, M)

    # Compute per-neuron weighted sums in the same naming as the original implementation
    V_norm_sq = (V * V).sum(dim=0)  # (M,)
    weighted_cosine_sum = 0.5 * (V_norm_sq - sum_a2)  # Σ_{i<j} a_i a_j cos 
    weight_sum = 0.5 * (sum_a * sum_a - sum_a2)      # Σ_{i<j} a_i a_j

    # Final monosemanticity scores with identical threshold semantics
    monosemanticity = torch.where(
        weight_sum > 1e-8,
        weighted_cosine_sum / (weight_sum + 1e-12),
        torch.zeros_like(weight_sum)
    )
    autoencoder.train()
    return monosemanticity