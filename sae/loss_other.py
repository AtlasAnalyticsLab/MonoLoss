import torch
import torch.nn.functional as F
from typing import Tuple, Dict
from tqdm import tqdm


def compute_monosemanticity_loss_batch(x: torch.Tensor, latents: torch.Tensor, 
                                       eps: float = 1e-8) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute monosemanticity loss on a batch using 1 - mono_score.
    
    Args:
        x: Input embeddings (B, D) - will be normalized to unit norm
        latents: Neuron activations (B, M) - should be non-negative
        eps: Small constant for numerical stability
    
    Returns:
        loss: 1 - mean(monosemanticity), range [0, 1]
        metrics: Dictionary with monosemanticity statistics
    """
    # Normalize embeddings to unit norm for cosine similarity
    x_norm = F.normalize(x, p=2, dim=1, eps=eps)  # (B, D)
    
    # Normalize activations per neuron to [0, 1] within batch
    min_acts = latents.min(dim=0, keepdim=True)[0]  # (1, M)
    max_acts = latents.max(dim=0, keepdim=True)[0]  # (1, M)
    a = (latents - min_acts) / (max_acts - min_acts + eps)  # (B, M)
    
    # Fast monosemanticity via streaming formula (same as compute_monosemanticity_fast)
    # V = X^T @ A, where rows are weighted embeddings per neuron
    V = x_norm.T @ a  # (D, M)
    
    # Numerator: Σ_{i<j} a_i·a_j·cos(e_i,e_j) = 0.5(||Σ a_i·e_i||² - Σ a_i²)
    V_norm_sq = (V * V).sum(dim=0)  # (M,)
    sum_a2 = (a * a).sum(dim=0)  # (M,)
    weighted_cosine_sum = 0.5 * (V_norm_sq - sum_a2)
    
    # Denominator: Σ_{i<j} a_i·a_j = 0.5((Σ a_i)² - Σ a_i²)
    sum_a = a.sum(dim=0)  # (M,)
    weight_sum = 0.5 * (sum_a * sum_a - sum_a2)
    
    # Monosemanticity score per neuron
    # mono_scores = torch.where(
    #     weight_sum > eps,
    #     weighted_cosine_sum / (weight_sum + eps),
    #     torch.zeros_like(weight_sum)
    # )  # (M,)
    B = a.shape[0]  # batch size
    num_pairs = B * (B - 1) / 2.0
    mono_scores = weighted_cosine_sum / num_pairs
    
    # Only consider active neurons
    active_mask = weight_sum > eps
    num_active = active_mask.sum().item()
    
    if num_active > 0:
        mono_mean = mono_scores[active_mask].mean()
        mono_median = mono_scores[active_mask].median()
    else:
        mono_mean = torch.tensor(0.0, device=x.device)
        mono_median = torch.tensor(0.0, device=x.device)
    
    # Loss: 1 - mono_score (minimize to maximize monosemanticity)
    loss = 1.0 - mono_mean
    
    metrics = {
        'mono_loss_val': loss.item(),
        'mono_score': mono_mean.item(),
        'mono_median': mono_median.item(),
        'mono_active_neurons': num_active,
    }
    
    return loss, metrics


@torch.no_grad()
def compute_monosemanticity_ref(autoencoder, dataloader, device='cuda', split_name='val', 
                                pair_batch_size: int = 100, eps: float = 1e-8, verbose: bool = False):
    """
    Reference MonoScore implementation using explicit nested loops over all pairs.
    
    
    Args:
        autoencoder: The autoencoder model
        dataloader: DataLoader for the dataset
        device: Device to run on
        split_name: Name of the split for logging
        pair_batch_size: Batch size for j loop (memory control)
        eps: Small constant for numerical stability
        verbose: If True, print progress messages
    
    Returns:
        monosemanticity: (M,) tensor of scores per neuron
    """
    autoencoder.eval()
    dataset = dataloader.dataset
    num_images = len(dataset)
    
    # Pass 1: Collect all embeddings and activations with normalization, also collect min and max values for activations
    if verbose:
        print(f"First pass: collecting all embeddings and activations ({split_name})...")
    all_embeddings = []
    all_activations = []
    min_vals = None
    max_vals = None
    num_neurons = None
    
    for batch in tqdm(dataloader, desc=f"Collecting data for {split_name}", disable=not verbose):
        if not isinstance(batch, torch.Tensor):
            batch = torch.from_numpy(batch)
        x = batch.to(device)
        
        _, latents, _ = autoencoder(x)
        
        if min_vals is None:
            num_neurons = latents.shape[1]
            min_vals = latents.min(dim=0, keepdim=True)[0]
            max_vals = latents.max(dim=0, keepdim=True)[0]
        else:
            batch_min = latents.min(dim=0, keepdim=True)[0]
            batch_max = latents.max(dim=0, keepdim=True)[0]
            min_vals = torch.min(min_vals, batch_min)
            max_vals = torch.max(max_vals, batch_max)
        
        all_embeddings.append(x)
        all_activations.append(latents)
    
    embeddings = torch.cat(all_embeddings, dim=0)  # (N, D)
    activations = torch.cat(all_activations, dim=0)  # (N, M)
    
    # Normalize activations per neuron to [0, 1]
    activations = (activations - min_vals) / (max_vals - min_vals + eps)
    
    if verbose:
        print(f"Dataset size: {num_images}")
        print(f"Number of neurons: {num_neurons}")
    
    # Pass 2: Nested loop over all pairs i < j
    if verbose:
        print(f"Second pass: computing pairwise monosemanticity (ref {split_name})...")
    weighted_cosine_similarity_sum = torch.zeros(num_neurons, device=device)
    weight_sum = torch.zeros(num_neurons, device=device)
    
    for i in tqdm(range(num_images), desc="Processing image pairs", disable=not verbose):
        for j_start in range(i + 1, num_images, pair_batch_size):  # Process in batches
            j_end = min(j_start + pair_batch_size, num_images)
            
            embeddings_i = embeddings[i]  # (D,)
            embeddings_j = embeddings[j_start:j_end]  # (batch, D)
            activations_i = activations[i]  # (M,)
            activations_j = activations[j_start:j_end]  # (batch, M)
            
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
    # monosemanticity = torch.where(
    #     weight_sum > eps,
    #     weighted_cosine_similarity_sum / weight_sum,
    #     torch.zeros_like(weight_sum)  
    # )
    num_pairs = num_images * (num_images - 1) / 2.0
    monosemanticity = weighted_cosine_similarity_sum / num_pairs
    
    autoencoder.train()
    return monosemanticity


@torch.no_grad()
def compute_monosemanticity_fast(autoencoder, dataloader, device='cuda', split_name='val', eps: float = 1e-8, verbose: bool = False):
    """
    Fast O(N·D·M) monosemanticity in single pass (no pairwise loops).

    Args:
        autoencoder: The autoencoder model
        dataloader: DataLoader for the dataset
        device: Device to run on
        split_name: Name of the split for logging
        eps: Small constant for numerical stability
        verbose: If True, print progress messages

    Returns:
        monosemanticity: (M,) tensor of scores per neuron
    """
    autoencoder.eval()

    dataset = dataloader.dataset

    # Pass 1: activation min/max per neuron for scaling
    if verbose:
        print(f"First pass: computing activation statistics (fast {split_name})...")
    min_vals = None
    max_vals = None
    num_neurons = None
    feature_dim = None

    for batch in tqdm(dataloader, desc=f"Computing stats for {split_name}", disable=not verbose):
        if not isinstance(batch, torch.Tensor):
            batch = torch.from_numpy(batch)
        x = batch.to(device)

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

    dataset_size = len(dataset)
    if verbose:
        print(f"Dataset size: {dataset_size}")
        print(f"Number of neurons: {num_neurons}")

    # Pass 2: stream accumulators
    if verbose:
        print(f"Second pass: streaming accumulators...")
    # Accumulators in FP32 for numerical stability
    sum_a = torch.zeros(num_neurons, device=device, dtype=torch.float32)
    sum_a2 = torch.zeros(num_neurons, device=device, dtype=torch.float32)
    V = torch.zeros(feature_dim, num_neurons, device=device, dtype=torch.float32)

    for batch in tqdm(dataloader, desc=f"Accumulating {split_name}", disable=not verbose):
        if not isinstance(batch, torch.Tensor):
            batch = torch.from_numpy(batch)
        x = batch.to(device)

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

    # Compute per-neuron weighted sums
    V_norm_sq = (V * V).sum(dim=0)  # (M,)
    weighted_cosine_sum = 0.5 * (V_norm_sq - sum_a2)  # Σ_{i<j} a_i a_j cos 
    weight_sum = 0.5 * (sum_a * sum_a - sum_a2)      # Σ_{i<j} a_i a_j

    # Final monosemanticity scores
    # monosemanticity = torch.where(
    #     weight_sum > 1e-8,
    #     weighted_cosine_sum / (weight_sum + 1e-12),
    #     torch.zeros_like(weight_sum)
    # )
    num_pairs = dataset_size * (dataset_size - 1) / 2.0
    monosemanticity = weighted_cosine_sum / num_pairs
    autoencoder.train()
    return monosemanticity