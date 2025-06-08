import torch
# import torch.nn as nn
# from torch.autograd import grad
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import cm
# import torch.nn.init as init
# import time
# from torch.optim.lr_scheduler import StepLR
# from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

    
def lennard_jones_energy_full_original(x, n_particles, n_dims, box_length=2*torch.pi, eps=1.0, rm=1.0, r_switch=0.95, r_cutoff=2.5, remove_diagonal=True):
    """
    Compute Lennard-Jones energy using linear approximation for r < r_cut.
    Args:
        x: Tensor of shape (B, N * D), flattened particle coordinates
        n_particles: Number of particles N
        n_dims: Number of spatial dimensions D
        box_length: Periodic box size
        eps, rm: LJ parameters
        r_cut: linear cutoff distance
        remove_diagonal: whether to exclude self-interactions
        
    Returns:
        lj_energy: Tensor of shape (B, 1)
    """

    # 1. Reshape x to (B, N, D)
    x = x.view(-1, n_particles, n_dims)

    # 2. Apply PBC to compute pairwise displacements: shape (B, N, N, D)
    delta = x.unsqueeze(2) - x.unsqueeze(1)  # [B, N, N, D]
    delta = delta - box_length * torch.round(delta / box_length)

    # 3. Optionally remove diagonal entries
    if remove_diagonal:
        B, N, _, D = delta.shape
        mask = ~torch.eye(N, dtype=torch.bool, device=x.device)  # [N, N]
        mask = mask.unsqueeze(0).expand(B, N, N)                 # [B, N, N]
        delta = delta[mask].view(B, N, N - 1, D)

    # 4. Compute distances r = ||delta||: shape (B, N, N-1)
    r = delta.pow(2).sum(dim=-1).sqrt()  # [B, N, N-1]

    # 5. Compute linear approximation: V = a * r + b where r < r_cut
    a = -10.0
    lj_at_cut = 4.0 * eps * ((rm / r_switch)**12 - (rm / r_switch)**6)
    b = lj_at_cut - a * r_switch
    V = a * r + b

    # 6. Apply full LJ where r >= r_cut
    mask = (r >= r_switch)
    if mask.any():
        r_big = r[mask]
        V[mask] = 4.0 * eps * ((rm / r_big)**12 - (rm / r_big)**6)
    
    V[r > r_cutoff] = 0.0

    # 7. Sum over all pairwise terms, divide by 2 to avoid double counting
    lj_energy = V.sum(dim=(-2, -1)) / 2  # shape (B,)
    
    return lj_energy[:, None]  # shape (B, 1)

def lennard_jones_energy_full(x, n_particles, n_dims, box_length=2*torch.pi, eps=1.0, rm=1.0, r_switch=0.95, r_cutoff=2.5, remove_diagonal=True):
    """
    Optimized computation of Lennard-Jones energy.
    Uses linear approximation for r < r_switch, LJ for r_switch <= r <= r_cutoff, and 0 for r > r_cutoff.
    Args:
        x: Tensor of shape (B, N * D), flattened particle coordinates
        n_particles: Number of particles N
        n_dims: Number of spatial dimensions D
        box_length: Periodic box size
        eps, rm: LJ parameters
        r_switch: Distance at which to switch from linear approximation to full LJ.
        r_cutoff: Cutoff distance beyond which potential is zero.
        remove_diagonal: If True, exclude self-interactions and sum unique pairs.
                         If False, include all N*N pairs (including self-interaction r_ii=0)
                         and divide sum by 2 (original behavior).
        
    Returns:
        lj_energy: Tensor of shape (B, 1)
    """
    B = x.shape[0]
    x_reshaped = x.view(B, n_particles, n_dims)

    # Compute pairwise displacements with periodic boundary conditions
    # delta_all_pairs[b, i, j, d] is component d of vector from particle i to particle j for batch b
    delta_all_pairs = x_reshaped.unsqueeze(2) - x_reshaped.unsqueeze(1)  # Shape: (B, N, N, D)
    delta_all_pairs = delta_all_pairs - box_length * torch.round(delta_all_pairs / box_length)

    if remove_diagonal:
        # Get indices for upper triangle (unique pairs, i < j)
        idx_i, idx_j = torch.triu_indices(n_particles, n_particles, offset=1, device=x.device)
        delta = delta_all_pairs[:, idx_i, idx_j, :]  # Shape: (B, N*(N-1)/2, D)
        # Compute distances r = ||delta||
        r = delta.pow(2).sum(dim=-1).sqrt()  # Shape: (B, N*(N-1)/2)
    else:
        # Use all N*N pairs, including self-interactions (r_ii = 0)
        delta = delta_all_pairs # Shape: (B, N, N, D)
        r = delta.pow(2).sum(dim=-1).sqrt() # Shape: (B, N, N)

    # Parameters for linear approximation V_lin(r) = a * r + b
    # This ensures V_lin(r_switch) = V_LJ(r_switch)
    a = -10.0  # Given slope for linear part
    # Calculate V_LJ(r_switch)
    # Clamp r_switch to avoid division by zero if it's pathologically small, though typically r_switch > 0
    r_switch_safe = max(r_switch, 1e-9) 
    lj_at_r_switch = 4.0 * eps * ((rm / r_switch_safe)**12 - (rm / r_switch_safe)**6)
    b = lj_at_r_switch - a * r_switch

    # Potential for r < r_switch (linear approximation)
    V_linear = a * r + b
    
    # Potential for r >= r_switch (full LJ)
    # Clamp r to avoid division by zero for r=0 if it were to be used by V_lj.
    # r_safe_for_lj is used for calculating V_lj part.
    r_safe_for_lj = r.clamp(min=1e-12) # Avoid division by zero or log(0)
    inv_r_lj = rm / r_safe_for_lj
    V_lj = 4.0 * eps * (inv_r_lj**12 - inv_r_lj**6)
    
    # Combine using torch.where: V = V_lj if r >= r_switch else V_linear
    V = torch.where(r >= r_switch, V_lj, V_linear)
    
    # Apply cutoff: V = 0 if r > r_cutoff
    V = torch.where(r > r_cutoff, torch.zeros_like(V), V)

    if remove_diagonal:
        # Sum over unique pairs
        lj_energy = V.sum(dim=-1)  # Shape: (B,)
    else:
        # Sum over all N*N pairs and divide by 2 (mimics original behavior for this case)
        lj_energy = V.sum(dim=(-2, -1)) / 2  # Shape: (B,)
    
    return lj_energy.unsqueeze(-1)  # Shape: (B, 1)