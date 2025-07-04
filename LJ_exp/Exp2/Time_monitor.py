import torch
import torch.nn as nn
from torch.autograd import grad
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import torch.nn.init as init
import time
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


def _standardize(kernel):
    """
    Makes sure that Var(W) = 1 and E[W] = 0
    """
    eps = 1e-6

    if len(kernel.shape) == 3:
        axis = [0, 1]  # last dimension is output dimension
    else:
        axis = 1

    var, mean = torch.var_mean(kernel, dim=axis, unbiased=True, keepdim=True)
    kernel = (kernel - mean) / (var + eps) ** 0.5
    return kernel


def he_orthogonal_init(tensor):
    """
    Generate a weight matrix with variance according to He initialization.
    Based on a random (semi-)orthogonal matrix neural networks
    are expected to learn better when features are decorrelated
    (stated by eg. "Reducing overfitting in deep networks by decorrelating representations",
    "Dropout: a simple way to prevent neural networks from overfitting",
    "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks")
    """
    tensor = torch.nn.init.orthogonal_(tensor)

    if len(tensor.shape) == 3:
        fan_in = tensor.shape[:-1].numel()
    else:
        fan_in = tensor.shape[1]

    with torch.no_grad():
        tensor.data = _standardize(tensor.data)
        tensor.data *= (1 / fan_in) ** 0.5

    return tensor

class Dense(torch.nn.Module):
    """
    Combines dense layer and scaling for swish activation.

    Parameters
    ----------
        units: int
            Output embedding size.
        activation: str
            Name of the activation function to use.
        bias: bool
            True if use bias.
    """

    def __init__(
        self, in_features, out_features, bias=False, activation=None, name=None
    ):
        super().__init__()

        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)
        self.reset_parameters()
        self.weight = self.linear.weight
        self.bias = self.linear.bias

        if isinstance(activation, str):
            activation = activation.lower()
        if activation in ["swish", "silu"]:
            self._activation = ScaledSiLU()
        elif activation is None:
            self._activation = torch.nn.Identity()
        else:
            raise NotImplementedError(
                "Activation function not implemented for GemNet (yet)."
            )

    def reset_parameters(self):
        he_orthogonal_init(self.linear.weight)
        if self.linear.bias is not None:
            self.linear.bias.data.fill_(0)

    def forward(self, x):
        x = self.linear(x)
        x = self._activation(x)
        return x


class ScaledSiLU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = 1 / 0.6
        self._activation = torch.nn.SiLU()

    def forward(self, x):
        return self._activation(x) * self.scale_factor


class ResidualLayer(torch.nn.Module):
    """
    Residual block with output scaled by 1/sqrt(2).

    Parameters
    ----------
        units: int
            Output embedding size.
        nLayers: int
            Number of dense layers.
        activation: str
            Name of the activation function to use.
    """

    def __init__(self, units: int, nLayers: int = 2, activation=None, name=None):
        super().__init__()
        self.dense_mlp = torch.nn.Sequential(
            *[
                Dense(units, units, activation=activation, bias=False)
                for i in range(nLayers)
            ]
        )
        self.inv_sqrt_2 = 1 / (2.0 ** 0.5)

    def forward(self, inputs):
        x = self.dense_mlp(inputs)
        x = inputs + x
        x = x * self.inv_sqrt_2
        return x

import math
import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GraphNorm, InstanceNorm

# from . import ResidualLayer, Dense, he_orthogonal_init

class CoorsNorm(nn.Module):
    def __init__(self, eps = 1e-8, scale_init = 1.):
        super().__init__()
        self.eps = eps
        scale = torch.zeros(1).fill_(scale_init)
        self.scale = nn.Parameter(scale)

    def forward(self, coors):
        norm = coors.norm(dim = -1, keepdim = True)
        normed_coors = coors / norm.clamp(min = self.eps)
        return normed_coors * self.scale

class Envelope(torch.nn.Module):
    """
    Envelope function that ensures a smooth cutoff.

    Parameters
    ----------
        p: int
            Exponent of the envelope function.
    """

    def __init__(self, p, name="envelope"):
        super().__init__()
        assert p > 0
        self.p = p
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, d_scaled):
        env_val = (
            1
            + self.a * d_scaled ** self.p
            + self.b * d_scaled ** (self.p + 1)
            + self.c * d_scaled ** (self.p + 2)
        )
        return torch.where(d_scaled < 1, env_val, torch.zeros_like(d_scaled))

def build_graph(globs, coors, feats = None, edges = None, cutoff = math.inf, mask = None):
    """
    Build the graph structure from coordinates
    
    Args:
        globs: global attributes [b, g]
        coors: coordinate [b, n, d]
        feats: optional node features [b, n, f]
        edges: optional edge features [b, n, n, e]
        cutoff_distance: maximum distance for edge creation
        mask: optional mask [b, n, n]
        
    Returns:
        A dictionary containing all graph information including triplet data
    """
    b, n, d = coors.shape
    device = coors.device
    cutoff = (2 * math.pi) ** 2 if cutoff > (2 * math.pi) ** 2 else cutoff
    
    # Compute relative coordinates for all pairs
    rel_coors = - coors[:,:,None,:] + coors[:,None,:,:]  #  - [b, i, 1, d] + [b, 1, j, d] = [b, i, j, d] vec from i to j
    
    # Apply periodic boundary conditions
    rel_coors = rel_coors - 2 * math.pi * torch.round(rel_coors / (2 * math.pi))
    
    # Compute chord distance
    # rel_dist = (2 - 2 * torch.cos(rel_coors)).sum(dim=-1, keepdim=True)  # [b, n, n, 1]
    # Compute geodesic distance
    rel_dists = (rel_coors ** 2).sum(dim = -1, keepdim = True)  # [b, i, j, 1]
    
    with torch.no_grad():
        # Create distance-based cutoff mask
        dist_mask = (rel_dists <= cutoff) & (rel_dists > 0)  # [b, i, j, 1]
        deg = dist_mask.sum(dim=-2).flatten()  # [b*i] - number of valid edges per node
        deg_max = deg.max()  
        deg_cum = deg.cumsum(dim=0) - deg  # [b*i] - cumulative sum of degrees
        
        # Combine with provided mask if it exists
        if mask is not None:
            mask = mask.unsqueeze(-1) & dist_mask
        else:
            mask = dist_mask
        del dist_mask
            
        # Vectorized batch processing using PyG's batch support
        mask_flat = mask.flatten()  # [b*i*j]
        
        # Use meshgrid to efficiently create source and target indices
        row, col = torch.meshgrid(
            torch.arange(n, device=device),
            torch.arange(n, device=device),
            indexing='ij'
        )
        source_indices = row.flatten()  # [n*n]
        target_indices = col.flatten()  # [n*n]
        
        # Create batch offsets more efficiently
        batch_index = torch.arange(b, device=device).repeat_interleave(n*n)  # [b*n*n]
        batch_offsets = batch_index * n  # [b*n*n]
        
        # Apply offsets directly
        batch_source = source_indices.repeat(b) + batch_offsets  # [b*n*n]
        batch_target = target_indices.repeat(b) + batch_offsets  # [b*n*n]
        
        # Filter edges based on the mask
        valid_edges = mask_flat.bool()
        batch_index = batch_index[valid_edges]  # [num_edges]
        edge_index = torch.stack([
            batch_source[valid_edges],
            batch_target[valid_edges]
        ])  # [2, num_edges]
        

    # Reshape coordinates and features based on the mask
    coors = coors.flatten(end_dim=-2)  # [b*n, d]
    if feats is not None:
        feats = feats.flatten(end_dim=-2)  # [b*n, f]

    # Reshape coordinates and distances based on the mask
    globs = globs[batch_index]  # [num_edges, g]
    rel_coors = rel_coors.reshape(-1, d)[valid_edges]  # [num_edges, d]
    rel_dists = rel_dists.reshape(-1, 1)[valid_edges]  # [num_edges, 1]
    if edges is not None:
        edges = edges.flatten(end_dim=-2)[valid_edges]  # [num_edges, e]
    
    # triplet_angles = calculate_neighbor_angles(
    #     rel_coors[triplet_index[0]],  # [num_triplets, d]
    #     rel_coors[triplet_index[1]]   # [num_triplets, d]
    # )  # [num_triplets, 2]

    # Return all the necessary components for graph operations
    return {
        'edge_index': edge_index,
        'globs': globs,
        'coors': coors,
        # 'triplet_index': triplet_index,
        # 'triplet_angles': triplet_angles,
        'feats': feats,
        'edges': edges,
        'rel_coors': rel_coors,
        'rel_dists': rel_dists,
        'b': b,
        'n': n,
        'd': d,
        # 'num_edges': num_edges,
        # 'num_triplets': num_triplets,
    }


# Based on E(n)-Equivariant Graph Neural Networks in torus space
class Coordinate_EGNN(MessagePassing):
    def __init__(self, glob_dim: int, feat_dim: int = 0, edge_dim: int = 0, hidden_dim: int = 64,
                 cutoff_distance: float = math.inf, 
                 norm_coors_scale_init: float = 1., 
                 envelope_exponent: int = 5):
        super().__init__(aggr="add")  # Use "add" aggregation
        
        self.cutoff_distance = (2 * math.pi) ** 2 if cutoff_distance > (2 * math.pi) ** 2 else cutoff_distance
        # Update edge_input_dim to include features from both source and target nodes
        edge_input_dim = glob_dim + 1 + (2 * feat_dim) + edge_dim
        
        self.edge_mlp = nn.Sequential(
            Dense(edge_input_dim, hidden_dim, activation='silu'),
            nn.SiLU(),
            ResidualLayer(hidden_dim, 2, activation='silu', name=None)
        )

        self.edge_gate = nn.Sequential(
            Dense(hidden_dim, 1),
            nn.Sigmoid()
        )

        self.coors_norm = CoorsNorm(scale_init = norm_coors_scale_init)

        self.vec_mlp = nn.Sequential(
            ResidualLayer(hidden_dim, 2, activation='silu', name=None),
            nn.SiLU(),
            Dense(hidden_dim, 1),
        )

        self.envelope = Envelope(envelope_exponent)

        self.inv_sqrt_2 = 1 / (2.0 ** 0.5)

    def forward(self, graph_dict):
        '''
        globs: global attributes B*D'
        coors: coordinate (vec) attributes B*N*D
        '''

        # Extract parameters from the provided graph_dict
        b, n, d = graph_dict['b'], graph_dict['n'], graph_dict['d']
        edge_index = graph_dict['edge_index']
        globs = graph_dict['globs']
        coors = graph_dict['coors']
        feats = graph_dict['feats']
        edges = graph_dict['edges']
        rel_coors = graph_dict['rel_coors']
        rel_dists = graph_dict['rel_dists']
        # num_edges = graph_dict['num_edges']
        # num_triplets = graph_dict['num_triplets']
        
        # Create edge attributes - start with global attributes and distances
        edge_attr = torch.cat([globs, rel_dists], dim=-1)
        dists_env = self.envelope(rel_dists / self.cutoff_distance)
        
        # Conditionally add edge features if they exist
        if edges is not None:
            edge_attr = torch.cat([edge_attr, edges], dim=-1)
        
        # Run message passing
        coor_update = self.propagate(
            edge_index,
            size=(b * n, b * n),  # Size of the graph
            feats=feats,
            rel_coors=rel_coors,
            dists_env = dists_env,
            edge_attr=edge_attr
        )
        
        coors = coors + coor_update

        rel_coors = - coors[edge_index[0]] + coors[edge_index[1]]  # [num_edges, d] vec from i to j
        rel_coors = rel_coors - 2 * math.pi * torch.round(rel_coors / (2 * math.pi))

        rel_dists = (rel_coors ** 2).sum(dim = -1, keepdim = True)  # [num_edges, 1]

        graph_dict['coors'] = coors
        graph_dict['rel_coors'] = rel_coors
        graph_dict['rel_dists'] = rel_dists
        
        return coors
    
    def message(self, feats_i, feats_j, rel_coors, dists_env, edge_attr):
        if feats_i is not None and feats_j is not None:
            # Concatenate features from both source and target nodes
            edge_attr = torch.cat([edge_attr, feats_i, feats_j], dim=-1)

        # Process edge attributes
        m_ij = self.edge_mlp(edge_attr)
        m_ij = m_ij * self.edge_gate(m_ij)  # Apply gating
        
        # Compute vector messages
        mij_hat = self.vec_mlp(m_ij)  # [num_edges, 1]

        # Normalize relative coordinates
        rel_coors_norm = self.coors_norm(rel_coors)
        
        # Scale relative coordinates
        return mij_hat * rel_coors_norm * dists_env  # [num_edges, d]
    
    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        # Use the default aggregation (sum)
        return super().aggregate(inputs, index, ptr, dim_size)
    
    def update(self, aggr_out):
        # No additional processing needed after aggregation
        return aggr_out


# Multi-layer version of CoordinateEGNN with optimized graph construction
class PINN2D(nn.Module):
    def __init__(self, glob_dim: int, feat_dim: int = 0, edge_dim: int = 0, hidden_dim: int = 64, 
                 norm_coors_scale_init: float = 1.0, cutoff_distance: float = math.inf, num_layers: int = 1):
        super().__init__()
        self.cutoff_distance = cutoff_distance
        
        # Create a ModuleList to hold multiple EGNN layers
        self.layers = nn.ModuleList([
            Coordinate_EGNN(
            # Coordinate_Triplet_EGNN(
                glob_dim=glob_dim, 
                feat_dim=feat_dim, 
                edge_dim=edge_dim, 
                hidden_dim=hidden_dim, 
                norm_coors_scale_init=norm_coors_scale_init, 
                cutoff_distance=cutoff_distance,
            )
            for _ in range(num_layers)
        ])
        
    def forward(self, globs, coors, shape, feats=None, edges=None, mask=None):
        """
        Apply multiple EGNN layers sequentially with a single graph construction
        
        Args:
            globs: global attributes (batch_size, D')
            coors: flattened coordinate attributes (total_size, d)
            shape: tuple of (batch_size, n_particles, dim) to reshape coordinates
            feats: node features (optional)
            edges: edge features (optional)
            mask: attention mask (optional)
        """
        # Extract shape information
        b, n, d = shape
        
        # Reshape flattened coordinates to (b, n, d) for graph building
        coors = coors.reshape(b, n, d)
        feats = feats.reshape(b, n, -1) if feats is not None else None
        edges = edges.reshape(b, n, n, -1) if edges is not None else None

        # Build graph structure once using the standalone function
        graph_dict = build_graph(globs, coors, feats, edges, self.cutoff_distance, mask)

        coors_residual = graph_dict['coors']
        
        # Apply each layer sequentially using the same graph structure
        for layer in self.layers:
            coors = layer(graph_dict)
        
        graph_dict['coors'] -= coors_residual
        
        return graph_dict['coors'].reshape(b, n*d)


class Time_residual(nn.Module):
    def __init__(self, num_hidden_layers=2, num_neurons_per_layer=128):
        super().__init__()
        
        # Input layer: 1D input (time)
        self.input_layer = nn.Linear(1, num_neurons_per_layer)

        # Hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Linear(num_neurons_per_layer, num_neurons_per_layer) 
            for _ in range(num_hidden_layers)
        ])
        
        # Output layer: 1D output (time residual)
        self.output_layer = nn.Linear(num_neurons_per_layer, 1)

        # Activation function
        self.activation = nn.Tanh()
        
        # Initialize weights using Xavier initialization
        nn.init.xavier_normal_(self.input_layer.weight.data)
        nn.init.zeros_(self.input_layer.bias.data)
        
        for layer in self.hidden_layers:
            nn.init.xavier_normal_(layer.weight.data)
            nn.init.zeros_(layer.bias.data)
            
        nn.init.xavier_normal_(self.output_layer.weight.data)
        nn.init.zeros_(self.output_layer.bias.data)

    def forward(self, t):
        """
        Forward pass through the network.
        
        Args:
            t (torch.Tensor): Time input of shape [batch_size, 1]
            
        Returns:
            torch.Tensor: Time residual output c_t of shape [batch_size, 1]
        """
        # Pass through network with residual connections
        outputs = self.activation(self.input_layer(t))
        
        for layer in self.hidden_layers:
            layer_output = self.activation(layer(outputs))
            outputs = layer_output + outputs  # Residual connection
        
        # Final output layer
        c_t = self.output_layer(outputs)
        
        return c_t
    
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

def vector_distance(samples,Velocity,f_t):
    # Samples is a tensor of shape (B, N*D）
    # Velocity_field is a tensor of shape (B, N*D)
    # Internal_energy_ is a tensor of shape (B,1)
    # Expected result is a tensor of shape (B,1)
    
    gradient = torch.autograd.grad(f_t, samples, grad_outputs=torch.ones_like(f_t), create_graph=True)[0]
    # Detect when gradient exists Nan, output samples and inter_energy

    if torch.isnan(gradient).any():
        nan_mask = torch.isnan(gradient).any(dim=1)
        print("NaN detected in gradient!")
        print("Problematic samples:\n", samples[nan_mask])
        print("Corresponding internal energy:\n", f_t[nan_mask])
        raise ValueError("NaN detected in gradient computation.")

    # Compute the distance
    result = torch.sum(gradient * Velocity,dim = 1)
    
    assert not torch.isnan(result).any(), "Result is NaN"
    result = result.unsqueeze(dim = 1)
    return result

def compute_divergence(Velocity, t, x, n_particles, n_dimension, n_pde_points):
    """
    model: the neural network representing the vector field
    t: torch tensor of shape [B, 1]
    x: torch tensor of shape [B, D]
    """
    x.requires_grad_(True)
    v = Velocity(t, x, (n_pde_points, n_particles, n_dimension))  # [B, D]
    
    divergence = torch.zeros(x.shape[0], device=x.device)
    for i in range(v.shape[1]):
        grad_v_i = torch.autograd.grad(
            outputs=v[:, i],
            inputs=x,
            grad_outputs=torch.ones_like(v[:, i]),
            create_graph=True,
            retain_graph=True
        )[0][:, i]  # 只取 ∂v_i/∂x_i
        divergence += grad_v_i

    return divergence.view(-1,1)  # shape [B]

def pde_formula(x, t_value, Velocity_model, Normalizing_model, n_particles, n_dimension, n_pde_points):
    '''
    t_value: torch tensor of shape [B, 1]
    x: torch tensor of shape [B, N*D]
    '''
    # First term is energy
    U = lennard_jones_energy_full(x, n_particles, n_dimension)
    f_t = t_value * U
    # check if U is NaN
    assert torch.isnan(f_t).sum() == 0, "Energy is NaN"

    shape = (n_pde_points, n_particles, n_dimension)
    velocity_field = Velocity_model(t_value, x, shape)
    assert torch.isnan(velocity_field).sum() == 0, "Velocity field is NaN"
    
    term2 = vector_distance(x, velocity_field, f_t)
    assert torch.isnan(term2).sum() == 0, "Term2 is NaN"
        
    trace_average = compute_divergence(Velocity_model, t_value, x, n_particles, n_dimension, n_pde_points)
    assert torch.isnan(trace_average).sum() == 0, "Trace average is NaN"

    C_t = Normalizing_model(t_value)
    assert torch.isnan(C_t).sum() == 0, "C_t is NaN"

    return torch.abs(U + term2 - trace_average +C_t) + (U + term2 - trace_average +C_t)**2

# Loss function for 2D PINN with time range [0.1, 1] and vector output correction
def compute_loss(Velocity, Time_residual, n_particles, n_dimension, device="cuda", bound = 2*torch.pi):
    timings = {}

    # Sample collocation points for PDE residual
    n_pde_points = 1028  # Number of points inside domain
    samples = torch.rand(n_pde_points, n_particles*n_dimension, device=device,requires_grad=True) * bound - bound/2  # Range [-10, 10]
    t_value = 1.0 - torch.rand(n_pde_points, 1, device=device)
    
    start_pde = time.time()
    pde_loss_t = pde_formula(samples, t_value, Velocity, Time_residual, n_particles, n_dimension, n_pde_points)
    pde_loss = torch.mean(pde_loss_t)
    timings['pde_loss_calc'] = time.time() - start_pde

    # Initial condition points at t=0.1 (smallest valid t)
    n_ic_points = 128
    shape = (n_ic_points, n_particles, n_dimension)
    samples_prior_ic = torch.rand(n_ic_points, n_particles*n_dimension, device=device)* bound - bound/2    # Range [-10, 10]
    t_ic = torch.zeros(n_ic_points, 1, device=device)      
    
    start_ic = time.time()
    u_ic_pred = Velocity(t_ic, samples_prior_ic, shape)
    u_ic_true = 1
    ic_loss = torch.sum((u_ic_pred - u_ic_true)**2,dim = 1)
    ic_loss = torch.mean(ic_loss)
    timings['ic_loss_calc'] = time.time() - start_ic
    
    # Boundary conditions
    n_bc_points = 128
    shape_bc = (n_bc_points, n_particles, n_dimension) # Use a different name for shape to avoid conflict
    
    start_bc = time.time()
    # Left boundary (x = -10)
    samples_prior_left = torch.rand(n_bc_points, n_particles*n_dimension, device=device)* bound - bound/2 
    samples_prior_left[:, 0] = -torch.pi   # Range [-10, 10]
    t_bc_left = torch.rand(n_bc_points, 1, device=device)  
    u_bc_left_pred = Velocity(t_bc_left, samples_prior_left, shape_bc)
    u_bc_left_true = torch.zeros_like(u_bc_left_pred, device=device)
    bc_loss_left = torch.sum((u_bc_left_pred - u_bc_left_true)**2,dim = 1)
    bc_loss_left = torch.mean(bc_loss_left)
    
    # Right boundary (x = 10)
    samples_prior_right = torch.rand(n_bc_points, n_particles*n_dimension, device=device)* bound - bound/2 
    samples_prior_right[:, 0] = torch.pi
    t_bc_right = torch.rand(n_bc_points, 1, device=device)   
    u_bc_right_pred = Velocity(t_bc_right, samples_prior_right, shape_bc)
    u_bc_right_true = torch.zeros_like(u_bc_right_pred, device=device)
    bc_loss_right = torch.sum((u_bc_right_pred - u_bc_right_true)**2,dim = 1)
    bc_loss_right = torch.mean(bc_loss_right)
    
    # Bottom boundary (y = -10)
    samples_prior_bottom = torch.rand(n_bc_points, n_particles*n_dimension, device=device)* bound - bound/2
    samples_prior_bottom[:, 1] = -torch.pi  
    t_bc_bottom = torch.rand(n_bc_points, 1, device=device) 
    u_bc_bottom_pred = Velocity(t_bc_bottom, samples_prior_bottom, shape_bc)
    u_bc_bottom_true = torch.zeros_like(u_bc_bottom_pred, device=device)
    bc_loss_bottom = torch.sum((u_bc_bottom_pred - u_bc_bottom_true)**2,dim = 1)
    bc_loss_bottom = torch.mean(bc_loss_bottom)
    
    # Top boundary (y = 10)
    samples_prior_top = torch.rand(n_bc_points, n_particles*n_dimension, device=device)* bound - bound/2
    samples_prior_top[:, 1] = torch.pi
    t_bc_top = torch.rand(n_bc_points, 1, device=device) 
    u_bc_top_pred = Velocity(t_bc_top, samples_prior_top, shape_bc)
    u_bc_top_true = torch.zeros_like(u_bc_top_pred, device=device)
    bc_loss_top = torch.sum((u_bc_top_pred - u_bc_top_true)**2,dim = 1)
    bc_loss_top = torch.mean(bc_loss_top)

    # Total boundary loss
    bc_loss = bc_loss_left + bc_loss_right + bc_loss_bottom + bc_loss_top
    timings['bc_loss_calc'] = time.time() - start_bc
    
    # Total loss
    total_loss = pde_loss + ic_loss + bc_loss 
    
    return total_loss, pde_loss, ic_loss, bc_loss, timings

# Training function
def train(n_particles, n_dimension, Velocity, Time_residual, optimizer, scheduler, device="cuda", n_epochs=10000, print_every=1000):
    start_time = time.time()
    loss_history = []
    
    # Accumulators for timings over print_every epochs
    acc_zero_grad_time = 0
    acc_pde_time = 0
    acc_ic_time = 0
    acc_bc_time = 0
    acc_backward_time = 0
    acc_optimizer_step_time = 0
    acc_other_time = 0
    
    for epoch in range(n_epochs):
        iter_start = time.time()
        
        # Time zero_grad operation
        zero_grad_start = time.time()
        optimizer.zero_grad()
        zero_grad_time = time.time() - zero_grad_start
        
        # Time forward pass and loss calculation
        forward_start = time.time()
        total_loss, pde_loss, ic_loss, bc_loss, loss_timings = compute_loss(Velocity, Time_residual, n_particles, n_dimension, device)
        forward_end = time.time()
        
        # Time backward pass
        backward_start = time.time()
        total_loss.backward()
        backward_time = time.time() - backward_start
        
        # Time optimizer step
        optimizer_step_start = time.time()
        optimizer.step()
        optimizer_step_time = time.time() - optimizer_step_start
        
        # Start timing "other operations" (scheduler, history logging, etc)
        other_start = time.time()
        # scheduler.step()  # Commented out in original code
        loss_history.append(total_loss.item())
        other_time = time.time() - other_start
        
        # Calculate total iteration time and any unaccounted time
        iter_end = time.time()
        iter_total = iter_end - iter_start
        
        # Accumulate timings
        acc_zero_grad_time += zero_grad_time
        acc_pde_time += loss_timings['pde_loss_calc']
        acc_ic_time += loss_timings['ic_loss_calc'] 
        acc_bc_time += loss_timings['bc_loss_calc']
        acc_backward_time += backward_time
        acc_optimizer_step_time += optimizer_step_time
        acc_other_time += other_time
        
        if epoch % print_every == 0:
            elapsed = time.time() - start_time
            
            # Calculate average times
            if epoch > 0:
                avg_zero_grad_time = acc_zero_grad_time / print_every
                avg_pde_time = acc_pde_time / print_every
                avg_ic_time = acc_ic_time / print_every
                avg_bc_time = acc_bc_time / print_every
                avg_backward_time = acc_backward_time / print_every
                avg_optimizer_step_time = acc_optimizer_step_time / print_every
                avg_other_time = acc_other_time / print_every
            else:
                avg_zero_grad_time = acc_zero_grad_time
                avg_pde_time = acc_pde_time
                avg_ic_time = acc_ic_time
                avg_bc_time = acc_bc_time
                avg_backward_time = acc_backward_time
                avg_optimizer_step_time = acc_optimizer_step_time
                avg_other_time = acc_other_time
                
            # Calculate average total time per iteration
            avg_total_per_epoch = (avg_zero_grad_time + avg_pde_time + avg_ic_time + 
                                 avg_bc_time + avg_backward_time + avg_optimizer_step_time + 
                                 avg_other_time)
            
            # Calculate measured time per epoch based on elapsed time
            if epoch > 0:
                measured_time_per_epoch = (elapsed - last_elapsed) / print_every
            else:
                measured_time_per_epoch = elapsed
                
            last_elapsed = elapsed
                
            # Print timing information
            print(f"Epoch {epoch}/{n_epochs} - Time: {elapsed:.2f}s - "
                  f"Loss: {total_loss.item():.6f}, PDE: {pde_loss.item():.6f}, "
                  f"IC: {ic_loss.item():.6f}, BC: {bc_loss.item():.6f}")
            
            print(f"  Avg Times (last {print_every} epochs):")
            print(f"    zero_grad: {avg_zero_grad_time*1000:.2f}ms, "
                  f"PDE Calc: {avg_pde_time*1000:.2f}ms, " 
                  f"IC Calc: {avg_ic_time*1000:.2f}ms, "
                  f"BC Calc: {avg_bc_time*1000:.2f}ms")
            print(f"    Backward: {avg_backward_time*1000:.2f}ms, "
                  f"Optimizer Step: {avg_optimizer_step_time*1000:.2f}ms, "
                  f"Other: {avg_other_time*1000:.2f}ms")
            print(f"    Sum of components: {avg_total_per_epoch*1000:.2f}ms, "
                  f"Measured per epoch: {measured_time_per_epoch*1000:.2f}ms")

            # Reset accumulators for the next interval
            acc_zero_grad_time = 0
            acc_pde_time = 0
            acc_ic_time = 0
            acc_bc_time = 0
            acc_backward_time = 0 
            acc_optimizer_step_time = 0
            acc_other_time = 0
            
    print(f"Training completed in {time.time() - start_time:.2f} seconds")
    return Velocity, Time_residual, loss_history

n_particles = 3
n_dimension = 2

n_layers_V = 2
n_neurons_V = 128

n_layers_T = 1
n_neurons_T = 128

device = torch.device("cpu")
Velocity = PINN2D(
    glob_dim=1,         
    feat_dim=0,        
    edge_dim=0,         
    hidden_dim=128,      
    norm_coors_scale_init=1.0,  
    cutoff_distance=1.5*torch.pi,  
    num_layers=1        
).to(device)

Time = Time_residual(n_layers_T, n_neurons_T).to(device)

optimizer = torch.optim.Adam(list(Velocity.parameters()) + list(Time.parameters()), lr=1e-3)
scheduler = StepLR(optimizer, step_size=1000, gamma=0.5)



# Velocity, Time, loss_history = train(
#     n_particles=n_particles,
#     n_dimension=n_dimension,
#     Velocity=Velocity,
#     Time_residual=Time,
#     optimizer=optimizer,
#     scheduler=scheduler,
#     device=device,
#     n_epochs=10000,
#     print_every=200
# )

# torch.save(Velocity.state_dict(), "Exp1_velocity_model.pth")
# torch.save(Time.state_dict(), "Exp1_time_model.pth")

# # Plot loss history
# plt.figure(figsize=(10, 6))
# plt.plot(loss_history)
# plt.yscale('log')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training Loss History')
# plt.grid(True)
# plt.show()

# # Add this function to check E(n) symmetry of the PINN2D network
# def check_E_n_symmetry(model, n_particles=3, n_dimension=2, n_samples=10, device="cpu"):
#     """
#     Check if the PINN2D model satisfies E(n) symmetry properties.
    
#     Args:
#         model: The PINN2D model to test
#         n_particles: Number of particles
#         n_dimension: Number of dimensions per particle
#         n_samples: Number of sample inputs to test
#         device: Device to run the test on
        
#     Returns:
#         Dictionary with error metrics for different symmetry transformations
#     """
#     print("Testing E(n) symmetry of the PINN2D network...")
#     model.eval()
    
#     # Generate random inputs
#     t = torch.ones((n_samples, 1), device=device)   # Fixed time for simplicity
#     shape = (n_samples, n_particles, n_dimension)
    
#     # Generate random coordinates in [-π, π]
#     coors = torch.rand(n_samples, n_particles*n_dimension, device=device) * 2*torch.pi - torch.pi
#     coors_reshaped = coors.reshape(n_samples, n_particles, n_dimension)
    
#     # Original forward pass
#     with torch.no_grad():
#         original_output = model(t, coors, shape)
#         original_output_reshaped = original_output.reshape(n_samples, n_particles, n_dimension)
    
#     results = {}
    
#     # 1. Test Translation Equivariance
#     print("\nTesting Translation Equivariance...")
#     translation_errors = []
    
#     for _ in range(5):  # Test with 5 different translation vectors
#         # Generate random translation vector (same for all particles in a batch)
#         trans_vec = (torch.rand(1, n_dimension, device=device) * 2 - 1) * torch.pi / 2
        
#         # Apply translation
#         translated_coors = coors_reshaped.clone()
#         for p in range(n_particles):
#             translated_coors[:, p, :] = translated_coors[:, p, :] + trans_vec
        
#         # Apply periodic boundary [−π, π]
#         translated_coors = torch.remainder(translated_coors + torch.pi, 2*torch.pi) - torch.pi
        
#         # Forward pass with translated coordinates
#         with torch.no_grad():
#             translated_output = model(t, translated_coors.reshape(n_samples, -1), shape)
#             translated_output_reshaped = translated_output.reshape(n_samples, n_particles, n_dimension)
        
#         # Apply the same translation to original output (no need to apply periodic boundary here)
#         expected_output = original_output_reshaped.clone()
#         for p in range(n_particles):
#             expected_output[:, p, :] = expected_output[:, p, :] + trans_vec
        
#         # Compute error
#         error = torch.norm(translated_output_reshaped - expected_output) / torch.norm(expected_output)
#         translation_errors.append(error.item())
#         print(f"  Translation vector: {trans_vec.squeeze().cpu().numpy()}, Error: {error.item():.6f}")
    
#     results['translation_error'] = sum(translation_errors) / len(translation_errors)
    
#     # 2. Test Rotation Equivariance (for 2D case)
#     if n_dimension == 2:
#         print("\nTesting Rotation Equivariance...")
#         rotation_errors = []
        
#         for angle in [0, torch.pi/4, torch.pi/2, torch.pi, 3*torch.pi/2]:  # Test with different angles
#             # Create 2D rotation matrix
#             cos_a, sin_a = torch.cos(angle), torch.sin(angle)
#             rotation_matrix = torch.tensor([
#                 [cos_a, -sin_a],
#                 [sin_a, cos_a]
#             ], device=device)
            
#             # Apply rotation
#             rotated_coors = coors_reshaped.clone()
#             for b in range(n_samples):
#                 for p in range(n_particles):
#                     rotated_coors[b, p, :] = torch.matmul(rotation_matrix, rotated_coors[b, p, :])
            
#             # Apply periodic boundary [−π, π]
#             rotated_coors = torch.remainder(rotated_coors + torch.pi, 2*torch.pi) - torch.pi
            
#             # Forward pass with rotated coordinates
#             with torch.no_grad():
#                 rotated_output = model(t, rotated_coors.reshape(n_samples, -1), shape)
#                 rotated_output_reshaped = rotated_output.reshape(n_samples, n_particles, n_dimension)
            
#             # Apply the same rotation to original output
#             expected_output = original_output_reshaped.clone()
#             for b in range(n_samples):
#                 for p in range(n_particles):
#                     expected_output[b, p, :] = torch.matmul(rotation_matrix, expected_output[b, p, :])
            
#             # Compute error
#             error = torch.norm(rotated_output_reshaped - expected_output) / torch.norm(expected_output)
#             rotation_errors.append(error.item())
#             print(f"  Rotation angle: {angle.item():.4f}, Error: {error.item():.6f}")
        
#         results['rotation_error'] = sum(rotation_errors) / len(rotation_errors)
    
#     # 3. Test Reflection Equivariance
#     print("\nTesting Reflection Equivariance...")
#     reflection_errors = []
    
#     # Reflect across each coordinate axis
#     for axis in range(n_dimension):
#         # Create reflection matrix (identity matrix with sign flip in one element)
#         reflection_matrix = torch.eye(n_dimension, device=device)
#         reflection_matrix[axis, axis] = -1
        
#         # Apply reflection
#         reflected_coors = coors_reshaped.clone()
#         for b in range(n_samples):
#             for p in range(n_particles):
#                 reflected_coors[b, p, :] = torch.matmul(reflection_matrix, reflected_coors[b, p, :])
        
#         # Apply periodic boundary [−π, π]
#         reflected_coors = torch.remainder(reflected_coors + torch.pi, 2*torch.pi) - torch.pi
        
#         # Forward pass with reflected coordinates
#         with torch.no_grad():
#             reflected_output = model(t, reflected_coors.reshape(n_samples, -1), shape)
#             reflected_output_reshaped = reflected_output.reshape(n_samples, n_particles, n_dimension)
        
#         # Apply the same reflection to original output
#         expected_output = original_output_reshaped.clone()
#         for b in range(n_samples):
#             for p in range(n_particles):
#                 expected_output[b, p, :] = torch.matmul(reflection_matrix, expected_output[b, p, :])
        
#         # Compute error
#         error = torch.norm(reflected_output_reshaped - expected_output) / torch.norm(expected_output)
#         reflection_errors.append(error.item())
#         print(f"  Reflection axis: {axis}, Error: {error.item():.6f}")
    
#     results['reflection_error'] = sum(reflection_errors) / len(reflection_errors)
    
#     # 4. Test Permutation Equivariance (symmetry for particle exchanges)
#     print("\nTesting Permutation Equivariance...")
#     permutation_errors = []
    
#     # Test multiple permutations
#     num_permutations = min(5, math.factorial(n_particles))
    
#     for _ in range(num_permutations):
#         # Generate a random permutation of the particles
#         perm_indices = torch.randperm(n_particles, device=device)
        
#         # Apply permutation to input coordinates - reshape and permute
#         perm_coors = coors_reshaped.clone()
#         perm_coors = perm_coors[:, perm_indices, :]
        
#         # Run model on permuted input
#         with torch.no_grad():
#             perm_output = model(t, perm_coors.reshape(n_samples, -1), shape)
#             perm_output_reshaped = perm_output.reshape(n_samples, n_particles, n_dimension)
        
#         # Apply the same permutation to original output for comparison
#         expected_output = original_output_reshaped.clone()
#         expected_output = expected_output[:, perm_indices, :]
        
#         # Compute error
#         error = torch.norm(perm_output_reshaped - expected_output) / torch.norm(expected_output)
#         permutation_errors.append(error.item())
#         print(f"  Permutation {perm_indices.cpu().numpy()}: Error: {error.item():.6f}")
    
#     results['permutation_error'] = sum(permutation_errors) / len(permutation_errors)
    
#     # Summary
#     print("\nE(n) Symmetry Test Results:")
#     print(f"  Average Translation Error: {results['translation_error']:.6f}")
#     if 'rotation_error' in results:
#         print(f"  Average Rotation Error: {results['rotation_error']:.6f}")
#     print(f"  Average Reflection Error: {results['reflection_error']:.6f}")
#     print(f"  Average Permutation Error: {results['permutation_error']:.6f}")
    
#     if (results['translation_error'] < 1e-5 and 
#         (n_dimension != 2 or results.get('rotation_error', 0) < 1e-5) and 
#         results['reflection_error'] < 1e-5 and
#         results['permutation_error'] < 1e-5):
#         print("\nPassed: The PINN2D network satisfies E(n) symmetry properties with high accuracy!")
#     else:
#         print("\nWarning: The network may not fully satisfy E(n) symmetry properties.")
#         print("This could be due to numerical errors or network design issues.")
    
#     return results

# # Add after model training to test symmetry
# def test_trained_model_symmetry(model_path="/Users/xuhengyuan/Downloads/PINN/Exp1_velocity_model.pth"):
#     """
#     Load a trained model and test its E(n) symmetry properties
#     """
#     print(f"Loading model from {model_path}...")
    
#     # Initialize model with same parameters
#     model = PINN2D(
#         glob_dim=1,         
#         feat_dim=0,        
#         edge_dim=0,         
#         hidden_dim=128,      
#         norm_coors_scale_init=1.0,  
#         cutoff_distance=1.5*torch.pi,  
#         num_layers=1        
#     ).to(device)
    
#     # Load saved parameters
#     model.load_state_dict(torch.load(model_path, map_location=device))
    
#     # Check E(n) symmetry
#     symmetry_results = check_E_n_symmetry(model, n_particles=3, n_dimension=2, device=device)
    
#     return symmetry_results

# # Add this at the end of the file to run the tests if needed
# if __name__ == "__main__":
#     # Either run with newly trained model
#     # test_trained_model_symmetry("Exp1_velocity_model.pth")
    
#     # Or with a specific model file
#     # test_trained_model_symmetry("/path/to/your/model.pth")
    
#     # Or check the fresh model's inherent symmetry (before training)
#     device = torch.device("cpu")
#     fresh_model = PINN2D(
#         glob_dim=1,         
#         feat_dim=0,        
#         edge_dim=0,         
#         hidden_dim=128,      
#         norm_coors_scale_init=1.0,  
#         cutoff_distance=1.5*torch.pi,  
#         num_layers=1        
#     ).to(device)
    
#     check_E_n_symmetry(fresh_model, device=device)