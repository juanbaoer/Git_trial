import torch
import torch.nn as nn
from torch.autograd import grad
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import torch.nn.init as init
import time
from torch.optim.lr_scheduler import StepLR
from torchdiffeq import odeint
import math



'''
Define the network
Initialization need n_particle, n_dimension, num_hidden_layers, num_neurons_per_layer
Forward required arguments: (t, samples_prior)
t: (batch, 1)
samples_prior: (batch, n_particle*n_dimension)
output: (batch, n_particle*n_dimension)
'''
class PINN2D(nn.Module):
    def __init__(self, n_particle, n_dimension, num_hidden_layers=6, num_neurons_per_layer=128):
        super().__init__()
        
        # Input layer: new input is 2ND+1 after transformation
        self.input_layer = nn.Linear(2*n_particle*n_dimension+1, num_neurons_per_layer)

        # Hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Linear(num_neurons_per_layer, num_neurons_per_layer) 
            for _ in range(num_hidden_layers)
        ])
        
        # Output layer: output ND dimensions
        self.output_layer = nn.Linear(num_neurons_per_layer, n_particle*n_dimension)

        # Activation function
        self.activation = nn.Tanh()
        
        # Initialize weights using Xavier initialization
        for layer in self.hidden_layers:
            nn.init.xavier_normal_(layer.weight.data)
            nn.init.zeros_(layer.bias.data)

    def forward(self, t, samples_prior):
        # Step 1: Transform samples_prior(high dimension) using sin/cos by setting every coordinates to its sin/cos 
        inputs = torch.cat([torch.cos(samples_prior), torch.sin(samples_prior),t], dim=1)

        # Step 3: Pass through network
        outputs = self.activation(self.input_layer(inputs))
        for layer in self.hidden_layers:
            outputs = self.activation(layer(outputs)) + outputs  
 
        # Step 4: Pass through output layer (4D output)
        outputs = self.output_layer(outputs)  

        # Step 6: Stack to return 2D output
        return outputs
    
'''
Define separate part for loss function
'''
def internal_energy(samples):
    # Input: samples (B, N*D)
    # Output: internal energy (B, 1)    
    N = samples.shape[1]
    x = samples[:,0:N//2]
    y = samples[:,N//2:N]
    # Compute internal energy
    energy =-(torch.cos(x) + torch.cos(y)) 
    energy = energy.sum(dim = 1)
    energy = energy.view(-1,1)
    return energy

def vector_distance(samples,Velocity_field,internal_energy_):
    # Samples is a tensor of shape (B, N*D）
    # Velocity_field is a tensor of shape (B, N*D)
    # Internal_energy_ is a tensor of shape (B,1)
    # Expected result is a tensor of shape (B,1)
    gradient = torch.autograd.grad(internal_energy_, samples, grad_outputs=torch.ones_like(internal_energy_),create_graph=True)[0]
    # Compute the distance
    result = torch.sum(gradient * Velocity_field,dim = 1)
    result = result.unsqueeze(dim = 1)
    return result

def coprod(y):
    n = y.shape[0]
    
    sum_y = y.sum(dim=0)  # shape (1,)
    
    sum_y_sq = (y**2).sum(dim=0)  # shape (1,)
    
    approx = (sum_y**2 - sum_y_sq) / (n * (n - 1))
    return approx

# Define the ODE dynamics function using the PINN2D network
def ode_dynamics(t, x, model):
    # Expand scalar t to a tensor of shape (batch, 1)
    t_batch = t * torch.ones(x.size(0), 1, device=x.device)
    dxdt = model(t_batch, x)
    return dxdt

def dynamics(t, x):
    return ode_dynamics(t, x, model)

def differentiable_trapz(y, x):
    """
    Differentiable trapezoidal integration in PyTorch.
    
    Parameters:
    - y: tensor of shape (T,) or (T, B) — values to integrate (e.g., loss at each time)
    - x: tensor of shape (T,) — time values (non-uniform supported)
    
    Returns:
    - Integrated result (scalar if y is 1D, else shape (B,))
    """
    # Ensure correct shape and device
    assert y.shape[0] == x.shape[0], "Time axis must match"
    
    dx = x[1:] - x[:-1]           # shape: (T-1,)
    avg_y = 0.5 * (y[:-1] + y[1:])  # shape: (T-1,) or (T-1, B)
    
    # Broadcasting dx with avg_y
    while dx.dim() < avg_y.dim():
        dx = dx.unsqueeze(-1)
    
    return torch.sum(avg_y * dx, dim=0)


def trace(fun, data, hutchinson_samples = None, other_data = None):
    """
    fun input: (B*D, B*D')
    data: B*D
    other_data: B*D'
    hutchinson_samples: N*D

    output: B*1
    """
    B, D = *data.shape,
    z = torch.zeros_like(data[..., :1], requires_grad=True) #B*1
    
    with torch.set_grad_enabled(True):
        if hutchinson_samples is None:
            hutchinson_samples = torch.eye(D, device = data.device) * math.sqrt(D)

        N = hutchinson_samples.shape[-2]
    
        data_expand = (data[..., None, :] + z[..., None, :] * hutchinson_samples).reshape(-1, D) #(B*N)*D
        if other_data is None:
            other_data_expand = None
            y = fun(data_expand) #(B*N)*D
        else:
            D_other = other_data.size()[-1]
            other_data_expand = other_data.unsqueeze(-2).expand(*other_data.shape[:-1], N, D_other).reshape(-1, D_other) #(B*N)*D'
            y = fun(data_expand, other_data_expand) #(B*N)*D

        return torch.autograd.grad(torch.sum(y.reshape(-1, N, D) * hutchinson_samples) / N, z, retain_graph=False)[0]

def L1(internal_energy, vector_distance, trace_vector):
    '''
    internal_energy: (B,1)
    vector_distance: (B,1)
    term3: (m,B,1), m = number of random vectors
    output: (1,)
    '''
    result = internal_energy + vector_distance - trace_vector # shape is (m,B,1)
    # Take the coprod over the m dimension
    result = coprod(result)

    return result.mean(dim=0)

def L2(internal_energy, vector_distance, trace_average):
    '''
    internal_energy: (B,1)
    vector_distance: (B,1)
    trace_average: (B,1)
    output: (1,)
    '''
    C_t = -internal_energy.sum(dim=0)
    product_term = (C_t+internal_energy)/(internal_energy.shape[0]-1)
    
    result = 2*(internal_energy + vector_distance - trace_average) * product_term
    
    return result.mean(dim=0)

def L3(internal_energy):
    '''
    internal_energy: (B,1)
    output: (1,)
    '''
    term = -internal_energy
    result = coprod(term)
    return result

def pde_los(x_t, t_values, t_weights, u, model):
    loss_t_list = []
    l3_values = []
    c_t_values = []  # Track C_t values per time step
    
    for i, t_value in enumerate(t_values):
        samples = x_t[i]  # shape (B, D)
        t = t_value * torch.ones(samples.size(0), 1, device=x_t.device)
        
        Velocity_field = model(t,samples)        
        
        internal_energy_val = internal_energy(samples)
        vector_distance_val = vector_distance(samples, Velocity_field, internal_energy_val)
        
        u = (torch.rand(5, n_dimension*n_particle) < 0.5).float() * 2 - 1
        def network_fn(samples):
            fixed_t = t_value*torch.ones(samples.size(0), 1, device=samples.device)
            return model(fixed_t, samples)
        
        trace_average = trace(network_fn, samples, hutchinson_samples=u)

        trace_list = [trace(network_fn, samples, hutchinson_samples=u[i:i+1, :])
                for i in range(u.shape[0])]

        trace_vector = torch.stack(trace_list, dim=0)

        # Calculate individual loss components
        l1_val = L1(internal_energy_val, vector_distance_val, trace_vector)
        l2_val = L2(internal_energy_val, vector_distance_val, trace_average)
        l3_val = L3(internal_energy_val)
        
        # Calculate C_t = sqrt(L3)
        c_t_val = torch.sqrt(l3_val)
        
        # Store values
        l3_values.append(l3_val)
        c_t_values.append(c_t_val)
        
        # Sum for total loss
        loss_t_val = l1_val + l2_val + l3_val
        loss_t_list.append(loss_t_val)

    # Stack all timestep losses
    loss_t_tensor = torch.stack(loss_t_list)
    l3_tensor = torch.stack(l3_values)
    c_t_tensor = torch.stack(c_t_values)
    
    # Compute overall PDE loss
    loss_pde = (1/2) * torch.sum(loss_t_tensor * t_weights)
    
    # Return loss and C_t values
    return loss_pde, (t_values.detach().cpu().numpy(), 
                     l3_tensor.detach().cpu().numpy(),
                     c_t_tensor.detach().cpu().numpy())

def compute_loss(u, t_values, t_weights, Velocity, n_dimension, n_particles, device="cuda"):
    ''' 
    u is the random vector from Rademacher distribution (N, n_particles*n_dimension)
    t_values is the time values (T, 1)
    Velocity is the instance of PINN network
    '''
    # Sample collocation points for PDE residual
    n_pde_points = 1024  # Number of points inside domain
    # Generate samples in the latent space
    samples_prior = torch.rand(n_pde_points, n_particles*n_dimension, device=device,requires_grad=True) * 2*torch.pi - torch.pi
    # Solve the ODE to get the trajectory
    x_t = odeint(dynamics, samples_prior, t_values)
    # Compute the PDE loss
    pde_loss, ct_data = pde_los(x_t, t_values, t_weights, u, Velocity)
    
    # Check for negative PDE loss
    if pde_loss < 0:
        print(f"WARNING: Negative PDE loss detected! pde_loss={pde_loss.item()}")
        l3_values = ct_data[1]
        print(f"L3 values: {l3_values}")
        print(f"C_t values: {ct_data[2]}")
        assert pde_loss >= 0, "PDE loss must be non-negative"
    
    # Rest of the compute_loss function remains the same
    # Initial condition points at t=0.1 (smallest valid t)
    n_ic_points = 256
    samples_prior_ic = torch.rand(n_ic_points, n_particles*n_dimension, device=device)*2*torch.pi-torch.pi    # Range [-10, 10]
    t_ic = torch.zeros(n_ic_points, 1, device=device)      # t = 0.1 (minimum time)
    u_ic_pred = Velocity(t_ic, samples_prior_ic)
    u_ic_true = 1
    ic_loss = torch.sum((u_ic_pred - u_ic_true)**2,dim = 1)
    ic_loss = torch.mean(ic_loss)
    
    # Boundary conditions
    n_bc_points = 256
    
    # Left boundary (x = -10)
    samples_prior_left = torch.rand(n_bc_points, n_particles*n_dimension, device=device)* 2*torch.pi - torch.pi 
    samples_prior_left[:, 0] = -torch.pi   # Range [-10, 10]
    t_bc_left = torch.rand(n_bc_points, 1, device=device)  # Range [0.1, 1]
    u_bc_left_pred = Velocity(t_bc_left, samples_prior_left)
    u_bc_left_true = torch.zeros_like(u_bc_left_pred, device=device)
    bc_loss_left = torch.sum((u_bc_left_pred - u_bc_left_true)**2,dim = 1)
    bc_loss_left = torch.mean(bc_loss_left)
    
    # Right boundary (x = 10)
    samples_prior_right = torch.rand(n_bc_points, n_particles*n_dimension, device=device)* 2*torch.pi - torch.pi 
    samples_prior_right[:, 0] = torch.pi
    t_bc_right = torch.rand(n_bc_points, 1, device=device)   
    u_bc_right_pred = Velocity(t_bc_right, samples_prior_right)
    u_bc_right_true = torch.zeros_like(u_bc_right_pred, device=device)
    bc_loss_right = torch.sum((u_bc_right_pred - u_bc_right_true)**2,dim = 1)
    bc_loss_right = torch.mean(bc_loss_right)
    
    # Bottom boundary (y = -10)
    samples_prior_bottom = torch.rand(n_bc_points, n_particles*n_dimension, device=device)* 2*torch.pi - torch.pi 
    samples_prior_bottom[:, 1] = -torch.pi  # Range [-10, 10]
    t_bc_bottom = torch.rand(n_bc_points, 1, device=device) 
    u_bc_bottom_pred = Velocity(t_bc_bottom, samples_prior_bottom)
    u_bc_bottom_true = torch.zeros_like(u_bc_bottom_pred, device=device)
    bc_loss_bottom = torch.sum((u_bc_bottom_pred - u_bc_bottom_true)**2,dim = 1)
    bc_loss_bottom = torch.mean(bc_loss_bottom)
    
    # Top boundary (y = 10)
    samples_prior_top = torch.rand(n_bc_points, n_particles*n_dimension, device=device)* 2*torch.pi - torch.pi
    samples_prior_top[:, 1] = torch.pi
    t_bc_top = torch.rand(n_bc_points, 1, device=device) 
    u_bc_top_pred = Velocity(t_bc_top, samples_prior_top)
    u_bc_top_true = torch.zeros_like(u_bc_top_pred, device=device)
    bc_loss_top = torch.sum((u_bc_top_pred - u_bc_top_true)**2,dim = 1)
    bc_loss_top = torch.mean(bc_loss_top)
    
    # Total boundary loss
    bc_loss = bc_loss_left + bc_loss_right + bc_loss_bottom + bc_loss_top
    
    # Total loss
    total_loss = pde_loss + ic_loss + bc_loss
    
    return total_loss, pde_loss, ic_loss, bc_loss, ct_data

# Save C_t data
def save_ct_data(ct_data, filename):
    """
    Save C_t data to a file.
    
    Parameters:
    - ct_data: Tuple containing (t_values, l3_values, c_t_values)
    - filename: Name of the file to save data to
    """
    import numpy as np
    t_values, l3_values, c_t_values = ct_data
    np.savez(filename, 
             t_values=t_values, 
             l3_values=l3_values,
             c_t_values=c_t_values)
    print(f"C_t data saved to {filename}")

# Plot C_t data
def plot_ct_data(ct_data, title="C_t Values Over Time", savefig=None):
    """
    Plot C_t values over time.
    
    Parameters:
    - ct_data: Tuple containing (t_values, l3_values, c_t_values)
    - title: Plot title
    - savefig: If provided, save the figure to this filename
    """
    t_values, l3_values, c_t_values = ct_data
    
    plt.figure(figsize=(10, 6))
    plt.plot(t_values, c_t_values, 'o-', linewidth=2, color='purple')
    plt.xlabel('Time')
    plt.ylabel('C_t Value')
    plt.title(title)
    plt.grid(True)
    
    if savefig:
        plt.savefig(savefig)
    plt.show()
    
    # Optionally plot L3 as well
    plt.figure(figsize=(10, 6))
    plt.plot(t_values, l3_values, 's-', linewidth=2, color='green')
    plt.xlabel('Time')
    plt.ylabel('L3 Value')
    plt.title('L3 Values Over Time')
    plt.grid(True)
    
    if savefig:
        plt.savefig(f"{savefig}_L3.png")
    plt.show()

# Modified train function to focus on C_t values
def train(t_values, t_weights, model, n_dimension, n_particle, device, optimizer, scheduler, n_epochs=10000, print_every=1000, save_ct=True):
    start_time = time.time()
    loss_history = []
    pde_loss_history = []
    ic_loss_history = []
    bc_loss_history = []
    ct_history = []
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        u = (torch.rand(3, 4) < 0.5).float() * 2 - 1
        
        # Get loss and C_t values
        total_loss, pde_loss, ic_loss, bc_loss, ct_data = compute_loss(
            u, t_values, t_weights, model, n_dimension, n_particle, device
        )

        # Save C_t data at regular intervals
        if epoch % (print_every * 5) == 0 or epoch == n_epochs - 1:
            ct_history.append((epoch, ct_data))
            save_ct_data(ct_data, f"/Users/xuhengyuan/Downloads/PINN/ct_data_epoch_{epoch}.npz")
            if epoch == n_epochs - 1:  # Plot on the final epoch
                plot_ct_data(ct_data, title=f"C_t Values at Epoch {epoch}", 
                            savefig=f"/Users/xuhengyuan/Downloads/PINN/ct_values_epoch_{epoch}")
        
        # Store loss components
        loss_history.append(total_loss.item())
        pde_loss_history.append(pde_loss.item())
        ic_loss_history.append(ic_loss.item())
        bc_loss_history.append(bc_loss.item())
        
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        
        if epoch % print_every == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch}/{n_epochs} - Time: {elapsed:.2f}s - "
                  f"Loss: {total_loss.item():.6f}, PDE: {pde_loss.item():.6f}, "
                  f"IC: {ic_loss.item():.6f}, BC: {bc_loss.item():.6f}")
            
            # Print C_t values at the current epoch
            if len(ct_history) > 0:
                _, latest_ct_data = ct_history[-1]
                latest_ct_values = latest_ct_data[2]
                print(f"Current C_t values: {latest_ct_values}")
    
    print(f"Training completed in {time.time() - start_time:.2f} seconds")
    
    # Save final C_t data
    if save_ct and ct_history:
        final_epoch, final_ct_data = ct_history[-1]
        save_ct_data(final_ct_data, f"/Users/xuhengyuan/Downloads/PINN/ct_data_final.npz")
        plot_ct_data(final_ct_data, title="Final C_t Values")

        # Also save loss history
        np.savez(f"/Users/xuhengyuan/Downloads/PINN/training_history.npz",
                total_loss=np.array(loss_history),
                pde_loss=np.array(pde_loss_history),
                ic_loss=np.array(ic_loss_history),
                bc_loss=np.array(bc_loss_history))

    return model, loss_history

n_particle = 1
n_dimension = 2
device = torch.device("mps" if torch.cuda.is_available() else "cpu")
model = PINN2D(n_particle, n_dimension).to(device)
t_start = 0
t_end = 1
n_timepoints = 3
t_values = torch.linspace(t_start, t_end, n_timepoints, device=device)
t_weights = torch.ones_like(t_values) / n_timepoints

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = StepLR(optimizer, step_size=200, gamma=0.99)
trained_model, loss_history = train(t_values, t_weights, model, n_dimension, n_particle, device, optimizer,scheduler,n_epochs=10000, print_every=200)

# Save the trained model
torch.save(trained_model.state_dict(), 'Vonmise_Ensemble_fitting.pth')

#log the model information
print(f"Model architecture: {model}")
# Save the loss history

# Plot loss history
plt.figure(figsize=(10, 6))
plt.plot(loss_history)
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss History')
plt.grid(True)
plt.savefig('loss_history.png')


