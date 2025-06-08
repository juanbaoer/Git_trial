import torch
import time
from GNN_exp import lennard_jones_energy_full_original, lennard_jones_energy_full

def run_benchmark(device_str="cpu"):
    device = torch.device(device_str)
    print(f"Running benchmark on {device}")

    # Parameters for the benchmark
    B = 1024  # Batch size
    N = 64   # Number of particles
    D = 2    # Number of dimensions
    box_length = 2 * torch.pi
    eps = 1.0
    rm = 1.0
    r_switch = 0.9 
    r_cutoff = 2.5
    
    # Generate random input data
    x = torch.rand(B, N * D, device=device) * box_length  # Particle coordinates

    # Warm-up runs (important for GPU)
    for _ in range(5):
        _ = lennard_jones_energy_full_original(x, N, D, box_length, eps, rm, r_switch, r_cutoff, remove_diagonal=True)
        _ = lennard_jones_energy_full(x, N, D, box_length, eps, rm, r_switch, r_cutoff, remove_diagonal=True)
        if device_str == "cuda":
            torch.cuda.synchronize()

    # Benchmark parameters
    num_runs = 20

    # --- Test case 1: remove_diagonal = True ---
    print("\n--- Testing with remove_diagonal = True ---")
    
    # Benchmark original function
    start_time = time.perf_counter()
    for _ in range(num_runs):
        energy_original_diag_true = lennard_jones_energy_full_original(x, N, D, box_length, eps, rm, r_switch, r_cutoff, remove_diagonal=True)
        if device_str == "cuda":
            torch.cuda.synchronize()
    end_time = time.perf_counter()
    avg_time_original_diag_true = (end_time - start_time) / num_runs
    print(f"Original function (remove_diagonal=True): {avg_time_original_diag_true:.6f} seconds per run")

    # Benchmark optimized function
    start_time = time.perf_counter()
    for _ in range(num_runs):
        energy_optimized_diag_true = lennard_jones_energy_full(x, N, D, box_length, eps, rm, r_switch, r_cutoff, remove_diagonal=True)
        if device_str == "cuda":
            torch.cuda.synchronize()
    end_time = time.perf_counter()
    avg_time_optimized_diag_true = (end_time - start_time) / num_runs
    print(f"Optimized function (remove_diagonal=True): {avg_time_optimized_diag_true:.6f} seconds per run")

    # Correctness check
    if torch.allclose(energy_original_diag_true, energy_optimized_diag_true, atol=1e-6):
        print("Correctness check (remove_diagonal=True): PASSED")
    else:
        print("Correctness check (remove_diagonal=True): FAILED")
        print("Original output:", energy_original_diag_true[0,:5].squeeze().tolist()) # Print first 5 for brevity
        print("Optimized output:", energy_optimized_diag_true[0,:5].squeeze().tolist())
        print("Difference:", torch.abs(energy_original_diag_true - energy_optimized_diag_true).max())


    # --- Test case 2: remove_diagonal = False ---
    print("\n--- Testing with remove_diagonal = False ---")
    # Benchmark original function
    start_time = time.perf_counter()
    for _ in range(num_runs):
        energy_original_diag_false = lennard_jones_energy_full_original(x, N, D, box_length, eps, rm, r_switch, r_cutoff, remove_diagonal=False)
        if device_str == "cuda":
            torch.cuda.synchronize()
    end_time = time.perf_counter()
    avg_time_original_diag_false = (end_time - start_time) / num_runs
    print(f"Original function (remove_diagonal=False): {avg_time_original_diag_false:.6f} seconds per run")

    # Benchmark optimized function
    start_time = time.perf_counter()
    for _ in range(num_runs):
        energy_optimized_diag_false = lennard_jones_energy_full(x, N, D, box_length, eps, rm, r_switch, r_cutoff, remove_diagonal=False)
        if device_str == "cuda":
            torch.cuda.synchronize()
    end_time = time.perf_counter()
    avg_time_optimized_diag_false = (end_time - start_time) / num_runs
    print(f"Optimized function (remove_diagonal=False): {avg_time_optimized_diag_false:.6f} seconds per run")

    # Correctness check
    if torch.allclose(energy_original_diag_false, energy_optimized_diag_false, atol=1e-6): # Increased atol slightly due to r_ii=0 path
        print("Correctness check (remove_diagonal=False): PASSED")
    else:
        print("Correctness check (remove_diagonal=False): FAILED")
        print("Original output:", energy_original_diag_false[0,:5].squeeze().tolist())
        print("Optimized output:", energy_optimized_diag_false[0,:5].squeeze().tolist())
        print("Difference:", torch.abs(energy_original_diag_false - energy_optimized_diag_false).max())
        
    print("\nBenchmark finished.")

if __name__ == "__main__":
    run_benchmark(device_str="cpu")
    if torch.cuda.is_available():
        run_benchmark(device_str="cuda")
    else:
        print("CUDA not available, skipping GPU benchmark.")
