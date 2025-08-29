import torch
import numpy as np
import time
import os
import ray

# ==============================================================================
# SETUP AND CONFIGURATION
# ==============================================================================
device = torch.device("cuda")
from main import create_tsp_instance, get_hamiltonian_pauli_strings

# ==============================================================================
# RAY ACTOR FOR STATE VECTOR SHARD
# ==============================================================================
@ray.remote(num_gpus=1)
class StateVectorShard:
    def __init__(self, shard_index, total_shards, n_qubits):
        self.shard_index = shard_index
        self.n_qubits = n_qubits
        self.dim = 2**n_qubits
        self.shard_size = self.dim // total_shards
        
        shard_data = np.ones(self.shard_size, dtype=np.complex128) / np.sqrt(self.dim)
        self.psi_shard = torch.tensor(shard_data, dtype=torch.cfloat).to(device)
        print(f"✅ Actor {self.shard_index}: Shard created on {device} with size {self.psi_shard.shape}")

    def get_shard(self):
        return self.psi_shard

    def apply_pauli_evolution_chunk(self, pauli_defs_chunk, coeffs_chunk, gamma, remote_shards):
        psi_local = self.psi_shard
        
        coeffs_chunk_np = np.array(coeffs_chunk, dtype=np.float64)

        for k in range(len(coeffs_chunk_np)):
            theta_np = coeffs_chunk_np[k] * float(gamma)
            
            # *** CORRECTION IS HERE ***
            # Convert theta to a PyTorch tensor before using it in torch functions
            theta = torch.tensor(theta_np, device=device)
            
            # This is a conceptual placeholder for the complex distributed operation
            P_psi = psi_local * -1 
            
            psi_local = torch.cos(theta) * psi_local - 1j * torch.sin(theta) * P_psi
            
        self.psi_shard = psi_local
        return self.shard_index

# ==============================================================================
# MAIN DISTRIBUTED EXECUTION BLOCK
# ==============================================================================
def main():
    print("--- Starting Distributed Quantum Simulation ---")
    ray.init() 

    NUM_CITIES = int(os.environ.get("NUM_CITIES", 5))
    n_qubits = NUM_CITIES**2
    P_LAYERS = 1

    print(f"Simulating TSP for {NUM_CITIES} cities ({n_qubits} qubits)...")
    
    dist_matrix = create_tsp_instance(NUM_CITIES)
    pauli_terms, _ = get_hamiltonian_pauli_strings(dist_matrix)
    qaoa_params = np.random.rand(2 * P_LAYERS) * np.pi
    
    num_nodes = len(ray.nodes())
    print(f"Ray cluster detected with {num_nodes} nodes.")
    
    shards = [StateVectorShard.remote(i, num_nodes, n_qubits) for i in range(num_nodes)]
    
    pauli_defs = [(term[1], float(term[0])) for term in pauli_terms]
    pauli_chunks = np.array_split(pauli_defs, num_nodes)

    start_time = time.time()
    for p in range(P_LAYERS):
        gamma = qaoa_params[p]
        
        futures = []
        for i in range(num_nodes):
            defs_chunk = [item[0] for item in pauli_chunks[i]]
            coeffs_chunk = [item[1] for item in pauli_chunks[i]]
            
            future = shards[i].apply_pauli_evolution_chunk.remote(defs_chunk, coeffs_chunk, gamma, shards)
            futures.append(future)
            
        ray.get(futures)
        print(f"Layer {p+1} cost evolution complete.")

    end_time = time.time()
    
    print(f"\nDistributed simulation complete in {end_time - start_time:.4f} seconds.")
    
    total_norm_sq = 0
    shard_data_refs = [shard.get_shard.remote() for shard in shards]
    for shard_data in ray.get(shard_data_refs):
        total_norm_sq += torch.sum(torch.abs(shard_data)**2).item()
        
    print(f"Final state vector norm: {np.sqrt(total_norm_sq):.4f}")
    print("--- Simulation Finished ---")

if __name__ == "__main__":
    main()
